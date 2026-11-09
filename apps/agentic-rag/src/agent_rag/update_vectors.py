"""Append local BGE vectors through the configured GPU Python; parent is stdlib-only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_rag.dense_backend import (
    MAX_LENGTH,
    _checked_json,
    _ordered_text_sha,
    _read_config,
    _read_jsonl,
    _sha_file,
    _validate_cache_metadata,
    _validate_index_rows,
    _validate_model_fingerprint,
    _validate_vector_file,
)

_OUTPUT_FILES = ("vectors.npy", "vectors.json", "dense-config.json")
_STAGES = {
    "validating_base",
    "index_validated",
    "loading_encoder",
    "encoding",
    "writing_vectors",
    "complete",
}
_WORKER_ERROR_CODES = {
    "DENSE_EXTENSION_WORKER_FAILED",
    "DENSE_DOCUMENT_TOO_LONG",
    "DENSE_RUNTIME_BINDING_MISMATCH",
    "DENSE_TOKENIZER_BINDING_MISMATCH",
    "DENSE_INDEX_SHA_MISMATCH",
    "DENSE_INDEX_PREFIX_MISMATCH",
    "DENSE_METADATA_SHA_MISMATCH",
    "DENSE_MODEL_SHA_MISMATCH",
    "DENSE_VECTOR_SHA_MISMATCH",
    "DENSE_CACHE_ORDER_TEXT_MISMATCH",
    "INVALID_DENSE_CONFIG",
    "INVALID_DENSE_INDEX",
    "INVALID_DENSE_JSONL",
    "INVALID_DENSE_CACHE_BINDING",
    "INVALID_DENSE_TOKEN_AUDIT",
    "INVALID_DENSE_MODEL_BINDING",
    "INVALID_DENSE_MODEL_FILE",
    "INVALID_DENSE_VECTOR_SHAPE",
    "INVALID_DENSE_VECTOR_NORMALIZATION",
    "INVALID_DENSE_EXTENSION_ENCODING_COUNT",
    "INVALID_DENSE_EXTENSION_DESTINATION",
    "INVALID_DENSE_EXTENSION_RESULT",
}
_ERROR_TYPES = (ValueError, RuntimeError, OSError, ImportError, Exception)


def _extension_timeout(config: dict) -> float:
    timeout = config.get("extension_timeout_seconds", 600)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError("INVALID_DENSE_EXTENSION_TIMEOUT_CONFIG")
    return timeout


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_progress(
    destination: Path, stage: str, *, added_units: int = 0, encoded_units: int = 0
) -> None:
    if stage not in _STAGES:
        raise ValueError("INVALID_DENSE_EXTENSION_STAGE")
    _atomic_json(
        destination / "progress.json",
        {"stage": stage, "added_units": added_units, "encoded_units": encoded_units},
    )


def _safe_stderr_tail(stderr: str | bytes | None) -> str:
    raw = stderr.decode("utf-8", errors="replace") if isinstance(stderr, bytes) else stderr
    safe = []
    for line in (raw or "")[-65536:].splitlines():
        try:
            row = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            not isinstance(row, dict)
            or row.get("event") != "dense_extension_diagnostic"
            or row.get("error_code") not in _WORKER_ERROR_CODES
            or row.get("error_type") not in {kind.__name__ for kind in _ERROR_TYPES}
        ):
            continue
        safe.append(json.dumps({key: row[key] for key in ("event", "error_code", "error_type")}))
    return ("\n".join(safe) + ("\n" if safe else "")).encode("utf-8")[-8192:].decode("utf-8")


def _record_failure(
    destination: Path,
    error: str,
    timeout: float,
    stderr: str | bytes | None = None,
    returncode: int | None = None,
) -> None:
    diagnostic = {
        "error": error,
        "timeout_seconds": timeout,
        "returncode": returncode,
        "stderr_tail": _safe_stderr_tail(stderr),
    }
    try:
        progress = json.loads((destination / "progress.json").read_text(encoding="utf-8"))
        if isinstance(progress, dict) and progress.get("stage") in _STAGES:
            diagnostic["stage"] = progress["stage"]
    except (OSError, ValueError):
        pass
    try:
        _atomic_json(destination / "worker-diagnostic.json", diagnostic)
    except OSError:
        pass
    # The destination was checked empty of these assets before the owned worker started.
    # A failed or timed-out build must not leave a configuration eligible for publication.
    try:
        (destination / "dense-config.json").unlink(missing_ok=True)
    except OSError:
        pass


def _worker_diagnostic(error: Exception) -> None:
    code = str(error).split(":", 1)[0]
    if code not in _WORKER_ERROR_CODES:
        code = "DENSE_EXTENSION_WORKER_FAILED"
    kind = next(kind.__name__ for kind in _ERROR_TYPES if isinstance(error, kind))
    print(
        json.dumps({"event": "dense_extension_diagnostic", "error_code": code, "error_type": kind}),
        file=sys.stderr,
        flush=True,
    )


def _base_assets(config: dict) -> tuple[list[dict], dict, int]:
    index = Path(config["index_path"])
    if _sha_file(index) != config["index_sha256"]:
        raise ValueError("DENSE_INDEX_SHA_MISMATCH")
    rows = _read_jsonl(index)
    _validate_index_rows(rows, config)
    meta = _checked_json(Path(config["vector_meta_path"]), config["vector_meta_sha256"])
    _validate_cache_metadata(meta, rows, "document")
    _validate_model_fingerprint(Path(config["model_path"]), meta["binding"]["model"])
    offset = _validate_vector_file(Path(config["vector_path"]), meta)
    return rows, meta, offset


def _extension_rows(index: Path, base_rows: list[dict]) -> list[dict]:
    rows = _read_jsonl(index)
    if len(rows) < len(base_rows):
        raise ValueError("DENSE_INDEX_PREFIX_MISMATCH")
    _validate_index_rows(rows, {"corpus_mode": "incremental-v1", "document_count": len(rows)})
    for base, row in zip(base_rows, rows):
        if any(base[key] != row[key] for key in ("unit_id", "file_id", "text")):
            raise ValueError("DENSE_INDEX_PREFIX_MISMATCH")
    return rows


def _check_destination(destination: Path, config: dict) -> None:
    if destination.is_relative_to(Path(config["model_path"]).resolve()):
        raise ValueError("INVALID_DENSE_EXTENSION_DESTINATION")
    if any((destination / name).exists() for name in _OUTPUT_FILES):
        raise FileExistsError("DENSE_EXTENSION_DESTINATION_EXISTS")
    destination.mkdir(parents=True, exist_ok=True)


def build_dense_extension(base_config_path: Path, index_path: Path, destination: Path) -> Path:
    """Build an immutable complete vector cache, encoding only the appended index rows."""
    base_config_path, index_path, destination = (
        Path(path).resolve() for path in (base_config_path, index_path, destination)
    )
    config = _read_config(base_config_path)
    timeout = _extension_timeout(config)
    _check_destination(destination, config)
    _write_progress(destination, "validating_base")
    base_rows, _, _ = _base_assets(config)
    rows = _extension_rows(index_path, base_rows)
    index_sha = _sha_file(index_path)
    _check_destination(destination, config)
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        process = subprocess.run(
            [
                config["python_executable"],
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                str(base_config_path),
                str(index_path),
                str(destination),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            shell=False,
            creationflags=flags,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        _record_failure(destination, "DENSE_EXTENSION_TIMEOUT", timeout, error.stderr)
        raise ValueError("DENSE_EXTENSION_TIMEOUT") from None
    except OSError:
        _record_failure(destination, "DENSE_EXTENSION_WORKER_FAILED", timeout)
        raise ValueError("DENSE_EXTENSION_WORKER_FAILED") from None
    if process.returncode != 0:
        _record_failure(
            destination,
            "DENSE_EXTENSION_WORKER_FAILED",
            timeout,
            process.stderr,
            process.returncode,
        )
        raise ValueError("DENSE_EXTENSION_WORKER_FAILED")
    result = destination / "dense-config.json"
    try:
        packet = json.loads(process.stdout)
        if not isinstance(packet, dict) or packet.get("config_path") != str(result):
            raise ValueError
        built = _read_config(result)
        if (
            built["index_path"] != str(index_path)
            or built["index_sha256"] != index_sha
            or built.get("corpus_mode") != "incremental-v1"
            or built.get("document_count") != len(rows)
            or built["vector_path"] != str(destination / "vectors.npy")
            or built["vector_meta_path"] != str(destination / "vectors.json")
            or built["model_path"] != config["model_path"]
            or built["python_executable"] != config["python_executable"]
        ):
            raise ValueError
        _base_assets(built)
    except (OSError, ValueError, TypeError, KeyError):
        _record_failure(
            destination,
            "INVALID_DENSE_EXTENSION_RESULT",
            timeout,
            process.stderr,
            process.returncode,
        )
        raise ValueError("INVALID_DENSE_EXTENSION_RESULT") from None
    return result


def _encode_new_documents(
    config: dict, binding: dict, rows: list[dict], *, progress: Callable[[int], None] | None = None
) -> tuple[list, list[dict]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer

    if (
        torch.__version__ != "2.6.0+cu124"
        or transformers.__version__ != "4.57.1"
        or not torch.cuda.is_available()
    ):
        raise ValueError("DENSE_RUNTIME_BINDING_MISMATCH")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_path"], local_files_only=True, trust_remote_code=False
    )
    if type(tokenizer).__name__ != binding["tokenizer_class"]:
        raise ValueError("DENSE_TOKENIZER_BINDING_MISMATCH")
    audit = []
    for row in rows:
        count = len(tokenizer(row["text"], truncation=False, padding=False)["input_ids"])
        if not 0 < count <= MAX_LENGTH:
            raise ValueError(f"DENSE_DOCUMENT_TOO_LONG: {count} > {MAX_LENGTH}")
        audit.append(
            {
                "id": row["unit_id"],
                "input_tokens": count,
                "used_tokens": count,
                "truncated": False,
                "truncated_tokens": 0,
                "max_length": MAX_LENGTH,
            }
        )
    model = AutoModel.from_pretrained(
        config["model_path"],
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.float16,
    )
    model.to(config["device"]).eval()
    if progress is not None:
        progress(0)
    encoded = []
    for row in rows:
        tokens = tokenizer(row["text"], padding=True, truncation=False, return_tensors="pt")
        tokens = {key: value.to(config["device"]) for key, value in tokens.items()}
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state[:, 0]
            vector = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        encoded.append(vector.cpu().tolist()[0])
        if progress is not None:
            progress(len(encoded))
    return encoded, audit


def _write_vectors(
    path: Path, base_path: Path, offset: int, count: int, added: list, base_sha: str
) -> None:
    header = repr({"descr": "<f4", "fortran_order": False, "shape": (count, 512)}).encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    with path.open("xb") as output, base_path.open("rb") as base:
        output.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        copied_sha = hashlib.sha256(base.read(offset))
        for chunk in iter(lambda: base.read(1024 * 1024), b""):
            output.write(chunk)
            copied_sha.update(chunk)
        if copied_sha.hexdigest() != base_sha:
            raise ValueError("DENSE_VECTOR_SHA_MISMATCH")
        for row in added:
            if len(row) != 512:
                raise ValueError("INVALID_DENSE_VECTOR_SHAPE")
            output.write(struct.pack("<512f", *row))


def _write_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _build_extension_worker(base_config_path: Path, index_path: Path, destination: Path) -> Path:
    config = _read_config(base_config_path)
    _check_destination(destination, config)
    _write_progress(destination, "validating_base")
    base_rows, base_meta, offset = _base_assets(config)
    rows = _extension_rows(index_path, base_rows)
    index_sha = _sha_file(index_path)
    added_rows = rows[len(base_rows) :]
    _write_progress(destination, "index_validated", added_units=len(added_rows))
    if added_rows:
        _write_progress(destination, "loading_encoder", added_units=len(added_rows))
        encoded, new_audit = _encode_new_documents(
            config,
            base_meta["binding"],
            added_rows,
            progress=lambda count: _write_progress(
                destination, "encoding", added_units=len(added_rows), encoded_units=count
            ),
        )
        _write_progress(
            destination, "encoding", added_units=len(added_rows), encoded_units=len(encoded)
        )
    else:
        encoded, new_audit = [], []
    if len(encoded) != len(added_rows) or len(new_audit) != len(added_rows):
        raise ValueError("INVALID_DENSE_EXTENSION_ENCODING_COUNT")
    meta = copy.deepcopy(base_meta)
    text_sha = _ordered_text_sha(rows)
    meta.update(
        {
            "ordered_unit_text_sha256": text_sha,
            "shape": [len(rows), 512],
            "token_audit": base_meta["token_audit"] + new_audit,
            "key": hashlib.sha256(
                json.dumps(
                    {"text_sha": text_sha, "config": meta["binding"]},
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest(),
        }
    )
    _write_progress(
        destination, "writing_vectors", added_units=len(added_rows), encoded_units=len(encoded)
    )
    vector_path = destination / "vectors.npy"
    _write_vectors(
        vector_path,
        Path(config["vector_path"]),
        offset,
        len(rows),
        encoded,
        base_meta["npy_sha256"],
    )
    meta["npy_sha256"] = _sha_file(vector_path)
    _validate_cache_metadata(meta, rows, "document")
    _validate_vector_file(vector_path, meta)
    if _sha_file(index_path) != index_sha:
        raise ValueError("DENSE_INDEX_SHA_MISMATCH")
    meta_path = destination / "vectors.json"
    _write_json(meta_path, meta)
    updated = dict(config)
    updated.update(
        {
            "corpus_mode": "incremental-v1",
            "document_count": len(rows),
            "index_path": str(index_path),
            "index_sha256": index_sha,
            "vector_path": str(vector_path),
            "vector_meta_path": str(meta_path),
            "vector_meta_sha256": _sha_file(meta_path),
        }
    )
    result = destination / "dense-config.json"
    _write_json(result, updated)
    _write_progress(
        destination, "complete", added_units=len(added_rows), encoded_units=len(encoded)
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        nargs=3,
        type=Path,
        required=True,
        metavar=("BASE_CONFIG", "INDEX", "DESTINATION"),
    )
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        built = _build_extension_worker(*parser.parse_args().worker)
    except Exception as error:
        _worker_diagnostic(error)
        print(json.dumps({"error": "DENSE_EXTENSION_WORKER_FAILED"}), flush=True)
        raise SystemExit(1) from None
    print(json.dumps({"config_path": str(built)}, ensure_ascii=False), flush=True)
