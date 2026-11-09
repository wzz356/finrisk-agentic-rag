"""Resident local BGE retrieval; the API process imports only the standard library."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import queue
import re
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

MODEL_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
MAX_LENGTH = 512
DOCUMENT_COUNT = 65475
PROTOCOL_VERSION = 1
_MAX_RESPONSE_BYTES = 1024 * 1024
_MAX_DIAGNOSTIC_BYTES = 64 * 1024
_DIAGNOSTIC_PHASES = {"worker_start", "runtime_import_start", "runtime_import",
                      "asset_validation", "index_load",
                      "model_load", "first_encoding", "worker_stop"}
_DIAGNOSTIC_ERRORS = {"DENSE_STARTUP_FAILED", "DENSE_WORKER_ERROR", "DENSE_RESPONSE_TIMEOUT",
                      "DENSE_PROTOCOL_ERROR", "DENSE_RUNTIME_BINDING_MISMATCH",
                      "DENSE_INDEX_SHA_MISMATCH", "DENSE_METADATA_SHA_MISMATCH",
                      "DENSE_MODEL_SHA_MISMATCH", "DENSE_VECTOR_SHA_MISMATCH"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _safe_diagnostic(value: Any) -> dict | None:
    if (not isinstance(value, dict) or value.get("event") != "diagnostic"
            or value.get("phase") not in _DIAGNOSTIC_PHASES
            or type(value.get("elapsed_ms")) not in (int, float)
            or not math.isfinite(value["elapsed_ms"]) or value["elapsed_ms"] < 0):
        return None
    row = {"event": "diagnostic", "phase": value["phase"],
           "elapsed_ms": round(value["elapsed_ms"], 3)}
    if value.get("error_code") in _DIAGNOSTIC_ERRORS:
        row["error_code"] = value["error_code"]
    return row


def _diagnostic(phase: str, elapsed_ms: float, *, error_code: str | None = None) -> None:
    row = _safe_diagnostic({"event": "diagnostic", "phase": phase,
                            "elapsed_ms": elapsed_ms, "error_code": error_code})
    if row is not None:
        print(json.dumps(row, ensure_ascii=False), file=sys.stderr, flush=True)


class DenseBackendError(RuntimeError):
    """A failed worker is unavailable until the service creates a new instance."""


class QueryTooLong(ValueError):
    def __init__(self, query_tokens: int) -> None:
        self.query_tokens = query_tokens
        super().__init__(f"QUERY_TOO_LONG: {query_tokens} > {MAX_LENGTH}")


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def validate_startup_timeout_seconds(value: Any) -> None:
    if value is not None and (type(value) not in (int, float)
            or not math.isfinite(value) or not 0 < value <= 600):
        raise ValueError("INVALID_DENSE_STARTUP_TIMEOUT")


def _read_config(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("INVALID_DENSE_CONFIG")
    required_paths = ("python_executable", "index_path", "vector_path", "vector_meta_path",
                      "model_path")
    for key in required_paths:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip() or not Path(value).is_absolute():
            raise ValueError("INVALID_DENSE_CONFIG")
    if (config.get("model_revision") != MODEL_REVISION
            or Path(config["model_path"]).name != MODEL_REVISION
            or config.get("device", "cuda") != "cuda"
            or config.get("precision", "float16") != "float16"
            or not all(_is_sha(config.get(key)) for key in (
                "index_sha256", "vector_meta_sha256"))):
        raise ValueError("INVALID_DENSE_CONFIG")
    if "corpus_mode" in config or "document_count" in config:
        if (config.get("corpus_mode") != "incremental-v1"
                or type(config.get("document_count")) is not int
                or config["document_count"] <= 0):
            raise ValueError("INVALID_DENSE_CONFIG")
    config.setdefault("device", "cuda")
    config.setdefault("precision", "float16")
    for key, default in (("startup_timeout_seconds", 120), ("request_timeout_seconds", 20)):
        value = config.setdefault(key, default)
        if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
            raise ValueError("INVALID_DENSE_CONFIG")
    subset = config.get("subset_file_ids")
    if subset is not None and (not isinstance(subset, list) or not subset
                              or any(not isinstance(item, str) or not item for item in subset)
                              or len(set(subset)) != len(subset)):
        raise ValueError("INVALID_DENSE_SUBSET")
    query_fields = ("frozen_queries_path", "query_vectors_path", "query_meta_path",
                    "query_meta_sha256")
    if any(key in config for key in query_fields):
        if (not all(isinstance(config.get(key), str) and config[key] for key in query_fields)
                or not _is_sha(config["query_meta_sha256"])
                or not all(Path(config[key]).is_absolute() for key in query_fields[:-1])):
            raise ValueError("INVALID_FROZEN_QUERY_CONFIG")
    return config


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checked_json(path: Path, expected_sha: str) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("DENSE_METADATA_SHA_MISMATCH")
    result = json.loads(raw)
    if not isinstance(result, dict):
        raise ValueError("INVALID_DENSE_METADATA")
    return result


def _ordered_text_sha(rows: list[dict]) -> str:
    # Byte-for-byte equivalent to the frozen experiment's ordered_text_sha.
    digest = hashlib.sha256()
    for row in rows:
        value = json.dumps([row["unit_id"], row["text"]], ensure_ascii=False,
                           separators=(",", ":"))
        digest.update(value.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _validate_cache_metadata(meta: dict, rows: list[dict], kind: str) -> None:
    binding = meta.get("binding")
    expected = {"tokenizer_class": "BertTokenizerFast", "token_audit_schema_version": 2,
                "max_length": MAX_LENGTH, "query_instruction": (
                    QUERY_INSTRUCTION if kind == "query" else ""),
                "pooling": "CLS", "normalization": "L2", "precision": "float16", "kind": kind}
    if (not isinstance(binding, dict) or any(binding.get(k) != v for k, v in expected.items())
            or not isinstance(binding.get("model"), dict)
            or binding["model"].get("revision") != MODEL_REVISION
            or binding.get("torch") != "2.6.0+cu124"
            or binding.get("transformers") != "4.57.1"
            or meta.get("shape") != [len(rows), 512]
            or not _is_sha(meta.get("npy_sha256"))):
        raise ValueError("INVALID_DENSE_CACHE_BINDING")
    text_sha = _ordered_text_sha(rows)
    key = hashlib.sha256(json.dumps({"text_sha": text_sha, "config": binding},
                                   sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    if meta.get("ordered_unit_text_sha256") != text_sha or meta.get("key") != key:
        raise ValueError("DENSE_CACHE_ORDER_TEXT_MISMATCH")
    audit = meta.get("token_audit")
    if not isinstance(audit, list) or len(audit) != len(rows):
        raise ValueError("INVALID_DENSE_TOKEN_AUDIT")
    for row, token in zip(rows, audit):
        if (not isinstance(token, dict) or token.get("id") != row["unit_id"]
                or type(token.get("input_tokens")) is not int
                or not 0 < token["input_tokens"] <= MAX_LENGTH
                or type(token.get("used_tokens")) is not int
                or token["used_tokens"] != token["input_tokens"]
                or token.get("truncated") is not False
                or type(token.get("truncated_tokens")) is not int
                or token["truncated_tokens"] != 0 or token.get("max_length") != MAX_LENGTH):
            raise ValueError("INVALID_DENSE_TOKEN_AUDIT")


def _validate_model_fingerprint(path: Path, fingerprint: dict) -> None:
    root = path.resolve()
    if (not root.is_dir() or root.name != MODEL_REVISION
            or not isinstance(fingerprint, dict)
            or fingerprint.get("revision") != MODEL_REVISION
            or not isinstance(fingerprint.get("path"), str)
            or Path(fingerprint["path"]).resolve() != root
            or not isinstance(fingerprint.get("files"), list)
            or not fingerprint["files"]):
        raise ValueError("INVALID_DENSE_MODEL_BINDING")
    for entry in fingerprint["files"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                or Path(entry["path"]).is_absolute()
                or not (root / entry["path"]).resolve().is_relative_to(root)
                or type(entry.get("size")) is not int or entry["size"] < 0
                or not _is_sha(entry.get("sha256"))):
            raise ValueError("INVALID_DENSE_MODEL_FILE")
    actual = []
    for item in sorted(root.rglob("*")):
        if item.is_file():
            if not item.resolve().is_relative_to(root):
                raise ValueError("INVALID_DENSE_MODEL_FILE")
            actual.append({"path": item.relative_to(root).as_posix(),
                           "size": item.stat().st_size, "sha256": _sha_file(item)})
    cache_sha = hashlib.sha256(json.dumps(actual, sort_keys=True).encode("utf-8")).hexdigest()
    if actual != fingerprint["files"] or cache_sha != fingerprint.get("cache_sha256"):
        raise ValueError("DENSE_MODEL_SHA_MISMATCH")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("INVALID_DENSE_JSONL")
    return rows


def _validate_index_rows(rows: list[dict], config: dict) -> None:
    expected = (config["document_count"] if config.get("corpus_mode") == "incremental-v1"
                else DOCUMENT_COUNT)
    if (len(rows) != expected
            or any(not isinstance(row.get("unit_id"), str) or not row["unit_id"]
                   or not isinstance(row.get("text"), str) or not row["text"]
                   or not isinstance(row.get("file_id"), str) for row in rows)
            or len({row["unit_id"] for row in rows}) != len(rows)):
        raise ValueError("INVALID_DENSE_INDEX")


def _validate_vector_file(path: Path, meta: dict) -> int:
    """Validate little-endian float32 NPY without a numerical runtime; return data offset."""
    if _sha_file(path) != meta["npy_sha256"]:
        raise ValueError("DENSE_VECTOR_SHA_MISMATCH")
    with path.open("rb") as stream:
        try:
            magic = stream.read(8)
            if magic[:6] != b"\x93NUMPY" or magic[6:] not in (b"\x01\x00", b"\x02\x00", b"\x03\x00"):
                raise ValueError
            length_size = 2 if magic[6] == 1 else 4
            raw_length = stream.read(length_size)
            if len(raw_length) != length_size:
                raise ValueError
            length = int.from_bytes(raw_length, "little")
            if not 0 < length <= 65536:
                raise ValueError
            raw_header = stream.read(length)
            if len(raw_header) != length or not raw_header.endswith(b"\n"):
                raise ValueError
            header = ast.literal_eval(raw_header.decode("utf-8" if magic[6] == 3 else "latin1"))
            shape = header.get("shape") if isinstance(header, dict) else None
            if (not isinstance(header, dict) or set(header) != {"descr", "fortran_order", "shape"}
                    or header.get("descr") != "<f4" or header.get("fortran_order") is not False
                    or not isinstance(shape, tuple) or len(shape) != 2
                    or any(type(value) is not int or value <= 0 for value in shape)
                    or list(shape) != meta["shape"] or shape[1] != 512):
                raise ValueError
            offset = stream.tell()
            if path.stat().st_size != offset + shape[0] * shape[1] * 4:
                raise ValueError
        except (ValueError, TypeError, SyntaxError, UnicodeError, OverflowError):
            raise ValueError("INVALID_DENSE_VECTOR_SHAPE") from None
        unpack = struct.Struct("<512f")
        for _ in range(shape[0]):
            values = unpack.unpack(stream.read(unpack.size))
            if (not all(math.isfinite(value) for value in values)
                    or abs(math.sqrt(math.fsum(value * value for value in values)) - 1) > 2e-4):
                raise ValueError("INVALID_DENSE_VECTOR_NORMALIZATION")
    return offset


def _validate_frozen_queries(path: Path, meta: dict) -> dict[str, dict]:
    queries = _read_jsonl(path)
    if len(queries) != 20:
        raise ValueError("INVALID_FROZEN_QUERY_COUNT")
    prepared, mapping = [], {}
    for position, row in enumerate(queries):
        text, query_id = row.get("text"), row.get("query_id")
        if (query_id != f"KBT{position + 1:03}"
                or not isinstance(text, str) or not text.strip() or text in mapping
                or row.get("text_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest()):
            raise ValueError("INVALID_FROZEN_QUERY_TEXT")
        prepared.append({"unit_id": query_id, "text": QUERY_INSTRUCTION + text})
        mapping[text] = {"position": position, "query_id": query_id}
    _validate_cache_metadata(meta, prepared, "query")
    for text, row in mapping.items():
        row["query_tokens"] = meta["token_audit"][row["position"]]["input_tokens"]
    return mapping


def _validate_input(text: str, top_k: int) -> None:
    if (not isinstance(text, str) or not text.strip()
            or type(top_k) is not int or not 1 <= top_k <= 100):
        raise ValueError("INVALID_DENSE_QUERY")


class ResidentDenseIndex:
    """Serialize JSONL requests to a single self-owned, hidden local worker."""

    def __init__(self, config_path: Path, *, startup_timeout_seconds: float | None = None) -> None:
        validate_startup_timeout_seconds(startup_timeout_seconds)
        self.config_path = Path(config_path).resolve()
        self.config = _read_config(self.config_path)
        self._lock = threading.Lock()
        self._responses: queue.Queue = queue.Queue(maxsize=4)
        self._process: Any = None
        self._reader: threading.Thread | None = None
        self._stderr_reader: threading.Thread | None = None
        self._diagnostic_lock = threading.Lock()
        self._stderr_tail: deque[bytes] = deque()
        self._stderr_size = 0
        self._started = time.monotonic()
        self._exit_code: int | None = None
        self._available = False
        self._sequence = 0
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        try:
            self._process = subprocess.Popen(
                [self.config["python_executable"], "-u", str(Path(__file__).resolve()),
                 "--worker", str(self.config_path)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="strict", bufsize=1, shell=False,
                creationflags=flags,
            )
            self._reader = threading.Thread(target=self._read_responses, daemon=True,
                                            name="bge-worker-output")
            self._reader.start()
            self._stderr_reader = threading.Thread(target=self._read_stderr,
                args=(self._process.stderr,), daemon=True, name="bge-worker-diagnostics")
            self._stderr_reader.start()
            ready = self._receive(self.config["startup_timeout_seconds"]
                if startup_timeout_seconds is None else startup_timeout_seconds)
            if (not isinstance(ready, dict) or ready.get("event") != "ready"
                    or type(ready.get("protocol")) is not int
                    or ready["protocol"] != PROTOCOL_VERSION
                    or ready.get("model_revision") != MODEL_REVISION):
                raise DenseBackendError("DENSE_STARTUP_FAILED")
            self._available = True
            _diagnostic("worker_start", (time.monotonic() - self._started) * 1000)
        except BaseException as error:
            self._close_owned_process()
            if not isinstance(error, Exception) or isinstance(error, DenseBackendError):
                raise
            raise DenseBackendError("DENSE_STARTUP_FAILED") from None

    def _read_stderr(self, stream: Any) -> None:
        if stream is None:
            return
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(errors="replace")
            while True:
                line = stream.readline(4096)
                if not line:
                    return
                try:
                    row = _safe_diagnostic(json.loads(line))
                except (TypeError, ValueError):
                    continue
                if row is None:
                    continue
                packet = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
                with self._diagnostic_lock:
                    self._stderr_tail.append(packet)
                    self._stderr_size += len(packet)
                    while self._stderr_size > _MAX_DIAGNOSTIC_BYTES:
                        self._stderr_size -= len(self._stderr_tail.popleft())
        except (OSError, ValueError):
            return

    def diagnostics(self) -> dict:
        with self._diagnostic_lock:
            return {"stderr_tail": b"".join(self._stderr_tail).decode("utf-8"),
                    "elapsed_ms": round((time.monotonic() - self._started) * 1000, 3),
                    "exit_code": self._exit_code}

    def _read_responses(self) -> None:
        try:
            stream = self._process.stdout
            while True:
                line = stream.readline(_MAX_RESPONSE_BYTES + 1)
                if not line or len(line) > _MAX_RESPONSE_BYTES or not line.endswith("\n"):
                    self._responses.put_nowait(None)
                    return
                value = json.loads(line)
                self._responses.put_nowait(value)
        except Exception:
            try:
                self._responses.put_nowait(None)
            except queue.Full:
                pass

    def _receive(self, timeout: float) -> dict:
        try:
            result = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise DenseBackendError("DENSE_RESPONSE_TIMEOUT") from None
        if not isinstance(result, dict):
            raise DenseBackendError("DENSE_PROTOCOL_ERROR")
        return result

    def _close_owned_process(self) -> None:
        self._available = False
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        except OSError:
            pass
        finally:
            self._exit_code = process.poll()
            if self._stderr_reader is not None:
                self._stderr_reader.join(timeout=0.2)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if self._reader is not None and self._reader is not threading.current_thread():
                self._reader.join(timeout=0.2)
            print(json.dumps({"event": "dense_worker_stopped", **self.diagnostics()},
                             ensure_ascii=False), file=sys.stderr, flush=True)

    def search(self, text: str, top_k: int = 100) -> dict:
        _validate_input(text, top_k)
        with self._lock:
            if not self._available:
                raise DenseBackendError("DENSE_UNAVAILABLE")
            self._sequence += 1
            try:
                request = {"id": self._sequence, "text": text, "top_k": top_k}
                self._process.stdin.write(json.dumps(request, ensure_ascii=False,
                                                     allow_nan=False) + "\n")
                self._process.stdin.flush()
                response = self._receive(self.config["request_timeout_seconds"])
                if (type(response.get("id")) is not int or response["id"] != self._sequence
                        or type(response.get("ok")) is not bool):
                    raise DenseBackendError("DENSE_PROTOCOL_ERROR")
                if not response["ok"]:
                    if (response.get("error") == "QUERY_TOO_LONG"
                            and type(response.get("query_tokens")) is int
                            and response["query_tokens"] > MAX_LENGTH
                            and response.get("max_length") == MAX_LENGTH):
                        raise QueryTooLong(response["query_tokens"])
                    raise DenseBackendError("DENSE_WORKER_ERROR")
                result = response.get("result")
                self._validate_result(result, top_k)
                return result
            except QueryTooLong:
                raise
            except Exception as error:
                self._close_owned_process()
                if isinstance(error, DenseBackendError):
                    raise
                raise DenseBackendError("DENSE_WORKER_ERROR") from None

    @staticmethod
    def _validate_result(result: Any, top_k: int) -> None:
        if not isinstance(result, dict):
            raise DenseBackendError("DENSE_PROTOCOL_ERROR")
        hits, meta = result.get("hits"), result.get("meta")
        if (not isinstance(hits, list) or not 0 < len(hits) <= top_k
                or not isinstance(meta, dict)
                or meta.get("mode") not in ("local_bge_encoding", "frozen_query_vector_replay")
                or meta.get("model_revision") != MODEL_REVISION
                or type(meta.get("query_tokens")) is not int
                or not 0 < meta["query_tokens"] <= MAX_LENGTH
                or meta.get("truncated") is not False):
            raise DenseBackendError("DENSE_PROTOCOL_ERROR")
        seen = set()
        for rank, row in enumerate(hits, 1):
            if (not isinstance(row, dict) or not isinstance(row.get("unit_id"), str)
                    or not row["unit_id"] or row["unit_id"] in seen
                    or type(row.get("rank")) is not int or row["rank"] != rank
                    or type(row.get("score")) not in (int, float)
                    or not math.isfinite(row["score"])):
                raise DenseBackendError("DENSE_PROTOCOL_ERROR")
            seen.add(row["unit_id"])

    def close(self) -> None:
        with self._lock:
            self._close_owned_process()

    def __enter__(self) -> ResidentDenseIndex:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class _DenseWorker:
    def __init__(self, config: dict) -> None:
        self.config = config
        started = time.monotonic()
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        _diagnostic("runtime_import_start", (time.monotonic() - started) * 1000)
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        _diagnostic("runtime_import", (time.monotonic() - started) * 1000)

        self.np, self.torch = np, torch
        if (torch.__version__ != "2.6.0+cu124" or transformers.__version__ != "4.57.1"
                or not torch.cuda.is_available()):
            raise ValueError("DENSE_RUNTIME_BINDING_MISMATCH")
        index = Path(config["index_path"])
        started = time.monotonic()
        if _sha_file(index) != config["index_sha256"]:
            raise ValueError("DENSE_INDEX_SHA_MISMATCH")
        rows = _read_jsonl(index)
        _validate_index_rows(rows, config)
        self.index_units = len(rows)
        meta = _checked_json(Path(config["vector_meta_path"]), config["vector_meta_sha256"])
        _validate_cache_metadata(meta, rows, "document")
        _validate_model_fingerprint(Path(config["model_path"]), meta["binding"]["model"])
        _diagnostic("asset_validation", (time.monotonic() - started) * 1000)
        started = time.monotonic()
        vectors = self._load_vectors(Path(config["vector_path"]), meta)
        subset = config.get("subset_file_ids")
        if subset is None:
            positions = list(range(len(rows)))
        else:
            wanted = set(subset)
            if not wanted <= {row["file_id"] for row in rows}:
                raise ValueError("UNKNOWN_DENSE_SUBSET_FILE")
            positions = [i for i, row in enumerate(rows) if row["file_id"] in wanted]
        self.positions = np.asarray(positions)
        self.unit_ids = [rows[i]["unit_id"] for i in positions]
        self.vectors = vectors if subset is None else vectors[self.positions]
        self.frozen_queries: dict[str, dict] = {}
        self.query_vectors = None
        if "frozen_queries_path" in config:
            query_meta = _checked_json(Path(config["query_meta_path"]), config["query_meta_sha256"])
            self.frozen_queries = _validate_frozen_queries(
                Path(config["frozen_queries_path"]), query_meta)
            if query_meta["binding"]["model"] != meta["binding"]["model"]:
                raise ValueError("FROZEN_QUERY_MODEL_MISMATCH")
            self.query_vectors = self._load_vectors(Path(config["query_vectors_path"]), query_meta)
        _diagnostic("index_load", (time.monotonic() - started) * 1000)
        started = time.monotonic()
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_path"], local_files_only=True, trust_remote_code=False)
        if type(self.tokenizer).__name__ != meta["binding"]["tokenizer_class"]:
            raise ValueError("DENSE_TOKENIZER_BINDING_MISMATCH")
        self.model = AutoModel.from_pretrained(config["model_path"], local_files_only=True,
                                              trust_remote_code=False, torch_dtype=torch.float16)
        self.model.to(config["device"]).eval()
        _diagnostic("model_load", (time.monotonic() - started) * 1000)

    def _load_vectors(self, path: Path, meta: dict) -> Any:
        if _sha_file(path) != meta["npy_sha256"]:
            raise ValueError("DENSE_VECTOR_SHA_MISMATCH")
        vectors = self.np.load(path, allow_pickle=False)
        if (vectors.ndim != 2 or list(vectors.shape) != meta["shape"]
                or vectors.dtype != self.np.dtype("float32")):
            raise ValueError("INVALID_DENSE_VECTOR_SHAPE")
        for start in range(0, len(vectors), 2048):
            batch = vectors[start:start + 2048]
            if (not self.np.isfinite(batch).all()
                    or not self.np.allclose(self.np.linalg.norm(batch, axis=1), 1,
                                           rtol=1e-4, atol=1e-4)):
                raise ValueError("INVALID_DENSE_VECTOR_NORMALIZATION")
        return vectors

    def _encode_query(self, text: str) -> tuple[Any, int]:
        started = time.monotonic()
        instructed = QUERY_INSTRUCTION + text
        unlimited = self.tokenizer(instructed, truncation=False, padding=False)
        count = len(unlimited["input_ids"])
        if count > MAX_LENGTH:
            raise QueryTooLong(count)
        tokens = self.tokenizer(instructed, padding=True, truncation=False, return_tensors="pt")
        tokens = {key: value.to(self.config["device"]) for key, value in tokens.items()}
        with self.torch.inference_mode():
            hidden = self.model(**tokens).last_hidden_state[:, 0]
            vectors = self.torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        result = vectors.cpu().numpy()[0], count
        if not getattr(self, "_first_encoding_reported", False):
            self._first_encoding_reported = True
            _diagnostic("first_encoding", (time.monotonic() - started) * 1000)
        return result

    def search(self, text: str, top_k: int) -> dict:
        _validate_input(text, top_k)
        frozen = self.frozen_queries.get(text)
        if frozen is not None:
            vector = self.query_vectors[frozen["position"]]
            count, mode = frozen["query_tokens"], "frozen_query_vector_replay"
        else:
            vector, count = self._encode_query(text)
            mode = "local_bge_encoding"
        scores = self.vectors @ vector
        order = self.np.lexsort((self.positions, -scores))[:top_k]
        hits = [{"unit_id": self.unit_ids[i], "score": float(scores[i]), "rank": rank}
                for rank, i in enumerate(order, 1)]
        meta = {"mode": mode, "model_revision": self.config["model_revision"],
                "query_tokens": count, "truncated": False, "max_length": MAX_LENGTH,
                "query_instruction": QUERY_INSTRUCTION, "pooling": "CLS", "normalization": "L2",
                "device": self.config["device"], "precision": self.config["precision"],
                "index_units": self.index_units, "searched_units": len(self.unit_ids),
                "external_retrieval_requests": 0}
        if frozen is not None:
            meta["query_id"] = frozen["query_id"]
        return {"hits": hits, "meta": meta}


def _serve_worker(config_path: Path) -> int:
    def emit(packet: dict) -> None:
        print(json.dumps(packet, ensure_ascii=False, allow_nan=False), flush=True)

    try:
        worker = _DenseWorker(_read_config(config_path))
    except Exception as error:
        code = str(error).split(":", 1)[0]
        _diagnostic("worker_start", 0, error_code=(code if code in _DIAGNOSTIC_ERRORS
                    else "DENSE_STARTUP_FAILED"))
        emit({"event": "error", "error": "DENSE_STARTUP_FAILED"})
        return 1
    emit({"event": "ready", "protocol": PROTOCOL_VERSION, "model_revision": MODEL_REVISION})
    sequence = 0
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if (not isinstance(request, dict) or type(request.get("id")) is not int
                    or request["id"] != sequence + 1 or set(request) != {"id", "text", "top_k"}):
                return 2
            sequence = request["id"]
            result = worker.search(request["text"], request["top_k"])
            emit({"id": sequence, "ok": True, "result": result})
        except QueryTooLong as error:
            emit({"id": sequence, "ok": False, "error": "QUERY_TOO_LONG",
                  "query_tokens": error.query_tokens, "max_length": MAX_LENGTH})
        except Exception:
            emit({"id": sequence, "ok": False, "error": "DENSE_WORKER_ERROR"})
            return 1
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=Path, required=True)
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
    raise SystemExit(_serve_worker(parser.parse_args().worker))
