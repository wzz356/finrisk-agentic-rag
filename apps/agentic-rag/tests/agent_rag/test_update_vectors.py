import ast
import hashlib
import importlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_rag import dense_backend as dense


def save_vectors(path, matrix, dtype="<f4"):
    shape = (len(matrix), len(matrix[0]))
    header = repr({"descr": dtype, "fortran_order": False, "shape": shape}).encode("ascii")
    header += b" " * ((64 - (10 + len(header) + 1) % 64) % 64) + b"\n"
    code = "f" if dtype == "<f4" else "d"
    with Path(path).open("wb") as stream:
        stream.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header)
        for row in matrix:
            stream.write(struct.pack("<" + code * len(row), *row))


def load_vectors(path):
    with Path(path).open("rb") as stream:
        assert stream.read(8) == b"\x93NUMPY\x01\x00"
        size = struct.unpack("<H", stream.read(2))[0]
        header = ast.literal_eval(stream.read(size).decode("ascii"))
        count, dimension = header["shape"]
        return [
            list(struct.unpack("<" + "f" * dimension, stream.read(4 * dimension)))
            for _ in range(count)
        ]


def zero_vectors(count):
    return [[0.0] * 512 for _ in range(count)]


def updates():
    assert importlib.util.find_spec("agent_rag.update_vectors") is not None, "增量向量模块未实现"
    return importlib.import_module("agent_rag.update_vectors")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def write_rows(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def audit(rows):
    return [
        {
            "id": row["unit_id"],
            "input_tokens": 4,
            "used_tokens": 4,
            "truncated": False,
            "truncated_tokens": 0,
            "max_length": 512,
        }
        for row in rows
    ]


def fixture_assets(tmp_path):
    root = tmp_path / "base"
    root.mkdir()
    model = root / dense.MODEL_REVISION
    model.mkdir()
    (model / "config.json").write_bytes(b"frozen-model")
    files = [
        {"path": "config.json", "size": 12, "sha256": hashlib.sha256(b"frozen-model").hexdigest()}
    ]
    fingerprint = {
        "path": str(model),
        "revision": dense.MODEL_REVISION,
        "files": files,
        "cache_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    }
    rows = [
        {"unit_id": "U1", "file_id": "F1", "text": "基础原文"},
        {"unit_id": "U2", "file_id": "F1", "text": "条件\n例外原文"},
    ]
    index = root / "index.jsonl"
    write_rows(index, rows)
    matrix = zero_vectors(2)
    matrix[0][0], matrix[1][1] = 1, 1
    vector = root / "vectors.npy"
    save_vectors(vector, matrix)
    binding = {
        "model": fingerprint,
        "tokenizer_class": "BertTokenizerFast",
        "token_audit_schema_version": 2,
        "max_length": 512,
        "query_instruction": "",
        "pooling": "CLS",
        "normalization": "L2",
        "precision": "float16",
        "kind": "document",
        "torch": "2.6.0+cu124",
        "transformers": "4.57.1",
    }
    text_sha = dense._ordered_text_sha(rows)
    meta = {
        "binding": binding,
        "ordered_unit_text_sha256": text_sha,
        "key": hashlib.sha256(
            json.dumps(
                {"text_sha": text_sha, "config": binding}, sort_keys=True, ensure_ascii=False
            ).encode()
        ).hexdigest(),
        "shape": [2, 512],
        "npy_sha256": dense._sha_file(vector),
        "token_audit": audit(rows),
    }
    metadata = root / "vectors.json"
    write_json(metadata, meta)
    config = {
        "python_executable": sys.executable,
        "index_path": str(index),
        "index_sha256": dense._sha_file(index),
        "vector_path": str(vector),
        "vector_meta_path": str(metadata),
        "vector_meta_sha256": dense._sha_file(metadata),
        "model_path": str(model),
        "model_revision": dense.MODEL_REVISION,
        "corpus_mode": "incremental-v1",
        "document_count": 2,
        "device": "cuda",
        "precision": "float16",
    }
    config_path = root / "dense-config.json"
    write_json(config_path, config)
    return config_path, config, rows, matrix


def extension(tmp_path, rows):
    path = tmp_path / "new-index.jsonl"
    write_rows(path, rows)
    return path


def fake_subprocess(monkeypatch, module):
    calls, encoded = [], []

    def encode(config, binding, rows, *, progress=None):
        encoded.extend(row["unit_id"] for row in rows)
        matrix = zero_vectors(len(rows))
        for row in matrix:
            row[2] = 1
        return matrix, audit(rows)

    def run(args, **kwargs):
        calls.append((args, kwargs))
        result = module._build_extension_worker(Path(args[-3]), Path(args[-2]), Path(args[-1]))
        return SimpleNamespace(
            returncode=0, stdout=json.dumps({"config_path": str(result)}), stderr=""
        )

    monkeypatch.setattr(module, "_encode_new_documents", encode)
    monkeypatch.setattr(module.subprocess, "run", run)
    return calls, encoded


def test_extension_only_encodes_added_rows_preserves_base_and_builds_bound_cache(
    tmp_path, monkeypatch
):
    module = updates()
    base_path, config, rows, matrix = fixture_assets(tmp_path)
    original = {p: p.read_bytes() for p in base_path.parent.rglob("*") if p.is_file()}
    new_rows = rows + [{"unit_id": "U3", "file_id": "F2", "text": "新增处罚原文"}]
    index = extension(tmp_path, new_rows)
    calls, encoded = fake_subprocess(monkeypatch, module)
    result = module.build_dense_extension(base_path, index, tmp_path / "extension")
    assert encoded == ["U3"]
    new_config = dense._read_config(result)
    assert new_config["corpus_mode"] == "incremental-v1" and new_config["document_count"] == 3
    assert new_config["index_path"] == str(index.resolve())
    assert new_config["model_path"] == config["model_path"]
    assert new_config["python_executable"] == config["python_executable"]
    meta = dense._checked_json(
        Path(new_config["vector_meta_path"]), new_config["vector_meta_sha256"]
    )
    dense._validate_cache_metadata(meta, new_rows, "document")
    assert (
        meta["binding"]["model"]
        == json.loads(Path(config["vector_meta_path"]).read_text())["binding"]["model"]
    )
    vectors = load_vectors(new_config["vector_path"])
    assert vectors[:2] == matrix
    assert dense._sha_file(Path(new_config["vector_path"])) == meta["npy_sha256"]
    assert all(path.read_bytes() == raw for path, raw in original.items())
    args, kwargs = calls[0]
    assert args[0] == config["python_executable"] and "--worker" in args
    assert kwargs["shell"] is False and kwargs["encoding"] == "utf-8"
    if sys.platform == "win32":
        assert kwargs["creationflags"] & subprocess.CREATE_NO_WINDOW


def test_no_added_rows_skips_encoder_but_still_checks_assets(tmp_path, monkeypatch):
    module = updates()
    base_path, config, rows, matrix = fixture_assets(tmp_path)
    _, encoded = fake_subprocess(monkeypatch, module)
    result = module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "same")
    assert encoded == []
    assert load_vectors(dense._read_config(result)["vector_path"]) == matrix


@pytest.mark.parametrize("change", ["delete", "reorder", "text", "file_id", "unit_id", "duplicate"])
def test_rejects_any_non_append_index_change_before_encoding(tmp_path, monkeypatch, change):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    changed = [dict(row) for row in rows]
    if change == "delete":
        changed.pop()
    elif change == "reorder":
        changed.reverse()
    elif change == "duplicate":
        changed.append(dict(changed[0]))
    else:
        changed[0][change] += "修改"
    _, encoded = fake_subprocess(monkeypatch, module)
    with pytest.raises(ValueError, match="PREFIX|INDEX"):
        module.build_dense_extension(base_path, extension(tmp_path, changed), tmp_path / "invalid")
    assert encoded == []


@pytest.mark.parametrize("asset", ["index", "metadata", "model", "vector"])
def test_rejects_changed_base_hashes_even_without_new_rows(tmp_path, monkeypatch, asset):
    module = updates()
    base_path, config, rows, _ = fixture_assets(tmp_path)
    path = {
        "index": Path(config["index_path"]),
        "metadata": Path(config["vector_meta_path"]),
        "model": Path(config["model_path"]) / "config.json",
        "vector": Path(config["vector_path"]),
    }[asset]
    path.write_bytes(path.read_bytes() + b" ")
    _, encoded = fake_subprocess(monkeypatch, module)
    with pytest.raises(ValueError, match="SHA_MISMATCH"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "invalid")
    assert encoded == []


@pytest.mark.parametrize("damage", ["shape", "dtype", "nan", "norm"])
def test_rejects_sha_bound_invalid_base_vectors(tmp_path, monkeypatch, damage):
    module = updates()
    base_path, config, rows, matrix = fixture_assets(tmp_path)
    if damage == "shape":
        matrix = [row[:511] for row in matrix]
    elif damage == "dtype":
        pass
    elif damage == "nan":
        matrix[0][0] = float("nan")
    else:
        matrix[0][0] = 0
    save_vectors(config["vector_path"], matrix, "<f8" if damage == "dtype" else "<f4")
    meta = json.loads(Path(config["vector_meta_path"]).read_text())
    meta["npy_sha256"] = dense._sha_file(Path(config["vector_path"]))
    write_json(Path(config["vector_meta_path"]), meta)
    config["vector_meta_sha256"] = dense._sha_file(Path(config["vector_meta_path"]))
    write_json(base_path, config)
    _, encoded = fake_subprocess(monkeypatch, module)
    with pytest.raises(ValueError, match="VECTOR_SHAPE|VECTOR_NORMALIZATION"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "invalid")
    assert encoded == []


def test_rejects_bound_token_audit_tampering(tmp_path, monkeypatch):
    module = updates()
    base_path, config, rows, _ = fixture_assets(tmp_path)
    meta_path = Path(config["vector_meta_path"])
    meta = json.loads(meta_path.read_text())
    meta["token_audit"][0]["truncated"] = True
    write_json(meta_path, meta)
    config["vector_meta_sha256"] = dense._sha_file(meta_path)
    write_json(base_path, config)
    _, encoded = fake_subprocess(monkeypatch, module)
    with pytest.raises(ValueError, match="TOKEN_AUDIT"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "invalid")
    assert encoded == []


@pytest.mark.parametrize(
    "changes",
    [
        {"corpus_mode": "other", "document_count": 2},
        {"corpus_mode": "incremental-v1"},
        {"document_count": 2},
        {"corpus_mode": "incremental-v1", "document_count": True},
        {"corpus_mode": "incremental-v1", "document_count": 0},
    ],
)
def test_incremental_count_requires_explicit_valid_protocol(tmp_path, changes):
    base_path, config, _, _ = fixture_assets(tmp_path)
    config.pop("corpus_mode")
    config.pop("document_count")
    config.update(changes)
    write_json(base_path, config)
    with pytest.raises(ValueError, match="CONFIG"):
        dense._read_config(base_path)


def test_legacy_config_cannot_bypass_frozen_document_count(tmp_path):
    base_path, config, rows, _ = fixture_assets(tmp_path)
    config.pop("corpus_mode")
    config.pop("document_count")
    write_json(base_path, config)
    legacy = dense._read_config(base_path)
    assert hasattr(dense, "_validate_index_rows"), "缺少共享索引计数验证"
    with pytest.raises(ValueError, match="INDEX"):
        dense._validate_index_rows(rows, legacy)
    dense._validate_index_rows(
        rows, {**legacy, "corpus_mode": "incremental-v1", "document_count": 2}
    )


def test_parent_module_can_import_without_numpy_torch_transformers(tmp_path):
    module = updates()
    source_root = str(Path(module.__file__).resolve().parents[1])
    script = (
        "import sys; sys.path.insert(0, "
        + repr(source_root)
        + "); import agent_rag.update_vectors; "
        + "assert not ({'numpy', 'torch', 'transformers'} & set(sys.modules))"
    )
    result = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_refuses_overwriting_existing_output_and_does_not_change_it(tmp_path, monkeypatch):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    destination = tmp_path / "extension"
    destination.mkdir()
    sentinel = destination / "vectors.npy"
    sentinel.write_bytes(b"keep")
    fake_subprocess(monkeypatch, module)
    with pytest.raises(FileExistsError):
        module.build_dense_extension(base_path, extension(tmp_path, rows), destination)
    assert sentinel.read_bytes() == b"keep"


def test_new_document_overlength_is_rejected_without_truncation_or_model_load(monkeypatch):
    module = updates()
    calls = []

    class BertTokenizerFast:
        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            return {"input_ids": list(range(513))}

    def unexpected_model(*args, **kwargs):
        pytest.fail("超长文档不应加载模型")

    torch = SimpleNamespace(
        __version__="2.6.0+cu124", cuda=SimpleNamespace(is_available=lambda: True)
    )
    transformers = SimpleNamespace(
        __version__="4.57.1",
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: BertTokenizerFast()),
        AutoModel=SimpleNamespace(from_pretrained=unexpected_model),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(ValueError, match="DOCUMENT_TOO_LONG"):
        module._encode_new_documents(
            {"model_path": "local", "device": "cuda"},
            {"tokenizer_class": "BertTokenizerFast"},
            [{"unit_id": "N", "text": "完整新增原文"}],
        )
    assert calls == [("完整新增原文", {"truncation": False, "padding": False})]


def test_unchanged_index_builds_through_real_standard_library_subprocess(tmp_path):
    module = updates()
    base_path, _, rows, matrix = fixture_assets(tmp_path)
    result = module.build_dense_extension(
        base_path, extension(tmp_path, rows), tmp_path / "real-worker"
    )
    config = dense._read_config(result)
    assert load_vectors(config["vector_path"]) == matrix
    assert config["document_count"] == 2


def test_output_directory_cannot_be_inside_frozen_model(tmp_path, monkeypatch):
    module = updates()
    base_path, config, rows, _ = fixture_assets(tmp_path)
    fake_subprocess(monkeypatch, module)
    destination = Path(config["model_path"]) / "new-vectors"
    with pytest.raises(ValueError, match="DESTINATION"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), destination)
    assert not destination.exists()


def test_worker_failure_never_returns_an_usable_config(tmp_path, monkeypatch):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="private text"),
    )
    with pytest.raises(ValueError, match="WORKER_FAILED") as caught:
        module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "failed")
    assert "private text" not in str(caught.value)
    assert not (tmp_path / "failed" / "dense-config.json").exists()


def test_new_encoding_keeps_local_cuda_fp16_cls_l2_and_full_text(monkeypatch):
    module = updates()
    calls = []
    fp16 = object()

    class Tensor:
        def to(self, device):
            assert device == "cuda"
            return self

    class Hidden:
        def __getitem__(self, item):
            assert item == (slice(None), 0)
            calls.append("CLS")
            return self

        def float(self):
            calls.append("float32-normalize")
            return self

    class Vector:
        def cpu(self):
            return self

        def tolist(self):
            return [[1.0] + [0.0] * 511]

    class BertTokenizerFast:
        def __call__(self, text, **kwargs):
            assert text == "完整正文\n条件和例外"
            assert kwargs.get("truncation") is False
            if kwargs.get("return_tensors") == "pt":
                assert kwargs["padding"] is True
                return {"input_ids": Tensor()}
            assert kwargs["padding"] is False
            return {"input_ids": [1, 2, 3, 4]}

    class Model:
        def to(self, device):
            assert device == "cuda"
            return self

        def eval(self):
            calls.append("eval")
            return self

        def __call__(self, **tokens):
            assert set(tokens) == {"input_ids"}
            return SimpleNamespace(last_hidden_state=Hidden())

    def tokenizer_load(path, **kwargs):
        assert path == "local-model"
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return BertTokenizerFast()

    def model_load(path, **kwargs):
        assert path == "local-model"
        assert kwargs == {"local_files_only": True, "trust_remote_code": False, "torch_dtype": fp16}
        return Model()

    def normalize(hidden, **kwargs):
        assert isinstance(hidden, Hidden) and kwargs == {"p": 2, "dim": 1}
        calls.append("L2")
        return Vector()

    from contextlib import nullcontext

    torch = SimpleNamespace(
        __version__="2.6.0+cu124",
        cuda=SimpleNamespace(is_available=lambda: True),
        float16=fp16,
        inference_mode=nullcontext,
        nn=SimpleNamespace(functional=SimpleNamespace(normalize=normalize)),
    )
    transformers = SimpleNamespace(
        __version__="4.57.1",
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_load),
        AutoModel=SimpleNamespace(from_pretrained=model_load),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    rows = [{"unit_id": "U3", "text": "完整正文\n条件和例外"}]
    progress = []
    vectors, tokens = module._encode_new_documents(
        {"model_path": "local-model", "device": "cuda"},
        {"tokenizer_class": "BertTokenizerFast", "torch": "2.6.0+cu124", "transformers": "4.57.1"},
        rows,
        progress=progress.append,
    )
    assert progress == [0, 1]
    assert vectors == [[1.0] + [0.0] * 511] and tokens == audit(rows)
    assert calls == ["eval", "CLS", "float32-normalize", "L2"]


def test_rejects_base_vector_changed_between_validation_and_copy(tmp_path, monkeypatch):
    module = updates()
    base_path, config, rows, matrix = fixture_assets(tmp_path)
    index = extension(tmp_path, rows + [{"unit_id": "U3", "file_id": "F2", "text": "新增原文"}])

    def encode(config, binding, added, *, progress=None):
        changed = [list(row) for row in matrix]
        changed[0][0], changed[0][3] = 0, 1
        save_vectors(config["vector_path"], changed)
        return [[1.0] + [0.0] * 511], audit(added)

    monkeypatch.setattr(module, "_encode_new_documents", encode)
    with pytest.raises(ValueError, match="VECTOR_SHA_MISMATCH"):
        module._build_extension_worker(base_path, index, tmp_path / "changed-in-flight")
    assert not (tmp_path / "changed-in-flight" / "dense-config.json").exists()


@pytest.mark.parametrize("timeout", [None, 120])
def test_worker_subprocess_has_bounded_timeout(tmp_path, monkeypatch, timeout):
    module = updates()
    base_path, config, rows, _ = fixture_assets(tmp_path)
    if timeout is not None:
        config["extension_timeout_seconds"] = timeout
        write_json(base_path, config)
    calls, _ = fake_subprocess(monkeypatch, module)
    module.build_dense_extension(base_path, extension(tmp_path, rows), tmp_path / "bounded")
    assert calls[0][1]["timeout"] == (600 if timeout is None else timeout)


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), 1801, "600"])
def test_invalid_worker_timeout_is_rejected_before_launch(tmp_path, monkeypatch, timeout):
    module = updates()
    base_path, config, rows, _ = fixture_assets(tmp_path)
    config["extension_timeout_seconds"] = timeout
    write_json(base_path, config)
    calls, _ = fake_subprocess(monkeypatch, module)
    with pytest.raises(ValueError, match="TIMEOUT_CONFIG"):
        module.build_dense_extension(
            base_path, extension(tmp_path, rows), tmp_path / "invalid-timeout"
        )
    assert calls == []


def test_timeout_records_safe_diagnostic_and_never_leaves_publishable_config(tmp_path, monkeypatch):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    destination = tmp_path / "timeout"

    def timeout(args, **kwargs):
        write_json(destination / "dense-config.json", {"partial": True})
        stderr = (
            json.dumps(
                {
                    "event": "dense_extension_diagnostic",
                    "error_code": "DENSE_RUNTIME_BINDING_MISMATCH",
                    "error_type": "ValueError",
                    "payload": "sk-private",
                }
            )
            + "\nBearer PRIVATE\n"
        ).encode()
        raise subprocess.TimeoutExpired(args, 600, stderr=stderr)

    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(ValueError, match="DENSE_EXTENSION_TIMEOUT"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), destination)
    assert not (destination / "dense-config.json").exists()
    diagnostic = json.loads((destination / "worker-diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["error"] == "DENSE_EXTENSION_TIMEOUT" and diagnostic["timeout_seconds"] == 600
    assert "DENSE_RUNTIME_BINDING_MISMATCH" in diagnostic["stderr_tail"]
    assert "PRIVATE" not in json.dumps(diagnostic) and "sk-private" not in json.dumps(diagnostic)
    assert "payload" not in json.dumps(diagnostic)


def test_nonzero_worker_exit_writes_bounded_allowlisted_stderr_diagnostic(tmp_path, monkeypatch):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    safe = (
        json.dumps(
            {
                "event": "dense_extension_diagnostic",
                "error_code": "DENSE_DOCUMENT_TOO_LONG",
                "error_type": "ValueError",
                "prompt": "PRIVATE",
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="private payload", stderr=safe * 5000 + "Bearer SECRET"
        ),
    )
    destination = tmp_path / "failed-diagnostic"
    with pytest.raises(ValueError, match="WORKER_FAILED"):
        module.build_dense_extension(base_path, extension(tmp_path, rows), destination)
    diagnostic = json.loads((destination / "worker-diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["returncode"] == 1 and diagnostic["error"] == "DENSE_EXTENSION_WORKER_FAILED"
    assert "DENSE_DOCUMENT_TOO_LONG" in diagnostic["stderr_tail"]
    assert len(diagnostic["stderr_tail"].encode("utf-8")) <= 8192
    assert not any(
        value in json.dumps(diagnostic) for value in ("PRIVATE", "SECRET", "prompt", "payload")
    )


def test_worker_progress_is_atomic_ordered_and_complete_only_after_config(tmp_path, monkeypatch):
    module = updates()
    base_path, _, rows, _ = fixture_assets(tmp_path)
    destination = tmp_path / "progress"
    stages = []
    replace = module.os.replace

    def observe(source, target):
        if Path(target).name == "progress.json":
            progress = json.loads(Path(source).read_text(encoding="utf-8"))
            stages.append(progress["stage"])
            if progress["stage"] == "complete":
                assert (destination / "dense-config.json").exists()
        return replace(source, target)

    monkeypatch.setattr(module.os, "replace", observe)
    fake_subprocess(monkeypatch, module)
    index = extension(tmp_path, rows + [{"unit_id": "U3", "file_id": "F2", "text": "新增正文"}])
    module.build_dense_extension(base_path, index, destination)
    assert stages[-6:] == [
        "validating_base",
        "index_validated",
        "loading_encoder",
        "encoding",
        "writing_vectors",
        "complete",
    ]
    progress = json.loads((destination / "progress.json").read_text(encoding="utf-8"))
    assert (
        progress["stage"] == "complete"
        and progress["added_units"] == 1
        and progress["encoded_units"] == 1
    )
    assert not list(destination.glob("*.tmp"))
