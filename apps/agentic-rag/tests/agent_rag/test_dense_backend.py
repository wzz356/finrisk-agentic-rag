import hashlib
import importlib
import importlib.util
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

REVISION = "7999e1d3359715c523056ef9478215996d62a620"
INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def backend():
    assert importlib.util.find_spec("agent_rag.dense_backend") is not None, "常驻 BGE 后端未实现"
    return importlib.import_module("agent_rag.dense_backend")


def config_file(tmp_path, **changes):
    config = {
        "python_executable": sys.executable,
        "index_path": str(tmp_path / "index.jsonl"), "index_sha256": "1" * 64,
        "vector_path": str(tmp_path / "vectors.npy"),
        "vector_meta_path": str(tmp_path / "vectors.json"), "vector_meta_sha256": "2" * 64,
        "model_path": str(tmp_path / REVISION), "model_revision": REVISION,
        "device": "cuda", "precision": "float16", "startup_timeout_seconds": 1,
        "request_timeout_seconds": 0.2,
    }
    config.update(changes)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


class FakeOutput:
    def __init__(self):
        self.lines = queue.Queue()
        self.closed = False

    def readline(self, limit=-1):
        return self.lines.get()

    def close(self):
        self.closed = True


class FakeInput:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def write(self, line):
        self.process.requests.append(json.loads(line))
        self.process.respond(json.loads(line))

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProcess:
    def __init__(self, respond=None, ready=None):
        self.stdout = FakeOutput()
        self.stdin = FakeInput(self)
        self.stderr = None
        self.requests = []
        self.terminated = False
        self.waited = False
        self.returncode = None
        self.respond = respond or self.normal_response
        self.stdout.lines.put(json.dumps(ready or {"event": "ready", "protocol": 1,
                                                 "model_revision": REVISION}) + "\n")

    def normal_response(self, request):
        self.emit({"id": request["id"], "ok": True, "result": {
            "hits": [{"unit_id": request["text"], "score": 0.75, "rank": 1}],
            "meta": {"mode": "local_bge_encoding", "model_revision": REVISION,
                     "query_tokens": 20, "truncated": False}}})

    def emit(self, response):
        self.stdout.lines.put(json.dumps(response) + "\n")

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -1
        self.stdout.lines.put("")

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode


def fake_resident(monkeypatch, tmp_path, process=None, **config):
    module = backend()
    process = process or FakeProcess()
    launches = []
    def launch(args, **kwargs):
        launches.append((args, kwargs))
        return process
    monkeypatch.setattr(module.subprocess, "Popen", launch)
    return module.ResidentDenseIndex(config_file(tmp_path, **config)), process, launches


@pytest.mark.parametrize('override', [None, 0.5, 300, 600])
def test_explicit_startup_wait_does_not_change_config_or_query_timeout(
    tmp_path, monkeypatch, override
):
    module = backend()
    path = config_file(tmp_path, startup_timeout_seconds=120, request_timeout_seconds=20)
    original = path.read_bytes()
    process = FakeProcess()
    launches, waits = [], []
    def launch(args, **kwargs):
        launches.append(args)
        return process
    receive = module.ResidentDenseIndex._receive
    def observe(self, timeout):
        waits.append(timeout)
        return receive(self, timeout)
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    monkeypatch.setattr(module.ResidentDenseIndex, '_receive', observe)
    resident = module.ResidentDenseIndex(path, startup_timeout_seconds=override)
    try:
        resident.search('query')
        assert waits == [120 if override is None else override, 20]
        assert resident.config['startup_timeout_seconds'] == 120
        assert path.read_bytes() == original
        assert launches[0][-2:] == ['--worker', str(path)]
    finally:
        resident.close()


@pytest.mark.parametrize('value', [True, False, 0, -1, 600.1, float('inf'), float('nan'), '300'])
def test_invalid_startup_override_is_rejected_before_launch(tmp_path, monkeypatch, value):
    module = backend()
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('worker launched'))
    with pytest.raises(ValueError, match='INVALID_DENSE_STARTUP_TIMEOUT'):
        module.ResidentDenseIndex(tmp_path/'missing-config', startup_timeout_seconds=value)


@pytest.mark.parametrize('error_type', [KeyboardInterrupt, SystemExit])
def test_startup_interruption_closes_owned_worker_and_reraises_original(
    tmp_path, monkeypatch, error_type
):
    module = backend()
    process = FakeProcess()
    failure = error_type('stop during startup')
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: process)
    def interrupted(self, timeout):
        raise failure
    monkeypatch.setattr(module.ResidentDenseIndex, '_receive', interrupted)
    try:
        with pytest.raises(error_type) as caught:
            module.ResidentDenseIndex(config_file(tmp_path))
        assert caught.value is failure
        assert process.terminated and process.waited
        assert process.stdin.closed and process.stdout.closed
    finally:
        if not process.terminated:
            process.terminate()


def test_worker_announces_runtime_import_before_heavy_dependency_import(monkeypatch, capsys):
    import builtins
    module = backend()
    original_import = builtins.__import__
    def intercepted(name, *args, **kwargs):
        if name == 'numpy':
            raise ImportError('stop before dependency import')
        return original_import(name, *args, **kwargs)
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY'):
        monkeypatch.setenv(name, 'before')
    monkeypatch.setattr(builtins, '__import__', intercepted)
    with pytest.raises(ImportError, match='stop before dependency import'):
        module._DenseWorker({})
    rows = [json.loads(line) for line in capsys.readouterr().err.splitlines()]
    assert len(rows) == 1 and rows[0]['phase'] == 'runtime_import_start'
    assert set(rows[0]) == {'event','phase','elapsed_ms'}


def test_parent_import_and_process_protocol_need_only_standard_library(monkeypatch, tmp_path):
    resident, process, launches = fake_resident(monkeypatch, tmp_path)
    try:
        assert resident.search("完整查询\n原文", 100)["hits"][0]["unit_id"] == "完整查询\n原文"
        args, kwargs = launches[0]
        assert args[0] == sys.executable
        assert args[-2:] == ["--worker", str(tmp_path / "config.json")]
        assert Path(args[2]).name == "dense_backend.py"
        assert kwargs["stderr"] == backend().subprocess.PIPE
        assert kwargs["encoding"] == "utf-8" and kwargs["shell"] is False
        if sys.platform == "win32":
            assert kwargs["creationflags"] & backend().subprocess.CREATE_NO_WINDOW
        assert process.requests[0] == {"id": 1, "text": "完整查询\n原文", "top_k": 100}
    finally:
        resident.close()
    assert process.terminated and process.waited


def test_requests_are_serialized_and_each_response_has_matching_id(monkeypatch, tmp_path):
    resident, process, _ = fake_resident(monkeypatch, tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(resident.search, ["A", "B", "C", "D"]))
        assert [r["hits"][0]["unit_id"] for r in results] == ["A", "B", "C", "D"]
        assert [r["id"] for r in process.requests] == [1, 2, 3, 4]
    finally:
        resident.close()


def test_timeout_discards_late_result_and_permanently_closes_own_process(monkeypatch, tmp_path):
    process = FakeProcess(respond=lambda request: None)
    resident, _, _ = fake_resident(monkeypatch, tmp_path, process, request_timeout_seconds=0.02)
    with pytest.raises(backend().DenseBackendError, match="TIMEOUT"):
        resident.search("first")
    process.normal_response(process.requests[0])
    with pytest.raises(backend().DenseBackendError, match="UNAVAILABLE"):
        resident.search("second")
    assert len(process.requests) == 1 and process.terminated and process.waited
    resident.close()


@pytest.mark.parametrize("ready", [
    {"event": "error", "error": "SECRET"},
    {"event": "ready", "protocol": 9, "model_revision": REVISION},
    {"event": "ready", "protocol": 1, "model_revision": "other"},
])
def test_bad_handshake_is_closed_without_echoing_worker_secrets(monkeypatch, tmp_path, ready):
    process = FakeProcess(ready=ready)
    with pytest.raises(backend().DenseBackendError) as caught:
        fake_resident(monkeypatch, tmp_path, process)
    assert "SECRET" not in str(caught.value) and process.terminated and process.waited


@pytest.mark.parametrize("bad", [
    {"id": 100, "ok": True, "result": {"hits": [], "meta": {}}},
    {"id": True, "ok": True, "result": {"hits": [], "meta": {}}},
    {"id": 1, "ok": True, "result": {"hits": [], "meta": {}}},
])
def test_protocol_errors_poison_connection(monkeypatch, tmp_path, bad):
    process = FakeProcess(respond=lambda request: process.emit(bad))
    resident, _, _ = fake_resident(monkeypatch, tmp_path, process)
    with pytest.raises(backend().DenseBackendError, match="PROTOCOL"):
        resident.search("query")
    assert process.terminated
    with pytest.raises(backend().DenseBackendError, match="UNAVAILABLE"):
        resident.search("again")


def test_overlength_query_is_rejected_without_poisoning_worker(monkeypatch, tmp_path):
    process = FakeProcess()
    def respond(request):
        if request["text"] == "long":
            process.emit({"id": request["id"], "ok": False, "error": "QUERY_TOO_LONG",
                          "query_tokens": 513, "max_length": 512})
        else:
            process.normal_response(request)
    process.respond = respond
    resident, _, _ = fake_resident(monkeypatch, tmp_path, process)
    try:
        with pytest.raises(ValueError, match="513.*512"):
            resident.search("long")
        assert resident.search("short")["hits"]
    finally:
        resident.close()


@pytest.mark.parametrize("text,top_k", [("", 100), (" ", 100), (None, 100), ("Q", 0),
                                        ("Q", 101), ("Q", True), ("Q", 1.5)])
def test_invalid_input_never_reaches_worker(monkeypatch, tmp_path, text, top_k):
    resident, process, _ = fake_resident(monkeypatch, tmp_path)
    try:
        with pytest.raises(ValueError):
            resident.search(text, top_k)
        assert process.requests == []
    finally:
        resident.close()


@pytest.mark.parametrize("changes", [
    {"model_revision": "other"}, {"device": "cpu"}, {"precision": "float32"},
    {"index_sha256": "invalid"}, {"request_timeout_seconds": 0},
    {"subset_file_ids": ["F1", "F1"]}, {"frozen_queries_path": "only-one-field"},
])
def test_config_rejects_unbound_or_unsupported_settings(tmp_path, changes):
    with pytest.raises(ValueError):
        backend()._read_config(config_file(tmp_path, **changes))


def test_ordered_hash_matches_original_unicode_and_newline_algorithm():
    rows = [{"unit_id": "U1", "text": "中国\n原文"}, {"unit_id": "U2", "text": " B "}]
    expected = hashlib.sha256('["U1","中国\\n原文"]\n["U2"," B "]\n'.encode()).hexdigest()
    assert backend()._ordered_text_sha(rows) == expected
    assert backend()._ordered_text_sha(rows[::-1]) != expected


def metadata(rows, kind="document"):
    binding = {"model": {"revision": REVISION}, "tokenizer_class": "BertTokenizerFast",
               "token_audit_schema_version": 2, "max_length": 512,
               "query_instruction": INSTRUCTION if kind == "query" else "",
               "pooling": "CLS", "normalization": "L2", "torch": "2.6.0+cu124",
               "transformers": "4.57.1", "precision": "float16", "kind": kind}
    text_sha = backend()._ordered_text_sha(rows)
    key = hashlib.sha256(json.dumps({"text_sha": text_sha, "config": binding},
                                   sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {"key": key, "binding": binding, "ordered_unit_text_sha256": text_sha,
            "shape": [len(rows), 512], "npy_sha256": "3" * 64,
            "token_audit": [{"id": r["unit_id"], "input_tokens": 20, "used_tokens": 20,
                             "truncated": False, "truncated_tokens": 0, "max_length": 512}
                            for r in rows]}


@pytest.mark.parametrize("field,value", [("shape", [2, 513]), ("ordered_unit_text_sha256", "bad"),
                                         ("key", "bad"), ("npy_sha256", "bad")])
def test_cache_metadata_rejects_wrong_shape_order_text_and_vector_binding(field, value):
    rows = [{"unit_id": "U1", "text": "A"}, {"unit_id": "U2", "text": "B"}]
    meta = metadata(rows)
    meta[field] = value
    with pytest.raises(ValueError):
        backend()._validate_cache_metadata(meta, rows, "document")


def test_valid_cache_metadata_and_audit_order_are_verified():
    rows = [{"unit_id": "U1", "text": "A"}, {"unit_id": "U2", "text": "B"}]
    meta = metadata(rows)
    backend()._validate_cache_metadata(meta, rows, "document")
    meta["token_audit"].reverse()
    with pytest.raises(ValueError):
        backend()._validate_cache_metadata(meta, rows, "document")


def test_frozen_query_binding_includes_instruction_and_only_original_text_matches(tmp_path):
    rows = [{"query_id": f"KBT{i:03}", "text": f"原文 {i}",
             "text_sha256": hashlib.sha256(f"原文 {i}".encode()).hexdigest()} for i in range(1, 21)]
    path = tmp_path / "queries.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    prepared = [{"unit_id": r["query_id"], "text": INSTRUCTION + r["text"]} for r in rows]
    meta = metadata(prepared, "query")
    mapping = backend()._validate_frozen_queries(path, meta)
    assert mapping["原文 1"]["position"] == 0 and mapping["原文 1"]["query_id"] == "KBT001"
    assert "原文 1 " not in mapping and "原文1" not in mapping
    meta["binding"]["query_instruction"] = ""
    with pytest.raises(ValueError):
        backend()._validate_frozen_queries(path, meta)


def test_model_files_are_sha_bound_and_path_cannot_escape(tmp_path):
    model = tmp_path / REVISION
    model.mkdir()
    (model / "config.json").write_bytes(b"model")
    files = [{"path": "config.json", "size": 5, "sha256": hashlib.sha256(b"model").hexdigest()}]
    fingerprint = {"path": str(model.resolve()), "revision": REVISION, "files": files,
                   "cache_sha256": hashlib.sha256(
                       json.dumps(files, sort_keys=True).encode()).hexdigest()}
    backend()._validate_model_fingerprint(model, fingerprint)
    (model / "config.json").write_bytes(b"other")
    with pytest.raises(ValueError):
        backend()._validate_model_fingerprint(model, fingerprint)
    fingerprint["files"][0]["path"] = "../outside"
    with pytest.raises(ValueError):
        backend()._validate_model_fingerprint(model, fingerprint)


class Array(list):
    def __neg__(self):
        return Array(-value for value in self)

    def __getitem__(self, item):
        result = super().__getitem__(item)
        return Array(result) if isinstance(item, slice) else result


class FakeNumpy:
    @staticmethod
    def lexsort(keys):
        positions, negative_scores = keys
        return Array(sorted(range(len(positions)),
                            key=lambda i: (negative_scores[i], positions[i])))


class Matrix:
    def __matmul__(self, vector):
        assert vector == [1.0, 0.0]
        return Array([0.75, 0.95, 0.95])


def test_worker_ranking_preserves_original_position_and_reports_frozen_replay():
    module = backend()
    worker = object.__new__(module._DenseWorker)
    worker.np = FakeNumpy()
    worker.vectors = Matrix()
    worker.positions = [3, 7, 11]
    worker.unit_ids = ["U3", "U7", "U11"]
    worker.index_units = 65478
    worker.config = {"model_revision": REVISION, "device": "cuda", "precision": "float16"}
    worker.frozen_queries = {
        "原文": {"position": 0, "query_id": "KBT001", "query_tokens": 20}}
    worker.query_vectors = [[1.0, 0.0]]
    def encode(text):
        assert text == "原文 "
        return [1.0, 0.0], 21
    worker._encode_query = encode
    replay = worker.search("原文", 2)
    assert replay["hits"] == [{"unit_id": "U7", "score": 0.95, "rank": 1},
                              {"unit_id": "U11", "score": 0.95, "rank": 2}]
    assert replay["meta"]["index_units"] == 65478
    assert replay["meta"]["mode"] == "frozen_query_vector_replay"
    assert replay["meta"]["query_id"] == "KBT001" and replay["meta"]["query_tokens"] == 20
    live = worker.search("原文 ", 1)
    assert live["meta"]["mode"] == "local_bge_encoding" and live["meta"]["query_tokens"] == 21
    assert live["meta"]["truncated"] is False


def test_worker_rejects_overlength_before_model_call_and_preserves_instruction():
    module = backend()
    worker = object.__new__(module._DenseWorker)
    def tokenize(text, **kwargs):
        assert text == INSTRUCTION + "查询"
        assert kwargs == {"truncation": False, "padding": False}
        return {"input_ids": list(range(513))}
    worker.tokenizer = tokenize
    with pytest.raises(module.QueryTooLong) as caught:
        worker._encode_query("查询")
    assert caught.value.query_tokens == 513


def test_worker_stderr_is_drained_bounded_and_only_safe_fields_are_exposed(monkeypatch, tmp_path):
    import io
    process = FakeProcess()
    packets = [json.dumps({'event': 'diagnostic', 'phase': 'asset_validation',
        'elapsed_ms': i, 'secret': 'Bearer SECRET'}) + '\n' for i in range(2000)]
    packets.append('SECRET header prompt traceback\n')
    process.stderr = io.StringIO(''.join(packets))
    resident, _, _ = fake_resident(monkeypatch, tmp_path, process)
    try:
        resident._stderr_reader.join(timeout=2)
        diagnostics = resident.diagnostics()
        assert len(diagnostics['stderr_tail'].encode('utf-8')) <= 65536
        assert 'SECRET' not in json.dumps(diagnostics)
        assert 'asset_validation' in diagnostics['stderr_tail']
        assert diagnostics['exit_code'] is None
        assert diagnostics['elapsed_ms'] >= 0
        assert resident.search('protocol still valid')['hits']
    finally:
        resident.close()
    assert resident.diagnostics()['exit_code'] == -1
    assert process.stderr.closed


def test_worker_stage_diagnostics_never_write_prompt_or_unapproved_error(capsys):
    module = backend()
    assert hasattr(module, '_diagnostic'), 'worker阶段必须独立写stderr'
    module._diagnostic('model_load', 1.25, error_code='DENSE_STARTUP_FAILED')
    module._diagnostic('SECRET', 2, error_code='Bearer SECRET')
    output = capsys.readouterr()
    assert output.out == ''
    rows = [json.loads(line) for line in output.err.splitlines()]
    assert rows == [{'event': 'diagnostic', 'phase': 'model_load', 'elapsed_ms': 1.25,
                     'error_code': 'DENSE_STARTUP_FAILED'}]
