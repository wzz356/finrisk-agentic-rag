import importlib
import importlib.util
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest


def model():
    assert importlib.util.find_spec("agent_rag.update_model") is not None, "独立生成客户端未实现"
    return importlib.import_module("agent_rag.update_model")


def response(content='{"cases": []}', *, usage=True):
    result = {
        "id": "local-fake-response",
        "model": "deepseek-v4.1-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content,
                },
            }
        ],
    }
    if usage:
        result["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 25,
            "prompt_cache_miss_tokens": 75,
        }
    return result


def client(tmp_path, transport=None, **kwargs):
    return model().UpdateModelClient(tmp_path / "generation.sqlite3", transport=transport, **kwargs)


def test_explicit_generation_payload_and_durable_success_cache(tmp_path):
    calls = []

    def send(payload, timeout):
        calls.append((payload, timeout))
        return response()

    first = client(tmp_path, send)
    assert (
        first.complete("规则", "原文\n保留条件", request_key="case:1", max_tokens=300)
        == '{"cases": []}'
    )
    payload, timeout = calls[0]
    assert payload == {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 300,
        "response_format": {"type": "json_object"},
        "stream": False,
        "messages": [
            {"role": "system", "content": "规则"},
            {"role": "user", "content": "原文\n保留条件"},
        ],
    }
    assert 0 < timeout <= 30
    second = client(tmp_path, lambda *_: pytest.fail("成功请求不能再次收费"))
    assert (
        second.complete("规则", "原文\n保留条件", request_key="case:1", max_tokens=300)
        == '{"cases": []}'
    )
    summary = second.summary()
    assert summary["purpose"] == "case_generation"
    assert summary["requests"] == summary["successful_requests"] == 1
    assert summary["settled_cost_cny"] == pytest.approx(0.000231)
    assert summary["reserved_cost_cny"] == 0


@pytest.mark.parametrize(
    "changed", [{"user": "另一个原文"}, {"system": "新规则"}, {"max_tokens": 512}]
)
def test_request_key_binds_entire_payload(tmp_path, changed):
    subject = client(tmp_path, lambda *_: response())
    subject.complete("规则", "原文", request_key="one", max_tokens=256)
    args = {"system": "规则", "user": "原文", "max_tokens": 256, **changed}
    with pytest.raises(RuntimeError, match="REQUEST_KEY_CONFLICT"):
        subject.complete(**args, request_key="one")
    assert subject.summary()["requests"] == 1


@pytest.mark.parametrize("change", [{"cap_cny": 2}, {"max_requests": 7}])
def test_existing_ledger_limits_are_immutable(tmp_path, change):
    client(tmp_path, lambda *_: response())
    with pytest.raises(RuntimeError, match="POLICY_MISMATCH"):
        client(tmp_path, lambda *_: response(), **change)


def test_budget_reserves_payload_bytes_and_output_before_send(tmp_path):
    calls = []
    subject = client(tmp_path, lambda *args: calls.append(args), cap_cny=0.001)
    with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
        subject.complete("规则", "原文", request_key="large", max_tokens=4096)
    assert calls == []
    assert subject.summary()["requests"] == 0


def test_unknown_usage_keeps_exact_reservation_and_cache(tmp_path):
    captured = []

    def send(payload, timeout):
        captured.append(payload)
        return response(usage=False)

    subject = client(tmp_path, send)
    assert (
        subject.complete("规则", "原文", request_key="unknown", max_tokens=100) == '{"cases": []}'
    )
    payload_bytes = len(
        json.dumps(captured[0], ensure_ascii=False, allow_nan=False).encode("utf-8")
    )
    expected = ((payload_bytes + 512) * 2 + 100 * 8) / 1_000_000
    assert subject.summary()["reserved_cost_cny"] == pytest.approx(expected)
    assert subject.summary()["usage_unknown_requests"] == 1
    restarted = client(tmp_path, lambda *_: pytest.fail("未知费用同键不能重新发送"))
    assert (
        restarted.complete("规则", "原文", request_key="unknown", max_tokens=100) == '{"cases": []}'
    )
    assert restarted.summary()["reserved_cost_cny"] == pytest.approx(expected)


def test_transport_error_is_safe_and_unknown_reservation_persists(tmp_path):
    calls = []
    secret = "sk-secretsecretsecretsecret"

    def send(*_):
        calls.append(1)
        raise TimeoutError("Authorization: Bearer " + secret)

    subject = client(tmp_path, send)
    with pytest.raises(RuntimeError, match="TRANSPORT_FAILED") as failure:
        subject.complete("规则", "原文", request_key="timeout")
    assert secret not in str(failure.value)
    with pytest.raises(RuntimeError, match="ALREADY_SENT"):
        client(tmp_path, send).complete("规则", "原文", request_key="timeout")
    assert calls == [1]
    assert subject.summary()["reserved_cost_cny"] > 0
    assert subject.summary()["usage_unknown_requests"] == 1
    assert secret.encode() not in (tmp_path / "generation.sqlite3").read_bytes()


@pytest.mark.parametrize(
    "change", ["model", "finish", "choices", "content", "json", "array", "nan"]
)
def test_usage_settles_before_rejecting_outer_response_or_json(tmp_path, change):
    result = response()
    if change == "model":
        result["model"] = "unverified-model"
    elif change == "finish":
        result["choices"][0]["finish_reason"] = "length"
    elif change == "choices":
        result["choices"] = []
    elif change == "content":
        result["choices"][0]["message"]["content"] = {}
    else:
        result["choices"][0]["message"]["content"] = {
            "json": "oops",
            "array": "[]",
            "nan": '{"x": NaN}',
        }[change]
    calls = []

    def send(*_):
        calls.append(1)
        return result

    subject = client(tmp_path, send)
    with pytest.raises(RuntimeError, match="INVALID_RESPONSE"):
        subject.complete("规则", "原文", request_key="invalid")
    assert subject.summary()["settled_cost_cny"] == pytest.approx(0.000231)
    assert subject.summary()["reserved_cost_cny"] == 0
    with pytest.raises(RuntimeError, match="ALREADY_SENT"):
        subject.complete("规则", "原文", request_key="invalid")
    assert calls == [1]


@pytest.mark.parametrize(
    "raw",
    [
        None,
        {"prompt_tokens": True},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_cache_hit_tokens": 1,
            "prompt_cache_miss_tokens": 1,
        },
    ],
)
def test_malformed_usage_is_not_refunded(tmp_path, raw):
    result = response()
    result["usage"] = raw
    subject = client(tmp_path, lambda *_: result)
    subject.complete("规则", "原文", request_key="bad-usage")
    assert subject.summary()["settled_cost_cny"] == 0
    assert subject.summary()["reserved_cost_cny"] > 0
    assert subject.summary()["usage_unknown_requests"] == 1


def test_pending_dispatch_survives_process_interruption_without_retry(tmp_path):
    def interrupted(*_):
        raise KeyboardInterrupt()

    subject = client(tmp_path, interrupted)
    with pytest.raises(KeyboardInterrupt):
        subject.complete("规则", "原文", request_key="interrupted")
    with pytest.raises(RuntimeError, match="ALREADY_SENT"):
        client(tmp_path, lambda *_: pytest.fail("中断后不能重发")).complete(
            "规则", "原文", request_key="interrupted"
        )
    assert subject.summary()["requests"] == 1
    assert subject.summary()["reserved_cost_cny"] > 0


def test_concurrent_clients_cannot_exceed_request_limit(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def send(*_):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return response()

    first = client(tmp_path, send, max_requests=1)
    second = client(tmp_path, send, max_requests=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        task = pool.submit(first.complete, "规则", "原文", request_key="first")
        assert started.wait(5)
        try:
            with pytest.raises(RuntimeError, match="REQUEST_LIMIT"):
                second.complete("规则", "原文", request_key="second")
        finally:
            release.set()
        assert task.result() == '{"cases": []}'
    assert calls == [1]


def test_concurrent_same_key_has_only_one_dispatch(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls = []

    def send(*_):
        calls.append(1)
        started.set()
        assert release.wait(5)
        return response()

    first, second = client(tmp_path, send), client(tmp_path, send)
    with ThreadPoolExecutor(max_workers=2) as pool:
        task = pool.submit(first.complete, "规则", "原文", request_key="same")
        assert started.wait(5)
        try:
            with pytest.raises(RuntimeError, match="ALREADY_SENT"):
                second.complete("规则", "原文", request_key="same")
        finally:
            release.set()
        assert task.result() == '{"cases": []}'
    assert calls == [1]


def test_concurrent_requests_cannot_overspend_reserved_budget(tmp_path):
    entered = threading.Event()
    release = threading.Event()

    def send(*_):
        entered.set()
        assert release.wait(5)
        return response(usage=False)

    first = client(tmp_path, send, cap_cny=0.04)
    second = client(tmp_path, send, cap_cny=0.04)
    with ThreadPoolExecutor(max_workers=2) as pool:
        task = pool.submit(first.complete, "规则", "原文", request_key="one")
        assert entered.wait(5)
        try:
            with pytest.raises(RuntimeError, match="BUDGET_EXHAUSTED"):
                second.complete("规则", "原文", request_key="two")
        finally:
            release.set()
        task.result()
    assert first.summary()["requests"] == 1


def test_audit_keeps_payload_response_and_no_authentication_fields(tmp_path, monkeypatch):
    secret = "sk-" + "secret" * 6
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)
    result = response()
    result["headers"] = {"Authorization": "Bearer " + secret}
    result["debug"] = secret
    subject = client(tmp_path, lambda *_: result)
    subject.complete("规则", "完整原文", request_key="audit")
    with sqlite3.connect(tmp_path / "generation.sqlite3") as connection:
        connection.row_factory = sqlite3.Row
        row = dict(connection.execute("SELECT * FROM requests").fetchone())
    assert json.loads(row["request_json"])["messages"][1]["content"] == "完整原文"
    assert json.loads(row["response_json"])["id"] == "local-fake-response"
    assert row["request_key"] == "audit"
    assert len(row["payload_sha256"]) == 64
    audit_text = row["request_json"] + row["response_json"]
    assert secret not in audit_text
    assert "Authorization" not in audit_text


def test_missing_or_ambiguous_key_never_dispatches_and_is_safe(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    subject = client(tmp_path)
    with pytest.raises(RuntimeError, match="KEY_UNAVAILABLE"):
        subject.complete("规则", "原文", request_key="no-key")
    assert subject.summary()["requests"] == 0
    key_file = tmp_path / "key.txt"
    key_file.write_text("sk-" + "a" * 20 + "\nsk-" + "b" * 20, encoding="utf-8")
    ambiguous = client(tmp_path, key_file=key_file)
    with pytest.raises(RuntimeError, match="KEY_UNAVAILABLE") as failure:
        ambiguous.complete("规则", "原文", request_key="no-key")
    assert "aaaa" not in str(failure.value)
    assert ambiguous.summary()["requests"] == 0


def test_explicit_key_file_wins_and_real_transport_uses_low_level_request(tmp_path, monkeypatch):
    module = model()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-" + "b" * 20)
    key_file = tmp_path / "key.txt"
    key_file.write_text("API_KEY=sk-" + "a" * 20, encoding="utf-8-sig")
    calls = []

    def request(payload, timeout, key):
        calls.append((payload, key))
        return response()

    monkeypatch.setattr(module.DeepSeekReranker, "_request", staticmethod(request))
    subject = client(tmp_path, key_file=key_file)
    subject.complete("规则", "原文", request_key="key-file")
    assert calls[0][1] == "sk-" + "a" * 20
    assert calls[0][1].encode() not in (tmp_path / "generation.sqlite3").read_bytes()


def test_foreign_sqlite_ledger_is_not_modified(tmp_path):
    path = tmp_path / "generation.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE old_rerank_ledger (purpose TEXT)")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="LEDGER_INVALID"):
        client(tmp_path, lambda *_: response())
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap_cny": True},
        {"cap_cny": float("nan")},
        {"cap_cny": -1},
        {"max_requests": True},
        {"max_requests": -1},
    ],
)
def test_invalid_configuration_is_rejected_without_ledger(tmp_path, kwargs):
    with pytest.raises(ValueError, match="CONFIGURATION"):
        client(tmp_path, lambda *_: response(), **kwargs)
    assert not (tmp_path / "generation.sqlite3").exists()


@pytest.mark.parametrize(
    "kwargs",
    [{"system": ""}, {"user": ""}, {"request_key": ""}, {"max_tokens": True}, {"max_tokens": 0}],
)
def test_invalid_request_is_rejected_before_reservation(tmp_path, kwargs):
    subject = client(tmp_path, lambda *_: pytest.fail("无效请求不能发送"))
    args = {"system": "规则", "user": "原文", "request_key": "valid", **kwargs}
    with pytest.raises(ValueError, match="INPUT"):
        subject.complete(**args)
    assert subject.summary()["requests"] == 0


def test_provider_usage_beyond_reservation_halts_further_dispatch(tmp_path):
    result = response()
    result["usage"] = {
        "prompt_tokens": 1_000_000,
        "completion_tokens": 1,
        "total_tokens": 1_000_001,
    }
    subject = client(tmp_path, lambda *_: result)
    with pytest.raises(RuntimeError, match="USAGE_EXCEEDED_RESERVATION"):
        subject.complete("规则", "原文", request_key="overrun")
    assert subject.summary()["settled_cost_cny"] == pytest.approx(2.000008)
    assert subject.summary()["reserved_cost_cny"] == 0
    with pytest.raises(RuntimeError, match="BUDGET_HALTED"):
        subject.complete("规则", "原文", request_key="next")
    assert subject.summary()["requests"] == 1


def test_response_audit_redacts_secret_dictionary_keys(tmp_path):
    secret = "sk-" + "sensitive" * 4
    result = response()
    result[secret] = "echo"
    subject = client(tmp_path, lambda *_: result)
    subject.complete("规则", "原文", request_key="redact-key")
    with sqlite3.connect(tmp_path / "generation.sqlite3") as connection:
        audit = connection.execute("SELECT response_json FROM requests").fetchone()[0]
    assert secret not in audit


@pytest.mark.parametrize("field", ["response_json", "content"])
def test_audit_or_completion_write_failure_does_not_erase_usage_or_retry(tmp_path, field):
    calls = []

    def send(*_):
        calls.append(1)
        return response()

    subject = client(tmp_path, send)
    with sqlite3.connect(tmp_path / "generation.sqlite3") as connection:
        connection.execute(f"""CREATE TRIGGER fail_update BEFORE UPDATE OF {field} ON requests
            BEGIN SELECT RAISE(ABORT, 'simulated write failure'); END""")
    with pytest.raises(RuntimeError, match="LEDGER_INVALID"):
        subject.complete("规则", "原文", request_key="write-fail")
    assert subject.summary()["settled_cost_cny"] == pytest.approx(0.000231)
    assert subject.summary()["reserved_cost_cny"] == 0
    with pytest.raises(RuntimeError, match="ALREADY_SENT"):
        client(tmp_path, send).complete("规则", "原文", request_key="write-fail")
    assert calls == [1]


def test_request_audit_failure_never_dispatches(tmp_path):
    calls = []
    subject = client(tmp_path, lambda *_: calls.append(1))
    with sqlite3.connect(tmp_path / "generation.sqlite3") as connection:
        connection.execute("""CREATE TRIGGER fail_insert BEFORE INSERT ON requests
            BEGIN SELECT RAISE(ABORT, 'simulated request audit failure'); END""")
    with pytest.raises(RuntimeError, match="LEDGER_INVALID"):
        subject.complete("规则", "原文", request_key="audit-fail")
    assert subject.summary()["requests"] == 0
    assert calls == []


def test_replay_only_returns_persisted_success_without_key_or_transport(tmp_path, monkeypatch):
    subject = client(tmp_path, lambda *_: response())
    expected = subject.complete("规则", "原文", request_key="cached")
    before = subject.summary()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    replay = client(tmp_path)
    assert replay.complete("规则", "原文", request_key="cached", replay_only=True) == expected
    assert replay.summary() == before


@pytest.mark.parametrize("injected_transport", [False, True])
def test_replay_only_missing_response_has_zero_dispatch_and_reservation(
    tmp_path, monkeypatch, injected_transport
):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    transport = (lambda *_: pytest.fail("只读回放不能发送")) if injected_transport else None
    subject = client(tmp_path, transport)
    before = subject.summary()
    with pytest.raises(RuntimeError, match="UPDATE_MODEL_REPLAY_NOT_AVAILABLE"):
        subject.complete("规则", "原文", request_key="missing", replay_only=True)
    assert subject.summary() == before
    assert subject.summary()["requests"] == 0
    assert subject.summary()["reserved_cost_cny"] == 0


def test_replay_only_still_rejects_conflicting_payload(tmp_path):
    subject = client(tmp_path, lambda *_: response())
    subject.complete("规则", "原文", request_key="bound")
    before = subject.summary()
    with pytest.raises(RuntimeError, match="REQUEST_KEY_CONFLICT"):
        subject.complete("规则", "修改后的原文", request_key="bound", replay_only=True)
    assert subject.summary() == before


@pytest.mark.parametrize("flag", [0, 1, None, "true"])
def test_replay_only_requires_boolean_before_reservation(tmp_path, flag):
    subject = client(tmp_path, lambda *_: pytest.fail("无效参数不能发送"))
    with pytest.raises(ValueError, match="INPUT"):
        subject.complete("规则", "原文", request_key="invalid-flag", replay_only=flag)
    assert subject.summary()["requests"] == 0


@pytest.mark.parametrize("known_usage", [False, True])
def test_replay_only_does_not_retry_previously_failed_requests(tmp_path, known_usage):
    result = response(usage=known_usage)
    result["choices"] = []
    subject = client(tmp_path, lambda *_: result)
    with pytest.raises(RuntimeError, match="INVALID_RESPONSE"):
        subject.complete("规则", "原文", request_key="failed")
    before = subject.summary()
    replay = client(tmp_path, lambda *_: pytest.fail("失败请求不能重新发送"))
    with pytest.raises(RuntimeError, match="ALREADY_SENT"):
        replay.complete("规则", "原文", request_key="failed", replay_only=True)
    assert replay.summary() == before
