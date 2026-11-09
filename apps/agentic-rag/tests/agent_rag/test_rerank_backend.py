import importlib
import importlib.util
import json
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest


def backend():
    assert importlib.util.find_spec("agent_rag.rerank_backend") is not None, "重排后端未实现"
    return importlib.import_module("agent_rag.rerank_backend")


def unit(number=1, text="允许\n部分领取，需符合合同条件。", file_id=None):
    return {"unit_id": f"U{number}", "file_id": file_id or f"F{number}", "text": text}


def response(payload, **changes):
    packet = json.loads(payload["messages"][1]["content"])
    opinion = {
        "pair_id": packet["pair_id"],
        "score": 85,
        "level": "specific_direct",
        "source_quote": packet["unit_text"],
        "reason": "支持部分领取的合同条件",
        "limitation": "不能推断其他合同权益",
        "human_approved": False,
    }
    opinion.update(changes)
    return {
        "id": "fake-response",
        "model": "deepseek-v4.1-flash",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps(opinion, ensure_ascii=False),
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "prompt_cache_hit_tokens": 1,
            "prompt_cache_miss_tokens": 99,
        },
    }


def control(tmp_path, **kwargs):
    return backend().RerankTaskControl(
        tmp_path / "ledger.json", **{"cap_cny": 1, "max_requests": 100, **kwargs}
    )


def test_payload_preserves_experiment_protocol_and_only_pair_text(tmp_path):
    module = backend()
    captured = []

    def send(payload, timeout_seconds):
        captured.append((payload, timeout_seconds))
        return response(payload)

    result = module.DeepSeekReranker("原规则", send).score_query(
        "完整原查询\n第二行", [unit()], control(tmp_path)
    )
    payload, timeout = captured[0]
    packet = json.loads(payload["messages"][1]["content"])
    assert set(packet) == {"pair_id", "query_text", "unit_text"}
    assert packet["query_text"] == "完整原查询\n第二行"
    assert packet["unit_text"] == unit()["text"]
    assert payload == {
        "model": "deepseek-flash",
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "原规则\n每次仅评下面一对，输出一个JSON对象，不用代码围栏。"
                    "reason和limitation各不超过50字，保持具体；"
                    "只取所需短连续原文quote，保留原换行。"
                ),
            },
            {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
        ],
    }
    assert 0 < timeout <= 30
    assert result["complete"] and result["scores"][0]["unit_id"] == "U1"
    assert result["scores"][0]["file_id"] == "F1"
    assert result["meta"]["rerank_mode"] == "live"
    assert result["meta"]["external_retrieval_requests"] == 1
    assert len(result["meta"]["rules_sha256"]) == 64


@pytest.mark.parametrize(
    "changes",
    [
        {"pair_id": "wrong"},
        {"score": True},
        {"score": 80.0},
        {"score": 79},
        {"source_quote": "不存在的引文"},
        {"source_quote": ""},
        {"human_approved": True},
        {"human_approved": 0},
        {"reason": " "},
        {"limitation": ""},
    ],
)
def test_rejects_invalid_opinion_without_fabricating_missing_score(tmp_path, changes):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload, **changes)

    result = (
        backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    )
    assert not result["complete"] and result["scores"] == []
    assert len(result["errors"]) == 1 and len(calls) == 2
    assert result["meta"]["external_retrieval_requests"] == 2


@pytest.mark.parametrize(
    "field,value", [("model", "other"), ("finish_reason", "length"), ("content", {})]
)
def test_rejects_unverified_model_and_incomplete_or_nontext_response(tmp_path, field, value):
    def send(payload, timeout_seconds):
        result = response(payload)
        if field == "model":
            result[field] = value
        elif field == "finish_reason":
            result["choices"][0][field] = value
        else:
            result["choices"][0]["message"][field] = value
        return result

    result = (
        backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    )
    assert not result["complete"] and not result["scores"]


def test_recovers_only_unique_whitespace_difference(tmp_path):
    def send(payload, timeout_seconds):
        return response(payload, source_quote="允许 部分领取")

    result = (
        backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], control(tmp_path))
    )
    assert result["scores"][0]["source_quote"] == "允许\n部分领取"
    assert result["scores"][0]["model_source_quote_raw"] == "允许 部分领取"
    assert result["scores"][0]["quote_whitespace_restored"] is True
    duplicate = (
        backend()
        .DeepSeekReranker("规则", send)
        .score_query(
            "查询",
            [unit(text="允许\n部分领取。允许\t部分领取。")],
            control(tmp_path, max_requests=100),
        )
    )
    assert not duplicate["complete"] and duplicate["scores"] == []


def test_cache_binds_original_query_unit_text_and_id_and_rules(tmp_path):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    module = backend()
    reranker = module.DeepSeekReranker("规则A", send, cache_size=2)
    budget = control(tmp_path)
    first = reranker.score_query("查询\n原文", [unit()], budget)
    cached = reranker.score_query("查询\n原文", [unit(file_id="F-new")], budget)
    assert first["scores"][0]["file_id"] == "F1"
    assert cached["scores"][0]["file_id"] == "F-new"
    assert cached["meta"]["cache_hits"] == 0 and cached["meta"]["external_retrieval_requests"] == 1
    cached["scores"][0]["score"] = 0
    assert reranker.score_query("查询\n原文", [unit()], budget)["scores"][0]["score"] == 85
    reranker.score_query("查询 原文", [unit()], budget)
    reranker.score_query("查询\n原文", [unit(2)], budget)
    reranker.score_query("查询\n原文", [unit(text="变更后的原文")], budget)
    module.DeepSeekReranker("规则B", send).score_query("查询\n原文", [unit()], budget)
    assert len(calls) == 6
    assert budget.meta()["total_external_retrieval_requests"] == 6


def test_failed_response_never_enters_success_cache_or_restores_pair_allowance(tmp_path):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload, pair_id="wrong")

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path)
    for _ in range(2):
        assert not reranker.score_query("查询", [unit()], budget)["complete"]
    assert len(calls) == 2
    assert not reranker._cache


def test_reservation_is_persisted_before_transport_and_usage_is_settled(tmp_path):
    module = backend()
    budget = control(tmp_path)

    def send(payload, timeout_seconds):
        ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
        body_bytes = len(json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"))
        assert ledger["reserved_cost_micro_cny"] == (body_bytes + 512) * 2 + 512 * 8
        assert len(ledger["request_reservations"]) == 1
        return response(payload)

    result = module.DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert result["complete"]
    meta = budget.meta()
    assert meta["settled_cost_micro_cny"] == 1 + 99 * 2 + 10 * 8
    assert meta["reserved_cost_micro_cny"] == 0
    assert meta["price_source"] == "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/"
    assert meta["price_checked_date"] == "2026-10-15"


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 3},
        {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_cache_hit_tokens": 2,
            "prompt_cache_miss_tokens": 0,
        },
    ],
)
def test_unknown_or_invalid_usage_keeps_reservation(tmp_path, usage):
    def send(payload, timeout_seconds):
        result = response(payload)
        result["usage"] = usage
        return result

    budget = control(tmp_path)
    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert result["complete"]
    assert budget.meta()["reserved_cost_micro_cny"] > 0
    assert budget.meta()["settled_cost_micro_cny"] == 0
    assert budget.meta()["unsettled_requests"] == 1


def test_restart_preserves_cumulative_request_cap_and_unknown_cost(tmp_path):
    module = backend()

    def send(payload, timeout_seconds):
        result = response(payload)
        result.pop("usage")
        return result

    module.DeepSeekReranker("规则", send).score_query(
        "查询", [unit()], control(tmp_path, max_requests=1)
    )
    restarted = control(tmp_path, max_requests=1)
    before = restarted.meta()["reserved_cost_micro_cny"]
    result = module.DeepSeekReranker("规则", send).score_query("另一查询", [unit()], restarted)
    assert not result["complete"] and result["meta"]["external_retrieval_requests"] == 0
    assert result["meta"]["total_external_retrieval_requests"] == 1
    assert restarted.meta()["reserved_cost_micro_cny"] == before
    with pytest.raises(ValueError):
        control(tmp_path, max_requests=2)


def test_cap_denies_request_before_transport(tmp_path):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    budget = control(tmp_path, cap_cny=0.001)
    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert not result["complete"] and calls == []
    assert budget.meta()["total_external_retrieval_requests"] == 0
    assert budget.meta()["booked_cost_micro_cny"] <= budget.meta()["cap_micro_cny"]


def test_concurrent_unknown_usage_reservations_cannot_exceed_cap(tmp_path):
    lock = threading.Lock()
    calls = []

    def send(payload, timeout_seconds):
        with lock:
            calls.append(payload)
        time.sleep(0.01)
        result = response(payload)
        result.pop("usage")
        return result

    budget = control(tmp_path, cap_cny=0.02)
    result = (
        backend()
        .DeepSeekReranker("规则", send)
        .score_query("查询", [unit(i) for i in range(30)], budget)
    )
    meta = budget.meta()
    assert not result["complete"] and 0 < len(calls) < 30
    assert meta["booked_cost_micro_cny"] <= meta["cap_micro_cny"]
    assert meta["total_external_retrieval_requests"] == len(calls)


def test_all_reranker_instances_share_concurrency_limit_eight(tmp_path):
    lock = threading.Lock()
    eight_entered = threading.Event()
    release = threading.Event()
    active = 0
    peak = 0

    def send(payload, timeout_seconds):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(active, peak)
            if active == 8:
                eight_entered.set()
        assert release.wait(5)
        with lock:
            active -= 1
        return response(payload)

    module = backend()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                module.DeepSeekReranker("规则", send).score_query,
                "查询",
                [unit(i) for i in range(12)],
                module.RerankTaskControl(tmp_path / f"ledger-{n}.json", 1, 100),
            )
            for n in range(2)
        ]
        assert eight_entered.wait(5)
        assert peak == 8
        release.set()
        results = [future.result(timeout=10) for future in futures]
    assert all(result["complete"] for result in results) and peak == 8


def test_cancel_stops_queue_and_retry_but_settles_inflight_success(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    calls = 0

    def send(payload, timeout_seconds):
        nonlocal calls
        with lock:
            calls += 1
            if calls == 8:
                entered.set()
        assert release.wait(5)
        return response(payload)

    budget = control(tmp_path)
    reranker = backend().DeepSeekReranker("规则", send)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(reranker.score_query, "查询", [unit(i) for i in range(30)], budget)
        assert entered.wait(5)
        budget.cancel()
        assert not budget.check()
        release.set()
        result = future.result(timeout=10)
    assert calls == 8 and len(result["scores"]) == 8
    assert not result["complete"] and budget.meta()["reserved_cost_micro_cny"] == 0
    assert budget.meta()["settled_cost_micro_cny"] == 8 * 279
    assert all(error["reason"] == "cancelled" for error in result["errors"])


def test_each_pair_has_at_most_one_retry_and_shares_task_limit(tmp_path):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        raise TimeoutError("fake timeout")

    budget = control(tmp_path, max_requests=3)
    result = (
        backend().DeepSeekReranker("规则", send).score_query("查询", [unit(1), unit(2)], budget)
    )
    assert not result["complete"] and len(calls) == 3
    per_pair = {}
    for payload in calls:
        pair_id = json.loads(payload["messages"][1]["content"])["pair_id"]
        per_pair[pair_id] = per_pair.get(pair_id, 0) + 1
    assert max(per_pair.values()) <= 2
    assert budget.meta()["unsettled_requests"] == 3


def test_cancel_after_failure_prevents_retry(tmp_path):
    budget = control(tmp_path)
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        budget.cancel()
        raise TimeoutError("fake timeout")

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert not result["complete"] and len(calls) == 1


def test_timeout_passed_to_transport_does_not_exceed_task_time(tmp_path):
    budget = control(tmp_path, timeout_seconds=0.2)

    def send(payload, timeout_seconds):
        assert 0 < timeout_seconds <= 0.2
        return response(payload)

    assert (
        backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)["complete"]
    )
    time.sleep(0.21)
    assert budget.remaining_seconds() == 0 and not budget.check()
    restarted = control(tmp_path, timeout_seconds=0.2)
    assert restarted.remaining_seconds() == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cap_cny": 0},
        {"cap_cny": float("nan")},
        {"max_requests": 0},
        {"max_requests": True},
        {"timeout_seconds": 0},
    ],
)
def test_live_control_requires_valid_explicit_caps(tmp_path, kwargs):
    with pytest.raises(ValueError):
        control(tmp_path, **kwargs)


def test_missing_key_does_not_open_transport_or_log_secrets(tmp_path, monkeypatch):
    module = backend()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setattr(
        module.urllib.request, "build_opener", lambda *args: pytest.fail("不得出网")
    )
    result = module.DeepSeekReranker("规则").score_query("查询", [unit()], control(tmp_path))
    assert not result["complete"] and result["meta"]["external_retrieval_requests"] == 0


def test_default_transport_uses_official_endpoint_no_redirect_and_redacts(tmp_path, monkeypatch):
    module = backend()
    key = "sk-fake-only-for-tests-0123456789"

    class FakeReply:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            return self.body

    class FakeOpener:
        def open(self, request, timeout):
            assert request.full_url == "https://api.deepseek.com/chat/completions"
            assert request.get_header("Authorization") == "Bearer " + key
            payload = json.loads(request.data)
            return FakeReply(
                json.dumps(response(payload, reason="理由" + key), ensure_ascii=False).encode(
                    "utf-8"
                )
            )

    def opener(handler):
        assert isinstance(handler, module.NoRedirect)
        with pytest.raises(ValueError):
            handler.redirect_request(None, None, None, None, None, None)
        return FakeOpener()

    monkeypatch.setattr(module.urllib.request, "build_opener", opener)
    result = module.DeepSeekReranker("规则", key_loader=lambda: key).score_query(
        "查询", [unit()], control(tmp_path)
    )
    assert result["complete"]
    assert key not in json.dumps(result) and key not in (tmp_path / "ledger.json").read_text(
        encoding="utf-8"
    )


def test_cache_hit_is_usable_after_request_cap_without_http(tmp_path):
    def send(payload, timeout_seconds):
        return response(payload)

    reranker = backend().DeepSeekReranker("规则", send)
    budget = control(tmp_path, max_requests=1)
    assert reranker.score_query("查询", [unit()], budget)["complete"]
    cached = reranker.score_query("查询", [unit()], budget)
    assert cached["complete"] and cached["meta"]["cache_hits"] == 1
    assert cached["meta"]["external_retrieval_requests"] == 0


def test_invalid_existing_ledger_is_rejected_instead_of_reset(tmp_path):
    control(tmp_path)
    data = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    data["reserved_cost_micro_cny"] = 99
    (tmp_path / "ledger.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        control(tmp_path)


def test_cancel_during_durable_reservation_does_not_dispatch(tmp_path, monkeypatch):
    budget = control(tmp_path)
    original_persist = budget._persist

    def cancel_after_write(ledger):
        original_persist(ledger)
        if ledger["request_reservations"]:
            budget.cancel_event.set()

    monkeypatch.setattr(budget, "_persist", cancel_after_write)
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert not result["complete"] and calls == []
    assert result["meta"]["external_retrieval_requests"] == 0
    assert budget.meta()["total_external_retrieval_requests"] == 0
    assert budget.meta()["reserved_cost_micro_cny"] == 0


def test_expired_during_durable_reservation_does_not_dispatch(tmp_path, monkeypatch):
    budget = control(tmp_path, timeout_seconds=0.05)
    original_persist = budget._persist

    def expire_after_write(ledger):
        original_persist(ledger)
        if ledger["request_reservations"]:
            time.sleep(0.06)

    monkeypatch.setattr(budget, "_persist", expire_after_write)
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert not result["complete"] and calls == []
    assert budget.meta()["total_external_retrieval_requests"] == 0
    assert budget.meta()["reserved_cost_micro_cny"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("price_source", "wrong"),
        ("cap_micro_cny", True),
        ("price_checked_date", None),
        ("max_requests", True),
        ("cap_cny", 999),
    ],
)
def test_rejects_modified_persistent_limits_and_price_provenance(tmp_path, field, value):
    control(tmp_path, cap_cny=0.000001, max_requests=1)
    path = tmp_path / "ledger.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data[field] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        control(tmp_path, cap_cny=0.000001, max_requests=1)


def test_key_file_is_explicit_and_ambiguous_file_never_dispatches(tmp_path, monkeypatch):
    module = backend()
    fake_key_path = tmp_path / "fake-key.txt"
    fake_key_path.write_text(
        "sk-fake-key-01234567890123456789 sk-second-key-0123456789012345", encoding="utf-8"
    )
    monkeypatch.setattr(
        module.urllib.request, "build_opener", lambda *args: pytest.fail("不得出网")
    )
    result = module.DeepSeekReranker("规则", key_file=fake_key_path).score_query(
        "查询", [unit()], control(tmp_path)
    )
    assert not result["complete"] and result["meta"]["external_retrieval_requests"] == 0


def test_usage_above_reservation_is_recorded_and_blocks_further_requests(tmp_path):
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        result = response(payload)
        result["usage"] = {
            "prompt_tokens": 1_000_000,
            "completion_tokens": 1_000_000,
            "total_tokens": 2_000_000,
        }
        return result

    budget = control(tmp_path)
    reranker = backend().DeepSeekReranker("规则", send)
    first = reranker.score_query("查询", [unit()], budget)
    assert first["complete"] and budget.meta()["settled_cost_micro_cny"] == 10_000_000
    assert budget.meta()["stop_reason"] == "usage_exceeded_reservation"
    second = reranker.score_query("另一查询", [unit()], budget)
    assert not second["complete"] and len(calls) == 1


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_stop_after_reservation_before_dispatch_causes_zero_http_attempts(
    tmp_path, monkeypatch, stop
):
    budget = control(tmp_path, timeout_seconds=0.2 if stop == "deadline" else 180)
    paused = threading.Event()
    release = threading.Event()
    reserve = budget._reserve

    def pause_after_reserve(*args):
        reserved = reserve(*args)
        paused.set()
        assert release.wait(5)
        return reserved

    monkeypatch.setattr(budget, "_reserve", pause_after_reserve)
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            backend().DeepSeekReranker("规则", send).score_query, "查询", [unit()], budget
        )
        assert paused.wait(5)
        if stop == "cancel":
            budget.cancel()
        else:
            time.sleep(0.25)
        assert not budget.check()
        release.set()
        result = future.result(timeout=5)
    assert not result["complete"] and calls == []
    assert result["meta"]["external_retrieval_requests"] == 0
    assert budget.meta()["total_external_retrieval_requests"] == 0
    assert budget.meta()["reserved_cost_micro_cny"] == 0


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
def test_committed_but_unstarted_transport_is_aborted_without_http_count(
    tmp_path, monkeypatch, stop
):
    budget = control(tmp_path, timeout_seconds=0.2 if stop == "deadline" else 180)
    assert hasattr(budget, "_commit_dispatch"), "缺少原子派发提交边界"
    paused = threading.Event()
    release = threading.Event()
    commit = budget._commit_dispatch

    def pause_after_commit(*args):
        result = commit(*args)
        paused.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(budget, "_commit_dispatch", pause_after_commit)
    calls = []

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            backend().DeepSeekReranker("规则", send).score_query, "查询", [unit()], budget
        )
        assert paused.wait(5)
        if stop == "cancel":
            budget.cancel()
        else:
            time.sleep(0.25)
        release.set()
        result = future.result(timeout=5)
    assert not result["complete"] and calls == []
    assert result["meta"]["external_retrieval_requests"] == 0
    assert budget.meta()["total_external_retrieval_requests"] == 0
    assert budget.meta()["total_dispatch_admissions"] == 1
    assert budget.meta()["reserved_cost_micro_cny"] == 0
    row = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))[
        "request_reservations"
    ][0]
    assert row["dispatch_state"] == "aborted"
    assert row["dispatch_committed_at"] and "transport_started_at" not in row


def test_ledger_distinguishes_dispatch_commit_and_actual_transport_start(tmp_path):
    budget = control(tmp_path)

    def send(payload, timeout_seconds):
        row = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))[
            "request_reservations"
        ][0]
        assert row["dispatch_state"] == "transport_started"
        assert row["dispatch_committed_at"] and row["transport_started_at"]
        assert budget.meta()["total_external_retrieval_requests"] == 1
        return response(payload)

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert result["complete"]
    assert result["meta"]["total_dispatch_admissions"] == 1


def test_legacy_ledger_migration_preserves_unknown_fee_and_request_count(tmp_path):
    def send(payload, timeout_seconds):
        result = response(payload)
        result.pop("usage")
        return result

    budget = control(tmp_path, max_requests=1)
    backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    path = tmp_path / "ledger.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    reserved = data["reserved_cost_micro_cny"]
    data["schema_version"] = "finrisk-rerank-budget/1.0"
    for row in data["request_reservations"]:
        for field in ("dispatch_state", "dispatch_committed_at", "transport_started_at"):
            row.pop(field)
    path.write_text(json.dumps(data), encoding="utf-8")
    restarted = control(tmp_path, max_requests=1)
    assert not restarted.check()
    assert restarted.meta()["total_external_retrieval_requests"] == 1
    assert restarted.meta()["reserved_cost_micro_cny"] == reserved


def test_malformed_legacy_reservation_is_rejected_as_invalid_ledger(tmp_path):
    control(tmp_path)
    path = tmp_path / "ledger.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = "finrisk-rerank-budget/1.0"
    data["request_reservations"] = [None]
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_EXISTING_RERANK_LEDGER"):
        control(tmp_path)


def attempt_audit(tmp_path):
    data = json.loads((tmp_path / "ledger.json.attempts.json").read_text(encoding="utf-8"))
    assert data["schema_version"] == "finrisk-rerank-attempts/1.0"
    return data["attempts"]


@pytest.mark.parametrize(
    "failure,state,error_code,http_status,usage_state,attempts",
    [
        ("http429", "finished_usage_unknown", "HTTP_429", 429, "unknown", 2),
        ("http503", "finished_usage_unknown", "HTTP_5XX", 503, "unknown", 2),
        ("http401", "finished_usage_unknown", "HTTP_4XX", 401, "unknown", 1),
        ("timeout", "finished_usage_unknown", "TRANSPORT_TIMEOUT", None, "unknown", 2),
        ("invalid_json", "finished_usage_unknown", "INVALID_JSON", None, "unknown", 2),
        ("quote", "settled", "QUOTE_MISMATCH", None, "known", 2),
        ("missing_usage", "finished_usage_unknown", None, None, "unknown", 1),
    ],
)
def test_attempt_audit_records_one_terminal_per_attempt_without_leaking_errors(
    tmp_path, failure, state, error_code, http_status, usage_state, attempts
):
    secret_body = "sensitive-response-body-with-prompt-and-auth"

    def send(payload, timeout_seconds):
        if failure.startswith("http"):
            raise urllib.error.HTTPError(
                "https://api.deepseek.com/chat/completions",
                int(failure[4:]), secret_body, {}, None,
            )
        if failure == "timeout":
            raise TimeoutError(secret_body)
        if failure == "invalid_json":
            raise json.JSONDecodeError(secret_body, "private response", 0)
        if failure == "quote":
            return response(payload, source_quote="引文错位")
        result = response(payload)
        result.pop("usage")
        return result

    budget = control(tmp_path)
    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    rows = attempt_audit(tmp_path)
    assert [row["ordinal"] for row in rows] == list(range(1, attempts + 1))
    assert [row["retry_index"] for row in rows] == list(range(attempts))
    assert [row["will_retry"] for row in rows] == ([True, False] if attempts == 2 else [False])
    assert all(row["state"] == state and row["completed_at"] for row in rows)
    assert all(row["http_status"] == http_status for row in rows)
    assert all(row["error_code"] == error_code for row in rows)
    assert all(row["usage_state"] == usage_state for row in rows)
    assert secret_body not in json.dumps(rows)
    assert result["meta"]["in_flight_requests"] == 0
    assert result["meta"]["finished_usage_unknown_requests"] == (
        attempts if usage_state == "unknown" else 0
    )
    assert result["meta"]["settled_requests"] == (attempts if usage_state == "known" else 0)
    ledger = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["schema_version"] == "finrisk-rerank-budget/1.1"
    expected_reserved = sum(
        row["reserved_micro_cny"]
        for row in ledger["request_reservations"]
        if not row.get("settlement")
    )
    assert result["meta"]["reserved_cost_micro_cny"] == expected_reserved
    assert result["meta"]["retained_reserved_micro_cny"] == expected_reserved


def test_attempt_audit_distinguishes_inflight_from_finished_unknown_and_survives_reopen(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    budget = control(tmp_path)

    def send(payload, timeout_seconds):
        entered.set()
        assert release.wait(5)
        result = response(payload)
        result.pop("usage")
        return result

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            backend().DeepSeekReranker("规则", send).score_query, "查询", [unit()], budget
        )
        assert entered.wait(5)
        try:
            row = attempt_audit(tmp_path)[0]
            assert row["state"] == "transport_started" and row["completed_at"] is None
            assert budget.meta()["in_flight_requests"] == 1
            assert budget.meta()["finished_usage_unknown_requests"] == 0
        finally:
            release.set()
        assert future.result(timeout=5)["complete"]
    assert budget.meta()["in_flight_requests"] == 0
    assert budget.meta()["finished_usage_unknown_requests"] == 1
    ledger_bytes = (tmp_path / "ledger.json").read_bytes()
    audit_bytes = (tmp_path / "ledger.json.attempts.json").read_bytes()
    restarted = control(tmp_path)
    assert restarted.meta()["finished_usage_unknown_requests"] == 1
    assert restarted.meta()["reserved_cost_micro_cny"] > 0
    assert (tmp_path / "ledger.json").read_bytes() == ledger_bytes
    assert (tmp_path / "ledger.json.attempts.json").read_bytes() == audit_bytes


def test_attempt_audit_records_cancelled_unsent_using_existing_release_rule(tmp_path, monkeypatch):
    budget = control(tmp_path)
    persist = budget._persist

    def cancel_after_reservation(ledger):
        persist(ledger)
        if ledger["request_reservations"]:
            budget.cancel_event.set()

    monkeypatch.setattr(budget, "_persist", cancel_after_reservation)
    result = backend().DeepSeekReranker(
        "规则", lambda *args: pytest.fail("取消后不得发送")
    ).score_query("查询", [unit()], budget)
    row = attempt_audit(tmp_path)[0]
    assert row["state"] == "aborted_unsent" and row["retry_index"] == 0
    assert row["completed_at"] and row["will_retry"] is False
    assert row["usage_state"] == "not_sent" and row["error_code"] == "CANCELLED"
    assert row["http_status"] is None
    assert result["meta"]["aborted_unsent_requests"] == 1
    assert result["meta"]["reserved_cost_micro_cny"] == 0
    assert result["meta"]["total_external_retrieval_requests"] == 0


def test_attempt_audit_does_not_claim_retry_when_cancelled_or_request_limit_reached(tmp_path):
    budget = control(tmp_path, max_requests=1)

    def send(payload, timeout_seconds):
        raise TimeoutError("fake timeout")

    backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert attempt_audit(tmp_path)[0]["will_retry"] is False
    assert budget.meta()["reserved_cost_micro_cny"] > 0


def test_unaudited_historical_unknown_usage_is_not_relabelled_or_rewritten(tmp_path):
    budget = control(tmp_path)

    def send(payload, timeout_seconds):
        result = response(payload)
        result.pop("usage")
        return result

    backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    (tmp_path / "ledger.json.attempts.json").unlink()
    before = (tmp_path / "ledger.json").read_bytes()
    meta = control(tmp_path).meta()
    assert meta["unaudited_requests"] == 1
    assert meta["in_flight_requests"] == 0 and meta["finished_usage_unknown_requests"] == 0
    assert meta["reserved_cost_micro_cny"] > 0
    assert (tmp_path / "ledger.json").read_bytes() == before
    assert not (tmp_path / "ledger.json.attempts.json").exists()


@pytest.mark.parametrize("outcome", ["success", "http503", "timeout", "invalid_json"])
def test_default_transport_preserves_structured_diagnostics_in_audit(
    tmp_path, monkeypatch, outcome
):
    module = backend()

    class Reply:
        status = 200

        def __init__(self, request):
            self.request = request

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self):
            if outcome == "invalid_json":
                return b"private-invalid-json"
            return json.dumps(response(json.loads(self.request.data))).encode("utf-8")

    class Opener:
        def open(self, request, timeout):
            if outcome == "http503":
                raise urllib.error.HTTPError(request.full_url, 503, "private-body", {}, None)
            if outcome == "timeout":
                raise urllib.error.URLError(TimeoutError("private-timeout"))
            return Reply(request)

    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *args: Opener())
    result = module.DeepSeekReranker(
        "规则", key_loader=lambda: "sk-fake-key-for-audit-123456789"
    ).score_query("查询", [unit()], control(tmp_path))
    rows = attempt_audit(tmp_path)
    assert all(row["http_status"] == (503 if outcome == "http503" else (
        None if outcome == "timeout" else 200
    )) for row in rows)
    assert all(row["error_code"] == {
        "success": None, "http503": "HTTP_5XX", "timeout": "TRANSPORT_TIMEOUT",
        "invalid_json": "INVALID_JSON",
    }[outcome] for row in rows)
    assert result["complete"] is (outcome == "success")
    assert "private-" not in json.dumps(rows)


@pytest.mark.parametrize("failed_state", ["transport_started", "settled"])
def test_attempt_audit_write_failure_never_creates_paid_retry(tmp_path, monkeypatch, failed_state):
    budget = control(tmp_path)
    persist_attempts = budget._persist_attempts
    failed = False
    calls = []

    def fail_once(audit):
        nonlocal failed
        if not failed and any(row["state"] == failed_state for row in audit["attempts"]):
            failed = True
            raise OSError("local audit unavailable")
        persist_attempts(audit)

    monkeypatch.setattr(budget, "_persist_attempts", fail_once)

    def send(payload, timeout_seconds):
        calls.append(payload)
        return response(payload)

    result = backend().DeepSeekReranker("规则", send).score_query("查询", [unit()], budget)
    assert not result["complete"]
    assert len(calls) == (0 if failed_state == "transport_started" else 1)
    assert budget.meta()["total_external_retrieval_requests"] == len(calls)
    assert budget.meta()["reserved_cost_micro_cny"] == 0
    rows = attempt_audit(tmp_path)
    assert len(rows) == 1 and rows[0]["completed_at"] and rows[0]["will_retry"] is False


def test_attempt_terminal_record_is_immutable_and_not_written_twice(tmp_path, monkeypatch):
    budget = control(tmp_path)
    backend().DeepSeekReranker("规则", lambda payload, _: response(payload)).score_query(
        "查询", [unit()], budget
    )
    before = (tmp_path / "ledger.json.attempts.json").read_bytes()
    monkeypatch.setattr(
        budget, "_persist_attempts", lambda *args: pytest.fail("终态不能重复写入")
    )
    budget._finish_attempt(
        1, http_status=503, error_code="HTTP_5XX", completed_at="changed", will_retry=True
    )
    assert (tmp_path / "ledger.json.attempts.json").read_bytes() == before


def test_terminal_audit_ledger_read_failure_does_not_repeat_paid_transport(tmp_path, monkeypatch):
    import inspect
    budget = control(tmp_path)
    original = budget._read
    failed = False
    calls = []

    def read():
        nonlocal failed
        if not failed and any(frame.function == '_finish_attempt' for frame in inspect.stack()):
            failed = True
            raise ValueError('INVALID_EXISTING_RERANK_LEDGER')
        return original()

    monkeypatch.setattr(budget, '_read', read)
    result = backend().DeepSeekReranker('规则', lambda payload, _: (calls.append(payload), response(payload))[1]).score_query('查询', [unit()], budget)
    assert len(calls) == 1
    assert not result['complete']


def test_cancel_during_attempt_start_audit_still_aborts_before_transport(tmp_path, monkeypatch):
    budget = control(tmp_path)
    persist_attempts = budget._persist_attempts

    def cancel_during_audit(audit):
        persist_attempts(audit)
        budget.cancel_event.set()

    monkeypatch.setattr(budget, "_persist_attempts", cancel_during_audit)
    result = backend().DeepSeekReranker(
        "规则", lambda *args: pytest.fail("审计期间取消不得发送")
    ).score_query("查询", [unit()], budget)
    assert not result["complete"] and result["meta"]["total_external_retrieval_requests"] == 0
    assert attempt_audit(tmp_path)[0]["state"] == "aborted_unsent"
    assert budget.meta()["reserved_cost_micro_cny"] == 0
