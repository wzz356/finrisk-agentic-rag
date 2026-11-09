"""Offline control-flow tests; real V2 cache/ledger, no model, network or GPU."""

import copy
import importlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_dev30_experiment import module as experiment_module
from test_knowledge_registry import version_snapshot


def paired_module():
    name = "scripts.agentic_dev30.paired_prefetch"
    assert importlib.util.find_spec(name), "paired prefetch control is required"
    return importlib.import_module(name)


def query(sample="X"):
    return {"sample_id": sample, "text": "保险材料" + sample, "user_context": {}}


def tasks_for(*samples):
    return [
        {"task_key": sample + ":" + arm, "arm": arm, "input": query(sample)}
        for sample in samples
        for arm in ("R1", "R2", "A")
    ]


def backend_fixture(tmp_path, *, complete=True, unknown=0, terminal=False):
    from agent_rag.v2_backend import HybridV2Backend
    from retrieval.sparse import RetrievalDocument

    class Snapshot:
        manifest = {"snapshot_id": "snap", "index_sha256": "a" * 64}
        units = [
            {"unit_id": "U1", "file_id": "F1", "text": "保险证据"},
            {"unit_id": "U2", "file_id": "F2", "text": "宣传证据"},
        ]

        def candidate(self, unit, rank, score):
            return {"unit_id": unit["unit_id"], "rank": rank, "score": score}

    class Dense:
        calls = []

        def close(self):
            pass

        def search(self, text, top_k):
            self.calls.append((text, top_k))
            return {
                "hits": [{"unit_id": "U1", "score": 1.0}, {"unit_id": "U2", "score": 0.5}],
                "meta": {},
            }

    class Reranker:
        protocol = "span-ref-v2"
        _cache_lock = threading.Lock()

        def __init__(self):
            self._cache = {"stale-pair": "old-material"}
            self.calls = []

        def score_query(self, text, packets, control):
            self.calls.append((text, control.ledger_path, dict(self._cache)))
            # Exercise real reservation/settlement boundaries, without a transport callable.
            ordinal, _ = control._reserve({"max_tokens": 512}, "fixture-known")
            control._commit_dispatch(ordinal)
            control._start_transport(ordinal)
            control._settle(
                ordinal, {"prompt_tokens": 1000, "completion_tokens": 0, "total_tokens": 1000}
            )
            if unknown:
                ordinal, _ = control._reserve({"max_tokens": 512}, "fixture-unknown")
                control._commit_dispatch(ordinal)
                control._start_transport(ordinal)
            scores = [
                {"unit_id": u["unit_id"], "file_id": u["file_id"], "score": 80 - i}
                for i, u in enumerate(packets if complete else packets[:1])
            ]
            self._cache[text] = "success"
            return {"scores": scores, "complete": complete, "meta": {}, "errors": [
                {"unit_id": u["unit_id"], "code": "INVALID_COMPLETION", "terminal": True,
                 "remaining_attempts": 0} for u in packets[1:]
            ] if terminal and not complete else []}

    snap = Snapshot()
    backend = HybridV2Backend(snap, Dense(), Reranker(), audit_root=tmp_path / "backend-audits")
    # Candidate generation is deterministic and exercises the real cache/coverage implementation.
    backend.sparse = SimpleNamespace(
        search=lambda text, top_k: [
            SimpleNamespace(
                document=RetrievalDocument(
                    unit_id=u["unit_id"], case_ids=(), text=u["text"], source="structured"
                ),
                rank=i + 1,
                score=1.0,
            )
            for i, u in enumerate(snap.units)
        ]
    )
    snap.backend = backend
    backend.cache["old-query"] = {"stale": True}
    return snap


def runner_fixture(tmp_path, **kwargs):
    p = paired_module()
    e = experiment_module()
    snap = backend_fixture(tmp_path, **kwargs)
    budget = e.RoundBudget(tmp_path / "round.json", "plan", 3_000_000)
    runner = p.SharedPrefetch(snap, tmp_path / "execution", budget, "plan")
    return e, snap, budget, runner


def test_prefetch_reserves_half_yuan_and_preserves_unknown_cost(tmp_path):
    e, snap, budget, runner = runner_fixture(tmp_path, unknown=5000)
    original_reserve = budget.reserve
    reservations = []

    def observe(key, reserve_micro=e.TASK_RESERVE_MICRO):
        original_reserve(key, reserve_micro)
        reservations.append(copy.deepcopy(budget.value["tasks"][-1]))

    budget.reserve = observe
    row = runner.ensure(query())
    assert reservations == [
        {
            "task_key": "PREFETCH:X",
            "status": "started",
            "known_micro_cny": 0,
            "unknown_micro_cny": 500_000,
        }
    ]
    assert row["known_micro_cny"] == 2000 and row["unknown_micro_cny"] == 5158
    assert budget.remaining() == 3_000_000 - 7158
    budget.reserve("X:R1")
    assert budget.value["tasks"][-1]["unknown_micro_cny"] == 1_500_000
    assert snap.backend.reranker.calls[0][2] == {}


def test_prefetch_once_per_material_clears_only_shared_caches_and_r1_a_reuse(tmp_path):
    _, snap, budget, runner = runner_fixture(tmp_path)
    first = runner.ensure(query())
    from agent_rag.rerank_backend import RerankTaskControl

    for arm in ("R1", "A"):
        control = RerankTaskControl(tmp_path / (arm + ".json"), 0.5, 240)
        cached = snap.backend.search(
            experiment_module().canonical({"text": query()["text"], "user_context": {}}), 3, control
        )
        assert cached["complete"] is True and cached["meta"]["cache_hits"] == 1
        assert cached["candidates"] == first["candidates"]
        assert runner.ensure(query()) == first
    before = (tmp_path / "R1.json").read_bytes()
    runner.ensure(query("Y"))
    assert (tmp_path / "R1.json").read_bytes() == before
    assert len(snap.backend.dense.calls) == len(snap.backend.reranker.calls) == 2
    assert all(call[2] == {} for call in snap.backend.reranker.calls)
    assert [r["task_key"] for r in budget.value["tasks"]] == ["PREFETCH:X", "PREFETCH:Y"]
    assert not snap.backend.is_cached(
        experiment_module().canonical({"text": query()["text"], "user_context": {}}), 3
    )
    rows = json.loads((tmp_path / "execution/shared-prefetch.json").read_text(encoding="utf-8"))
    assert len(rows["items"]) == 2
    assert len(rows["items"][0]["ranking"]) == 2
    assert Path(rows["items"][0]["audit_ref"]).is_file()
    assert "reference" not in json.dumps(rows)


@pytest.mark.parametrize("next_sample", ["X", "Y"])
def test_cancelled_search_must_return_before_cached_reuse_or_new_material(tmp_path, next_sample):
    from agent_rag.rerank_backend import RerankTaskControl

    _, snapshot, budget, runner = runner_fixture(tmp_path)
    runner.ensure(query())
    old_control = RerankTaskControl(tmp_path / "old-task.json", 0.5, 240)
    entered, release = threading.Event(), threading.Event()
    original_score = snapshot.backend.reranker.score_query
    late_query = "old task supplementary search"
    errors = []

    def delayed_score(text, packets, control):
        result = original_score(text, packets, control)
        if text == late_query:
            # Usage has settled, but successful-pair and whole-query writes can still arrive.
            entered.set()
            if not release.wait(timeout=5):
                raise TimeoutError("test search was not released")
            with snapshot.backend.reranker._cache_lock:
                snapshot.backend.reranker._cache["late-old-pair"] = "old-task-success"
        return result

    snapshot.backend.reranker.score_query = delayed_score

    def old_search():
        try:
            snapshot.backend.search(late_query, 3, old_control)
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=old_search, daemon=True)
    worker.start()
    try:
        assert entered.wait(timeout=5)
        old_control.cancel()
        assert old_control.meta()["reserved_cost_micro_cny"] == 0
        old_ledger_before = old_control.ledger_path.read_bytes()
        ledger_before = copy.deepcopy(budget.value)
        cache_before = copy.deepcopy(snapshot.backend.cache)
        with pytest.raises(ValueError, match="PREFETCH_BACKEND_SEARCH_ACTIVE"):
            runner.ensure(query(next_sample))
        assert budget.value == ledger_before
        assert snapshot.backend.cache == cache_before
        assert old_control.ledger_path.read_bytes() == old_ledger_before
    finally:
        release.set()
        worker.join(timeout=5)
    assert not worker.is_alive() and errors == []
    assert snapshot.backend.is_cached(late_query, 3)
    assert snapshot.backend.reranker._cache["late-old-pair"] == "old-task-success"
    # A failed boundary stops this round even after the old thread eventually returns.
    with pytest.raises(ValueError, match="PREFETCH_BACKEND_SEARCH_ACTIVE"):
        runner.ensure(query(next_sample))
    assert len(budget.value["tasks"]) == 1
    assert runner.value["search_boundary_failure"]["active_searches"] == 1


def test_partial_prefetch_fails_without_starting_any_arm_or_replaying(tmp_path):
    _, snap, budget, runner = runner_fixture(tmp_path, complete=False)
    started = []
    with pytest.raises(ValueError, match="PREFETCH_INCOMPLETE"):
        for task in tasks_for("X"):
            runner.ensure(task["input"])
            budget.reserve(task["task_key"])
            started.append(task["task_key"])
    assert started == []
    assert budget.value["tasks"][0]["status"] == "finished"
    assert len(snap.backend.reranker.calls) == 1
    with pytest.raises(ValueError, match="PREFETCH"):
        runner.ensure(query())
    assert len(snap.backend.reranker.calls) == 1
    record = json.loads((tmp_path / "execution/shared-prefetch.json").read_text(encoding="utf-8"))
    assert record["items"][0]["status"] == "failed"


def test_missing_prefetch_ledger_keeps_entire_reserve_and_no_search_when_budget_exhausted(tmp_path):
    _, snap, budget, runner = runner_fixture(tmp_path)

    def fail(*args):
        raise OSError("offline failure")

    snap.backend.search = fail
    with pytest.raises(OSError):
        runner.ensure(query())
    # Control created a valid empty ledger: zero requests really occurred.
    assert budget.value["tasks"][0]["known_micro_cny"] == 0
    budget.reserve("other")
    budget.finish("other", 1_500_000, 1_100_000)
    with pytest.raises(ValueError, match="ROUND_BUDGET_EXHAUSTED"):
        runner.ensure(query("Y"))


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_unreadable_prefetch_ledger_cannot_release_unproven_usage(tmp_path, monkeypatch, failure):
    _, snapshot, budget, runner = runner_fixture(tmp_path)
    if failure == "missing":

        def fail(*args, **kwargs):
            raise OSError("ledger creation failed")

        monkeypatch.setattr("agent_rag.rerank_backend.RerankTaskControl", fail)
    else:

        def fail(text, top_k, control):
            # Plausible totals do not prove zero usage when the durable ledger is corrupt.
            control.ledger_path.write_text(
                json.dumps({"settled_cost_micro_cny": 0, "reserved_cost_micro_cny": 0}),
                encoding="utf-8",
            )
            raise OSError("corrupted during search")

        snapshot.backend.search = fail
    with pytest.raises((OSError, ValueError)):
        runner.ensure(query())
    assert budget.value["tasks"][0]["unknown_micro_cny"] == 500_000
    assert budget.value["tasks"][0]["status"] == "finished"
    assert runner.records["X"]["status"] == "failed"


def test_task_timing_and_hypothetical_costs_keep_actual_wall_and_actual_round_cost(tmp_path):
    p = paired_module()
    shared = {"elapsed_seconds": 10.0, "known_micro_cny": 2000, "unknown_micro_cny": 5000}
    raw = {"wall_seconds": 2.0, "known_micro_cny": 1000, "unknown_micro_cny": 0}
    result = p.task_result_fields("R1", raw, shared)
    assert raw["wall_seconds"] == 2.0
    assert result["actual_task_seconds"] == 2.0
    assert result["shared_prefetch_seconds"] == 10.0
    assert result["comparable_e2e_seconds"] == 12.0
    assert result["method_cost_estimate"]["known_micro_cny"] == 3000
    assert result["method_cost_estimate"]["unknown_micro_cny"] == 5000
    assert result["method_cost_estimate"]["not_additive_to_round_actual"] is True
    assert p.task_result_fields("R2", raw, shared)["comparable_e2e_seconds"] is None


def prepare_fixture(tmp_path, monkeypatch):
    e = experiment_module()
    project = tmp_path / "project"
    rules_dir = project / "resources/agent_rag"
    rules_dir.mkdir(parents=True)
    rules = rules_dir / "deepseek-rerank-span-rules.md"
    rules.write_text("rules", encoding="utf-8")
    configs = project / "configs/agent_rag"
    configs.mkdir(parents=True)
    (configs / "evaluation-system-prompt.txt").write_text("prompt", encoding="utf-8")
    refs = tmp_path / "reference"
    (refs / "inputs").mkdir(parents=True)
    (refs / "evaluation").mkdir()
    queries = [query("X" + str(i)) for i in range(30)]
    references = [
        {
            "sample_id": q["sample_id"],
            "review_id": "S02" if i == 0 else "C" + str(i),
            "human_approved": True,
            "difficulty": "simple",
            "reference": {"overall_judgment": "normal", "supported_tags": []},
            "input_sha256": e.obj_sha({"text": q["text"], "user_context": {}}),
        }
        for i, q in enumerate(queries)
    ]
    for relative, value in [
        ("inputs/queries.jsonl", queries),
        ("evaluation/references.jsonl", references),
    ]:
        (refs / relative).write_text(
            "\n".join(e.canonical(row) for row in value) + "\n", encoding="utf-8"
        )
    manifest = {
        "human_approved_count": 30,
        "files": [
            {"path": relative, "sha256": e.file_sha(refs / relative)}
            for relative in ("inputs/queries.jsonl", "evaluation/references.jsonl")
        ],
    }
    e.save(refs / "manifest.json", manifest)
    e.save(project / "resources/agent_rag/evaluation/dev30-human-reference-manifest.json", manifest)
    snapshot = version_snapshot(tmp_path / "registry", "v1")
    value = e.load(snapshot / "manifest.json")
    value.update(corpus_scope="incremental-v1", rerank_mode="live")
    value["backend"]["assets"]["rules"] = {"path": str(rules), "sha256": e.file_sha(rules)}
    e.save(snapshot / "manifest.json", value)
    monkeypatch.setattr(
        e,
        "frozen_runtime",
        lambda project: [
            {
                "path": str(
                    Path(__file__).resolve().parents[2] / "scripts/agentic_dev30/paired_prefetch.py"
                ),
                "sha256": e.file_sha(
                    Path(__file__).resolve().parents[2] / "scripts/agentic_dev30/paired_prefetch.py"
                ),
            }
        ],
    )
    description = {
        "schema_version": e.EVALUATION_VERSION,
        "retrieval_reliability": e.RETRIEVAL_RELIABILITY,
        "evaluation_protocol_sha256": "a" * 64,
        "system_prompt_sha256": e.file_sha(configs / "evaluation-system-prompt.txt"),
    }
    monkeypatch.setattr(
        e.subprocess, "check_output", lambda *a, **kw: json.dumps(description).encode()
    )
    monkeypatch.setattr(
        "agent_rag.dense_backend.ResidentDenseIndex",
        lambda *a, **kw: pytest.fail("offline prepare started a GPU worker"),
    )
    monkeypatch.setattr(
        "agent_rag.rerank_backend.DeepSeekReranker",
        lambda *a, **kw: pytest.fail("offline prepare constructed a paid client"),
    )
    return e, project, refs, snapshot


def test_prepare_requires_explicit_incremental_mode_and_freezes_prefetch_limits(
    tmp_path, monkeypatch
):
    e, project, refs, snapshot = prepare_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="FULL_LIVE_V2_SNAPSHOT_REQUIRED"):
        e.prepare(project, refs, snapshot, tmp_path / "legacy", "node", review_ids=["S02"])
    plan = e.prepare(
        project,
        refs,
        snapshot,
        tmp_path / "new",
        "node",
        review_ids=["S02"],
        snapshot_mode="incremental-v1",
        paired_prefetch=True,
    )
    assert plan["max_rerank_requests"] == 4 * 240
    assert plan["max_judgment_requests"] == 13
    assert plan["paired_prefetch"]["max_requests"] == 240
    assert plan["paired_prefetch"]["cap_cny"] == 0.5
    assert plan["paired_prefetch"]["deadline_seconds"] == 180
    assert plan["paired_prefetch"]["task_count"] == 1
    assert plan["network_calls_during_preparation"] == 0
    assert any(Path(row["path"]).name == "paired_prefetch.py" for row in plan["frozen_files"])
    assert (
        plan["timing_policy"]["comparable_e2e_seconds"]
        == "shared_prefetch_seconds + actual_task_seconds; R1/A only; "
        "composed estimate, not measured cold start"
    )


@pytest.mark.parametrize(
    "damage", ["mode", "count", "subset", "source_count", "schema", "model", "vector"]
)
def test_incremental_prepare_rejects_corruption_before_output_or_calls(
    tmp_path, monkeypatch, damage
):
    e, project, refs, snapshot = prepare_fixture(tmp_path, monkeypatch)
    manifest = e.load(snapshot / "manifest.json")
    dense_asset = manifest["backend"]["assets"]["dense_config"]
    config = e.load(dense_asset["path"])
    if damage == "mode":
        config["corpus_mode"] = "full9639"
    elif damage == "count":
        config["document_count"] += 1
    elif damage == "subset":
        config["subset_file_ids"] = ["f1"]
    elif damage in {"source_count", "schema"}:
        manifest["source_count" if damage == "source_count" else "schema_version"] = (
            999 if damage == "source_count" else "wrong"
        )
    elif damage == "model":
        (Path(config["model_path"]) / "config.json").write_bytes(b"bad")
    else:
        Path(config["vector_path"]).write_bytes(b"bad")
    e.save(dense_asset["path"], config)
    dense_asset["sha256"] = e.file_sha(dense_asset["path"])
    e.save(snapshot / "manifest.json", manifest)
    with pytest.raises(ValueError):
        e.prepare(
            project,
            refs,
            snapshot,
            tmp_path / "new",
            "node",
            review_ids=["S02"],
            snapshot_mode="incremental-v1",
            paired_prefetch=True,
        )
    assert not (tmp_path / "new").exists()


@pytest.mark.parametrize(
    "damage", [None, "null_subset", "vector", "model", "subset", "incremental", "source_count"]
)
def test_full_prepare_validates_all_assets_and_rejects_subset(tmp_path, monkeypatch, damage):
    e, project, refs, snapshot = prepare_fixture(tmp_path, monkeypatch)
    p = paired_module()
    # Keep the real asset, model and span validators; only shrink the historic corpus cardinality.
    monkeypatch.setattr("agent_rag.dense_backend.DOCUMENT_COUNT", 1)
    monkeypatch.setattr(p, "FULL_SOURCE_COUNT", 1, raising=False)
    manifest = e.load(snapshot / "manifest.json")
    manifest["corpus_scope"] = "full9639"
    dense_asset = manifest["backend"]["assets"]["dense_config"]
    dense = e.load(dense_asset["path"])
    if damage != "incremental":
        del dense["corpus_mode"], dense["document_count"]
    if damage == "vector":
        Path(dense["vector_path"]).write_bytes(b"corrupt")
    elif damage == "model":
        (Path(dense["model_path"]) / "config.json").write_bytes(b"corrupt")
    elif damage == "subset":
        dense["subset_file_ids"] = ["f1"]
    elif damage == "null_subset":
        dense["subset_file_ids"] = None
    elif damage == "source_count":
        manifest["source_count"] = 9639
    e.save(dense_asset["path"], dense)
    dense_asset["sha256"] = e.file_sha(dense_asset["path"])
    e.save(snapshot / "manifest.json", manifest)
    if damage in {None, "null_subset"}:
        plan = e.prepare(project, refs, snapshot, tmp_path / "new", "node", review_ids=["S02"])
        assert plan["snapshot_mode"] == "full9639"
        assert "paired_prefetch" not in plan
        assert plan["max_rerank_requests"] == 3 * 240
    else:
        with pytest.raises(ValueError):
            e.prepare(project, refs, snapshot, tmp_path / "new", "node", review_ids=["S02"])
        assert not (tmp_path / "new").exists()


def test_incremental_prepare_accepts_null_subset(tmp_path, monkeypatch):
    e, project, refs, snapshot = prepare_fixture(tmp_path, monkeypatch)
    manifest = e.load(snapshot / "manifest.json")
    dense_asset = manifest["backend"]["assets"]["dense_config"]
    dense = e.load(dense_asset["path"])
    dense["subset_file_ids"] = None
    e.save(dense_asset["path"], dense)
    dense_asset["sha256"] = e.file_sha(dense_asset["path"])
    e.save(snapshot / "manifest.json", manifest)
    plan = e.prepare(
        project,
        refs,
        snapshot,
        tmp_path / "new",
        "node",
        review_ids=["S02"],
        snapshot_mode="incremental-v1",
        paired_prefetch=True,
    )
    assert plan["paired_prefetch"]["task_count"] == 1


@pytest.mark.parametrize('override', [None, 300])
def test_prepare_freezes_optional_dense_parent_wait_without_changing_snapshot(
    tmp_path, monkeypatch, override
):
    e, project, refs, snapshot = prepare_fixture(tmp_path, monkeypatch)
    manifest = e.load(snapshot/'manifest.json')
    config = Path(manifest['backend']['assets']['dense_config']['path'])
    before = config.read_bytes()
    plan = e.prepare(project, refs, snapshot, tmp_path/'new', 'node', review_ids=['S02'],
        snapshot_mode='incremental-v1', paired_prefetch=True,
        dense_startup_timeout_seconds=override)
    if override is None:
        assert 'dense_startup_timeout_seconds' not in plan
    else:
        assert plan['dense_startup_timeout_seconds'] == 300
    assert config.read_bytes() == before
    assert plan['limits']['task_deadline_seconds'] == 270
    assert plan['max_judgment_requests'] == 13 and plan['max_rerank_requests'] == 960


@pytest.mark.parametrize('value', [True, 0, 601, float('inf'), float('nan'), '300'])
def test_prepare_rejects_invalid_dense_wait_before_reading_assets(tmp_path, value):
    e = experiment_module()
    with pytest.raises(ValueError, match='INVALID_DENSE_STARTUP_TIMEOUT'):
        e.prepare(tmp_path, tmp_path/'reference', tmp_path/'snapshot', tmp_path/'new', 'node',
            dense_startup_timeout_seconds=value)


@pytest.mark.parametrize(
    "damage", ["max_requests", "limits", "cache", "timing", "inputs", "search_boundary"]
)
def test_execute_rejects_tampered_prefetch_binding_before_runtime_access(
    tmp_path, monkeypatch, damage
):
    e, snapshot, observed = execute_fixture(tmp_path, monkeypatch)
    plan = e.load(tmp_path / "plan.json")
    if damage == "max_requests":
        plan["paired_prefetch"]["max_requests"] = 241
    elif damage == "limits":
        plan["limits"]["prefetch_rerank_requests"] = 241
    elif damage == "cache":
        plan["cache_policy"] = "unbound"
    elif damage == "timing":
        plan["timing_policy"] = {}
    elif damage == "search_boundary":
        plan["paired_prefetch"]["search_boundary"] = "cancel-bit-only"
    else:
        plan["tasks"][0]["input"]["text"] += "changed"
    e.save(tmp_path / "plan.json", plan)
    approval = e.load(tmp_path / "approved.json")
    approval.update(plan_sha256=e.file_sha(tmp_path / "plan.json"))
    e.save(tmp_path / "approved.json", approval)
    with pytest.raises(ValueError, match="PREFETCH"):
        e.execute(
            tmp_path / "plan.json", tmp_path / "approved.json", tmp_path / "node", tmp_path / "key"
        )
    assert not (tmp_path / "execution").exists()
    assert observed["admitted"] == []


def execute_fixture(tmp_path, monkeypatch, *, complete=True):
    """Run the actual scheduler; substitute only process/server/model boundaries."""
    import uvicorn

    import agent_rag.task_service
    import agent_rag.v2_runtime
    from agent_rag.rerank_backend import RerankTaskControl

    e, p = experiment_module(), paired_module()
    project = Path(__file__).resolve().parents[2]
    snapshot = backend_fixture(tmp_path, complete=complete)
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    rule = project / "resources/agent_rag/deepseek-rerank-span-rules.md"
    e.save(
        snapshot_dir / "manifest.json",
        {
            "snapshot_id": "snap",
            "backend": {
                "mode": "live",
                "rerank_protocol": "span-ref-v2",
                "assets": {"rules": {"path": str(rule), "sha256": e.file_sha(rule)}},
            },
        },
    )
    description = {
        "schema_version": e.EVALUATION_VERSION,
        "retrieval_reliability": e.RETRIEVAL_RELIABILITY,
        "evaluation_protocol_sha256": "a" * 64,
        "system_prompt_sha256": e.file_sha(
            project / "configs/agent_rag/evaluation-system-prompt.txt"
        ),
    }
    plan = {
        "schema_version": "finrisk-three-arm-experiment/1.0",
        "project_root": str(project),
        "snapshot_dir": str(snapshot_dir),
        "snapshot_id": "snap",
        "resilience_policy": e.RESILIENCE_POLICY,
        "evaluation_runtime_description": description,
        "evaluation_protocol_sha256": "a" * 64,
        "retrieval_binding": e.snapshot_rerank_binding(
            e.load(snapshot_dir / "manifest.json"), project
        ),
        "tasks": tasks_for("X", "Y"),
        "task_count": 6,
        "max_judgment_requests": 26,
        "max_rerank_requests": 8 * 240,
        "paired_prefetch": p.configuration(tasks_for("X", "Y"), "snap"),
        "cache_policy": p.CACHE_POLICY,
        "timing_policy": p.TIMING_POLICY,
        "limits": {
            "task_rerank_requests": 240,
            "task_rerank_cap_cny": 0.5,
            "task_model_ledger_cap_cny": 1,
            "round_task_reserve_cny": 1.5,
            "prefetch_rerank_requests": 240,
            "prefetch_rerank_cap_cny": 0.5,
            "prefetch_deadline_seconds": 180,
            "round_prefetch_reserve_cny": 0.5,
        },
        "frozen_files": [{"path": str(Path(p.__file__)), "sha256": e.file_sha(p.__file__)}],
        "reference_file": str(tmp_path / "references.jsonl"),
    }
    references = [
        {
            "sample_id": s,
            "difficulty": "simple",
            "reference": {"overall_judgment": "normal", "supported_tags": []},
        }
        for s in ("X", "Y")
    ]
    (tmp_path / "references.jsonl").write_text(
        "\n".join(e.canonical(r) for r in references), encoding="utf-8"
    )
    e.save(tmp_path / "plan.json", plan)
    e.save(
        tmp_path / "approved.json",
        {
            "approval_status": "approved",
            "plan_sha256": e.file_sha(tmp_path / "plan.json"),
            "human_statement": "offline control test",
            "approved_at": "2026-10-17",
            "external_data_scope": e.EXTERNAL_SCOPE,
            "max_judgment_requests": 26,
            "max_rerank_requests": 8 * 240,
            "total_cap_cny": 3,
        },
    )
    (tmp_path / "node").write_text("offline substitute", encoding="utf-8")
    (tmp_path / "key").write_text("offline substitute", encoding="utf-8")
    observed = {"admitted": [], "processes": [], "cached": [], "runtime_loads": []}

    class Service:
        def __init__(self, root, snap):
            self.root, self.snapshot = root, snap
            self.controls, self.records, self.bundles = {}, {}, {}

        def admit(self, data, key):
            observed["admitted"].append((data, key))
            tid = str(len(self.records)).zfill(32)
            q = {"text": data["text"], "user_context": data["user_context"]}
            self.bundles[tid] = {
                "task_id": tid,
                "input": {**q, "input_sha256": e.obj_sha(q)},
                "config": {"retrieval_profile": "v2-hybrid-flash-v1"},
                "execution_contract": {
                    "prompt_sha256": e.file_sha(project / "configs/agent_rag/system-prompt.txt")
                },
            }
            self.records[tid] = {"task_id": tid, "task_status": "running"}
            self.controls[tid] = RerankTaskControl(self.root / tid / "rerank-ledger.json", 0.5, 240)
            return dict(self.records[tid]), False

        def bundle(self, tid):
            return copy.deepcopy(self.bundles[tid])

        def cancel_retrieval(self, tid):
            self.controls[tid].cancel()

    def process(args, **kwargs):
        bundle = e.load(args[args.index("--bundle") + 1])
        arm = args[args.index("--arm") + 1]
        observed["processes"].append(arm)
        if arm in {"R1", "A"}:
            q = {k: bundle["input"][k] for k in ("text", "user_context")}
            cached = snapshot.backend.search(
                e.canonical(q),
                3,
                RerankTaskControl(Path(bundle["task_dir"]) / "rerank-ledger.json", 0.5, 240),
            )
            observed["cached"].append(
                (q["text"], arm, cached["meta"].get("cache_hits", 0), cached["candidates"])
            )
        directory = Path(bundle["task_dir"])
        e.save(
            directory / "report.json",
            {
                "task_id": bundle["task_id"],
                "task_status": "completed",
                "execution_mode": "live",
                "versions": {
                    "input_sha256": bundle["input"]["input_sha256"],
                    "snapshot_id": "snap",
                },
                "overall_judgment": "normal",
                "risk_points": [],
            },
        )
        e.save(
            Path(kwargs["env"]["FINRISK_AGENT_TRIAL_LEDGER"]),
            {"settled_cost_micro_cny": 0, "reserved_cost_micro_cny": 0},
        )
        return SimpleNamespace(returncode=0)

    class Socket:
        def bind(self, address):
            pass

        def listen(self, backlog):
            pass

        def getsockname(self):
            return ("127.0.0.1", 12345)

        def close(self):
            pass

    class Server:
        started = True

        def __init__(self, config):
            pass

        def run(self, **kwargs):
            pass

    monkeypatch.setattr(agent_rag.task_service, "TaskService", Service)
    def load_snapshot(root, **kwargs):
        observed['runtime_loads'].append((root, kwargs))
        return snapshot
    monkeypatch.setattr(agent_rag.v2_runtime, "load_v2_snapshot", load_snapshot)
    monkeypatch.setattr(e, "local_tool_app", lambda service: object())
    monkeypatch.setattr(e.socket, "socket", Socket)
    monkeypatch.setattr(uvicorn, "Config", lambda *a, **kw: None)
    monkeypatch.setattr(uvicorn, "Server", Server)
    monkeypatch.setattr(e.subprocess, "run", process)
    monkeypatch.setattr(e, "write_retrieval_comparison", lambda *a: None)
    return e, snapshot, observed


@pytest.mark.parametrize('value', [None, True, 0, 601, float('inf'), float('nan'), '300'])
def test_execute_rejects_invalid_frozen_dense_wait_before_runtime_or_key_access(
    tmp_path, monkeypatch, value
):
    e, _, observed = execute_fixture(tmp_path, monkeypatch)
    plan = e.load(tmp_path/'plan.json')
    plan['dense_startup_timeout_seconds'] = value
    e.save(tmp_path/'plan.json', plan)
    approval = e.load(tmp_path/'approved.json')
    approval['plan_sha256'] = e.file_sha(tmp_path/'plan.json')
    e.save(tmp_path/'approved.json', approval)
    with pytest.raises(ValueError, match='INVALID_DENSE_STARTUP_TIMEOUT'):
        e.execute(tmp_path/'plan.json', tmp_path/'approved.json',
            tmp_path/'no-node', tmp_path/'no-key')
    assert observed['runtime_loads'] == [] and not (tmp_path/'execution').exists()


@pytest.mark.parametrize('requested', [None, 300, 600])
def test_execute_uses_only_frozen_dense_startup_wait(tmp_path, monkeypatch, requested):
    e, _, observed = execute_fixture(tmp_path, monkeypatch)
    plan = e.load(tmp_path/'plan.json')
    plan['dense_startup_timeout_seconds'] = 300
    e.save(tmp_path/'plan.json', plan)
    approval = e.load(tmp_path/'approved.json')
    approval['plan_sha256'] = e.file_sha(tmp_path/'plan.json')
    e.save(tmp_path/'approved.json', approval)
    args = (tmp_path/'plan.json', tmp_path/'approved.json', tmp_path/'node', tmp_path/'key')
    if requested == 600:
        with pytest.raises(ValueError, match='DENSE_STARTUP_TIMEOUT_BINDING_MISMATCH'):
            e.execute(*args, dense_startup_timeout_seconds=requested)
        assert observed['runtime_loads'] == []
    else:
        e.execute(*args, dense_startup_timeout_seconds=requested)
        assert observed['runtime_loads'][0][1] == {'startup_timeout_seconds':300}


def test_actual_execute_prefetches_each_material_once_and_preserves_scores(tmp_path, monkeypatch):
    e, snapshot, observed = execute_fixture(tmp_path, monkeypatch)
    e.execute(
        tmp_path / "plan.json", tmp_path / "approved.json", tmp_path / "node", tmp_path / "key"
    )
    assert observed["processes"] == ["R1", "R2", "A"] * 2
    assert len(snapshot.backend.reranker.calls) == 2
    assert all(row[2] == 1 for row in observed["cached"])
    assert observed["cached"][0][3] == observed["cached"][1][3]
    assert observed["cached"][2][3] == observed["cached"][3][3]
    assert [data["engine"] for data, key in observed["admitted"]] == [
        "controlled_agent",
        "fixed_rag",
        "controlled_agent",
    ] * 2
    execution = tmp_path / "execution"
    round_ledger = e.load(execution / "round-ledger.json")
    assert len(round_ledger["tasks"]) == 8
    assert sum(row["known_micro_cny"] for row in round_ledger["tasks"]) == 4000
    scores = e.load(execution / "automatic-comparison.json")
    for item in scores["items"]:
        task = e.load(execution / (item["task_key"].replace(":", "_") + ".json"))
        assert item["end_to_end_seconds"] == task["wall_seconds"] == task["actual_task_seconds"]
        if item["arm"] in {"R1", "A"}:
            assert (
                task["comparable_e2e_seconds"]
                == task["wall_seconds"] + task["shared_prefetch_seconds"]
            )
        else:
            assert task["comparable_e2e_seconds"] is None
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        e.execute(
            tmp_path / "plan.json", tmp_path / "approved.json", tmp_path / "node", tmp_path / "key"
        )
    assert len(observed["processes"]) == 6


def test_actual_execute_never_admits_or_runs_an_arm_after_partial_prefetch(tmp_path, monkeypatch):
    e, snapshot, observed = execute_fixture(tmp_path, monkeypatch, complete=False)
    with pytest.raises(ValueError, match="PREFETCH_INCOMPLETE"):
        e.execute(
            tmp_path / "plan.json", tmp_path / "approved.json", tmp_path / "node", tmp_path / "key"
        )
    assert observed["admitted"] == observed["processes"] == []
    assert len(snapshot.backend.reranker.calls) == 1
    assert e.load(tmp_path / "execution/status.json")["status"] == "stopped"
    ledger = e.load(tmp_path / "execution/round-ledger.json")
    assert len(ledger["tasks"]) == 1 and ledger["tasks"][0]["task_key"] == "PREFETCH:X"


def test_execute_without_prefetch_configuration_keeps_legacy_behavior(tmp_path, monkeypatch):
    e, snapshot, observed = execute_fixture(tmp_path, monkeypatch)
    plan = e.load(tmp_path / "plan.json")
    del plan["paired_prefetch"], plan["timing_policy"]
    plan.update(cache_policy="legacy resident cache", max_rerank_requests=6 * 240)
    e.save(tmp_path / "plan.json", plan)
    approval = e.load(tmp_path / "approved.json")
    approval.update(plan_sha256=e.file_sha(tmp_path / "plan.json"), max_rerank_requests=6 * 240)
    e.save(tmp_path / "approved.json", approval)
    e.execute(
        tmp_path / "plan.json", tmp_path / "approved.json", tmp_path / "node", tmp_path / "key"
    )
    assert observed["processes"] == ["R1", "R2", "A"] * 2
    assert not (tmp_path / "execution/shared-prefetch.json").exists()
    ledger = e.load(tmp_path / "execution/round-ledger.json")
    assert len(ledger["tasks"]) == 6
    assert all(not row["task_key"].startswith("PREFETCH:") for row in ledger["tasks"])
    assert "comparable_e2e_seconds" not in e.load(tmp_path / "execution/X_R1.json")


def test_exhausted_local_partial_prefetch_is_frozen_equally_without_retry(tmp_path):
    e, snap, budget, runner = runner_fixture(tmp_path, complete=False, terminal=True)
    first = runner.ensure(query())
    assert first['status'] == 'partial'
    assert first['ranking_coverage']['scored_units'] == 1
    assert first['ranking_coverage']['failed_units'] == 1
    from agent_rag.rerank_backend import RerankTaskControl
    text = e.canonical({'text': query()['text'], 'user_context': {}})
    for arm in ('R1', 'R2', 'A'):
        control = RerankTaskControl(tmp_path/(arm+'.json'), 0.5, 240)
        cached = snap.backend.search(text, 3, control)
        assert cached['complete'] is False
        assert cached['ranking_coverage'] == first['ranking_coverage']
        assert cached['candidates'] == first['candidates']
        assert cached['meta']['external_retrieval_requests'] == 0
        assert cached['meta']['cache_hits'] == 1
        assert control.meta()['external_retrieval_requests'] == 0
        assert runner.ensure(query()) == first
    assert len(snap.backend.reranker.calls) == 1
    assert len(budget.value['tasks']) == 1
    # Returned views cannot mutate the frozen copy.
    cached['candidates'].clear()
    assert snap.backend.search(text, 3)['candidates'] == first['candidates']


@pytest.mark.parametrize('damage', ['coverage_count', 'nonterminal', 'transport_open',
                                  'unsupported_code', 'audit_binding', 'empty_scores',
                                  'failed_ids', 'audit_complete', 'audit_coverage',
                                  'query_binding'])
def test_partial_prefetch_never_admits_unbounded_or_inconsistent_results(tmp_path, damage):
    _, snap, _, runner = runner_fixture(tmp_path, complete=False, terminal=True)
    original = snap.backend.search
    def damaged(*args, **kwargs):
        result = original(*args, **kwargs)
        if damage == 'coverage_count':
            result['ranking_coverage']['scored_units'] = 2
        elif damage == 'nonterminal':
            result['ranking_coverage']['attempts_exhausted'] = False
        elif damage == 'transport_open':
            result['meta']['transport_circuit_open'] = True
        elif damage == 'unsupported_code':
            result['ranking_coverage']['failure_counts'] = {'UNCLASSIFIED_RERANK_FAILURE': 1}
        elif damage == 'audit_binding':
            result['audit']['failed_unit_ids'] = []
        elif damage == 'failed_ids':
            result['ranking_coverage']['failed_unit_ids'] = ['foreign-unit']
        elif damage == 'audit_complete':
            result['audit']['complete'] = True
        elif damage == 'audit_coverage':
            result['audit']['ranking_coverage'] = {}
        elif damage == 'query_binding':
            result['audit']['query_text_sha256'] = '0'*64
        else:
            result['candidates'] = []
        return result
    snap.backend.search = damaged
    with pytest.raises(ValueError, match='PREFETCH_INCOMPLETE'):
        runner.ensure(query())
    assert runner.records['X']['status'] == 'failed'
