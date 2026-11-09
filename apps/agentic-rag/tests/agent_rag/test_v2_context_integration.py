import copy
import importlib.util
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
from test_v2_context import POLICY, fixture_context
from test_v2_snapshot import fixture_assets
from test_v2_tools import payload

from agent_rag.contracts import canonical_json
from agent_rag.input_registry import InputRegistry, text_hash
from agent_rag.tools import KnowledgeTools
from agent_rag.v2_backend import HybridV2Backend
from agent_rag.v2_context import ParentContextExpander, validate_context_expansion
from agent_rag.v2_snapshot import PROFILE, V2KnowledgeSnapshot, file_hash


def context_snapshot():
    snapshot, unit, _ = fixture_context()
    second = copy.deepcopy(unit)
    second.update(unit_id="U2", behavior_id="B2", behavior_ids=["B2"])
    second["associations"][0]["behavior_id"] = "B2"
    source = copy.deepcopy(snapshot.by_file["F1"])
    source.update(official_file_id="F2", source_id="S2")
    third = copy.deepcopy(unit)
    third.update(unit_id="U3", file_id="F2", case_id="C2", case_ids=["C2"])
    third["associations"][0]["case_id"] = "C2"
    snapshot.units += [second, third]
    snapshot.by_file["F2"] = source
    snapshot.manifest = {
        "snapshot_id": "v2-context-test", "index_sha256": "index", "retrieval_profile": PROFILE
    }
    snapshot.candidate = MethodType(V2KnowledgeSnapshot.candidate, snapshot)
    snapshot.read = lambda sid: next(
        (source, source["full_text"])
        for source in snapshot.by_file.values() if source["source_id"] == sid
    )
    snapshot.search = lambda *args: snapshot.backend.search(*args)
    return snapshot


class Dense:
    def __init__(self):
        self.calls = []

    def search(self, text, top_k=100):
        self.calls.append((text, top_k))
        return {"hits": [
            {"unit_id": uid, "rank": rank, "score": score}
            for rank, (uid, score) in enumerate([("U3", .9), ("U2", .8), ("U1", .7)], 1)
        ], "meta": {"mode": "test"}}


class Scorer:
    supports_query_binding = True

    def __init__(self):
        self.calls = []

    def score_query(self, text, units, control, *, query_binding=None):
        self.calls.append(copy.deepcopy((text, units, control, query_binding)))
        return {
            "scores": [{"unit_id": unit["unit_id"], "file_id": unit["file_id"],
                        "score": {"U1": 80, "U2": 90, "U3": 85}[unit["unit_id"]]}
                       for unit in units],
            "complete": True,
            "meta": {"external_retrieval_requests": len(units), "cache_hits": 0},
        }


def test_disabled_backend_keeps_original_key_and_structure():
    snapshot = context_snapshot()
    backend = HybridV2Backend(snapshot, Dense(), Scorer())
    expected = text_hash(canonical_json({
        "text": "不得承诺收益", "top_k": 2, "snapshot": "v2-context-test", "profile": PROFILE,
        "rerank_protocol": "quote-copy-v1", "rules_sha256": None, "protocol_fingerprint": None,
    }))
    assert backend.cache_key("不得承诺收益", 2) == expected
    result = backend.search("不得承诺收益", 2)
    assert "context_expansion" not in result["meta"]
    assert "context_expansion" not in result["audit"]
    assert all("context_expansion" not in candidate for candidate in result["candidates"])


def test_enabled_expands_only_final_files_without_changing_scoring_or_ranks(monkeypatch):
    import agent_rag.v2_backend as module

    calls, constructions = [], []

    class TracedExpander(ParentContextExpander):
        def __init__(self, snapshot, policy):
            constructions.append(policy)
            super().__init__(snapshot, policy)

        def expand(self, unit, candidate):
            assert "context_expansion" not in candidate
            calls.append(unit["unit_id"])
            return super().expand(unit, candidate)

    monkeypatch.setattr(module, "ParentContextExpander", TracedExpander, raising=False)
    snapshot = context_snapshot()
    before = copy.deepcopy((snapshot.units, snapshot.by_file))
    off_dense, on_dense, off_scorer, on_scorer = Dense(), Dense(), Scorer(), Scorer()
    off = HybridV2Backend(snapshot, off_dense, off_scorer)
    on = HybridV2Backend(snapshot, on_dense, on_scorer, context_expansion=POLICY)
    baseline = off.search("不得承诺收益", 1)
    result = on.search("不得承诺收益", 1)
    assert off_dense.calls == on_dense.calls == [("不得承诺收益", 100)]
    assert off_scorer.calls == on_scorer.calls
    assert calls == ["U2"]
    assert constructions == [validate_context_expansion(POLICY)]
    assert result["audit"]["stages"] == baseline["audit"]["stages"]
    assert result["audit"]["file_rankings"] == baseline["audit"]["file_rankings"]
    assert result["ranking_coverage"] == baseline["ranking_coverage"]
    assert result["meta"]["external_retrieval_requests"] == 3
    candidate = result["candidates"][0]
    stripped = copy.deepcopy(candidate)
    stripped["evidence"].pop()
    stripped.pop("context_expansion")
    assert stripped == baseline["candidates"][0]
    assert candidate["context_expansion"]["status"] == "expanded"
    stats = result["meta"]["context_expansion"]
    assert stats["expanded_candidates"] == 1 and stats["skipped_candidates"] == 0
    assert stats["policy_sha256"] == text_hash(canonical_json(validate_context_expansion(POLICY)))
    assert result["audit"]["context_expansion"] == stats
    on.search("不得承诺收益", 1)
    assert calls == ["U2"] and len(on_scorer.calls) == 1
    assert (snapshot.units, snapshot.by_file) == before


def test_context_policy_cache_isolation_and_skipped_statistics():
    snapshot = context_snapshot()
    off = HybridV2Backend(snapshot, Dense(), Scorer())
    on = HybridV2Backend(snapshot, Dense(), Scorer(), context_expansion=POLICY)
    small = HybridV2Backend(
        snapshot, Dense(), Scorer(), context_expansion=POLICY | {"max_parent_chars": 1}
    )
    explicit = HybridV2Backend(
        snapshot, Dense(), Scorer(), context_expansion=validate_context_expansion(POLICY)
    )
    assert len({backend.cache_key("q", 2) for backend in [off, on, small]}) == 3
    assert on.cache_key("q", 2) == explicit.cache_key("q", 2)
    value = small.search("不得承诺收益", 2)
    stats = value["meta"]["context_expansion"]
    assert stats["expanded_candidates"] == 0 and stats["skipped_candidates"] == 2
    assert stats["skip_reasons"] == {"PARENT_TOO_LONG": 2}


@pytest.mark.parametrize("bad", [False, {}, POLICY | {"max_parent_chars": True}])
def test_runtime_rejects_bad_context_before_assets_key_scorer_or_gpu(tmp_path, monkeypatch, bad):
    import agent_rag.v2_runtime as runtime

    snapshot = SimpleNamespace(manifest={"backend": {"mode": "live", "context_expansion": bad}})
    monkeypatch.setattr(runtime, "V2KnowledgeSnapshot", lambda root: snapshot)
    monkeypatch.setattr(runtime, "DeepSeekReranker", lambda *a, **kw: pytest.fail("scorer started"))
    monkeypatch.setattr(runtime, "ResidentDenseIndex", lambda *a, **kw: pytest.fail("GPU started"))
    with pytest.raises(ValueError, match="INVALID_CONTEXT_EXPANSION"):
        runtime.load_v2_snapshot(tmp_path)


@pytest.mark.parametrize("policy", [None, POLICY])
def test_runtime_passes_only_enabled_normalized_policy(tmp_path, monkeypatch, policy):
    import agent_rag.v2_runtime as runtime

    rules = tmp_path / "rules"
    rules.write_text("rules", encoding="utf-8")
    backend = {"mode": "live", "assets": {name: {"path": str(rules), "sha256": file_hash(rules)}
                                         for name in ["rules", "dense_config"]}}
    if policy is not None:
        backend["context_expansion"] = policy
    snapshot = SimpleNamespace(manifest={"backend": backend})
    calls = []
    monkeypatch.setenv("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE", "1")
    monkeypatch.setattr(runtime, "V2KnowledgeSnapshot", lambda root: snapshot)
    monkeypatch.setattr(runtime, "DeepSeekReranker", lambda *a, **kw: object())
    monkeypatch.setattr(runtime, "ResidentDenseIndex", lambda *a, **kw: object())
    monkeypatch.setattr(runtime, "HybridV2Backend", lambda *a, **kw: calls.append(kw))
    runtime.load_v2_snapshot(tmp_path)
    expected = {"audit_root": tmp_path / "retrieval-audits"}
    if policy:
        expected["context_expansion"] = validate_context_expansion(policy)
    assert calls == [expected]


def tool_with_candidates(candidates, *, stats=None, audit_dir=None):
    snapshot = context_snapshot()

    class Backend:
        def is_cached(self, text, top_k):
            return False

        def search(self, text, top_k, control=None):
            meta = {"stage_status": {key: "completed" for key in ["bm25", "bge", "rrf", "rerank"]}}
            if stats is not None:
                meta["context_expansion"] = copy.deepcopy(stats)
            return {"candidates": copy.deepcopy(candidates), "complete": True, "meta": meta,
                    "audit": {"context_expansion": copy.deepcopy(stats)} if stats else {}}

    snapshot.backend = Backend()
    return KnowledgeTools(snapshot, InputRegistry("赠送购物卡。", task_id="context-task"),
                          audit_dir=audit_dir)


def encoded_size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))


def padded_candidates(target_size, monkeypatch, *, count=1):
    import agent_rag.tools as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 0)
    snapshot = context_snapshot()
    candidates = [snapshot.candidate(snapshot.units[index * 2], index + 1, 90 - index)
                  for index in range(count)]
    baseline = tool_with_candidates(candidates).call("search_cases", payload())
    candidates[0]["title"] += "长" * (target_size - encoded_size(baseline))
    baseline = tool_with_candidates(candidates).call("search_cases", payload())
    assert encoded_size(baseline) == target_size
    assert len(baseline["data"]["queries"][0]["candidates"]) == count
    return snapshot, candidates, baseline


@pytest.mark.parametrize("target_size", [11920, 11999, 12000])
def test_tool_optional_diagnostics_and_meta_never_displace_baseline_evidence(
    tmp_path, monkeypatch, target_size
):
    snapshot, candidates, baseline = padded_candidates(target_size, monkeypatch)
    expander = ParentContextExpander(snapshot, POLICY)
    enhanced = [expander.expand(snapshot.units[0], candidates[0])]
    parent_id = enhanced[0]["context_expansion"]["added_evidence_ids"][0]
    tool = tool_with_candidates(
        enhanced, stats={"policy": "verified-parent-v1", "extra": "诊" * 600},
        audit_dir=tmp_path / "audit",
    )
    response = tool.call("search_cases", payload())
    assert response["status"] == "ok", response
    row = response["data"]["queries"][0]
    assert len(row["candidates"]) == 1 and row["returned_candidates_truncated"] is False
    assert encoded_size(response) <= 12000
    candidate = row["candidates"][0]
    assert candidate["evidence"] == candidates[0]["evidence"]
    assert candidate["required_evidence_ids"] == candidates[0]["required_evidence_ids"]
    assert baseline["data"]["queries"][0]["candidates"][0]["evidence"] == candidate["evidence"]
    audit = json.loads((tmp_path / "audit" / "Q0001.json").read_text(encoding="utf-8"))
    assert audit["context_expansion"]["budget_omitted_candidates"] == 1
    assert parent_id in audit["context_expansion"]["budget_omitted_evidence_ids"]
    denied = tool.call("read_source", {"source_id": candidate["source_id"],
                                       "anchor_evidence_id": parent_id, "max_chars": 500})
    assert denied["error"]["code"] == "INVALID_EVIDENCE_REFERENCE"
    for evidence_id in candidate["required_evidence_ids"]:
        assert evidence_id in tool.evidence


def test_tool_omits_lowest_parent_first_and_keeps_returned_parent_registered(monkeypatch):
    snapshot, candidates, _ = padded_candidates(10700, monkeypatch, count=2)
    expander = ParentContextExpander(snapshot, POLICY)
    enhanced = [expander.expand(snapshot.units[index * 2], candidate)
                for index, candidate in enumerate(candidates)]
    tool = tool_with_candidates(enhanced, stats={"policy": "verified-parent-v1"})
    response = tool.call("search_cases", payload())
    returned = response["data"]["queries"][0]["candidates"]
    assert len(returned) == 2 and encoded_size(response) <= 12000
    assert returned[0]["context_expansion"]["status"] == "expanded"
    assert returned[1]["context_expansion"]["status"] == "budget_omitted"
    kept_id = returned[0]["context_expansion"]["added_evidence_ids"][0]
    removed_id = enhanced[1]["context_expansion"]["added_evidence_ids"][0]
    assert kept_id in tool.evidence and removed_id not in tool.evidence


def test_tool_removes_skipped_diagnosis_before_dropping_candidate(monkeypatch):
    snapshot, candidates, _ = padded_candidates(12000, monkeypatch)
    enhanced = ParentContextExpander(snapshot, POLICY | {"max_parent_chars": 1}).expand(
        snapshot.units[0], candidates[0]
    )
    response = tool_with_candidates([enhanced], stats={"skipped_candidates": 1}).call(
        "search_cases", payload()
    )
    assert response["status"] == "ok"
    assert response["data"]["queries"][0]["candidates"] == candidates
    assert encoded_size(response) <= 12000


def test_tool_preserves_baseline_registration_for_trimmed_original_candidates(monkeypatch):
    snapshot, candidates, _ = padded_candidates(12000, monkeypatch, count=2)
    candidates[0]["title"] += "再超限" * 500
    enhanced = [ParentContextExpander(snapshot, POLICY).expand(snapshot.units[index * 2], candidate)
                for index, candidate in enumerate(candidates)]
    tool = tool_with_candidates(enhanced)
    response = tool.call("search_cases", payload())
    assert response["data"]["queries"][0]["returned_candidates_truncated"]
    for candidate in candidates:
        assert all(eid in tool.evidence for eid in candidate["required_evidence_ids"])
    assert all(candidate["context_expansion"]["added_evidence_ids"][0] not in tool.evidence
               for candidate in enhanced)


def test_later_budget_omission_keeps_previously_returned_parent_readable(monkeypatch):
    snapshot = context_snapshot()
    original = snapshot.candidate(snapshot.units[0], 1, 90)
    expander = ParentContextExpander(snapshot, POLICY)
    first = expander.expand(snapshot.units[0], original)
    parent_id = first["context_expansion"]["added_evidence_ids"][0]
    tool = tool_with_candidates([first])
    response = tool.call("search_cases", payload())
    assert response["data"]["queries"][0]["candidates"][0]["context_expansion"]["status"] == (
        "expanded"
    )
    _, candidates, _ = padded_candidates(12000, monkeypatch)
    second = expander.expand(snapshot.units[0], candidates[0])
    tool.snapshot.backend = tool_with_candidates([second]).snapshot.backend
    response = tool.call("search_cases", payload())
    assert response["status"] == "ok" and encoded_size(response) <= 12000
    returned = response["data"]["queries"][0]["candidates"][0]
    assert all(item["evidence_id"] != parent_id for item in returned["evidence"])
    assert parent_id in tool.evidence
    read = tool.call("read_source", {"source_id": "S1", "anchor_evidence_id": parent_id,
                                     "max_chars": 500})
    assert read["status"] == "ok"


def prepare_module():
    path = Path(__file__).resolve().parents[2] / "scripts/prepare-agent-rag-context.py"
    assert path.is_file(), "离线派生清单脚本尚未实现"
    spec = importlib.util.spec_from_file_location("prepare_agent_rag_context", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_snapshot(tmp_path):
    sources, units, dictionary, _, _ = fixture_assets(tmp_path)
    snapshot = V2KnowledgeSnapshot.create(sources, units, dictionary, tmp_path / "source")
    rule = tmp_path / "rules.md"
    rule.write_text("offline rules", encoding="utf-8")
    snapshot.manifest["backend"] = {
        "mode": "live", "rerank_protocol": "span-ref-v2", "assets": {
            "rules": {"path": str(rule), "sha256": file_hash(rule)},
            "dense_config": {"path": str(rule), "sha256": file_hash(rule)},
        }
    }
    manifest = snapshot.root / "manifest.json"
    manifest.write_text(canonical_json(snapshot.manifest), encoding="utf-8")
    return snapshot


def test_prepare_derives_new_identity_without_runtime_or_source_registry_mutation(
    tmp_path, monkeypatch
):
    module = prepare_module()
    snapshot = source_snapshot(tmp_path)
    current = tmp_path / "current.json"
    current.write_text('{"snapshot":"original"}', encoding="utf-8")
    original_manifest = (snapshot.root / "manifest.json").read_bytes()
    monkeypatch.delenv("FINRISK_AGENT_ALLOW_RETRIEVAL_LIVE", raising=False)
    monkeypatch.delenv("FINRISK_AGENT_DEEPSEEK_KEY_FILE", raising=False)
    result = module.prepare(snapshot.root, tmp_path / "derived")
    derived = json.loads((tmp_path / "derived" / "manifest.json").read_text(encoding="utf-8"))
    assert derived["assets"] == snapshot.manifest["assets"]
    assert derived["backend"]["assets"] == snapshot.manifest["backend"]["assets"]
    assert derived["backend"]["rerank_protocol"] == "span-ref-v2"
    assert derived["backend"]["context_expansion"] == validate_context_expansion(POLICY)
    assert derived["derived_from"]["manifest_sha256"] == file_hash(snapshot.root / "manifest.json")
    assert derived["derived_from"]["snapshot_id"] == snapshot.manifest["snapshot_id"]
    assert derived["snapshot_id"] != snapshot.manifest["snapshot_id"]
    assert derived["snapshot_id"].startswith("v2-") and len(derived["snapshot_id"]) == 27
    assert result["snapshot_id"] == derived["snapshot_id"]
    assert (snapshot.root / "manifest.json").read_bytes() == original_manifest
    assert current.read_text(encoding="utf-8") == '{"snapshot":"original"}'
    assert list((tmp_path / "derived").iterdir()) == [tmp_path / "derived" / "manifest.json"]
    module.prepare(snapshot.root, tmp_path / "same")
    same = json.loads((tmp_path / "same" / "manifest.json").read_text(encoding="utf-8"))
    assert same["snapshot_id"] == derived["snapshot_id"]
    module.prepare(snapshot.root, tmp_path / "different", POLICY | {"max_added_chars": 5})
    different = json.loads((tmp_path / "different" / "manifest.json").read_text(encoding="utf-8"))
    assert different["snapshot_id"] != derived["snapshot_id"]


@pytest.mark.parametrize("name", ["sources", "index", "backend"])
def test_prepare_rejects_tampered_assets_before_destination_creation(tmp_path, name):
    module = prepare_module()
    snapshot = source_snapshot(tmp_path)
    asset = (snapshot.manifest["backend"]["assets"]["rules"] if name == "backend"
             else snapshot.manifest["assets"][name])
    Path(asset["path"]).write_text("篡改", encoding="utf-8")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        module.prepare(snapshot.root, tmp_path / "derived")
    assert not (tmp_path / "derived").exists()


@pytest.mark.parametrize("existing", ["empty_directory", "file", "source"])
def test_prepare_never_overwrites_existing_destination(tmp_path, existing):
    module = prepare_module()
    snapshot = source_snapshot(tmp_path)
    destination = snapshot.root if existing == "source" else tmp_path / "derived"
    if existing == "empty_directory":
        destination.mkdir()
    elif existing == "file":
        destination.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="SNAPSHOT_ALREADY_EXISTS"):
        module.prepare(snapshot.root, destination)


def test_prepare_rejects_bad_policy_before_destination_creation(tmp_path):
    module = prepare_module()
    snapshot = source_snapshot(tmp_path)
    with pytest.raises(ValueError, match="INVALID_CONTEXT_EXPANSION"):
        module.prepare(snapshot.root, tmp_path / "derived", POLICY | {"max_added_chars": 601})
    assert not (tmp_path / "derived").exists()
