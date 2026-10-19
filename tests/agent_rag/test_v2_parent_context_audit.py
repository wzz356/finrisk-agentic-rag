import copy
import csv
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

from agent_rag.contracts import load_contract
from agent_rag.input_registry import text_hash
from agent_rag.v2_snapshot import V2KnowledgeSnapshot, file_hash

PROJECT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT / "scripts/evaluate_v2_parent_context.py"


def audit_module():
    assert SCRIPT.is_file(), "离线父上下文审计 CLI 尚未实现"
    spec = importlib.util.spec_from_file_location("v2_parent_context_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_snapshot(tmp_path, modes=("positive", "zero", "covered", "skip", "error")):
    sources, units = [], []
    for number, mode in enumerate(modes, 1):
        parent = "甲公司😀提示：不得承诺收益；以合同为准。"
        text = "标题\n" + parent + "\n尾部。"
        fid, sid, uid = f"F{number}", f"S{number}", f"U{number}"
        source = {
            "official_file_id": fid, "source_id": sid, "full_text": text,
            "text_sha256": text_hash(text), "text_version": 1,
        }
        fact = parent if mode == "zero" else "不得承诺收益"
        fact_span = {"start": text.index(fact), "end": text.index(fact) + len(fact)}
        span = {"start": text.index(parent), "end": text.index(parent) + len(parent)}
        unit = {
            "unit_id": uid, "file_id": fid, "text": fact, "source_text": fact,
            "source_sha256": text_hash(text), "source_spans": [fact_span],
            "parent_span": copy.deepcopy(span), "parent_source_spans": [copy.deepcopy(span)],
            "parent_source_text": parent, "context_spans": [copy.deepcopy(span)],
            "context_source_text": parent,
            "context_segments": [{"span": copy.deepcopy(span), "text": parent}],
            "case_id": "C1", "case_ids": ["C1"], "behavior_id": "B1",
            "behavior_ids": ["B1"], "metadata_unknown": mode == "skip",
            "context_ambiguous": False, "prefix_evidence_ambiguous": False,
            "associations": [{
                "case_id": "C1", "behavior_id": "B1", "party_position_ambiguous": False,
                "qualifier_position_ambiguous": False,
            }],
        }
        if mode == "covered":
            left = {"start": span["start"], "end": fact_span["end"]}
            right = {"start": fact_span["start"], "end": span["end"]}
            unit["associations"][0].update(
                party_quote=text[left["start"]:left["end"]], party_source_spans=[left],
                qualifier_source_links=[{
                    "span": right, "quote": text[right["start"]:right["end"]],
                    "kind": "condition",
                }],
            )
        if mode == "error":
            unit["source_text"] = "错误原文"
        sources.append(source)
        units.append(unit)
    source_path, index_path = tmp_path / "sources.jsonl", tmp_path / "index.jsonl"
    source_path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in sources),
                           encoding="utf-8")
    index_path.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in units),
                          encoding="utf-8")
    dictionary = tmp_path / "dictionary.csv"
    with dictionary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["risk_tag", "description", "category"])
        writer.writeheader()
        for tag in load_contract("model-output.schema.json")["$defs"]["tag"]["enum"]:
            writer.writerow({"risk_tag": tag, "description": "合成定义", "category": "测试"})
    snapshot = V2KnowledgeSnapshot.create(
        source_path, index_path, dictionary, tmp_path / "snapshot"
    )
    manifest = snapshot.manifest | {"derived_from": {"snapshot_id": "historical-parent"}}
    (snapshot.root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot.root, units


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ranking_file(tmp_path, units):
    audit = {
        "query_text": "保险😀赠送", "query_text_sha256": text_hash("保险😀赠送"),
        "snapshot_id": "historical-snapshot",
        "top20": [{
            "unit_id": u["unit_id"], "file_id": u["file_id"],
            "unit_text_sha256": text_hash(u["text"]), "rank": i, "score": 90 - i,
        } for i, u in enumerate(units, 1)],
    }
    path = tmp_path / "ranking.json"
    path.write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
    return path, audit


def test_all_units_count_errors_separately_and_preserve_snapshot(tmp_path):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    destination = tmp_path / "audit"
    summary = module.audit_snapshot(root, destination)
    assert summary == read_json(destination / "summary.json")
    assert summary["status"] == "failed"
    assert summary["denominators"] == {"sources": 5, "units": 5, "sources_with_units": 5}
    assert summary["counts"] == {"expanded": 3, "skipped": 1, "errors": 1}
    assert summary["reason_counts"]["skipped"] == {"UNSAFE_METADATA": 1}
    assert summary["reason_counts"]["errors"] == {"SOURCE_QUOTE_MISMATCH": 1}
    added = summary["added_context"]
    assert added["added_evidence_windows"] == 2
    assert added["units_with_added_windows"] == 2
    assert added["positive_added_units"] == 1
    assert added["zero_added_expanded_units"] == 2
    assert added["unique_added_chars"]["count"] == 3
    assert added["unique_added_chars"]["total"] > 0
    assert added["unique_added_chars"]["min"] == 0
    assert summary["policy"] == {
        "policy": "verified-parent-v1", "max_parent_chars": 900, "max_added_chars": 600,
    }
    assert summary["snapshot"]["derived_from"] == {"snapshot_id": "historical-parent"}
    assert summary["api_calls"] == summary["retrieval_calls"] == summary["rerank_calls"] == 0
    assert summary["input_integrity"]["unchanged"] is True
    assert set(summary["code_sha256"]) == {"script", "v2_context", "v2_snapshot"}
    assert len(summary["policy_sha256"]) == 64
    rows = [json.loads(line) for line in (destination / "units.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == 5
    assert rows[-1]["status"] == "error"
    assert rows[-1]["phase"] == "candidate"
    assert rows[-1]["error_type"] == "ValueError"
    examples = read_json(destination / "examples.json")
    assert examples[0]["diagnostic"]["added_chars"] > 0
    assert any(row["diagnostic"]["status"] == "skipped" for row in examples)
    source_rows = {
        s["source_id"]: s for s in map(json.loads, (tmp_path / "sources.jsonl").read_text(
            encoding="utf-8").splitlines())
    }
    for example in examples:
        base, enhanced = example["base_candidate"], example["enhanced_candidate"]
        assert base["required_evidence_ids"] == enhanced["required_evidence_ids"]
        assert enhanced["evidence"][:len(base["evidence"])] == base["evidence"]
        stripped = module.strip_parent_context(enhanced, "audit_round_trip")
        stripped.pop("context_expansion")
        assert stripped == base
        for ev in enhanced["evidence"]:
            source = source_rows[ev["source_id"]]
            assert ev["quote"] == source["full_text"][ev["start"]:ev["end"]]
            assert ev["text_sha256"] == source["text_sha256"]
            assert ev["text_version"] == str(source["text_version"])
    assert all(p.read_bytes() == data for p, data in before.items())


def test_existing_destination_is_rejected_without_overwriting(tmp_path):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    destination = tmp_path / "audit"
    destination.mkdir()
    sentinel = destination / "summary.json"
    sentinel.write_text("保留", encoding="utf-8")
    with pytest.raises(ValueError, match="AUDIT_DESTINATION_EXISTS"):
        module.audit_snapshot(root, destination)
    assert sentinel.read_text(encoding="utf-8") == "保留"


def test_manifest_policy_is_effective_and_recorded(tmp_path):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    manifest = read_json(root / "manifest.json")
    manifest["backend"] = {"context_expansion": {
        "policy": "verified-parent-v1", "max_added_chars": 1, "max_parent_chars": 100,
    }}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    summary = module.audit_snapshot(root, tmp_path / "audit")
    assert summary["status"] == "passed"
    assert summary["policy"] == manifest["backend"]["context_expansion"]
    assert summary["counts"] == {"expanded": 0, "skipped": 1, "errors": 0}
    assert summary["reason_counts"]["skipped"] == {"ADDED_CHARS_EXCEEDED": 1}


@pytest.mark.parametrize("policy", [{}, False, []])
def test_invalid_manifest_policy_is_not_hidden_by_default(tmp_path, policy):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    manifest = read_json(root / "manifest.json")
    manifest["backend"] = {"context_expansion": policy}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="INVALID_CONTEXT_EXPANSION"):
        module.audit_snapshot(root, tmp_path / "audit")
    assert not (tmp_path / "audit").exists()


def test_in_memory_source_corruption_is_an_audit_error(tmp_path, monkeypatch):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    original = module.ParentContextExpander.expand

    def corrupt_source(self, unit, base):
        result = original(self, unit, base)
        source = self.snapshot.by_file[unit["file_id"]]
        source["full_text"] = source["full_text"].replace("尾部", "伪造")
        return result

    monkeypatch.setattr(module.ParentContextExpander, "expand", corrupt_source)
    summary = module.audit_snapshot(root, tmp_path / "audit")
    assert summary["status"] == "failed"
    assert summary["counts"] == {"expanded": 0, "skipped": 0, "errors": 1}
    assert summary["reason_counts"]["errors"] == {"AUDIT_SOURCE_HASH_MISMATCH": 1}


def test_nonfinite_frozen_score_does_not_produce_nonstandard_output_json(tmp_path):
    module = audit_module()
    root, units = fixture_snapshot(tmp_path, ("positive",))
    ranking, data = ranking_file(tmp_path, units)
    data["top20"][0]["score"] = float("nan")
    ranking.write_text(json.dumps(data), encoding="utf-8")
    summary = module.audit_snapshot(root, tmp_path / "audit", [ranking])
    assert summary["status"] == "failed"

    def reject_nonfinite(value):
        raise ValueError(value)

    replay = json.loads((tmp_path / "audit/ranking-replay.json").read_text(encoding="utf-8"),
                        parse_constant=reject_nonfinite)
    assert replay["audits"][0]["status"] == "failed"
    assert replay["audits"][0]["candidates"] == []


@pytest.mark.parametrize("number", ["1e400", "-1e400"])
def test_overflow_json_score_retains_failure_report_and_input_binding(tmp_path, number):
    module = audit_module()
    root, units = fixture_snapshot(tmp_path, ("positive",))
    ranking, data = ranking_file(tmp_path, units)
    ranking.write_text(json.dumps(data).replace('"score": 89', '"score": ' + number),
                       encoding="utf-8")
    original_sha256 = file_hash(ranking)
    destination = tmp_path / "audit"
    summary = module.audit_snapshot(root, destination, [ranking])
    assert summary["status"] == "failed"
    assert read_json(destination / "summary.json") == summary
    audit = read_json(destination / "ranking-replay.json")["audits"][0]
    assert audit["status"] == "failed"
    assert audit["audit_path"] == str(ranking.resolve())
    assert audit["audit_sha256"] == original_sha256 == file_hash(ranking)
    assert audit["error_type"] == "ValueError"
    assert audit["error_message"] == "RANKING_NON_FINITE_NUMBER:" + number
    assert audit["candidates"] == []
    assert audit["frozen_top20"] is None


@pytest.mark.parametrize("corruption", ["quote", "required", "rank", "original", "throws"])
def test_expansion_corruption_and_exceptions_remain_errors(tmp_path, monkeypatch, corruption):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    original = module.ParentContextExpander.expand

    def broken(self, unit, base):
        if corruption == "throws":
            raise RuntimeError("unexpected expansion crash")
        result = original(self, unit, base)
        if corruption == "quote":
            result["evidence"][-1]["quote"] = "伪造"
        elif corruption == "required":
            result["required_evidence_ids"] = []
        elif corruption == "rank":
            result["rank"] += 1
        else:
            result["evidence"][0]["evidence_roles"] = ["modified"]
        return result

    monkeypatch.setattr(module.ParentContextExpander, "expand", broken)
    summary = module.audit_snapshot(root, tmp_path / "audit")
    assert summary["status"] == "failed"
    assert summary["counts"] == {"expanded": 0, "skipped": 0, "errors": 1}
    row = json.loads((tmp_path / "audit/units.jsonl").read_text(encoding="utf-8"))
    assert row["status"] == "error"
    assert row["phase"] == ("expand" if corruption == "throws" else "validation")


def test_after_hash_detects_input_change_and_marks_audit_failed(tmp_path, monkeypatch):
    module = audit_module()
    root, _ = fixture_snapshot(tmp_path, ("positive",))
    original = module.ParentContextExpander.expand

    def mutating(self, unit, base):
        result = original(self, unit, base)
        with (root / "manifest.json").open("a", encoding="utf-8") as handle:
            handle.write(" ")
        return result

    monkeypatch.setattr(module.ParentContextExpander, "expand", mutating)
    summary = module.audit_snapshot(root, tmp_path / "audit")
    assert summary["status"] == "failed"
    assert summary["input_integrity"]["unchanged"] is False
    assert summary["input_integrity"]["changed_inputs"] == ["manifest"]


def test_frozen_ranking_replays_same_units_scores_and_original_snapshot_id(tmp_path):
    module = audit_module()
    root, units = fixture_snapshot(tmp_path, ("positive", "skip"))
    ranking, original = ranking_file(tmp_path, units)
    second = tmp_path / "second-ranking.json"
    second.write_bytes(ranking.read_bytes())
    summary = module.audit_snapshot(root, tmp_path / "audit", [ranking, second])
    assert summary["status"] == "passed"
    replay = read_json(tmp_path / "audit/ranking-replay.json")
    assert replay["retrieval_performed"] is False
    assert replay["rerank_performed"] is False
    assert len(replay["audits"]) == 2
    first = replay["audits"][0]
    assert first["status"] == "passed"
    assert first["audit_sha256"] == file_hash(ranking)
    assert first["historical_snapshot_id"] == original["snapshot_id"]
    assert first["current_snapshot_id"] == summary["snapshot"]["snapshot_id"]
    assert first["frozen_top20"] == original["top20"]
    assert len(first["candidates"]) == len(original["top20"])
    for row, hit in zip(first["candidates"], original["top20"]):
        assert row["binding"] == {
            key: hit[key] for key in ["unit_id", "file_id", "unit_text_sha256"]
        }
        assert row["base_candidate"]["rank"] == row["enhanced_candidate"]["rank"] == hit["rank"]
        assert row["base_candidate"]["score"] == row["enhanced_candidate"]["score"] == hit["score"]


@pytest.mark.parametrize("bad", [
    "query_hash", "unit_id", "file_id", "text_hash", "duplicate", "rank", "score", "nan",
])
def test_bad_ranking_binding_is_reported_as_failure_without_partial_success(tmp_path, bad):
    module = audit_module()
    root, units = fixture_snapshot(tmp_path, ("positive", "skip"))
    ranking, data = ranking_file(tmp_path, units)
    hit = data["top20"][1]
    if bad == "query_hash":
        data["query_text_sha256"] = "0" * 64
    elif bad == "unit_id":
        hit["unit_id"] = "missing-unit"
    elif bad == "file_id":
        hit["file_id"] = "wrong-file"
    elif bad == "text_hash":
        hit["unit_text_sha256"] = "0" * 64
    elif bad == "duplicate":
        data["top20"][1] = data["top20"][0] | {"rank": 2}
    elif bad == "rank":
        hit["rank"] = 1
    else:
        hit["score"] = float("nan") if bad == "nan" else "90"
    ranking.write_text(json.dumps(data), encoding="utf-8")
    summary = module.audit_snapshot(root, tmp_path / "audit", [ranking])
    assert summary["status"] == "failed"
    assert summary["ranking_audits"] == {"total": 1, "passed": 0, "failed": 1}
    audit = read_json(tmp_path / "audit/ranking-replay.json")["audits"][0]
    assert audit["status"] == "failed"
    assert audit["error_type"] == "ValueError"
    assert audit["candidates"] == []
    assert audit["frozen_top20"] == read_json(ranking)["top20"] or bad == "nan"


def test_cli_offline_and_nonzero_on_candidate_error(tmp_path):
    import sys

    audit_module()
    root, _ = fixture_snapshot(tmp_path, ("error",))
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--snapshot", str(root),
        "--destination", str(tmp_path / "audit"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 1
    assert read_json(tmp_path / "audit/summary.json")["counts"]["errors"] == 1
