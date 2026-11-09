import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from agent_rag.input_registry import text_hash


def create_snapshot(*args):
    assert importlib.util.find_spec("agent_rag.v2_snapshot"), "V2来源快照适配尚未实现"
    return importlib.import_module("agent_rag.v2_snapshot").V2KnowledgeSnapshot.create(*args)


def fixture_assets(tmp_path, *, bad_span=False):
    text = "机构甲😀\n赠送购物卡。\n" + "其他事项。" * 120 + "\n仅限合同约定之外。"
    party = "机构甲😀"
    fact = "赠送购物卡。"
    condition = "仅限合同约定之外。"
    source = {
        "official_file_id": "F000001",
        "source_id": "FRV2-SOURCE-F000001-abc",
        "full_text": text,
        "text_sha256": text_hash(text),
        "source_sha256": text_hash(text),
        "text_version": 1,
        "publication_status": "candidate",
        "human_review_status": "pending",
    }

    def span(q):
        return {"start": text.index(q), "end": text.index(q) + len(q)}

    unit = {
        "unit_id": "u1",
        "file_id": "F000001",
        "text": fact,
        "source_text": fact,
        "source_sha256": text_hash(text),
        "source_spans": [span(fact)],
        "case_ids": ["C1"],
        "behavior_ids": ["B1"],
        "case_id": "C1",
        "behavior_id": "B1",
        "associations": [
            {
                "case_id": "C1",
                "behavior_id": "B1",
                "actor_role": "institution_violation",
                "party_quote": party,
                "party_source_spans": [span(party)],
                "quote": fact,
                "quote_source_spans": [span(fact)],
                "qualifier_source_links": [
                    {"span": span(condition), "kind": "condition", "quote": condition}
                ],
            }
        ],
    }
    if bad_span:
        unit["source_spans"][0]["end"] = len(text) + 1
    sources, units = tmp_path / "sources.jsonl", tmp_path / "units.jsonl"
    sources.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
    units.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    dictionary = (
        Path(__file__).resolve().parents[2] / "contracts/agent_rag/risk_type_dictionary.csv"
    )
    return sources, units, dictionary, text, unit


def test_multiple_source_spans_keep_unicode_positions_and_relationships(tmp_path):
    sources, units, dictionary, text, unit = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    candidate = snapshot.candidate(unit, 1, 90)
    assert candidate["source_id"] == "FRV2-SOURCE-F000001-abc"
    assert candidate["file_id"] == "F000001"
    assert candidate["associations"][0]["behavior_id"] == "B1"
    assert len(candidate["required_evidence_ids"]) == 3
    assert len(candidate["evidence"]) == 3
    for evidence in candidate["evidence"]:
        assert evidence["quote"] == text[evidence["start"] : evidence["end"]]
        assert evidence["source_verified"] is True
    assert candidate["annotation_status"] == "candidate_pending_human_acceptance"
    assert len(snapshot.definitions) == 27


def test_source_asset_change_is_rejected_at_read(tmp_path):
    sources, units, dictionary, _, unit = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    sources.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        snapshot.read("FRV2-SOURCE-F000001-abc")


def test_unknown_and_old_source_ids_are_not_guessed(tmp_path):
    sources, units, dictionary, _, _ = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="SOURCE_NOT_ALLOWED"):
        snapshot.read("OFFTXT-same-title")


def test_invalid_span_cannot_become_verified_evidence(tmp_path):
    sources, units, dictionary, _, unit = fixture_assets(tmp_path, bad_span=True)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="INVALID_SOURCE_SPAN"):
        snapshot.candidate(unit, 1, 90)


def test_stale_unit_source_sha_rejected(tmp_path):
    sources, units, dictionary, _, unit = fixture_assets(tmp_path)
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    unit["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        snapshot.candidate(unit, 1, 90)


@pytest.mark.parametrize(
    "role", ["fact", "party", "qualifier", "association_quote", "unit_qualifier"]
)
def test_in_range_misaligned_quote_is_rejected_in_a_new_frozen_snapshot(tmp_path, role):
    sources, units, dictionary, text, unit = fixture_assets(tmp_path)
    wrong = {"start": text.index("其他事项。"), "end": text.index("其他事项。") + 4}
    association = unit["associations"][0]
    if role == "fact":
        unit["source_spans"] = [{"start": 0, "end": 4}]
    elif role == "party":
        association["party_source_spans"] = [wrong]
    elif role == "qualifier":
        association["qualifier_source_links"][0]["span"] = wrong
    elif role == "association_quote":
        association["quote_source_spans"] = [wrong]
    else:
        unit["qualifier_source_links"] = [
            {"field": "conditions", "quote": "仅限合同约定之外。", "source_spans": [wrong]}
        ]
    # Freeze the altered index too: this is not a stale asset/hash failure.
    units.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="SOURCE_QUOTE_MISMATCH"):
        snapshot.candidate(snapshot.units[0], 1, 90)


@pytest.mark.parametrize("role", ["fact", "party", "qualifier"])
def test_spans_without_their_reference_quote_cannot_claim_verified(tmp_path, role):
    sources, units, dictionary, _, unit = fixture_assets(tmp_path)
    if role == "fact":
        del unit["source_text"]
    elif role == "party":
        del unit["associations"][0]["party_quote"]
    else:
        del unit["associations"][0]["qualifier_source_links"][0]["quote"]
    units.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    with pytest.raises(ValueError, match="MISSING_SOURCE_QUOTE"):
        snapshot.candidate(snapshot.units[0], 1, 90)


def test_real_index_shape_repeated_occurrences_each_match_one_quote(tmp_path):
    sources, units, dictionary, text, unit = fixture_assets(tmp_path)
    fact = unit["source_text"]
    repeated = text + fact
    source = json.loads(sources.read_text(encoding="utf-8"))
    source.update(
        full_text=repeated, text_sha256=text_hash(repeated), source_sha256=text_hash(repeated)
    )
    unit["source_sha256"] = text_hash(repeated)
    unit["source_spans"].append({"start": len(text), "end": len(repeated)})
    unit["context_spans"] = unit["source_spans"]
    unit["context_segments"] = [{"span": span, "text": fact} for span in unit["source_spans"]]
    association = unit["associations"][0]
    association["quote_source_spans"] = unit["source_spans"]
    qualifier = association["qualifier_source_links"][0]
    qualifier["source_spans"] = [qualifier.pop("span")]
    qualifier["field"] = "conditions"
    unit["qualifier_source_links"] = [qualifier]
    sources.write_text(json.dumps(source, ensure_ascii=False) + "\n", encoding="utf-8")
    units.write_text(json.dumps(unit, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    candidate = snapshot.candidate(snapshot.units[0], 1, 90)
    assert [ev["quote"] for ev in candidate["evidence"] if "fact" in ev["evidence_roles"]] == [
        fact,
        fact,
    ]
    assert all(ev["source_verified"] for ev in candidate["evidence"])


@pytest.mark.parametrize("bad", ["span", "segment_text", "joined_text"])
def test_source_segments_bind_each_disjoint_piece_and_marker_free_body(tmp_path, bad):
    sources, units, dictionary, text, unit = fixture_assets(tmp_path)
    first = unit["source_spans"][0]
    last = unit["associations"][0]["qualifier_source_links"][0]["span"]
    unit["source_spans"] = [first, last]
    unit["source_segments"] = [
        {"span": first, "text": text[first["start"] : first["end"]]},
        {"span": last, "text": text[last["start"] : last["end"]]},
    ]
    unit["source_text"] = "".join(segment["text"] for segment in unit["source_segments"])
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    good = snapshot.candidate(unit, 1, 90)
    assert len([ev for ev in good["evidence"] if "fact" in ev["evidence_roles"]]) == 2
    if bad == "span":
        unit["source_spans"][1] = first
    elif bad == "segment_text":
        unit["source_segments"][1]["text"] = "伪造限定"
    else:
        unit["source_text"] += "伪造"
    with pytest.raises(ValueError, match="SOURCE_QUOTE_MISMATCH"):
        snapshot.candidate(unit, 1, 90)


def test_optional_absent_party_and_qualifier_produce_no_claimed_evidence(tmp_path):
    sources, units, dictionary, _, unit = fixture_assets(tmp_path)
    association = unit["associations"][0]
    association.pop("party_quote")
    association.pop("party_source_spans")
    association.pop("qualifier_source_links")
    snapshot = create_snapshot(sources, units, dictionary, tmp_path / "snapshot")
    candidate = snapshot.candidate(unit, 1, 90)
    assert candidate["associations"][0]["party_evidence_ids"] == []
    assert candidate["associations"][0]["qualifier_evidence_ids"] == []
    assert len(candidate["evidence"]) == 1
