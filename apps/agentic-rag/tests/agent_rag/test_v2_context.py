import copy
import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from agent_rag.contracts import canonical_json
from agent_rag.input_registry import text_hash
from agent_rag.knowledge_snapshot import make_evidence
from agent_rag.v2_snapshot import V2KnowledgeSnapshot

POLICY = {"policy": "verified-parent-v1"}


def context_module():
    assert importlib.util.find_spec("agent_rag.v2_context"), "V2父上下文投影器尚未实现"
    return importlib.import_module("agent_rag.v2_context")


def fixture_context(parent="甲公司😀提示：不得承诺收益；实际以合同约定为准，法定例外除外。"):
    fact = "不得承诺收益"
    text = "文书标题\n" + parent + "\n文书尾部。"
    start = text.index(fact)
    fact_span = {"start": start, "end": start + len(fact)}
    parent_start = text.index(parent)
    parent_span = {"start": parent_start, "end": parent_start + len(parent)}
    source = {
        "official_file_id": "F1",
        "source_id": "S1",
        "title": "测试文书",
        "full_text": text,
        "text_sha256": text_hash(text),
        "text_version": "1",
    }
    unit = {
        "unit_id": "U1",
        "file_id": "F1",
        "source_sha256": source["text_sha256"],
        "text": fact,
        "source_text": fact,
        "source_spans": [fact_span],
        "parent_source_spans": [copy.deepcopy(parent_span)],
        "parent_span": copy.deepcopy(parent_span),
        "parent_source_text": parent,
        "context_spans": [copy.deepcopy(parent_span)],
        "context_source_text": parent,
        "context_segments": [{"span": copy.deepcopy(parent_span), "text": parent}],
        "case_id": "C1",
        "case_ids": ["C1"],
        "behavior_id": "B1",
        "behavior_ids": ["B1"],
        "metadata_unknown": False,
        "context_ambiguous": False,
        "prefix_evidence_ambiguous": False,
        "associations": [
            {
                "case_id": "C1",
                "behavior_id": "B1",
                "party_position_ambiguous": False,
                "qualifier_position_ambiguous": False,
                "party_quote": "甲公司😀",
                "party_source_spans": [{"start": parent_start, "end": parent_start + 4}],
            }
        ],
        "tags": ["原始标签"],
    }
    snapshot = SimpleNamespace(by_file={"F1": source}, units=[unit])
    candidate = V2KnowledgeSnapshot.candidate(snapshot, unit, 2, 88)
    candidate["tags"] = ["原始标签"]
    return snapshot, unit, candidate


def expand(snapshot, unit, candidate, **limits):
    module = context_module()
    policy = module.validate_context_expansion(POLICY | limits)
    return module.ParentContextExpander(snapshot, policy).expand(unit, candidate)


def assert_skipped(snapshot, unit, candidate, reason, **limits):
    before = copy.deepcopy(candidate)
    result = expand(snapshot, unit, candidate, **limits)
    assert result["context_expansion"]["status"] == "skipped"
    assert result["context_expansion"]["reason"] == reason
    assert result["context_expansion"]["parent_evidence_ids"] == []
    assert result["context_expansion"]["added_evidence_ids"] == []
    assert result["context_expansion"]["added_chars"] == 0
    assert result["evidence"] == before["evidence"]
    assert result["required_evidence_ids"] == before["required_evidence_ids"]
    assert candidate == before


def test_adds_exact_complete_unicode_parent_without_changing_required_or_input():
    snapshot, unit, candidate = fixture_context()
    before = copy.deepcopy((snapshot.by_file, snapshot.units, unit, candidate))
    result = expand(snapshot, unit, candidate)
    original_keys = set(candidate)
    assert {key: result[key] for key in original_keys if key != "evidence"} == {
        key: candidate[key] for key in original_keys if key != "evidence"
    }
    assert result["evidence"][:-1] == candidate["evidence"]
    parent = result["evidence"][-1]
    assert parent["quote"] == unit["parent_source_text"]
    assert parent["quote"] == snapshot.by_file["F1"]["full_text"][parent["start"] : parent["end"]]
    assert "实际以合同约定为准，法定例外除外。" in parent["quote"]
    assert parent["evidence_roles"] == ["context"]
    assert parent["evidence_id"] not in candidate["required_evidence_ids"]
    diagnosis = result["context_expansion"]
    assert diagnosis["status"] == "expanded"
    assert diagnosis["reason"] == "VERIFIED_PARENT"
    assert diagnosis["parent_evidence_ids"] == [parent["evidence_id"]]
    assert diagnosis["added_evidence_ids"] == [parent["evidence_id"]]
    assert diagnosis["added_chars"] == len(parent["quote"]) - 10
    assert (snapshot.by_file, snapshot.units, unit, candidate) == before
    result["associations"][0]["case_id"] = "mutated"
    assert candidate["associations"][0]["case_id"] == "C1"


def test_policy_normalization_hash_and_disabled_copy():
    module = context_module()
    policy = module.validate_context_expansion(POLICY)
    assert policy == POLICY | {"max_parent_chars": 900, "max_added_chars": 600}
    assert module.validate_context_expansion(None) is None
    assert module.validate_context_expansion(policy) == policy
    snapshot, unit, candidate = fixture_context()
    expander = module.ParentContextExpander(snapshot, policy)
    assert expander.policy_sha256 == text_hash(canonical_json(policy))
    assert expander.expand(unit, candidate)["context_expansion"]["policy_sha256"] == (
        expander.policy_sha256
    )
    disabled = module.ParentContextExpander(snapshot, None).expand(unit, candidate)
    assert disabled == candidate
    assert disabled is not candidate
    disabled["evidence"].clear()
    assert candidate["evidence"]


@pytest.mark.parametrize(
    "bad",
    [
        False,
        True,
        0,
        1,
        "verified-parent-v1",
        [],
        {},
        {"policy": "other"},
        POLICY | {"unknown": 1},
        POLICY | {"policy_sha256": "forged"},
        POLICY | {"max_parent_chars": False},
        POLICY | {"max_parent_chars": 0},
        POLICY | {"max_parent_chars": 901},
        POLICY | {"max_parent_chars": 1.0},
        POLICY | {"max_parent_chars": "900"},
        POLICY | {"max_added_chars": True},
        POLICY | {"max_added_chars": 0},
        POLICY | {"max_added_chars": 601},
        POLICY | {"max_added_chars": None},
    ],
)
def test_invalid_policy_is_rejected(bad):
    with pytest.raises(ValueError, match="INVALID_CONTEXT_EXPANSION"):
        context_module().validate_context_expansion(bad)


@pytest.mark.parametrize(
    "field", ["metadata_unknown", "context_ambiguous", "prefix_evidence_ambiguous"]
)
@pytest.mark.parametrize("value", [True, None, 0, "false"])
def test_unit_safety_flags_must_be_explicit_false(field, value):
    snapshot, unit, candidate = fixture_context()
    unit[field] = value
    assert_skipped(snapshot, unit, candidate, "UNSAFE_METADATA")


@pytest.mark.parametrize(
    "field", ["metadata_unknown", "context_ambiguous", "prefix_evidence_ambiguous"]
)
def test_missing_safety_marker_cannot_prove_safe(field):
    snapshot, unit, candidate = fixture_context()
    del unit[field]
    assert_skipped(snapshot, unit, candidate, "UNSAFE_METADATA")


@pytest.mark.parametrize("field", ["party_position_ambiguous", "qualifier_position_ambiguous"])
def test_ambiguous_association_is_not_expanded(field):
    snapshot, unit, candidate = fixture_context()
    unit["associations"][0][field] = True
    assert_skipped(snapshot, unit, candidate, "UNSAFE_METADATA")


def test_hard_split_and_long_markers_do_not_reject_a_complete_short_parent():
    snapshot, unit, candidate = fixture_context()
    unit.update(context_long=True, input_hard_split=True)
    assert expand(snapshot, unit, candidate)["context_expansion"]["status"] == "expanded"


@pytest.mark.parametrize(
    "field, value",
    [
        ("case_ids", []),
        ("case_ids", ["C1", "C2"]),
        ("behavior_ids", []),
        ("behavior_ids", ["B1", "B2"]),
        ("associations", []),
        ("associations", [{"case_id": "C2", "behavior_id": "B1"}]),
        ("associations", [{"case_id": "C1", "behavior_id": "B2"}]),
    ],
)
def test_only_single_case_behavior_and_matching_associations_expand(field, value):
    snapshot, unit, candidate = fixture_context()
    unit[field] = value
    reason = "NON_UNIQUE_CASE_BEHAVIOR" if field != "associations" else "ASSOCIATION_MISMATCH"
    assert_skipped(snapshot, unit, candidate, reason)


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_id", "S2"),
        ("file_id", "F2"),
        ("unit_id", "U2"),
        ("case_id", "C2"),
        ("behavior_id", "B2"),
        ("case_ids", ["C2"]),
        ("behavior_ids", ["B2"]),
    ],
)
def test_candidate_identity_must_match_unit_and_source(field, value):
    snapshot, unit, candidate = fixture_context()
    candidate[field] = value
    assert_skipped(snapshot, unit, candidate, "IDENTITY_MISMATCH")


@pytest.mark.parametrize("bad", ["missing_source", "unit_hash", "source_hash", "source_file"])
def test_source_identity_and_content_hash_are_checked(bad):
    snapshot, unit, candidate = fixture_context()
    if bad == "missing_source":
        snapshot.by_file.clear()
        reason = "SOURCE_NOT_FOUND"
    elif bad == "unit_hash":
        unit["source_sha256"] = "0" * 64
        reason = "SOURCE_HASH_MISMATCH"
    elif bad == "source_hash":
        snapshot.by_file["F1"]["full_text"] += "篡改"
        reason = "SOURCE_HASH_MISMATCH"
    else:
        snapshot.by_file["F1"]["official_file_id"] = "F2"
        reason = "IDENTITY_MISMATCH"
    assert_skipped(snapshot, unit, candidate, reason)


@pytest.mark.parametrize(
    "bad",
    [
        "repeated_fact",
        "repeated_context",
        "repeated_parent",
        "missing_parent_span",
        "bool_start",
        "out_of_range",
        "parent_span_mismatch",
        "context_span_mismatch",
        "missing_segments",
        "extra_segment",
        "segment_span",
        "segment_text",
        "parent_text",
        "context_text",
        "fact_quote",
        "parent_not_covering_fact",
    ],
)
def test_parent_and_context_coordinates_and_text_must_all_match(bad):
    snapshot, unit, candidate = fixture_context()
    if bad == "repeated_fact":
        unit["source_spans"] *= 2
        reason = "INVALID_SOURCE_SPAN"
    elif bad == "repeated_context":
        unit["context_spans"] *= 2
        reason = "INVALID_PARENT_SPAN"
    elif bad == "repeated_parent":
        unit["parent_source_spans"] *= 2
        reason = "INVALID_PARENT_SPAN"
    elif bad == "missing_parent_span":
        del unit["parent_span"]
        reason = "INVALID_PARENT_SPAN"
    elif bad == "bool_start":
        unit["parent_source_spans"][0]["start"] = True
        reason = "INVALID_PARENT_SPAN"
    elif bad == "out_of_range":
        unit["parent_source_spans"][0]["end"] = 10000
        reason = "INVALID_PARENT_SPAN"
    elif bad == "parent_span_mismatch":
        unit["parent_span"]["start"] += 1
        reason = "INVALID_PARENT_SPAN"
    elif bad == "context_span_mismatch":
        unit["context_spans"][0]["start"] += 1
        reason = "INVALID_PARENT_SPAN"
    elif bad == "missing_segments":
        del unit["context_segments"]
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "extra_segment":
        unit["context_segments"] *= 2
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "segment_span":
        unit["context_segments"][0]["span"]["start"] += 1
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "segment_text":
        unit["context_segments"][0]["text"] += "伪造"
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "parent_text":
        unit["parent_source_text"] += "伪造"
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "context_text":
        unit["context_source_text"] += "伪造"
        reason = "SOURCE_QUOTE_MISMATCH"
    elif bad == "fact_quote":
        unit["source_text"] += "伪造"
        reason = "SOURCE_QUOTE_MISMATCH"
    else:
        unit["source_spans"] = [{"start": 0, "end": 4}]
        unit["source_text"] = snapshot.by_file["F1"]["full_text"][:4]
        reason = "PARENT_NOT_CONTAIN_FACT"
    assert_skipped(snapshot, unit, candidate, reason)


@pytest.mark.parametrize(
    "field,value",
    [
        ("quote", "伪造原锚点"),
        ("text_sha256", "0" * 64),
        ("source_id", "S2"),
        ("text_version", "2"),
        ("source_verified", False),
        ("evidence_id", "EV-forged"),
        ("start", True),
    ],
)
def test_invalid_existing_evidence_cannot_reduce_added_budget(field, value):
    snapshot, unit, candidate = fixture_context()
    candidate["evidence"][0][field] = value
    assert_skipped(snapshot, unit, candidate, "INVALID_CANDIDATE_EVIDENCE")


def test_missing_required_evidence_cannot_expand():
    snapshot, unit, candidate = fixture_context()
    candidate["required_evidence_ids"].append("EV-missing")
    assert_skipped(snapshot, unit, candidate, "INVALID_CANDIDATE_EVIDENCE")


@pytest.mark.parametrize("limit_field", ["max_parent_chars", "max_added_chars"])
def test_whole_parent_is_skipped_when_one_character_over_budget(limit_field):
    snapshot, unit, candidate = fixture_context()
    count = len(unit["parent_source_text"])
    limit = count - 1 if limit_field == "max_parent_chars" else count - 11
    reason = "PARENT_TOO_LONG" if limit_field == "max_parent_chars" else "ADDED_CHARS_EXCEEDED"
    assert_skipped(snapshot, unit, candidate, reason, **{limit_field: limit})


def test_parent_hard_cap_of_900_is_enforced_without_truncating():
    snapshot, unit, candidate = fixture_context("甲公司😀" + "不得承诺收益" + "例" * 891)
    assert_skipped(snapshot, unit, candidate, "PARENT_TOO_LONG")


def test_added_chars_are_union_of_original_source_intervals_inside_parent():
    snapshot, unit, candidate = fixture_context()
    source = snapshot.by_file["F1"]
    start, end = unit["parent_span"]["start"], unit["parent_span"]["end"]
    for left, right in [(start, start + 15), (start + 10, end - 2)]:
        item = make_evidence(source, left, right, source["full_text"])
        candidate["evidence"].append(item)
        candidate["required_evidence_ids"].append(item["evidence_id"])
    result = expand(snapshot, unit, candidate, max_added_chars=2)
    assert result["context_expansion"]["status"] == "expanded"
    assert result["context_expansion"]["added_chars"] == 2


@pytest.mark.parametrize(
    "case_ids, file_id, touching, reject",
    [
        (["C2"], "F1", False, True),
        (["C1", "C2"], "F1", False, True),
        (["C1"], "F1", False, False),
        ([], "F1", False, False),
        (["C2"], "F2", False, False),
        (["C2"], "F1", True, False),
    ],
)
def test_only_overlapping_other_known_cases_in_same_source_block_expansion(
    case_ids, file_id, touching, reject
):
    snapshot, unit, candidate = fixture_context()
    parent = unit["parent_span"]
    start = parent["end"] if touching else parent["end"] - 4
    snapshot.units.append(
        {
            "unit_id": "U2",
            "file_id": file_id,
            "case_ids": case_ids,
            "case_id": case_ids[0] if case_ids else "F1:SOURCE",
            "source_spans": [{"start": start, "end": start + 4}],
        }
    )
    if reject:
        assert_skipped(snapshot, unit, candidate, "CROSS_CASE_OVERLAP")
    else:
        assert expand(snapshot, unit, candidate)["context_expansion"]["status"] == "expanded"


def test_existing_parent_is_reused_and_never_removed_as_original_evidence():
    snapshot, unit, candidate = fixture_context()
    source = snapshot.by_file["F1"]
    span = unit["parent_span"]
    item = make_evidence(source, span["start"], span["end"], source["full_text"])
    candidate["evidence"].append(item)
    result = expand(snapshot, unit, candidate)
    assert result["evidence"] == candidate["evidence"]
    assert result["context_expansion"]["parent_evidence_ids"] == [item["evidence_id"]]
    assert result["context_expansion"]["added_evidence_ids"] == []
    assert result["context_expansion"]["added_chars"] == 0
    restored = context_module().strip_parent_context(result, "tool_payload_budget")
    assert restored["evidence"] == candidate["evidence"]


def test_repeat_expansion_is_idempotent_and_remains_reversible():
    snapshot, unit, candidate = fixture_context()
    result = expand(snapshot, unit, candidate)
    repeated = expand(snapshot, unit, result)
    assert repeated == result
    restored = context_module().strip_parent_context(repeated, "tool_payload_budget")
    assert restored["evidence"] == candidate["evidence"]
    assert restored["required_evidence_ids"] == candidate["required_evidence_ids"]
    assert restored["context_expansion"]["status"] == "budget_omitted"
    assert restored["context_expansion"]["reason"] == "tool_payload_budget"
    assert restored["context_expansion"]["added_evidence_ids"] == []
    assert restored["context_expansion"]["parent_evidence_ids"] == []
    assert restored["context_expansion"]["added_chars"] == 0
    assert result["context_expansion"]["status"] == "expanded"


def test_strip_never_removes_required_even_if_named_in_added_diagnostics():
    snapshot, unit, candidate = fixture_context()
    result = expand(snapshot, unit, candidate)
    parent_id = result["context_expansion"]["added_evidence_ids"][0]
    result["required_evidence_ids"].append(parent_id)
    result["context_expansion"]["added_evidence_ids"] += candidate["required_evidence_ids"]
    before = copy.deepcopy(result)
    restored = context_module().strip_parent_context(result, "tool_payload_budget")
    assert restored["evidence"] == before["evidence"]
    assert restored["required_evidence_ids"] == before["required_evidence_ids"]
    assert result == before


def test_strip_without_our_diagnostics_is_only_a_deep_copy():
    _, _, candidate = fixture_context()
    module = context_module()
    stripped = module.strip_parent_context(candidate, "tool_payload_budget")
    assert stripped == candidate
    assert stripped is not candidate
    candidate["context_expansion"] = {
        "policy": "other",
        "added_evidence_ids": [candidate["evidence"][0]["evidence_id"]],
    }
    assert module.strip_parent_context(candidate, "tool_payload_budget") == candidate


def test_multiple_party_associations_do_not_establish_single_subject_context():
    snapshot, unit, candidate = fixture_context()
    second = copy.deepcopy(unit["associations"][0])
    second["party_quote"] = "另一主体"
    unit["associations"].append(second)
    assert_skipped(snapshot, unit, candidate, "ASSOCIATION_MISMATCH")


def test_historical_evidence_ambiguity_from_hard_split_does_not_change_candidate_flags():
    snapshot, unit, _ = fixture_context()
    unit.update(evidence_ambiguous=True, context_long=True, input_hard_split=True)
    candidate = V2KnowledgeSnapshot.candidate(snapshot, unit, 2, 88)
    assert candidate["context_ambiguous"] is True
    result = expand(snapshot, unit, candidate)
    assert result["context_expansion"]["status"] == "expanded"
    assert result["context_ambiguous"] is True
    assert result["annotation_status"] == candidate["annotation_status"]
