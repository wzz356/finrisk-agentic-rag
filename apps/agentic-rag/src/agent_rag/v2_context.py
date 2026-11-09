"""只把精确绑定且归属安全的完整父段补为可撤销上下文。"""

import copy
from collections import defaultdict
from typing import Any

from agent_rag.contracts import canonical_json
from agent_rag.input_registry import text_hash
from agent_rag.knowledge_snapshot import make_evidence

POLICY = "verified-parent-v1"


def validate_context_expansion(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("policy") != POLICY
        or set(value) - {"policy", "max_parent_chars", "max_added_chars"}
    ):
        raise ValueError("INVALID_CONTEXT_EXPANSION")
    result = {"policy": POLICY}
    for field, maximum in [("max_parent_chars", 900), ("max_added_chars", 600)]:
        limit = value.get(field, maximum)
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise ValueError("INVALID_CONTEXT_EXPANSION")
        result[field] = limit
    return result


def _span(value: Any, length: int, reason: str) -> tuple[int, int]:
    if not isinstance(value, dict):
        raise ValueError(reason)
    start, end = value.get("start"), value.get("end")
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= length:
        raise ValueError(reason)
    return start, end


def _single_span(value: Any, length: int, reason: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(reason)
    return _span(value[0], length, reason)


def _single_id(value: Any) -> str:
    if (
        not isinstance(value, list)
        or len(value) != 1
        or not isinstance(value[0], str)
        or not value[0]
    ):
        raise ValueError("NON_UNIQUE_CASE_BEHAVIOR")
    return value[0]


def _covered_chars(spans: list[tuple[int, int]], parent: tuple[int, int]) -> int:
    left, right = parent
    covered, last_end = 0, left
    for start, end in sorted(spans):
        start, end = max(left, start, last_end), min(right, end)
        if start < end:
            covered += end - start
            last_end = end
    return covered


class ParentContextExpander:
    def __init__(self, snapshot: Any, policy: dict[str, Any] | None) -> None:
        self.snapshot = snapshot
        self.policy = validate_context_expansion(policy)
        self.policy_sha256 = text_hash(canonical_json(self.policy)) if self.policy else None
        self._ownership: dict[str, list[tuple[set[str], Any]]] = defaultdict(list)
        for unit in snapshot.units:
            cases = unit.get("case_ids")
            if not isinstance(cases, list) or not cases:
                continue
            known_cases = {case for case in cases if isinstance(case, str) and case}
            if known_cases:
                for span in unit.get("source_spans", []):
                    self._ownership[unit.get("file_id")].append((known_cases, copy.deepcopy(span)))

    def _diagnosis(self, status: str, reason: str, **values: Any) -> dict[str, Any]:
        return {
            "policy": POLICY,
            "policy_sha256": self.policy_sha256,
            "status": status,
            "reason": reason,
            "parent_evidence_ids": [],
            "added_evidence_ids": [],
            "added_chars": 0,
            **values,
        }

    def _validated_parent(
        self, unit: dict[str, Any], candidate: dict[str, Any]
    ) -> tuple[dict[str, Any], tuple[int, int], list[dict[str, Any]]]:
        source = self.snapshot.by_file.get(unit.get("file_id"))
        if not isinstance(source, dict):
            raise ValueError("SOURCE_NOT_FOUND")
        text = source.get("full_text")
        if (
            not isinstance(text, str)
            or text_hash(text) != source.get("text_sha256")
            or unit.get("source_sha256") != source.get("text_sha256")
        ):
            raise ValueError("SOURCE_HASH_MISMATCH")
        case, behavior = _single_id(unit.get("case_ids")), _single_id(unit.get("behavior_ids"))
        if (
            source.get("official_file_id") != unit.get("file_id")
            or not isinstance(source.get("source_id"), str)
            or not source["source_id"]
            or not isinstance(source.get("text_version"), str)
            or candidate.get("source_id") != source["source_id"]
            or ("source_id" in unit and unit["source_id"] != source["source_id"])
            or any(candidate.get(key) != unit.get(key) for key in ["unit_id", "file_id"])
            or any(candidate.get(key) != unit.get(key) for key in ["case_ids", "behavior_ids"])
            or unit.get("case_id") not in (None, case)
            or unit.get("behavior_id") not in (None, behavior)
            or candidate.get("case_id") != unit.get("case_id")
            or candidate.get("behavior_id") != unit.get("behavior_id")
        ):
            raise ValueError("IDENTITY_MISMATCH")
        for associations in [unit.get("associations"), candidate.get("associations")]:
            if (
                not isinstance(associations, list)
                or len(associations) != 1
                or any(
                    not isinstance(row, dict)
                    or row.get("case_id") != case
                    or row.get("behavior_id") != behavior
                    for row in associations
                )
            ):
                raise ValueError("ASSOCIATION_MISMATCH")
        if any(
            unit.get(field) is not False
            for field in ["metadata_unknown", "context_ambiguous", "prefix_evidence_ambiguous"]
        ) or any(
            association.get(field) is not False
            for association in unit["associations"]
            for field in ["party_position_ambiguous", "qualifier_position_ambiguous"]
        ):
            raise ValueError("UNSAFE_METADATA")
        fact = _single_span(unit.get("source_spans"), len(text), "INVALID_SOURCE_SPAN")
        parent = _single_span(unit.get("parent_source_spans"), len(text), "INVALID_PARENT_SPAN")
        context = _single_span(unit.get("context_spans"), len(text), "INVALID_PARENT_SPAN")
        if (
            context != parent
            or _span(unit.get("parent_span"), len(text), "INVALID_PARENT_SPAN") != parent
        ):
            raise ValueError("INVALID_PARENT_SPAN")
        segments = unit.get("context_segments")
        quote = text[parent[0] : parent[1]]
        if (
            not isinstance(segments, list)
            or len(segments) != 1
            or not isinstance(segments[0], dict)
            or segments[0].get("span") != unit["parent_source_spans"][0]
            or segments[0].get("text") != quote
            or unit.get("parent_source_text") != quote
            or unit.get("context_source_text") != quote
            or unit.get("source_text") != text[fact[0] : fact[1]]
        ):
            raise ValueError("SOURCE_QUOTE_MISMATCH")
        if not parent[0] <= fact[0] < fact[1] <= parent[1]:
            raise ValueError("PARENT_NOT_CONTAIN_FACT")
        if parent[1] - parent[0] > self.policy["max_parent_chars"]:
            raise ValueError("PARENT_TOO_LONG")
        for cases, span in self._ownership.get(unit["file_id"], []):
            if cases - {case}:
                other = _span(span, len(text), "INVALID_SOURCE_SPAN")
                if max(parent[0], other[0]) < min(parent[1], other[1]):
                    raise ValueError("CROSS_CASE_OVERLAP")
        evidence, required = candidate.get("evidence"), candidate.get("required_evidence_ids")
        if not isinstance(evidence, list) or not isinstance(required, list):
            raise ValueError("INVALID_CANDIDATE_EVIDENCE")
        ids = []
        for item in evidence:
            start, end = _span(item, len(text), "INVALID_CANDIDATE_EVIDENCE")
            expected = make_evidence(source, start, end, text)
            if (
                any(
                    item.get(key) != expected[key]
                    for key in ["evidence_id", "source_id", "text_sha256", "text_version", "quote"]
                )
                or item.get("source_verified") is not True
            ):
                raise ValueError("INVALID_CANDIDATE_EVIDENCE")
            ids.append(item["evidence_id"])
        if (
            len(ids) != len(set(ids))
            or any(not isinstance(eid, str) or eid not in ids for eid in required)
            or make_evidence(source, fact[0], fact[1], text)["evidence_id"] not in required
        ):
            raise ValueError("INVALID_CANDIDATE_EVIDENCE")
        return source, parent, evidence

    def expand(self, unit: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(candidate)
        if self.policy is None:
            return result
        try:
            source, parent, evidence = self._validated_parent(unit, candidate)
            parent_item = make_evidence(source, parent[0], parent[1], source["full_text"])
            parent_id = parent_item["evidence_id"]
            previous = candidate.get("context_expansion", {})
            previous_added = set()
            if (
                isinstance(previous, dict)
                and previous.get("policy") == POLICY
                and previous.get("policy_sha256") == self.policy_sha256
                and previous.get("status") == "expanded"
                and previous.get("parent_evidence_ids") == [parent_id]
                and previous.get("added_evidence_ids") == [parent_id]
                and parent_id not in candidate["required_evidence_ids"]
            ):
                previous_added.add(parent_id)
            covered = _covered_chars(
                [
                    (item["start"], item["end"])
                    for item in evidence
                    if item["evidence_id"] not in previous_added
                ],
                parent,
            )
            added_chars = parent[1] - parent[0] - covered
            if added_chars > self.policy["max_added_chars"]:
                raise ValueError("ADDED_CHARS_EXCEEDED")
        except ValueError as error:
            result["context_expansion"] = self._diagnosis("skipped", str(error))
            return result
        added_ids = list(previous_added)
        if all(item["evidence_id"] != parent_id for item in evidence):
            parent_item["evidence_roles"] = ["context"]
            result["evidence"].append(parent_item)
            added_ids.append(parent_id)
        result["context_expansion"] = self._diagnosis(
            "expanded",
            "VERIFIED_PARENT",
            parent_evidence_ids=[parent_id],
            added_evidence_ids=added_ids,
            added_chars=added_chars,
        )
        return result


def strip_parent_context(candidate: dict[str, Any], reason: str) -> dict[str, Any]:
    result = copy.deepcopy(candidate)
    diagnosis = result.get("context_expansion")
    if not isinstance(diagnosis, dict) or diagnosis.get("policy") != POLICY:
        return result
    if diagnosis.get("status") != "expanded":
        return result
    required = set(result.get("required_evidence_ids", []))
    removable = set(diagnosis.get("added_evidence_ids", [])) - required
    result["evidence"] = [
        item for item in result.get("evidence", []) if item.get("evidence_id") not in removable
    ]
    diagnosis.update(
        status="budget_omitted",
        reason=reason,
        added_evidence_ids=[],
        added_chars=0,
        parent_evidence_ids=[
            eid for eid in diagnosis.get("parent_evidence_ids", []) if eid not in removable
        ],
    )
    return result
