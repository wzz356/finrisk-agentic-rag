"""Provider-facing shared source table; the full evidence contract stays on the host."""

from __future__ import annotations

from typing import Any


def compact_review_payload(items: list[dict[str, Any]], *, lean: bool = False) -> dict[str, Any]:
    sources: dict[str, dict[str, Any]] = {}
    aliases: dict[str, str] = {}
    scopes: dict[str, str] = {}
    targets = []
    case_bank: dict[str, dict[str, Any]] = {}
    for item in items:
        context = item["context"]
        scope = str(context["scope_id"])
        scopes.setdefault(scope, f"p{len(scopes) + 1}")
        allowed = []
        target_source = None
        for source in context["source_sentences"]:
            sid = source["sentence_id"]
            aliases.setdefault(sid, f"s{len(aliases) + 1}")
            alias = aliases[sid]
            sources.setdefault(
                sid,
                {
                    "id": alias,
                    "text": source["text"],
                    "speaker_role": source["speaker_role"],
                    "scope": scopes[scope],
                },
            )
            allowed.append(alias)
            if source["is_target"]:
                target_source = alias
        target = {
            "sentence_id": item["sentence_id"],
            "scope": scopes[scope],
            "target_source": target_source,
            "source_ids": allowed,
            "scene": item["scene"],
            "stance": context["stance"],
            "speaker_role": context["speaker_role"],
            "review_reasons": context.get("review_reasons", []),
            "context_truncated": context["context_truncated"],
            "local_scores": [item["local_stage1_score"], item["local_stage2_score"]],
            "local_labels": item["local_labels"],
            "local_evidence": item["local_evidence"],
        }
        if item.get("reference_cases"):
            target["reference_case_ids"] = []
            for case in item["reference_cases"]:
                case_bank.setdefault(case["case_id"], case)
                target["reference_case_ids"].append(case["case_id"])
        if item.get("mechanism_candidates"):
            target["mechanism_candidates"] = item["mechanism_candidates"]
        if lean:
            for key in ["local_scores", "local_evidence", "review_reasons", "scene"]:
                target.pop(key, None)
        targets.append(target)
    result = {"source_sentences": list(sources.values()), "targets": targets}
    if case_bank:
        result["reference_cases"] = list(case_bank.values())
    if lean and items:
        result["scene"] = items[0]["scene"]
    return result
