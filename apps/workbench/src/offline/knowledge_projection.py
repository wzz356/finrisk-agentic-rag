"""Lossless application metadata projection of canonical case fields."""

from __future__ import annotations

import re
from datetime import date
from typing import Any


def normalize_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*(\d{4})[-年/.](\d{1,2})[-月/.](\d{1,2})日?\s*", value)
    if not match:
        return None
    try:
        return date(*(int(v) for v in match.groups())).isoformat()
    except ValueError:
        return None


def institution_type(name: str, party_type: str | None = None) -> str | None:
    if party_type == "person":
        return "个人责任人"
    for pattern, label in [
        (r"保险资产管理|保险资管", "保险资管机构"),
        (r"保险经纪", "保险经纪机构"),
        (r"保险代理", "保险代理机构"),
        (r"保险公估", "保险公估机构"),
        (r"人寿|寿险|养老保险|健康保险", "人身险公司"),
        (r"财产保险|财险|农业保险", "财产险公司"),
        (r"再保险", "再保险公司"),
        (r"保险", "保险机构"),
        (r"银行|信用合作|信用社", "银行机构"),
    ]:
        if re.search(pattern, name):
            return label
    return "其他机构" if party_type == "institution" else None


def project_metadata(case: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    parties = case.get("parties") or []
    party = parties[0] if parties else {}
    penalties = case.get("penalties") or []
    contents = []
    for penalty in penalties:
        kind = str(penalty.get("type") or "")
        quote = penalty.get("amount_quote") or penalty.get("basis_quote")
        amount = penalty.get("amount_yuan")
        if quote and len(str(quote).strip()) <= 180:
            text = str(quote).strip()
            contents.append(text if kind in text else f"{kind}：{text}")
        elif amount is not None:
            contents.append(f"{kind}{amount:g}元" if isinstance(amount, (int, float)) else kind)
        else:
            contents.append(kind)
    field_sources = dict(case.get("field_evidence") or {})
    field_sources["penalty_content"] = list(
        dict.fromkeys(eid for p in penalties for eid in p.get("evidence_ids", []))
    )
    field_sources["party_name"] = field_sources.pop("party", [])
    relations = []
    for p in parties:
        if case.get("regulator") and penalties:
            relations.append(
                {
                    "source": case["regulator"],
                    "relation": "作出处罚",
                    "target": p.get("name"),
                    "evidence_ids": field_sources.get("regulator", [])
                    + field_sources["penalty_content"],
                    "origin": "canonical_case",
                }
            )
        for name in case.get("related_party_names") or []:
            if isinstance(name, str) and name != p.get("name"):
                relations.append(
                    {
                        "source": p.get("name"),
                        "relation": "文书记载关联主体",
                        "target": name,
                        "origin": "canonical_related_party_names",
                    }
                )
    text = document.get("full_text", "")
    published = normalize_date(document.get("publish_date"))
    if not published:
        match = re.search(
            r"(?:发布日期|发布时间|公示日期)[：:\s]*(\d{4}[年./-]\d{1,2}[月./-]\d{1,2}日?)", text
        )
        published = normalize_date(match[1]) if match else None
    return {
        "institution_type": institution_type(str(party.get("name") or ""), party.get("party_type")),
        "party_type": party.get("party_type"),
        "penalty_date": normalize_date(case.get("penalty_date")),
        "publish_date": published,
        "penalty_content": "；".join(dict.fromkeys(contents)),
        "penalty_amounts": penalties,
        "related_parties": parties,
        "entity_relations": relations,
        "source_url": document.get("source_url"),
        "field_sources": field_sources,
        "field_quality": {k: "有原文证据" if v else "待补来源" for k, v in field_sources.items()},
        "field_evidence_state": case.get("field_evidence_state"),
        "is_insurance_related": True,
        "metadata_version": "case-fields-20261019",
    }
