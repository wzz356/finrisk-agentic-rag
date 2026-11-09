"""Read normalized releases, skip existing IDs and validate original evidence."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from offline.knowledge_projection import institution_type, normalize_date
from offline.penalty_crawler.polite_http import validate_url

JsonObject = dict[str, Any]


def rows(archive: zipfile.ZipFile, name: str) -> Iterator[JsonObject]:
    with archive.open(name) as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_new_sources(
    archive_path: Path, existing_ids: set[str]
) -> tuple[list[JsonObject], JsonObject]:
    with zipfile.ZipFile(archive_path) as archive:
        cases = []
        total = duplicates = 0
        for case in rows(archive, "data/cases.jsonl"):
            total += 1
            if case["case_id"] in existing_ids:
                duplicates += 1
            else:
                cases.append(case)
        if not cases:
            return [], {
                "upstream_case_count": total,
                "existing_case_ids": duplicates,
                "candidate_cases": 0,
                "candidate_sources": 0,
            }
        fids = {c["official_file_id"] for c in cases}
        case_ids = {c["case_id"] for c in cases}
        sources = {
            r["official_file_id"]: r
            for r in rows(archive, "data/source_inventory.jsonl")
            if r["official_file_id"] in fids
        }
        documents = {
            r["official_file_id"]: r
            for r in rows(archive, "data/documents.jsonl")
            if r["official_file_id"] in fids
        }
        behaviors: dict[str, list[JsonObject]] = defaultdict(list)
        for row in rows(archive, "data/behaviors.jsonl"):
            if row.get("case_id") in case_ids:
                behaviors[row["case_id"]].append(row)
        evidence: dict[str, list[JsonObject]] = defaultdict(list)
        for row in rows(archive, "data/evidence.jsonl"):
            if row.get("official_file_id") in fids:
                evidence[row["official_file_id"]].append(row)
        grouped: dict[str, JsonObject] = {}
        for case in cases:
            fid = case["official_file_id"]
            group = grouped.setdefault(
                fid,
                {
                    "file_id": fid,
                    "source": sources[fid],
                    "document": documents[fid],
                    "evidence": evidence[fid],
                    "cases": [],
                },
            )
            group["cases"].append({"case": case, "behaviors": behaviors[case["case_id"]]})
        return list(grouped.values()), {
            "upstream_case_count": total,
            "existing_case_ids": duplicates,
            "candidate_cases": len(cases),
            "candidate_sources": len(grouped),
        }


def validate_source(group: JsonObject) -> str:
    document, source = group["document"], group["source"]
    text = document["full_text"]
    if not text or hashlib.sha256(text.encode("utf-8")).hexdigest() != document["text_sha256"]:
        raise ValueError("发布包来源全文指纹不一致")
    if source.get("insurance_relevance") is not True:
        raise ValueError("发布包来源未确认保险相关性")
    url = str((source.get("original_mapping_metadata") or {}).get("source_url") or "")
    validate_url(url)
    for evidence in group["evidence"]:
        start, end = evidence["start_char"], evidence["end_char"]
        if (
            evidence.get("offset_unit") != "unicode_codepoint"
            or evidence.get("text_sha256") != document["text_sha256"]
            or not 0 <= start < end <= len(text)
            or text[start:end] != evidence["verbatim"]
        ):
            raise ValueError("发布包证据区间或逐字引文不一致")
    return url


def case_payload(item: JsonObject, text: str, taxonomy: set[str]) -> dict[str, object]:
    case, behaviors = item["case"], item["behaviors"]
    parties = case.get("parties") or []
    if len(parties) != 1 or not parties[0].get("name"):
        raise ValueError("发布包案件卡须有一个可核对的处罚主体")
    party = parties[0]
    facts = list(
        dict.fromkeys(str(b.get("regulatory_finding") or b.get("action") or "") for b in behaviors)
    )
    if not facts or any(not fact or fact not in text for fact in facts):
        raise ValueError("发布包处罚事实未对应原文")
    for field in ("penalty_doc_no", "regulator"):
        if case.get(field) and case[field] not in text:
            raise ValueError(f"发布包{field}未对应原文")
    tags = list(dict.fromkeys(tag for b in behaviors for tag in b.get("risk_tags", [])))
    if set(tags) - taxonomy:
        raise ValueError("发布包标签与本地27类标签不一致")
    basis = []
    for behavior in behaviors:
        for entry in behavior.get("risk_tag_basis", []):
            for quote in entry.get("quotes", []):
                if quote not in text:
                    raise ValueError("发布包标签引文未对应原文")
                basis.append(
                    {
                        "risk_tag": entry["tag"],
                        "reason": str(entry.get("reason") or "发布包原文标签依据"),
                        "quote": quote,
                    }
                )
    if set(tags) - {entry["risk_tag"] for entry in basis}:
        raise ValueError("发布包标签缺少逐字原文依据")
    penalties = []
    for penalty in case.get("penalties", []):
        quote = str(penalty.get("quote") or penalty.get("amount_quote") or "")
        value = re.sub(r"^行政处罚内容\s*", "", quote).strip()
        if value and value in text:
            penalties.append(value)
    if not penalties:
        raise ValueError("发布包处罚内容缺少可核对的原文")
    return {
        "upstream_case_id": case["case_id"],
        "is_insurance_related": True,
        "party_name": party["name"],
        "party_type": party.get("party_type"),
        "penalty_amounts": case.get("penalties", []),
        "institution_type": institution_type(party["name"], party.get("party_type")),
        "penalty_doc_no": case.get("penalty_doc_no"),
        "regulator": case.get("regulator"),
        "penalty_date": normalize_date(case.get("penalty_date")),
        "violation_behavior": "；".join(facts),
        "fact_quotes": facts,
        "penalty_content": "；".join(penalties),
        "risk_tags": tags,
        "risk_tag_basis": basis,
        "case_summary": case["case_summary"],
        "case_title": f"{party['name']} {case.get('penalty_doc_no') or ''}".strip(),
        "legal_basis_refs": [
            {"law_name": r.get("law") or r.get("law_name"), "article": r.get("article")}
            for r in case.get("legal_refs", [])
        ],
        "confidence": 0.0,
        "keywords": [],
        "marketing_expressions": [],
    }
