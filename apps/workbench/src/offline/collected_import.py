"""Reuse archived collection text and its existing labels without chat or web requests."""

from __future__ import annotations

import hashlib
import re
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from offline.knowledge_projection import institution_type, normalize_date
from offline.penalty_crawler.polite_http import validate_url
from offline.release_import import rows
from risk.detection import RuleRiskDetector

PREFIX = "data/raw/cases/crawl/"
LEGACY_CASES = "data/processed/knowledge/case_kb_v2_1_2026-08-23/data/cases.jsonl"
JsonObject = dict[str, Any]

# Only field headers are normalized. The values remain literal source substrings.
HEADERS = {
    "party_name": r"(?:被处罚当事人|当事人)",
    "penalty_doc_no": r"行政处罚决定书文号",
    "violation_behavior": r"主要违法违规事实(?:[ \t]*(?:\(案由\)|（案由）))?",
    "penalty_content": r"行政处罚(?:内容|决定)(?!书)",
    "regulator": r"作出处罚决定的[ \t\n]*机关名称",
    "penalty_date": r"作出处罚决定的日期",
    "legal_basis": r"行政处罚依据",
    "source_title": r"来源公示",
}
HEADER_PATTERN = re.compile(
    r"(?m)^[ \t]*(?P<header>"
    + "|".join(f"(?P<{name}>{pattern})" for name, pattern in HEADERS.items())
    + r")(?:[ \t]*[:：][ \t]*|[ \t]*\n)"
)


@dataclass(frozen=True)
class CollectedDocument:
    member: str
    raw: bytes
    metadata: JsonObject
    legacy: JsonObject | None


def archive_documents(path: Path) -> Iterator[CollectedDocument]:
    """Resolve archive members internally; never follow paths saved on the old computer."""
    with zipfile.ZipFile(path) as archive:
        ledger = {str(row["sha1"]): row for row in rows(archive, PREFIX + "ingested.jsonl")}
        ids = {str(row["file_id"]) for row in ledger.values()}
        legacy = {
            str(row["file_id"]): row
            for row in rows(archive, LEGACY_CASES)
            if str(row.get("file_id")) in ids
        }
        for metadata in rows(archive, PREFIX + "index.jsonl"):
            filename = PurePosixPath(str(metadata["txt_path"]).replace("\\", "/")).name
            column = int(metadata["column"])
            if column not in {4113, 4114, 4115}:
                raise ValueError("非支持的官网采集栏目")
            member = f"{PREFIX}nfra/{column}/{filename}"
            raw = archive.read(member)
            text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
            sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if sha1 != metadata["sha1"]:
                raise ValueError(f"历史采集文件校验失败: {member}")
            validate_url(str(metadata["source_url"]))
            old_id = str(ledger.get(sha1, {}).get("file_id") or "")
            yield CollectedDocument(member, raw, metadata, legacy.get(old_id))


def parse_fields(text: str) -> dict[str, str]:
    matches = list(HEADER_PATTERN.finditer(text))
    result: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = next(name for name in HEADERS if match.group(name) is not None)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[match.end() : end].strip()
        if key == "violation_behavior":
            value = re.sub(r"^[（(]案由[）)]\s*", "", value)
        if value:
            result[key] = value
    if not result.get("violation_behavior"):
        # A minority are delivery notices rather than public-disclosure tables.
        party = re.match(r"^([^\n：:]{2,30})[：:]", text)
        facts = re.search(r"经查，(.+?)。根据", text, re.S)
        penalty = re.search(r"决定对你作出[^。]+行政处罚", text)
        if party and facts and penalty:
            result.update(
                party_name=party.group(1),
                violation_behavior=facts.group(1),
                penalty_content=penalty.group(0),
            )
            regulators = re.findall(r"(?m)^国家金融监督管理总局[^\n]*监管(?:分)?局$", text)
            dates = re.findall(r"(?m)^\d{4}年\d{1,2}月\d{1,2}日$", text)
            if regulators:
                result["regulator"] = regulators[-1]
            if dates:
                result["penalty_date"] = dates[-1]
    return result


def literal_value(value: object, text: str) -> str | None:
    """Recover exact whitespace from the source instead of accepting rewritten quotations."""
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    if value in text:
        return value
    pattern = r"\s+".join(re.escape(word) for word in value.split())
    match = re.search(pattern, text)
    return match.group(0) if match else None


def is_insurance_case(fields: dict[str, str]) -> bool:
    party = fields.get("party_name", "")
    facts = fields.get("violation_behavior", "")
    if re.search(r"保险|寿险|财险|人寿|财保", party):
        return True
    if "资产管理" in party and re.search(
        r"保险|新华|太平|华夏久盈|民生通惠|中再", fields.get("legal_basis", "") + party
    ):
        return True
    # Laws and regulator names do not establish the subject's insurance relevance.
    return bool(re.search(r"保险|投保人|保单|保费|理赔", facts))


def collected_payload(
    document: CollectedDocument,
    text: str,
    taxonomy: set[str],
    detector: RuleRiskDetector,
) -> dict[str, object] | None:
    fields = parse_fields(text)
    legacy = document.legacy or {}
    for name in (
        "party_name",
        "violation_behavior",
        "penalty_content",
        "penalty_doc_no",
        "regulator",
    ):
        if name not in fields:
            value = literal_value(legacy.get(name), text)
            if value:
                fields[name] = value
    if not is_insurance_case(fields):
        return None
    for name in ("party_name", "violation_behavior", "penalty_content"):
        if not fields.get(name):
            raise ValueError(f"已采集材料缺少{name}")
    party, facts = fields["party_name"], fields["violation_behavior"]
    if len(party) > 300:
        # Keep the whole notice as a grouped document, with the first named subject as its anchor.
        party = next((line.strip() for line in party.splitlines() if line.strip()), party)
    basis: list[dict[str, str]] = []
    for entry in legacy.get("risk_tag_basis") or []:
        quote = literal_value(entry.get("quote"), facts)
        tag = entry.get("risk_tag")
        if tag in taxonomy and quote:
            basis.append(
                {
                    "risk_tag": tag,
                    "reason": str(entry.get("reason") or "既有原文标签")[:300],
                    "quote": quote[:500],
                }
            )
    if not basis:
        labels = [tag for tag in detector.detect(facts).risk_labels if tag in taxonomy]
        if not labels and "其他" in taxonomy:
            labels = ["其他"]
        basis = [
            {"risk_tag": tag, "reason": "基于采集表格违法事实的本地规则标签", "quote": facts[:500]}
            for tag in labels
        ]
    tags = list(dict.fromkeys(entry["risk_tag"] for entry in basis))
    return {
        "import_kind": "legacy_structured" if legacy else "collected_fields",
        "upstream_case_id": legacy.get("case_id"),
        "is_insurance_related": True,
        "party_name": party,
        "institution_type": institution_type(party, legacy.get("party_type")),
        "penalty_doc_no": fields.get("penalty_doc_no"),
        "regulator": fields.get("regulator"),
        "penalty_date": normalize_date(fields.get("penalty_date") or legacy.get("penalty_date")),
        "violation_behavior": facts,
        "fact_quotes": [facts],
        "penalty_content": fields["penalty_content"],
        "risk_tags": tags,
        "risk_tag_basis": basis,
        "case_summary": f"{facts}。{fields['penalty_content']}"[:1000],
        "case_title": f"{party} {fields.get('penalty_doc_no', '')}"[:500].strip(),
        "confidence": 0.0,
        "keywords": [],
        "marketing_expressions": [],
    }
