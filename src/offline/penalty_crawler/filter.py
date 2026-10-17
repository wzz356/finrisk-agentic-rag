"""采集案例的保险相关性筛选、银行当事人排除与去重。

口径（2026-08-22 与案例库侧确认）：原始语料为保险业处罚（财险/寿险/中介/个人），
银行代销保险被罚的文书虽含"保险"字样但当事人是银行，不入库；寿险公司
"银行保险部/银行渠道部"的个人责任人案是保险口径，不误伤；保险资管公司属
保险业机构，放行。

去重三级：采集侧 state.json（已见 docId）→ 本模块 ingested.jsonl 台账 sha1 →
案例库侧 case_id = "U" + sha256(文件字节)[:15]（与 CaseManagementService 同一算法，
重复时 CaseRepository.add_case 会抛 DuplicateCaseError 兜底）。
"""

from __future__ import annotations

import json
import re
from hashlib import sha1, sha256
from pathlib import Path

INSURANCE_KEYWORD = "保险"
BANK_PARTY_RE = re.compile(
    r"当事人[：:][^\n]*(?:银行股份|银行(?:分行|支行)|农村商业|村镇银行|信用合作社|农村信用)"
)


def sha1_text(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()


def case_id_for_file(path: Path) -> str:
    """与 CaseManagementService.ingest_file 相同的 case_id 推导（文件字节 SHA-256）。"""
    return f"U{sha256(path.read_bytes()).hexdigest()[:15].upper()}"


def load_existing_case_ids(cases_path: Path) -> set[str]:
    ids: set[str] = set()
    if not cases_path.is_file():
        return ids
    with cases_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(str(json.loads(line).get("case_id") or ""))
            except json.JSONDecodeError:
                continue
    ids.discard("")
    return ids


def load_ingested_sha1(crawl_dir: Path) -> set[str]:
    keys: set[str] = set()
    ledger = crawl_dir / "ingested.jsonl"
    if ledger.is_file():
        with ledger.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line).get("sha1")
                except json.JSONDecodeError:
                    continue
                if value:
                    keys.add(str(value))
    return keys


def collect_pending(
    crawl_dir: Path,
    *,
    ingested_sha1: set[str],
    existing_case_ids: set[str],
    include_all: bool = False,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """读 index.jsonl，筛出未入库的待入拆分案。返回 (待入列表, 统计)。"""
    stats = {
        "indexed": 0,
        "already_ingested": 0,
        "non_insurance": 0,
        "bank_party": 0,
        "missing_file": 0,
    }
    pending: list[dict[str, object]] = []
    seen_sha1: set[str] = set()
    index_path = crawl_dir / "index.jsonl"
    if not index_path.is_file():
        return pending, stats
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            stats["indexed"] += 1
            txt = Path(str(row["txt_path"]))
            candidates = [txt]
            if not txt.is_absolute():
                candidates.append(crawl_dir / txt)
            # 跨机器搬移后绝对路径失效时，按 栏目/文件名 在本包内重定位
            candidates.append(crawl_dir / "nfra" / str(row.get("column") or "") / txt.name)
            txt = next((path for path in candidates if path.is_file()), candidates[-1])
            if not txt.is_file():
                stats["missing_file"] += 1
                continue
            text = txt.read_text(encoding="utf-8")
            digest = sha1_text(text)
            if digest in ingested_sha1 or digest in seen_sha1:
                stats["already_ingested"] += 1
                continue
            if case_id_for_file(txt) in existing_case_ids:
                stats["already_ingested"] += 1
                continue
            if not include_all and INSURANCE_KEYWORD not in text:
                stats["non_insurance"] += 1
                continue
            if not include_all and BANK_PARTY_RE.search(text):
                stats["bank_party"] += 1
                continue
            seen_sha1.add(digest)
            pending.append(
                {
                    "txt_path": txt,
                    "sha1": digest,
                    "title": str(row.get("title") or ""),
                    "source_url": str(row.get("source_url") or ""),
                }
            )
    return pending, stats
