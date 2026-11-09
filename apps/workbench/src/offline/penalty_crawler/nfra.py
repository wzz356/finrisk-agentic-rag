"""国家金融监督管理总局处罚栏目：有日期下界、有预算的增量采集。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

from .common import html_to_text
from .polite_http import CollectionHTTPError, PoliteHTTPClient, SitePaused, atomic_json
from .table_split import extract_table_cases

BASE = "https://www.nfra.gov.cn"
LIST_URL = (
    BASE
    + "/cn/static/data/DocInfo/SelectDocByItemIdAndChild/"
    + "data_itemId={item_id},pageIndex={page},pageSize=18.json"
)
DETAIL_URL = BASE + "/cn/static/data/DocInfo/SelectByDocId/data_docId={doc_id}.json"
LIST_API = BASE + "/cbircweb/DocInfo/SelectDocByItemIdAndChild"
DETAIL_API = BASE + "/cbircweb/DocInfo/SelectByDocId"
PAGE_URL = BASE + "/cn/view/pages/ItemDetail.html?docId={doc_id}&itemId={item_id}&generaltype=9"
COLUMNS = {4113: "总局机关", 4114: "省监管局", 4115: "监管分局"}
PAGE_SIZE = 18
JsonObject = dict[str, Any]


@dataclass
class CollectionBatch:
    documents: dict[str, JsonObject] = field(default_factory=dict)
    pending_details: dict[str, JsonObject] = field(default_factory=dict)
    columns: dict[str, JsonObject] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped_old: int = 0
    skipped_known: int = 0
    skipped_bank: int = 0
    paused: bool = False
    retry_at: float = 0

    @property
    def complete(self) -> bool:
        return (
            bool(self.columns)
            and all(row.get("complete") for row in self.columns.values())
            and not self.errors
        )


def publication_date(value: object) -> date | None:
    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})", str(value or ""))
    if not match:
        return None
    try:
        return date(*(int(x) for x in match.groups()))
    except ValueError:
        return None


def _payload(client: PoliteHTTPClient, static_url: str, api: str, params: JsonObject) -> JsonObject:
    try:
        return client.get_json(static_url)
    except CollectionHTTPError as error:
        if error.status not in {404, 410}:
            raise
        # Only a genuinely missing CDN object uses the documented dynamic API.
        return client.get_json(api + "?" + urlencode(params))


def _list(client: PoliteHTTPClient, column: int, page: int) -> tuple[list[JsonObject], int | None]:
    payload = _payload(
        client,
        LIST_URL.format(item_id=column, page=page),
        LIST_API,
        {"itemId": column, "pageSize": PAGE_SIZE, "pageIndex": page},
    )
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise CollectionHTTPError("官网列表结构变化，未推进采集水位")
    rows = data["rows"]
    if any(not isinstance(row, dict) or not str(row.get("docId", "")).isdigit() for row in rows):
        raise CollectionHTTPError("官网列表包含无效文书ID")
    total = data.get("total")
    return rows, int(str(total)) if str(total).isdigit() else None


def _signature(rows: list[JsonObject]) -> str:
    return sha256(
        json.dumps([(r.get("docId"), r.get("publishDate")) for r in rows], sort_keys=True).encode()
    ).hexdigest()


def _download(client: PoliteHTTPClient, root: Path, column: int, row: JsonObject) -> JsonObject:
    doc_id = int(row["docId"])
    payload = _payload(client, DETAIL_URL.format(doc_id=doc_id), DETAIL_API, {"docId": doc_id})
    data = payload.get("data")
    if not isinstance(data, dict) or not isinstance(data.get("docClob"), str):
        raise CollectionHTTPError(f"文书 {doc_id} 无可解析正文，保留待重试")
    clob = data["docClob"]
    text = html_to_text(clob)
    if not text.strip():
        raise CollectionHTTPError(f"文书 {doc_id} 正文为空，保留待重试")
    title = str(row.get("docSubtitle") or row.get("docTitle") or "")
    published = publication_date(row.get("publishDate"))
    if published is None:
        raise CollectionHTTPError("文书缺少发布日期")
    # A publication date is not used as a substitute for a missing penalty date.
    parts = extract_table_cases(clob, source_title=title) or [text]
    directory = root / "documents" / str(doc_id)
    directory.mkdir(parents=True, exist_ok=True)
    raw_path = directory / f"{sha256(clob.encode()).hexdigest()}.html"
    raw_path.write_text(clob, encoding="utf-8")
    entries = []
    for index, part in enumerate(parts, 1):
        path = directory / f"part_{index:03d}.txt"
        path.write_text(part, encoding="utf-8")
        entries.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": sha256(re.sub(r"\s+", "", part).encode()).hexdigest(),
                "status": "pending",
            }
        )
    document = {
        "doc_id": doc_id,
        "column": column,
        "column_name": COLUMNS[column],
        "title": title,
        "publish_date": published.isoformat(),
        "source_url": PAGE_URL.format(doc_id=doc_id, item_id=column),
        "document_no": data.get("documentNo"),
        "raw_path": str(raw_path.relative_to(root)),
        "parts": entries,
    }
    atomic_json(directory / "metadata.json", document)
    return document


def collect_window(
    client: PoliteHTTPClient,
    root: Path,
    *,
    since: date,
    until: date,
    columns: tuple[int, ...] = (4113, 4114, 4115),
    known_documents: dict[str, JsonObject] | None = None,
    pending_details: dict[str, JsonObject] | None = None,
    max_pages: int = 3,
    max_details: int = 12,
) -> CollectionBatch:
    if since > until or max_pages < 1 or max_details < 1:
        raise ValueError("采集日期或预算不合法")
    if not columns or any(column not in COLUMNS for column in columns):
        raise ValueError("未配置的官网栏目")
    result = CollectionBatch(columns={str(c): {"pages": 0, "complete": False} for c in columns})
    known = known_documents or {}
    result.pending_details = dict(pending_details or {})
    details = 0
    attempted: set[str] = set()

    def fetch_detail(column: int, row: JsonObject) -> bool:
        nonlocal details
        key = f"nfra:{row['docId']}"
        result.pending_details[key] = {"column": column, "row": row}
        atomic_json(root / "pending_details.json", result.pending_details)
        if details >= max_details or key in attempted:
            result.columns[str(column)]["detail_budget_reached"] = details >= max_details
            return False
        attempted.add(key)
        details += 1
        try:
            result.documents[key] = _download(client, root, column, row)
            del result.pending_details[key]
            atomic_json(root / "pending_details.json", result.pending_details)
            return True
        except SitePaused:
            raise
        except CollectionHTTPError as error:
            if not error.status or error.status >= 500:
                raise
            result.errors.append(f"文书 {row['docId']}: {error}")
            return False

    for column in columns:
        info = result.columns[str(column)]
        first_rows = None
        valid = True
        try:
            queued = list(result.pending_details.items())
            for key, task in queued:
                if int(task["column"]) != column:
                    continue
                if key in known:
                    result.pending_details.pop(key, None)
                elif not fetch_detail(column, task["row"]):
                    valid = False
            for page in range(1, max_pages + 1):
                rows, total = _list(client, column, page)
                info["pages"] += 1
                if first_rows is None:
                    first_rows = rows
                dated = [publication_date(row.get("publishDate")) for row in rows]
                if any(value is None for value in dated):
                    result.errors.append(f"栏目 {column} 第 {page} 页存在未知发布日期")
                    valid = False
                for row, published in zip(rows, dated):
                    if published is None:
                        continue
                    if published < since:
                        result.skipped_old += 1
                        continue
                    if published > until:
                        continue
                    key = f"nfra:{row['docId']}"
                    if key in known or key in result.documents:
                        result.skipped_known += 1
                        continue
                    title = str(row.get("docSubtitle") or row.get("docTitle") or "")
                    if "保险" not in title and any(
                        token in title for token in ("银行", "信用社", "信托公司", "财务公司")
                    ):
                        result.skipped_bank += 1
                        continue
                    if not fetch_detail(column, row):
                        valid = False
                old_boundary = bool(rows) and all(
                    value is not None and value < since for value in dated
                )
                exhausted = not rows or (total is not None and page * PAGE_SIZE >= total)
                if old_boundary or exhausted:
                    info["complete"] = valid
                    break
            if first_rows is not None and first_rows:
                check_rows, _ = _list(client, column, 1)
                if _signature(first_rows) != _signature(check_rows):
                    info["complete"] = False
                    result.errors.append(f"栏目 {column} 在采集期间更新，次日重叠补查")
        except CollectionHTTPError as error:
            result.errors.append(str(error))
            info["complete"] = False
            result.paused = isinstance(error, SitePaused)
            result.retry_at = error.retry_at
            break  # No alternative hosts/endpoints or fast retries on blocking/network failure.
    atomic_json(root / "pending_details.json", result.pending_details)
    return result
