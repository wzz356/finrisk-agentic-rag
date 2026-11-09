"""金监总局行政处罚栏目采集。

实测要点：
- 列表/详情优先走 CDN 静态 JSON（pageSize 固定 18，其他值 404）；
- **CDN 只预生成前 3 页**，第 4 页起 404，需回退动态接口
  （GET /cbircweb/DocInfo/SelectByDocIdAndChild?itemId=..&pageSize=18&pageIndex=N）；
- 总局机关（4113）为批量"公示列表"（一案一行摘要），4114/4115 多为单案"信息公开表"；
- 增量模式：列表按发布日期倒序，遇到一整页全是已见 docId 即停。
"""

from pathlib import Path
from typing import Any, TypedDict

from .common import (
    FetchError,
    StateStore,
    append_jsonl,
    html_to_text,
    load_json,
    load_json_get,
    sha1_text,
    write_text,
)
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

COLUMNS = {
    4113: "总局机关",
    4114: "省监管局",
    4115: "监管分局",
}
PAGE_SIZE = 18


class ColumnStats(TypedDict):
    column: int
    pages: int
    new: int
    skipped: int
    empty: int
    errors: int


def _list_page(item_id: int, page: int, delay: float) -> dict[str, Any]:
    """列表：先试 CDN 静态 JSON，404/失败回退动态接口。"""
    try:
        return load_json(LIST_URL.format(item_id=item_id, page=page), delay=delay)
    except (FetchError, ValueError):
        return load_json_get(
            LIST_API,
            {"itemId": item_id, "pageSize": PAGE_SIZE, "pageIndex": page},
            delay=delay,
        )


def _doc_text(doc_id: int, delay: float) -> tuple[str, dict[str, str], str]:
    """取详情 JSON（CDN 失败回退动态接口），返回 (纯文本正文, 详情元数据, docClob 原文)。"""
    try:
        payload = load_json(DETAIL_URL.format(doc_id=doc_id), delay=delay)
    except (FetchError, ValueError):
        payload = load_json_get(DETAIL_API, {"docId": doc_id}, delay=delay)
    data = payload.get("data") or {}
    clob = data.get("docClob") or ""
    meta = {
        "document_no": data.get("documentNo") or "",
        "doc_source": data.get("docSource") or "",
        "pdf_url": data.get("pdfFileUrl") or "",
    }
    if not clob.strip():
        return "", meta, ""
    return html_to_text(clob), meta, clob


def crawl_column(
    item_id: int,
    out_dir: Path,
    state: StateStore,
    *,
    full: bool = False,
    max_pages: int = 0,
    delay: float = 1.5,
) -> ColumnStats:
    """抓取单个栏目。返回统计；失败记录写 errors.jsonl。"""
    text_dir = out_dir / "nfra" / str(item_id)
    index_path = out_dir / "index.jsonl"
    errors_path = out_dir / "errors.jsonl"
    stats: ColumnStats = {
        "column": item_id,
        "pages": 0,
        "new": 0,
        "skipped": 0,
        "empty": 0,
        "errors": 0,
    }

    page = 1
    total_pages: int | None = None
    while True:
        if max_pages and page > max_pages:
            break
        if total_pages is not None and page > total_pages:
            break
        try:
            payload = _list_page(item_id, page, delay)
        except (FetchError, ValueError) as exc:
            append_jsonl(
                errors_path,
                {
                    "source": "nfra",
                    "column": item_id,
                    "page": page,
                    "error": str(exc),
                },
            )
            stats["errors"] += 1
            break
        data = payload.get("data") or {}
        rows = data.get("rows") or []
        if not rows:
            break
        if total_pages is None:
            total = int(data.get("total") or 0)
            total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
        stats["pages"] += 1

        page_all_seen = True
        for row in rows:
            doc_id = row.get("docId")
            if not doc_id:
                continue
            key = f"nfra:{item_id}:{doc_id}"
            if key in state:
                stats["skipped"] += 1
                continue
            page_all_seen = False
            try:
                text, meta, clob = _doc_text(doc_id, delay)
            except (FetchError, ValueError) as exc:
                append_jsonl(errors_path, {"source": "nfra", "doc_id": doc_id, "error": str(exc)})
                stats["errors"] += 1
                continue
            if not text:
                stats["empty"] += 1
                state.add(key)
                continue
            title = row.get("docSubtitle") or row.get("docTitle") or ""
            publish_date = row.get("publishDate") or ""
            # 汇总公示表拆行：一案一条 txt；拆不出则保留整篇
            cases = (
                extract_table_cases(clob, source_title=title, publish_date=publish_date[:10])
                if clob
                else []
            )
            parts = cases if cases else [text]
            for part_index, part in enumerate(parts, 1):
                txt_name = f"{doc_id}.txt" if len(parts) == 1 else f"{doc_id}_r{part_index}.txt"
                write_text(text_dir / txt_name, part)
                append_jsonl(
                    index_path,
                    {
                        "source": "nfra",
                        "column": item_id,
                        "column_name": COLUMNS.get(item_id, str(item_id)),
                        "doc_id": doc_id,
                        "part": part_index if len(parts) > 1 else None,
                        "split_from_list": bool(cases),
                        "title": title,
                        "publish_date": publish_date,
                        "document_no": meta["document_no"],
                        "source_url": PAGE_URL.format(doc_id=doc_id, item_id=item_id),
                        "pdf_url": meta["pdf_url"],
                        "txt_path": str(text_dir / txt_name),
                        "sha1": sha1_text(part),
                    },
                )
                stats["new"] += 1
            state.add(key)
        state.save()

        # 增量模式：整页都是已见文书 → 之后全是旧数据，收工
        if not full and page_all_seen:
            break
        page += 1
    return stats


def crawl_all(
    out_dir: Path,
    *,
    full: bool = False,
    max_pages: int = 0,
    delay: float = 1.5,
    columns: list[int] | None = None,
) -> list[ColumnStats]:
    state = StateStore(out_dir / "state.json")
    results: list[ColumnStats] = []
    for item_id in columns or list(COLUMNS):
        results.append(
            crawl_column(item_id, out_dir, state, full=full, max_pages=max_pages, delay=delay)
        )
    return results
