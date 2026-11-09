"""只采集到隔离库存；不执行旧脚本、不调用模型、不发布知识版本。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

from offline.penalty_crawler import nfra


def _date(value: str) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("INVALID_DATE")
    return date.fromisoformat(value)


def _official_url(value) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname in {"nfra.gov.cn", "www.nfra.gov.cn"}
        and parsed.username is None
        and parsed.password is None
        and parsed.netloc.lower() in {"nfra.gov.cn", "www.nfra.gov.cn"}
    )


def _result(**fields) -> dict:
    return {
        "complete": False,
        "documents": 0,
        "new": 0,
        "duplicates": 0,
        "baseline_duplicates": 0,
        "errors": [],
        "next_pages": {},
        **fields,
    }


def _count(result: dict, ingested: dict) -> None:
    key = {"new": "new", "duplicate": "duplicates", "baseline_duplicate": "baseline_duplicates"}[
        ingested["disposition"]
    ]
    result["documents"] += 1
    result[key] += 1


def _error(result: dict, code: str, exc=None, **context) -> None:
    # I/O异常的filename可能包含服务器本地目录，不进入管理API状态。
    message = (
        f"{type(exc).__name__}: {exc.strerror or 'I/O error'}"
        if isinstance(exc, OSError) and exc.filename
        else str(exc)
    )
    result["errors"].append(
        {"code": code, **context, **({"error": message} if exc is not None else {})}
    )


def _finish(store, batch_id: str, result: dict) -> dict:
    result["completed_at"] = datetime.now(UTC).isoformat() if result["complete"] else None
    store.set_batch_status(batch_id, "collected" if result["complete"] else "incomplete", result)
    return result


def _history_path(root: Path, row: dict) -> Path:
    value = row.get("txt_path")
    if not isinstance(value, str) or not value:
        raise ValueError("MISSING_SOURCE_PATH")
    original = Path(value)
    candidate = (original if original.is_absolute() else root / original).resolve()
    if candidate.is_relative_to(root) and candidate.is_file():
        return candidate
    # 旧绝对路径搬家时只按已登记栏目和basename定位，不遍历其他目录。
    column = row.get("column")
    if str(column) in {str(c) for c in nfra.COLUMNS}:
        fallback = (root / "nfra" / str(column) / original.name).resolve()
        if fallback.is_relative_to(root) and fallback.is_file():
            return fallback
    raise ValueError(
        "SOURCE_PATH_OUTSIDE_ROOT" if not candidate.is_relative_to(root) else "SOURCE_FILE_MISSING"
    )


def import_history(
    store, source_root: Path, batch_id: str, *, baseline_hashes: set[str] | None = None
) -> dict:
    """逐行复制已登记的NFRA原件，坏行记错后继续；不为缺失原件生成正文。"""
    root = Path(source_root).resolve()
    result = _result(kind="history", since=None, until=None, columns=[])
    store.set_batch_status(batch_id, "collecting")
    index = (root / "index.jsonl").resolve()
    try:
        if not index.is_relative_to(root):
            raise ValueError("INDEX_PATH_OUTSIDE_ROOT")
        with index.open("rb") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if not raw_line.strip():
                    continue
                try:
                    row = json.loads(raw_line.decode("utf-8-sig"))
                    if not isinstance(row, dict):
                        raise ValueError("INVALID_INDEX_ROW")
                    if not _official_url(row.get("source_url")):
                        raise ValueError("UNTRUSTED_SOURCE_URL")
                    path = _history_path(root, row)
                    raw = path.read_bytes()
                    metadata = {
                        **row,
                        "origin": "legacy_import",
                        "txt_path": str(path),
                        "original_filename": path.name,
                        "import_origin": "history",
                        "source_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                    _count(
                        result,
                        store.ingest(batch_id, raw, metadata, baseline_hashes=baseline_hashes),
                    )
                except (ValueError, OSError, KeyError, TypeError) as exc:
                    _error(result, "HISTORY_ROW_ERROR", exc, line=line_number)
    except (ValueError, OSError) as exc:
        _error(result, "HISTORY_INDEX_ERROR", exc)
    result["complete"] = not result["errors"]
    return _finish(store, batch_id, result)


def _rows(payload: dict, page: int) -> tuple[list, int | None]:
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("rows"), list):
        raise ValueError("INVALID_LIST_RESPONSE")
    rows = data["rows"]
    total = data.get("total")
    if total is None:
        total_pages = None
    else:
        if isinstance(total, bool) or not str(total).isdigit():
            raise ValueError("INVALID_LIST_TOTAL")
        total_pages = max(1, (int(total) + nfra.PAGE_SIZE - 1) // nfra.PAGE_SIZE)
        if not rows and int(total) > (page - 1) * nfra.PAGE_SIZE:
            raise ValueError("UNEXPECTED_EMPTY_PAGE")
    return rows, total_pages


def _archive_html(store, clob: str) -> tuple[str, str]:
    raw = clob.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    directory = (store.root / "raw_html").resolve()
    if not directory.is_relative_to(store.root.resolve()):
        raise ValueError("RAW_HTML_PATH_OUTSIDE_STORE")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".html")
    try:
        with path.open("xb") as stream:
            stream.write(raw)
    except FileExistsError:
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(directory)
            or path.read_bytes() != raw
        ):
            raise ValueError("RAW_HTML_INTEGRITY_ERROR") from None
    return digest, str(path)


def _record_coverage(store, since: str, until: str, suffix: str = "") -> None:
    """已完成窄区间可先保存，但不能越过回补锚点或连续水位前的缺口。"""
    ranges_key, watermark_key = "crawl_completed_ranges" + suffix, "crawl_success_until" + suffix
    ranges = [*(store.get_value(ranges_key) or []), {"since": since, "until": until}]
    merged = []
    for item in sorted(ranges, key=lambda item: (item["since"], item["until"])):
        if (
            merged
            and _date(item["since"]).toordinal() <= _date(merged[-1]["until"]).toordinal() + 1
        ):
            merged[-1]["until"] = max(merged[-1]["until"], item["until"])
        else:
            merged.append(dict(item))
    store.set_value(ranges_key, merged)
    previous = store.get_value(watermark_key)
    cursor = previous or store.get_value("backfill_since")
    if not cursor:
        return
    covered = bool(previous)
    for item in merged:
        # 有连续水位时次日相接即连续；尚无水位时首个区间必须实际覆盖锚点。
        frontier = _date(cursor).toordinal() + int(covered)
        if _date(item["since"]).toordinal() <= frontier and item["until"] >= cursor:
            cursor, covered = item["until"], True
    if covered:
        store.set_value(watermark_key, cursor)


def _new_scan() -> dict:
    return {"schema": 2, "page": 1, "phase": "scan", "complete": False, "signatures": {}}


def _page_signature(payload: dict, rows: list) -> str:
    # 排序身份/日期/标题改变及总数改变都要求重扫，忽略阅读量等易变展示字段。
    identity = [
        [row.get(key) for key in ("docId", "publishDate", "docTitle", "docSubtitle")]
        if isinstance(row, dict)
        else {"invalid_row": row}
        for row in rows
    ]
    total = payload["data"].get("total")
    raw = json.dumps(
        {"rows": identity, "total": int(total) if total is not None else None},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def crawl_range(
    store,
    batch_id: str,
    *,
    since: str,
    until: str,
    max_pages: int = 2,
    columns: list[int] | None = None,
    delay: float = 1.5,
    baseline_hashes: set[str] | None = None,
) -> dict:
    """按发布日期闭区间有界采集；失败页不推进，重试靠全文SHA去重。"""
    start, end = _date(since), _date(until)
    selected = list(nfra.COLUMNS) if columns is None else columns
    if (
        start > end
        or type(max_pages) is not int
        or not 1 <= max_pages <= 100
        or not isinstance(selected, list)
        or not selected
        or any(type(c) is not int or c not in nfra.COLUMNS for c in selected)
        or len(set(selected)) != len(selected)
        or isinstance(delay, bool)
        or not isinstance(delay, (int, float))
        or not math.isfinite(delay)
        or delay < 0
    ):
        raise ValueError("INVALID_CRAWL_RANGE_OR_LIMIT")
    result = _result(
        kind="crawl",
        since=since,
        until=until,
        columns=list(selected),
        pages={},
        phases={},
        restarted_columns={},
    )
    store.set_batch_status(batch_id, "collecting")
    keys = {c: f"crawl_cursor:{since}:{until}:{c}" for c in selected}
    states = {}
    for column in selected:
        saved = store.get_value(keys[column])
        # 旧页号未保存整轮列表签名，不能用它宣布已确认完成。
        states[column] = (
            saved if isinstance(saved, dict) and saved.get("schema") == 2 else _new_scan()
        )
    # 一轮全部完成后，再请求同一范围会从头核验，保留相同docId的修订正文。
    if all(state.get("complete") for state in states.values()):
        states = {c: _new_scan() for c in selected}
    for column in selected:
        store.set_value(keys[column], states[column])
    for column in selected:
        state = states[column]
        page_number = state["page"]
        result["pages"][str(column)] = 0
        if state.get("complete"):
            result["phases"][str(column)] = "complete"
            continue
        for _ in range(max_pages):
            try:
                payload = nfra._list_page(column, page_number, delay)
                rows, total_pages = _rows(payload, page_number)
            except Exception as exc:
                _error(result, "LIST_PAGE_ERROR", exc, column=column, page=page_number)
                break
            result["pages"][str(column)] += 1
            finished = not rows or (total_pages is not None and page_number >= total_pages)
            if state["phase"] == "confirm":
                if state["signatures"].get(str(page_number)) != _page_signature(payload, rows):
                    # 本轮预算内不追逐页漂移；下次从第一页重新采正文并生成候选序列。
                    state = _new_scan()
                    states[column], page_number = state, 1
                    result["restarted_columns"][str(column)] = "LIST_CHANGED"
                    store.set_value(keys[column], state)
                    break
                page_number += 1
                state.update(page=page_number, complete=finished)
                store.set_value(keys[column], state)
                if finished:
                    break
                continue
            page_errors = len(result["errors"])
            for row in rows:
                context = {"column": column, "page": page_number}
                if not isinstance(row, dict):
                    _error(result, "INVALID_LIST_ROW", **context)
                    continue
                doc_id = row.get("docId")
                if isinstance(doc_id, bool) or not str(doc_id).isdigit() or int(doc_id) < 1:
                    _error(result, "INVALID_DOC_ID", **context)
                    continue
                context["doc_id"] = doc_id
                published = row.get("publishDate")
                try:
                    publish_day = _date(published[:10] if isinstance(published, str) else None)
                except ValueError:
                    _error(
                        result,
                        "UNKNOWN_PUBLISH_DATE",
                        publish_date=published,
                        source_url=nfra.PAGE_URL.format(doc_id=doc_id, item_id=column),
                        **context,
                    )
                    continue
                if not start <= publish_day <= end:
                    continue
                try:
                    text, meta, clob = nfra._doc_text(int(doc_id), delay)
                    if not isinstance(clob, str) or not clob.strip():
                        raise ValueError("EMPTY_DOCUMENT_HTML")
                    html_digest, html_path = _archive_html(store, clob)
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError("EMPTY_DOCUMENT_TEXT")
                    title = row.get("docSubtitle") or row.get("docTitle") or ""
                    cases = nfra.extract_table_cases(
                        clob, source_title=title, publish_date=publish_day.isoformat()
                    )
                    parts = cases or [text]
                    for part_number, part in enumerate(parts, 1):
                        metadata = {
                            **meta,
                            "origin": "official_crawl",
                            "source": "nfra",
                            "column": column,
                            "column_name": nfra.COLUMNS[column],
                            "doc_id": doc_id,
                            "part": part_number if len(parts) > 1 else None,
                            "split_from_list": bool(cases),
                            "title": title,
                            "publish_date": published,
                            "date_status": "verified",
                            "source_url": nfra.PAGE_URL.format(doc_id=doc_id, item_id=column),
                            "source_hash": html_digest,
                            "raw_html_sha256": html_digest,
                            "raw_html_path": html_path,
                            "scope_review_status": "pending",
                        }
                        _count(
                            result,
                            store.ingest(
                                batch_id,
                                part.encode("utf-8"),
                                metadata,
                                baseline_hashes=baseline_hashes,
                            ),
                        )
                except Exception as exc:
                    _error(result, "DOCUMENT_ERROR", exc, **context)
            if len(result["errors"]) != page_errors:
                break
            state["signatures"][str(page_number)] = _page_signature(payload, rows)
            # 首遍只构成候选快照；完整第二遍列表序列相同才判完成。
            state.update(
                page=1 if finished else page_number + 1, phase="confirm" if finished else "scan"
            )
            page_number = state["page"]
            store.set_value(keys[column], state)
        result["phases"][str(column)] = "complete" if state.get("complete") else state["phase"]
        if not states[column].get("complete"):
            result["next_pages"][str(column)] = page_number
    result["complete"] = not result["errors"] and all(s.get("complete") for s in states.values())
    if result["complete"]:
        for column in selected:
            _record_coverage(store, since, until, f":{column}")
        if set(selected) == set(nfra.COLUMNS):
            _record_coverage(store, since, until)
    return _finish(store, batch_id, result)
