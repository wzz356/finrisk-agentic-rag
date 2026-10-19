import hashlib
import json
from pathlib import Path

import pytest


class FakeStore:
    def __init__(self, root):
        self.root = root
        self.values = {"backfill_since": "2026-10-01"}
        self.raw = {}
        self.sources = []
        self.status = {}

    def get_value(self, key):
        return self.values.get(key)

    def set_value(self, key, value):
        self.values[key] = json.loads(json.dumps(value))

    def ingest(self, batch_id, raw, metadata, baseline_hashes=None):
        digest = hashlib.sha256(raw.decode("utf-8-sig").encode()).hexdigest()
        disposition = (
            "baseline_duplicate"
            if digest in (baseline_hashes or set())
            else "duplicate"
            if digest in self.raw
            else "new"
        )
        self.raw[digest] = raw
        self.sources.append((batch_id, raw, metadata))
        return {"digest": digest, "disposition": disposition}

    def set_batch_status(self, batch_id, status, details=None):
        self.status[batch_id] = (status, details)


def module():
    from agent_rag import collection_crawler

    return collection_crawler


def history_row(path, **kwargs):
    return {
        "source": "nfra",
        "column": 4113,
        "doc_id": 1,
        "part": 2,
        "publish_date": "2026-10-01",
        "source_hash": "legacy-hash",
        "source_url": "https://www.nfra.gov.cn/cn/view/pages/ItemDetail.html?docId=1",
        "txt_path": str(path),
        **kwargs,
    }


def write_index(root, rows):
    (root / "index.jsonl").write_text(
        "\n".join(json.dumps(r) if isinstance(r, dict) else r for r in rows) + "\n",
        encoding="utf-8",
    )


def test_history_relocated_path_preserves_bytes_metadata_and_bank(tmp_path):
    root = tmp_path / "history"
    leaf = root / "nfra" / "4113" / "old.txt"
    leaf.parent.mkdir(parents=True)
    raw = b"\xef\xbb\xbf" + "甲银行保险代理业务\r\n原件".encode()
    leaf.write_bytes(raw)
    write_index(root, [history_row(Path("Z:/moved/crawl/old.txt"))])
    store = FakeStore(tmp_path / "store")
    result = module().import_history(store, root, "history")
    assert result["complete"] is True and result["new"] == 1
    _, actual, metadata = store.sources[0]
    assert actual == raw
    assert all(
        metadata[key] == history_row(leaf)[key]
        for key in ("doc_id", "part", "publish_date", "source_hash", "source_url")
    )
    assert metadata["origin"] == "legacy_import"
    assert store.status["history"][0] == "collected"


def test_history_bad_lines_outside_paths_and_untrusted_urls_are_reported(tmp_path):
    root = tmp_path / "history"
    root.mkdir()
    inside = root / "good.txt"
    inside.write_text("合法原件", encoding="utf-8")
    outside = tmp_path / "secret.txt"
    outside.write_text("不可读", encoding="utf-8")
    write_index(
        root,
        [
            history_row(inside),
            "{bad",
            history_row(outside),
            history_row(inside, source_url="https://nfra.gov.cn.attacker.invalid/a"),
            history_row(root / "missing.txt"),
        ],
    )
    store = FakeStore(tmp_path / "store")
    result = module().import_history(store, root, "h")
    assert result["complete"] is False and result["documents"] == 1
    assert len(result["errors"]) == 4
    assert len(store.sources) == 1


def test_history_repeated_rows_and_baseline_are_counted(tmp_path):
    leaf = tmp_path / "same.txt"
    leaf.write_bytes("原件正文".encode())
    row = history_row(leaf)
    write_index(tmp_path, [row, row])
    store = FakeStore(tmp_path / "store")
    result = module().import_history(
        store, tmp_path, "h", baseline_hashes={hashlib.sha256(leaf.read_bytes()).hexdigest()}
    )
    assert result["documents"] == 2
    assert result["baseline_duplicates"] == 2 and result["duplicates"] == 0
    assert result["new"] == 0


def test_history_prefers_real_in_root_path_over_basename_fallback(tmp_path):
    direct = tmp_path / "registered.txt"
    direct.write_bytes("登记路径正文".encode())
    fallback = tmp_path / "nfra" / "4113" / direct.name
    fallback.parent.mkdir(parents=True)
    fallback.write_bytes("同名其他正文".encode())
    write_index(tmp_path, [history_row(direct)])
    store = FakeStore(tmp_path / "store")
    assert module().import_history(store, tmp_path, "h")["complete"] is True
    assert store.sources[0][1] == direct.read_bytes()


def test_missing_history_index_reports_io_failure_without_local_path(tmp_path):
    result = module().import_history(FakeStore(tmp_path / "store"), tmp_path, "missing")
    assert result["complete"] is False and result["errors"][0]["code"] == "HISTORY_INDEX_ERROR"
    assert tmp_path.name not in result["errors"][0].get("error", "")


def install_pages(monkeypatch, pages, details=None):
    crawler = module()
    calls = []

    def listing(column, page, delay):
        calls.append((column, page))
        value = pages[(column, page)]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(crawler.nfra, "_list_page", listing)

    def document(doc_id, delay):
        value = (details or {}).get(
            doc_id, (f"银行保险文书{doc_id}", {}, f"<p>原始HTML{doc_id}</p>")
        )
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(crawler.nfra, "_doc_text", document)
    return calls


def page(*rows, total=18):
    return {"data": {"rows": list(rows), "total": total}}


def row(doc_id, date="2026-10-02 10:01:00"):
    return {"docId": doc_id, "docTitle": "银行保险处罚", "publishDate": date}


def crawl(store, batch="c", **kwargs):
    kwargs.setdefault("max_pages", 4)
    return module().crawl_range(
        store, batch, since="2026-10-01", until="2026-10-03", columns=[4113], delay=0, **kwargs
    )


def test_crawl_filters_publication_dates_and_archives_entire_html(monkeypatch, tmp_path):
    install_pages(
        monkeypatch, {(4113, 1): page(row(1), row(2, "2026-09-30"), row(3, "2026-10-04"))}
    )
    store = FakeStore(tmp_path)
    result = crawl(store)
    assert result["complete"] is True and result["new"] == 1
    metadata = store.sources[0][2]
    assert metadata["doc_id"] == 1 and metadata["publish_date"] == "2026-10-02 10:01:00"
    assert metadata["source_url"].startswith("https://www.nfra.gov.cn/")
    # 内部保存绝对路径，由库存API统一移除，不能把相对服务器路径交给页面。
    assert Path(metadata["raw_html_path"]).is_absolute()
    html_files = list((tmp_path / "raw_html").rglob("*.html"))
    assert len(html_files) == 1 and html_files[0].read_text(encoding="utf-8") == "<p>原始HTML1</p>"
    assert store.values["crawl_success_until:4113"] == "2026-10-03"
    assert "crawl_success_until" not in store.values
    assert store.status["c"][0] == "collected"


def test_limit_is_per_column_and_resume_is_scoped_to_range(monkeypatch, tmp_path):
    calls = install_pages(
        monkeypatch,
        {
            (4113, 1): page(row(1), total=36),
            (4113, 2): page(row(2), total=36),
            (4114, 1): page(row(3), total=36),
            (4114, 2): page(row(4), total=36),
        },
    )
    store = FakeStore(tmp_path)
    first = module().crawl_range(
        store,
        "a",
        since="2026-10-01",
        until="2026-10-03",
        columns=[4113, 4114],
        max_pages=1,
        delay=0,
    )
    assert first["complete"] is False and first["next_pages"] == {"4113": 2, "4114": 2}
    assert "crawl_success_until" not in store.values
    assert store.status["a"][0] == "incomplete"
    second = module().crawl_range(
        store,
        "b",
        since="2026-10-01",
        until="2026-10-03",
        columns=[4113, 4114],
        max_pages=1,
        delay=0,
    )
    assert second["complete"] is False and second["new"] == 2
    assert second["next_pages"] == {"4113": 1, "4114": 1}
    assert second["phases"] == {"4113": "confirm", "4114": "confirm"}
    for batch in ("confirmation-a", "confirmation-b"):
        last = module().crawl_range(
            store,
            batch,
            since="2026-10-01",
            until="2026-10-03",
            columns=[4113, 4114],
            max_pages=1,
            delay=0,
        )
    assert last["complete"] is True
    assert calls == [
        (4113, 1),
        (4114, 1),
        (4113, 2),
        (4114, 2),
        (4113, 1),
        (4114, 1),
        (4113, 2),
        (4114, 2),
    ]
    module().crawl_range(
        store, "other", since="2026-10-02", until="2026-10-03", columns=[4113], max_pages=1, delay=0
    )
    assert calls[-1] == (4113, 1)


def test_failed_detail_keeps_page_and_retries_successful_rows_by_text_hash(monkeypatch, tmp_path):
    pages = {(4113, 1): page(row(1), row(2), total=36), (4113, 2): page(row(3), total=36)}
    details = {2: RuntimeError("detail failed")}
    calls = install_pages(monkeypatch, pages, details)
    store = FakeStore(tmp_path)
    first = crawl(store, max_pages=1)
    assert first["complete"] is False and first["next_pages"] == {"4113": 1}
    assert first["new"] == 1 and len(first["errors"]) == 1
    assert "crawl_success_until" not in store.values
    details.pop(2)
    second = crawl(store, "retry", max_pages=4)
    assert second["complete"] is True and second["new"] == 2 and second["duplicates"] == 1
    assert calls == [(4113, 1), (4113, 1), (4113, 2), (4113, 1), (4113, 2)]


def test_failed_list_page_and_unknown_date_do_not_advance(monkeypatch, tmp_path):
    pages = {(4113, 1): RuntimeError("network failed")}
    install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    result = crawl(store)
    assert result["complete"] is False and result["next_pages"] == {"4113": 1}
    pages[(4113, 1)] = page(row(1, ""), row(2))
    result = crawl(store)
    assert result["complete"] is False and result["next_pages"] == {"4113": 1}
    assert result["errors"][0]["code"] == "UNKNOWN_PUBLISH_DATE"
    assert result["new"] == 1


def test_repeat_range_checks_body_revisions_and_baseline(monkeypatch, tmp_path):
    details = {1: ("原始正文", {}, "<p>完整原件</p>")}
    calls = install_pages(monkeypatch, {(4113, 1): page(row(1))}, details)
    store = FakeStore(tmp_path)
    hashes = {hashlib.sha256("原始正文".encode()).hexdigest()}
    assert crawl(store, baseline_hashes=hashes)["baseline_duplicates"] == 1
    details[1] = ("修订正文", {}, "<p>修订完整原件</p>")
    assert crawl(store, "revision")["new"] == 1
    assert calls == [(4113, 1), (4113, 1), (4113, 1), (4113, 1)]
    assert len(list((tmp_path / "raw_html").rglob("*.html"))) == 2


def test_table_parts_keep_link_to_original_html(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(4113, 1): page(row(1))})
    monkeypatch.setattr(
        module().nfra, "extract_table_cases", lambda *args, **kw: ["甲银行保险原件", "乙保险原件"]
    )
    store = FakeStore(tmp_path)
    result = crawl(store)
    assert result["new"] == 2 and [s[2]["part"] for s in store.sources] == [1, 2]
    assert len({s[2]["raw_html_sha256"] for s in store.sources}) == 1


def test_three_columns_are_required_for_global_watermark_and_it_never_regresses(
    monkeypatch, tmp_path
):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    for until in ("2026-10-03", "2026-10-02"):
        result = module().crawl_range(store, until, since="2026-10-01", until=until, delay=0)
        assert result["complete"] is True and result["completed_at"]
    assert store.values["crawl_success_until"] == "2026-10-03"


def test_resume_skips_completed_columns_until_whole_range_completes(monkeypatch, tmp_path):
    pages = {(4113, 1): page(row(1)), (4114, 1): RuntimeError("offline")}
    calls = install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    arguments = dict(since="2026-10-01", until="2026-10-03", columns=[4113, 4114], delay=0)
    first = module().crawl_range(store, "a", **arguments)
    assert first["next_pages"] == {"4114": 1}
    pages[(4114, 1)] = page(row(2))
    second = module().crawl_range(store, "b", **arguments)
    assert second["complete"] is True and second["new"] == 1
    assert calls == [(4113, 1), (4113, 1), (4114, 1), (4114, 1), (4114, 1)]


def test_narrow_future_range_does_not_skip_backfill_gap(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    result = module().crawl_range(store, "future", since="2026-10-10", until="2026-10-18", delay=0)
    assert result["complete"] is True
    assert store.get_value("crawl_success_until") is None
    assert store.get_value("crawl_success_until:4113") is None
    assert store.get_value("crawl_completed_ranges") == [
        {"since": "2026-10-10", "until": "2026-10-18"}
    ]
    result = module().crawl_range(store, "bridge", since="2026-10-01", until="2026-10-10", delay=0)
    assert result["complete"] is True
    assert store.get_value("crawl_success_until") == "2026-10-18"
    assert store.get_value("crawl_success_until:4113") == "2026-10-18"


def test_existing_continuous_watermark_waits_for_missing_interval(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    store.set_value("crawl_success_until", "2026-10-03")
    module().crawl_range(store, "future", since="2026-10-10", until="2026-10-18", delay=0)
    assert store.get_value("crawl_success_until") == "2026-10-03"
    module().crawl_range(store, "bridge", since="2026-10-03", until="2026-10-10", delay=0)
    assert store.get_value("crawl_success_until") == "2026-10-18"


def test_no_backfill_anchor_means_no_inferred_continuous_watermark(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    store.values.pop("backfill_since")
    result = module().crawl_range(
        store, "unanchored", since="2026-10-01", until="2026-10-03", delay=0
    )
    assert result["complete"] is True
    assert store.get_value("crawl_success_until") is None
    assert store.get_value("crawl_completed_ranges")


def test_adjacent_closed_ranges_merge_without_requiring_overlapping_day(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    module().crawl_range(store, "head", since="2026-10-01", until="2026-10-03", delay=0)
    module().crawl_range(store, "tail", since="2026-10-04", until="2026-10-18", delay=0)
    assert store.get_value("crawl_success_until") == "2026-10-18"
    assert store.get_value("crawl_completed_ranges") == [
        {"since": "2026-10-01", "until": "2026-10-18"}
    ]


def test_even_one_missing_day_prevents_continuous_watermark(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    module().crawl_range(store, "tail", since="2026-10-02", until="2026-10-18", delay=0)
    assert store.get_value("crawl_success_until") is None  # 首锚10月1日没有被覆盖。
    module().crawl_range(store, "anchor", since="2026-10-01", until="2026-10-01", delay=0)
    assert store.get_value("crawl_success_until") == "2026-10-18"
    module().crawl_range(store, "future", since="2026-10-20", until="2026-10-22", delay=0)
    assert store.get_value("crawl_success_until") == "2026-10-18"  # 仍缺19日。


def test_upper_date_boundary_does_not_overflow_during_coverage_merge(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(column, 1): page(row(column)) for column in (4113, 4114, 4115)})
    store = FakeStore(tmp_path)
    store.set_value("backfill_since", "9999-12-31")
    for batch in ("first", "repeat"):
        result = module().crawl_range(store, batch, since="9999-12-31", until="9999-12-31", delay=0)
        assert result["complete"] is True
    assert store.get_value("crawl_success_until") == "9999-12-31"


def test_page_deletion_requires_confirmation_and_recovers_skipped_boundary(monkeypatch, tmp_path):
    # 首扫page1为1..18，随后源删1；若直接续page2=20..36会永久漏19。
    pages = {
        (4113, 1): page(*(row(i) for i in range(1, 19)), total=36),
        (4113, 2): page(*(row(i) for i in range(20, 37)), total=35),
    }
    calls = install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    first = crawl(store, max_pages=1)
    assert first["complete"] is False
    pages[(4113, 1)] = page(*(row(i) for i in range(2, 20)), total=35)
    second = crawl(store, "tail", max_pages=1)
    assert second["complete"] is False and second["next_pages"] == {"4113": 1}
    assert store.get_value("crawl_success_until:4113") is None
    changed = crawl(store, "verify-change", max_pages=1)
    assert changed["complete"] is False and changed["next_pages"] == {"4113": 1}
    assert changed["restarted_columns"] == {"4113": "LIST_CHANGED"}
    assert calls == [(4113, 1), (4113, 2), (4113, 1)]
    final = crawl(store, "stable", max_pages=4)
    assert final["complete"] is True
    assert any(source[2]["doc_id"] == 19 for source in store.sources)
    assert store.get_value("crawl_success_until:4113") == "2026-10-03"


def test_new_head_document_is_discovered_before_completion(monkeypatch, tmp_path):
    pages = {(4113, 1): page(row(1), total=18)}
    install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    assert crawl(store, max_pages=1)["complete"] is False
    pages[(4113, 1)] = page(row(99), row(1), total=18)
    assert crawl(store, "changed", max_pages=1)["complete"] is False
    result = crawl(store, "rescan", max_pages=2)
    assert result["complete"] is True
    assert any(source[2]["doc_id"] == 99 for source in store.sources)


def test_confirmation_changed_row_shape_restarts_instead_of_crashing(monkeypatch, tmp_path):
    pages = {(4113, 1): page(row(1))}
    install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    assert crawl(store, max_pages=1)["complete"] is False
    pages[(4113, 1)] = page(None)
    result = crawl(store, "invalid-confirmation", max_pages=1)
    assert result["complete"] is False and result["next_pages"] == {"4113": 1}


def test_small_budget_progress_survives_reopened_state_until_stable_full_scan(
    monkeypatch, tmp_path
):
    pages = {(4113, index): page(row(index), total=54) for index in (1, 2, 3)}
    calls = install_pages(monkeypatch, pages)
    store = FakeStore(tmp_path)
    for index in range(6):
        result = crawl(store, str(index), max_pages=1)
        assert result["complete"] is (index == 5)
        assert result["pages"] == {"4113": 1}
        # 模拟跨进程/跨日从JSON状态恢复，不依赖内存标志或每天从头重扫。
        reopened = FakeStore(tmp_path)
        reopened.values = json.loads(json.dumps(store.values))
        reopened.raw = store.raw
        reopened.sources = store.sources
        store = reopened
    assert calls == [(4113, 1), (4113, 2), (4113, 3), (4113, 1), (4113, 2), (4113, 3)]
    assert store.get_value("crawl_success_until:4113") == "2026-10-03"


def test_legacy_page_number_without_confirmation_evidence_restarts_safely(monkeypatch, tmp_path):
    calls = install_pages(monkeypatch, {(4113, 1): page(row(1))})
    store = FakeStore(tmp_path)
    store.set_value("crawl_cursor:2026-10-01:2026-10-03:4113", {"page": 8, "complete": False})
    assert crawl(store)["complete"] is True
    assert calls == [(4113, 1), (4113, 1)]


@pytest.mark.parametrize(
    "payload",
    [{"data": {}}, {"data": {"rows": [], "total": 18}}, {"data": {"rows": [], "total": -1}}],
)
def test_malformed_and_unexpected_empty_pages_remain_pending(monkeypatch, tmp_path, payload):
    install_pages(monkeypatch, {(4113, 1): payload})
    result = crawl(FakeStore(tmp_path))
    assert result["complete"] is False and result["next_pages"] == {"4113": 1}
    assert result["errors"][0]["code"] == "LIST_PAGE_ERROR"


def test_fully_seen_page_does_not_hide_later_new_document(monkeypatch, tmp_path):
    install_pages(
        monkeypatch, {(4113, 1): page(row(1), total=36), (4113, 2): page(row(2), total=36)}
    )
    store = FakeStore(tmp_path)
    store.ingest("prior", "银行保险文书1".encode(), {})
    result = crawl(store)
    assert result["complete"] is True and result["duplicates"] == 1 and result["new"] == 1


def test_unknown_order_does_not_stop_on_entirely_old_page(monkeypatch, tmp_path):
    install_pages(
        monkeypatch,
        {(4113, 1): page(row(1, "2026-09-01"), total=36), (4113, 2): page(row(2), total=36)},
    )
    result = crawl(FakeStore(tmp_path))
    assert result["complete"] is True and result["new"] == 1


def test_empty_text_is_pending_but_received_html_is_preserved(monkeypatch, tmp_path):
    install_pages(monkeypatch, {(4113, 1): page(row(1))}, {1: ("", {}, "<p>原始异常HTML</p>")})
    store = FakeStore(tmp_path)
    result = crawl(store)
    assert result["complete"] is False and result["documents"] == 0
    assert result["errors"][0]["code"] == "DOCUMENT_ERROR"
    assert len(list((tmp_path / "raw_html").glob("*.html"))) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"since": "2026-1-01"},
        {"since": "2026-02-30"},
        {"until": "2026-09-30"},
        {"max_pages": 0},
        {"max_pages": True},
        {"columns": [999]},
        {"columns": []},
        {"delay": -1},
    ],
)
def test_invalid_ranges_make_no_network_request(monkeypatch, tmp_path, kwargs):
    monkeypatch.setattr(module().nfra, "_list_page", lambda *a: pytest.fail("unexpected network"))
    arguments = {"since": "2026-10-01", "until": "2026-10-03", "columns": [4113], **kwargs}
    with pytest.raises(ValueError):
        module().crawl_range(FakeStore(tmp_path), "c", **arguments)
