"""Deterministic collection tests. No official website or model is called here."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from types import SimpleNamespace

import pytest

from application.collection_service import SHANGHAI, CollectionService
from offline.penalty_crawler.nfra import DETAIL_API, DETAIL_URL, collect_window
from offline.penalty_crawler.polite_http import (
    CollectionHTTPError,
    PoliteHTTPClient,
    RequestBudgetExceeded,
    Response,
    SitePaused,
    validate_url,
)
from settings import AppSettings


class FakeClock:
    def __init__(self):
        self.value = 1_792_300_000.0
        self.sleeps = []

    def now(self):
        return self.value

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.value += duration


def make_http(tmp_path, responses, *, budget=12):
    clock, calls = FakeClock(), []

    def transport(url, headers, timeout):
        calls.append((url, headers, clock.now()))
        if url.endswith("/robots.txt"):
            return Response(404, b"", {}, url)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status, body, response_headers = response
        return Response(status, body.encode(), response_headers, url)

    client = PoliteHTTPClient(
        tmp_path,
        transport=transport,
        now=clock.now,
        sleep=clock.sleep,
        jitter=0,
        max_requests=budget,
    )
    return client, clock, calls, transport


def test_conditional_cache_and_serial_delay(tmp_path):
    client, clock, calls, _ = make_http(
        tmp_path,
        [
            (
                200,
                '{"data": 1}',
                {"ETag": '"v1"', "Last-Modified": "Mon, 19 Oct 2026 00:00:00 GMT"},
            ),
            (304, "", {}),
        ],
    )
    url = "https://www.nfra.gov.cn/list.json"
    assert client.get_json(url) == client.get_json(url) == {"data": 1}
    assert client.cache_hits == 1
    assert all(b[2] - a[2] >= 3 for a, b in zip(calls, calls[1:]))
    assert calls[-1][1]["If-None-Match"] == '"v1"'
    assert calls[-1][1]["If-Modified-Since"]
    assert "FinRiskCollector" in calls[-1][1]["User-Agent"]


@pytest.mark.parametrize(
    "status,headers", [(403, {}), (429, {"Retry-After": "3600"}), (503, {"Retry-After": "3600"})]
)
def test_block_stops_without_retries_and_survives_restart(tmp_path, status, headers):
    client, clock, calls, transport = make_http(tmp_path, [(status, "blocked", headers)])
    with pytest.raises(SitePaused) as captured:
        client.get_json("https://www.nfra.gov.cn/list.json")
    assert captured.value.retry_at > clock.now()
    restarted = PoliteHTTPClient(tmp_path, transport=transport, now=clock.now, sleep=clock.sleep)
    with pytest.raises(SitePaused):
        restarted.get_json("https://www.nfra.gov.cn/another.json")
    assert len(calls) == 2  # robots + one list, no alternate URL or retry.


def test_captcha_stops_and_request_budget_includes_robots(tmp_path):
    client, _, _, _ = make_http(tmp_path, [(200, "<html>安全验证 captcha</html>", {})])
    with pytest.raises(SitePaused):
        client.get_json("https://www.nfra.gov.cn/list.json")
    client2, _, calls, _ = make_http(tmp_path / "second", [(200, "{}", {})], budget=2)
    client2.get_json("https://www.nfra.gov.cn/list.json")
    with pytest.raises(RequestBudgetExceeded):
        client2.get_json("https://www.nfra.gov.cn/list2.json")
    assert len(calls) == 2


def test_robots_disallow_prevents_list_request(tmp_path):
    calls = []

    def transport(url, headers, timeout):
        calls.append(url)
        return Response(200, b"User-agent: *\nDisallow: /\n", {}, url)

    client = PoliteHTTPClient(tmp_path, transport=transport)
    with pytest.raises(SitePaused, match="robots"):
        client.get_json("https://www.nfra.gov.cn/list.json")
    assert len(calls) == 1


@pytest.mark.parametrize(
    "url",
    [
        "http://www.nfra.gov.cn/x",
        "https://www.nfra.gov.cn.evil.test/x",
        "https://user@www.nfra.gov.cn/x",
        "https://www.nfra.gov.cn:8443/x",
    ],
)
def test_only_registered_official_https_host_is_allowed(url):
    with pytest.raises(CollectionHTTPError):
        validate_url(url)


def row(doc_id=1, published="2026-10-19", title="甲保险公司行政处罚"):
    return {"docId": doc_id, "publishDate": published, "docTitle": title}


class FakeNFRA:
    def __init__(self, pages=None, *, detail_error=None, drift=False):
        self.pages = pages or {1: [row()]}
        self.detail_error, self.drift = detail_error, drift
        self.urls = []
        self.requests = self.cache_hits = 0

    def get_json(self, url):
        self.urls.append(url)
        self.requests += 1
        if "SelectDocByItemIdAndChild" in url:
            page = int(url.split("pageIndex=")[1].split(",")[0])
            rows = self.pages.get(page, [])
            if self.drift and self.urls.count(url) > 1:
                rows = [row(99)]
            return {"data": {"rows": rows, "total": max(self.pages) * 18}}
        if self.detail_error:
            raise self.detail_error
        return {
            "data": {
                "docClob": "<p>当事人：甲保险公司</p>"
                "<p>主要违法违规事实：欺骗投保人。</p><p>行政处罚内容：罚款。</p>"
            }
        }


def collect(tmp_path, client, **kwargs):
    return collect_window(
        client,
        tmp_path,
        since=date(2026, 10, 19),
        until=date(2026, 10, 19),
        columns=(4114,),
        max_pages=2,
        **kwargs,
    )


def test_date_boundary_avoids_all_old_details(tmp_path):
    client = FakeNFRA({1: [row(1, "2026-10-18"), row(2, "2025-01-01")]})
    batch = collect(tmp_path, client)
    assert batch.complete and batch.skipped_old == 2
    assert not batch.documents
    assert all("SelectByDocId" not in url for url in client.urls)
    assert len(client.urls) == 2


def test_mixed_pinned_dates_and_known_docs_do_not_stop_on_one_old_row(tmp_path):
    client = FakeNFRA({1: [row(1, "2020-01-01"), row(2)], 2: [row(3, "2026-10-18")]})
    batch = collect(tmp_path, client)
    assert batch.complete and set(batch.documents) == {"nfra:2"}
    assert (tmp_path / "documents/2/metadata.json").is_file()
    assert not (tmp_path / "documents/1").exists()
    second = FakeNFRA({1: [row(2)]})
    again = collect(tmp_path, second, known_documents=batch.documents)
    assert again.skipped_known == 1 and not again.documents
    assert all("SelectByDocId" not in url for url in second.urls)


def test_unknown_date_or_listing_drift_does_not_advance_watermark(tmp_path):
    unknown = collect(tmp_path / "unknown", FakeNFRA({1: [row(1, "unknown")]}))
    assert not unknown.complete and unknown.errors and not unknown.documents
    drift = collect(tmp_path / "drift", FakeNFRA(drift=True))
    assert not drift.complete and any("采集期间更新" in error for error in drift.errors)


def test_detail_budget_leaves_window_incomplete(tmp_path):
    batch = collect(tmp_path, FakeNFRA({1: [row(1), row(2)]}), max_details=1)
    assert len(batch.documents) == 1 and not batch.complete
    assert batch.columns["4114"]["detail_budget_reached"]
    assert "nfra:2" in batch.pending_details
    following = collect(
        tmp_path,
        FakeNFRA({1: [row(99, "2026-10-18")]}),
        known_documents=batch.documents,
        pending_details=batch.pending_details,
    )
    assert "nfra:2" in following.documents and not following.pending_details


def test_http_403_never_uses_dynamic_detail_fallback(tmp_path):
    client = FakeNFRA(detail_error=SitePaused("blocked", status=403, retry_at=100))
    batch = collect(tmp_path, client)
    assert batch.paused and not batch.complete
    assert not any(url.startswith(DETAIL_API) for url in client.urls)


def test_only_missing_cdn_object_uses_dynamic_fallback(tmp_path):
    class MissingCDN(FakeNFRA):
        def get_json(self, url):
            if url == DETAIL_URL.format(doc_id=1):
                self.urls.append(url)
                raise CollectionHTTPError("not found", status=404)
            return super().get_json(url)

    client = MissingCDN()
    batch = collect(tmp_path, client)
    assert batch.complete
    assert any(url.startswith(DETAIL_API) for url in client.urls)


def settings(tmp_path):
    return AppSettings(
        project_root=tmp_path,
        case_kb_root=tmp_path / "kb",
        risk_keyword_path=tmp_path / "risk",
        raw_case_upload_dir=tmp_path / "uploads",
        collection_columns="4114",
        collection_enabled=True,
    )


class FakeCases:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self.repository = SimpleNamespace(count_collected_documents=lambda on, *, zone: 0)

    async def ingest_file(self, path, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("model temporarily unavailable")
        return SimpleNamespace(cases=[SimpleNamespace(case_id="new1")])


def at(day, hour=5, minute=0):
    return datetime(2026, 10, day, hour, minute, tzinfo=SHANGHAI)


def test_status_reports_beijing_date_and_persisted_daily_count(tmp_path):
    cases = FakeCases()
    calls = []

    def count(on, *, zone):
        calls.append((on, zone))
        return 7

    cases.repository.count_collected_documents = count
    service = CollectionService(settings(tmp_path), cases)
    value = service.status(datetime(2026, 10, 22, 16, 5, tzinfo=UTC))
    assert value["today"] == "2026-10-23"
    assert value["today_crawled_cases"] == 7
    assert calls == [(date(2026, 10, 23), SHANGHAI)]
    assert not service.state_path.exists()


def test_activation_date_is_immutable_and_first_day_makes_no_requests(tmp_path):
    config = settings(tmp_path)
    service = CollectionService(config, FakeCases())
    service.activate(at(19))
    client = FakeNFRA()
    result = asyncio.run(service.run_once(now=at(19), client=client))
    assert result["status"] == "waiting" and client.requests == 0
    restarted = CollectionService(replace(config, collection_start_date="2020-01-01"), FakeCases())
    assert restarted.activate(at(21))["start_date"] == "2026-10-19"
    assert restarted.plan(at(21))["since"] == "2026-10-19"


def test_once_daily_restart_and_success_watermark(tmp_path):
    config = settings(tmp_path)
    service = CollectionService(config, FakeCases())
    service.activate(at(19))
    client = FakeNFRA({1: [row(1, "2026-10-18")]})
    result = asyncio.run(service.run_once(now=at(20), client=client))
    assert result["status"] == "success"
    restarted = CollectionService(config, FakeCases())
    assert restarted.status(at(20))["successful_until"] == "2026-10-19"
    second = asyncio.run(restarted.run_once(now=at(20), client=FakeNFRA()))
    assert second == {"status": "already_attempted_today", "requests": 0}
    assert restarted.status(at(20))["next_run_at"].startswith("2026-10-21T04:30")


def test_failed_ingestion_retries_saved_original_without_redownload(tmp_path):
    config = settings(tmp_path)
    cases = FakeCases(fail=True)
    service = CollectionService(config, cases)
    service.activate(at(19))
    first = asyncio.run(service.run_once(now=at(20), client=FakeNFRA()))
    assert first["status"] == "partial" and first["pending_parts"] == 1
    assert service.status(at(20))["successful_until"] is None
    cases.fail = False
    client = FakeNFRA()
    second = asyncio.run(service.run_once(now=at(21), client=client))
    assert second["created_cases"] == 1 and second["pending_parts"] == 0
    assert all("SelectByDocId" not in url for url in client.urls)
    assert cases.calls[-1]["origin"] == "official_crawl"
    assert cases.calls[-1]["source_metadata"]["publish_date"] == "2026-10-19"


def test_incomplete_page_limit_and_model_budget_are_visible(tmp_path):
    config = replace(settings(tmp_path), collection_max_pages=1, collection_max_details=1)
    service = CollectionService(config, FakeCases())
    service.activate(at(19))
    result = asyncio.run(
        service.run_once(now=at(20), client=FakeNFRA({1: [row(1), row(2)], 2: [row(3)]}))
    )
    assert result["status"] == "partial" and result["created_cases"] == 1
    assert service.status(at(20))["successful_until"] is None


def test_schedule_does_not_run_before_daily_time(tmp_path):
    service = CollectionService(settings(tmp_path), FakeCases())
    state = service.activate(at(19))
    assert service._next_run(state, at(20, 4, 29)) == at(20, 4, 30)
    assert service._next_run(state, at(20, 4, 31)) == at(20, 4, 31)


def test_document_receipt_recovers_after_interruption(tmp_path):
    config = settings(tmp_path)
    service = CollectionService(config, FakeCases())
    service.activate(at(19))
    downloaded = collect(config.collection_root, FakeNFRA())
    assert downloaded.documents
    result = asyncio.run(service.run_once(now=at(20), client=FakeNFRA()))
    assert result["downloaded_documents"] == 0 and result["created_cases"] == 1


def test_collection_cannot_run_concurrently_in_two_processes(tmp_path):
    from storage.file_lock import FileLock

    service = CollectionService(settings(tmp_path), FakeCases())
    service.activate(at(19))
    with FileLock(service.root / ".job.lock", timeout=0):
        result = asyncio.run(service.run_once(now=at(20), client=FakeNFRA()))
    assert result["status"] == "already_running"


def test_lifespan_scheduler_dispatches_due_job_and_stops(tmp_path, monkeypatch):
    import application.collection_service as module

    service = CollectionService(settings(tmp_path), FakeCases())
    service.activate(at(19))
    calls = []

    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return at(20)

    async def fake_run(**kwargs):
        calls.append(kwargs)
        service._stop.set()

    monkeypatch.setattr(module, "datetime", ClockDateTime)
    monkeypatch.setattr(service, "run_once", fake_run)

    async def run():
        service.start()
        await service._task
        await service.stop()

    asyncio.run(run())
    assert len(calls) == 1 and calls[0]["now"] == at(20)
