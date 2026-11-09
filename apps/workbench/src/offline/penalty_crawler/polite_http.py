"""Bounded official-site access: identify, cache, space requests and stop on blocking."""

from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "FinRiskCollector/1.0 (+https://github.com/qby1210/finrisk-framework)"
OFFICIAL_HOSTS = frozenset({"www.nfra.gov.cn"})


class CollectionHTTPError(RuntimeError):
    def __init__(self, message: str, *, status: int = 0, retry_at: float = 0) -> None:
        super().__init__(message)
        self.status, self.retry_at = status, retry_at


class RequestBudgetExceeded(CollectionHTTPError):
    pass


class SitePaused(CollectionHTTPError):
    pass


@dataclass
class Response:
    status: int
    body: bytes
    headers: dict[str, str]
    url: str


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class _OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        request: urllib.request.Request,
        fp: Any,
        code: int,
        message: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        validate_url(newurl)
        return super().redirect_request(request, fp, code, message, headers, newurl)


def validate_url(url: str) -> str:
    parsed = urlparse(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in OFFICIAL_HOSTS
        or parsed.username
        or parsed.password
        or parsed.port not in (None, 443)
    ):
        raise CollectionHTTPError("采集地址必须为已配置的官方 HTTPS 站点")
    return str(parsed.hostname)


def _transport(url: str, headers: dict[str, str], timeout: float) -> Response:
    opener = urllib.request.build_opener(_OfficialRedirect())
    request = urllib.request.Request(url, headers=headers)
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        validate_url(response.geturl())
        body = response.read(8 * 1024 * 1024 + 1)
        if len(body) > 8 * 1024 * 1024:
            raise CollectionHTTPError("官网响应超过单次大小限制")
        return Response(
            response.code,
            body,
            {k.lower(): v for k, v in response.headers.items()},
            response.geturl(),
        )


class PoliteHTTPClient:
    def __init__(
        self,
        cache_dir: Path,
        *,
        delay: float = 3,
        jitter: float = 2,
        max_requests: int = 32,
        timeout: float = 25,
        transport: Callable[[str, dict[str, str], float], Response] = _transport,
        now: Callable[[], float] = time.time,
        sleep: Callable[[float], object] = time.sleep,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        self.cache_dir = cache_dir
        self.delay, self.jitter = max(3.0, delay), max(0.0, jitter)
        self.max_requests, self.timeout = max_requests, timeout
        self.transport, self.now, self.sleep = transport, now, sleep
        self.cancelled = cancelled
        self.requests = self.cache_hits = 0
        self._last_request: dict[str, float] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self.state_path = cache_dir / "host_state.json"
        self.state = (
            json.loads(self.state_path.read_text("utf-8")) if self.state_path.is_file() else {}
        )

    def _paused(self, host: str) -> None:
        until = float(self.state.get(host, {}).get("retry_at", 0))
        if until > self.now():
            raise SitePaused("官网访问已暂停，等待下次允许时间", retry_at=until)

    def _pause(self, host: str, response: Response) -> None:
        value = response.headers.get("retry-after", "")
        retry_at = self.now() + (86400 if response.status == 403 else 21600)
        if value:
            try:
                retry_at = max(self.now() + 60, self.now() + int(value))
            except ValueError:
                try:
                    retry_at = max(self.now() + 60, parsedate_to_datetime(value).timestamp())
                except (ValueError, TypeError, OverflowError):
                    pass
        self.state[host] = {"retry_at": retry_at, "status": response.status}
        atomic_json(self.state_path, self.state)
        raise SitePaused(
            "官网限流或访问验证，已停止本轮访问", status=response.status, retry_at=retry_at
        )

    def _request(self, url: str) -> Response:
        if self.cancelled():
            raise CollectionHTTPError("任务停止，采集进度已保留")
        host = validate_url(url)
        self._paused(host)
        if self.requests >= self.max_requests:
            raise RequestBudgetExceeded("已达到本轮官网请求上限")
        host_state = self.state.get(host, {})
        last = max(self._last_request.get(host, 0), float(host_state.get("last_request", 0)))
        delay = last + self.delay + random.uniform(0, self.jitter) - self.now()
        if delay > 0:
            self.sleep(delay)
        if self.cancelled():
            raise CollectionHTTPError("任务停止，采集进度已保留")
        key = sha256(url.encode()).hexdigest()
        metadata_path = self.cache_dir / f"{key}.json"
        body_path = self.cache_dir / f"{key}.body"
        cached = (
            json.loads(metadata_path.read_text("utf-8"))
            if metadata_path.is_file() and body_path.is_file()
            else {}
        )
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "application/json,text/html;q=0.9,*/*;q=0.5",
            "Accept-Language": "zh-CN,zh;q=0.9",
        }
        if cached.get("etag"):
            headers["If-None-Match"] = cached["etag"]
        if cached.get("last_modified"):
            headers["If-Modified-Since"] = cached["last_modified"]
        self.requests += 1
        self._last_request[host] = self.now()
        self.state[host] = {**host_state, "last_request": self.now()}
        atomic_json(self.state_path, self.state)
        try:
            response = self.transport(url, headers, self.timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise CollectionHTTPError(f"官网连接失败: {type(error).__name__}") from error
        response.headers = {k.lower(): v for k, v in response.headers.items()}
        if response.status in {403, 429} or (
            response.status == 503 and "retry-after" in response.headers
        ):
            self._pause(host, response)
        stripped = response.body.lstrip().lower()
        if stripped.startswith((b"<!doctype html", b"<html")):
            text = response.body.decode("utf-8", errors="replace").lower()
            if any(
                word in text
                for word in ("访问过于频繁", "安全验证", "人机验证", "captcha", "验证您是真人")
            ):
                self._pause(host, response)
        if response.status == 304 and cached:
            self.cache_hits += 1
            return Response(200, body_path.read_bytes(), cached.get("headers", {}), url)
        if response.status != 200:
            raise CollectionHTTPError(f"官网返回 HTTP {response.status}", status=response.status)
        if "no-store" not in response.headers.get("cache-control", "").lower():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            body_path.write_bytes(response.body)
            atomic_json(
                metadata_path,
                {
                    "etag": response.headers.get("etag"),
                    "last_modified": response.headers.get("last-modified"),
                    "headers": response.headers,
                    "url": url,
                },
            )
        return response

    def _check_robots(self, url: str) -> None:
        host = validate_url(url)
        if host not in self._robots:
            path = self.cache_dir / f"robots_{host}.json"
            cached = json.loads(path.read_text("utf-8")) if path.is_file() else {}
            if float(cached.get("expires_at", 0)) > self.now():
                content = cached.get("text")
            else:
                try:
                    content = self._request(f"https://{host}/robots.txt").body.decode(
                        "utf-8", errors="replace"
                    )
                except CollectionHTTPError as error:
                    if error.status not in {404, 410}:
                        raise
                    content = None
                atomic_json(path, {"text": content, "expires_at": self.now() + 86400})
            parser = None
            if content is not None:
                parser = RobotFileParser()
                parser.parse(content.splitlines())
                interval = parser.crawl_delay(USER_AGENT) or parser.crawl_delay("*")
                if interval:
                    self.delay = max(self.delay, float(interval))
            self._robots[host] = parser
        parser = self._robots[host]
        if parser is not None and not parser.can_fetch(USER_AGENT, url):
            raise SitePaused("robots.txt 未允许采集此地址")

    def get_json(self, url: str) -> dict[str, Any]:
        self._check_robots(url)
        response = self._request(url)
        try:
            result = json.loads(response.body.decode("utf-8-sig"))
        except (ValueError, UnicodeError) as error:
            raise CollectionHTTPError("官网返回内容不是预期 JSON，停止本轮访问") from error
        if not isinstance(result, dict):
            raise CollectionHTTPError("官网 JSON 结构异常")
        return cast(dict[str, Any], result)
