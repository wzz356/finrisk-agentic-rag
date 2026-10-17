"""采集器公共件：HTTP 抓取（重试/编码探测）、Word-HTML 转纯文本、状态与落盘。

仅依赖标准库。所有站点共用纪律：
- 带 UA、限速（默认 1.5s/请求）、超时与指数退避重试；
- 编码不统一（NFRA 4115 为 UTF-8、4114 为 GBK、PBOC 详情页为 GBK），一律探测解码；
- 失败只记录 errors.jsonl，不中断整体抓取。
"""

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path
from typing import Any, cast

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
DEFAULT_DELAY = 1.5
DEFAULT_TIMEOUT = 20
MAX_RETRIES = 3


class FetchError(Exception):
    pass


def fetch(url: str, *, delay: float = DEFAULT_DELAY, timeout: int = DEFAULT_TIMEOUT) -> bytes:
    """GET 抓取，带限速与指数退避重试。最终失败抛 FetchError。"""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        if delay > 0:
            time.sleep(delay)
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return cast(bytes, response.read())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(delay * (attempt + 1) * 2)
    raise FetchError(f"抓取失败（{MAX_RETRIES} 次重试后）: {url} -> {last_error}")


def decode_text(data: bytes) -> str:
    """按候选编码探测解码；全部失败时 UTF-8 + replace 兜底。"""
    for encoding in ("utf-8", "gb18030"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def load_json(url: str, *, delay: float = DEFAULT_DELAY) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(decode_text(fetch(url, delay=delay))))


def load_json_get(
    url: str,
    params: dict[str, object],
    *,
    delay: float = DEFAULT_DELAY,
) -> dict[str, Any]:
    """GET + query string 取 JSON（NFRA 动态接口回退用）。"""
    full_url = url + "?" + urllib.parse.urlencode(params)
    return cast(dict[str, Any], json.loads(decode_text(fetch(full_url, delay=delay))))


_TAG_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG_XML = re.compile(r"<xml>.*?</xml>", re.S | re.I)
_TAG_STYLE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.S | re.I)
_TAG_BREAK = re.compile(r"</(?:p|tr|div|li|h\d)>|<br\s*/?>", re.I)
_TAG_CELL = re.compile(r"</t[dh]>", re.I)
_TAG_ANY = re.compile(r"<[^>]+>")
_WS_LINES = re.compile(r"\n{3,}")
_WS_SPACES = re.compile(r"[ \t\xa0　]+")


def html_to_text(html: str) -> str:
    """HTML（含 Word 导出的表格）转纯文本：单元格空格分隔、行/段落换行。

    先剥 Word 头部垃圾：条件注释（<!--[if mso]>...<![endif]-->）与 <xml> 元数据块，
    否则 strip 标签后会残留 cbircwsy.../样式表等噪声行。
    """
    text = _TAG_COMMENT.sub("", html)
    text = _TAG_XML.sub("", text)
    text = _TAG_STYLE.sub("", text)
    text = _TAG_BREAK.sub("\n", text)
    text = _TAG_CELL.sub(" ", text)
    text = _TAG_ANY.sub("", text)
    text = unescape(text)
    lines = [_WS_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _WS_LINES.sub("\n\n", text).strip()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class StateStore:
    """已见文书键的持久化（state.json），增量模式下用于去重与断点。"""

    def __init__(self, path: Path):
        self.path = path
        self.seen: set[str] = set()
        if path.is_file():
            try:
                self.seen = set(json.loads(path.read_text(encoding="utf-8")).get("seen", []))
            except (json.JSONDecodeError, OSError):
                self.seen = set()

    def __contains__(self, key: str) -> bool:
        return key in self.seen

    def add(self, key: str) -> None:
        self.seen.add(key)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"seen": sorted(self.seen)}, ensure_ascii=False, indent=1)
        self.path.write_text(payload, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
