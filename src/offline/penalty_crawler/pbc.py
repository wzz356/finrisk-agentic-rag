"""人民银行"行政处罚公示"栏目采集（银罚决字批次，总行层面）。

实测要点：
- 列表为静态 HTML：第 1 页 index.html，第 2..N 页为 "<模块号>-<页码>.html"，
  模块号从页面"下一页"按钮的 onclick/tagname 中解析（如 d80f41dc-2.html）；
- 详情页正文在 <div id="xxgk_con"> 内，页面声明 UTF-8 但实际为 GBK，需探测解码；
- 单条常为一批文号（如 银罚决字〔2026〕27-36号），保持一份 txt，
  多案拆分交给下游 parser.parse_document；
- 增量模式：整页条目全部已见即停。
"""

import re
from pathlib import Path
from typing import TypedDict

from .common import (
    FetchError,
    StateStore,
    append_jsonl,
    decode_text,
    fetch,
    html_to_text,
    sha1_text,
    write_text,
)

BASE = "http://www.pbc.gov.cn"
COLUMN_PATH = "/zhengwugongkai/4081330/4081344/4081407/4081705"
LIST_FIRST = BASE + COLUMN_PATH + "/index.html"

_ITEM_RE = re.compile(
    r'<a[^>]+href="(' + re.escape(COLUMN_PATH) + r'/([^/"]+)/index\.html)"[^>]*'
    r'(?:title="([^"]*)")?[^>]*>.*?</a>\s*<span[^>]*class="date"[^>]*>([^<]*)</span>',
    re.S,
)
_NEXT_PAGE_RE = re.compile(r'tagname="' + re.escape(COLUMN_PATH) + r'/([0-9a-f]+)-(\d+)\.html"')
_TOTAL_PAGES_RE = re.compile(r'totalpage="(\d+)"|</font>\s*/\s*<b>(\d+)</b>')
_CONTENT_START = 'id="xxgk_con"'
_CONTENT_END = re.compile(r'<div[^>]+class="[^"]*(?:foot|list_page)[^"]*"')
_ARTICLE_ID_RE = re.compile(r"[0-9a-z]+(?=-\d+\.html)")


class CrawlStats(TypedDict):
    source: str
    pages: int
    new: int
    skipped: int
    empty: int
    errors: int


def _parse_list(html: str) -> list[dict[str, str]]:
    items = []
    for match in _ITEM_RE.finditer(html):
        path, slug, title, date = match.groups()
        items.append(
            {
                "slug": slug,
                "title": (title or "").strip(),
                "date": date.strip(),
                "url": f"{BASE}{path}",  # path 捕获组已含结尾 /index.html
            }
        )
    return items


def _detail_text(html: str) -> str:
    start = html.find(_CONTENT_START)
    region = html[start:] if start >= 0 else html
    # start 落在 <div ...> 标签内部，先跳到标签结束，避免残留 `id="xxgk_con" ...>` 半截标签
    tag_end = region.find(">")
    if tag_end >= 0:
        region = region[tag_end + 1 :]
    end_match = _CONTENT_END.search(region)
    if end_match:
        region = region[: end_match.start()]
    text = html_to_text(region)
    # 去掉面包屑导航行
    return "\n".join(line for line in text.splitlines() if not line.startswith("所在位置")).strip()


def crawl(
    out_dir: Path,
    *,
    full: bool = False,
    max_pages: int = 0,
    delay: float = 1.5,
) -> CrawlStats:
    state = StateStore(out_dir / "state.json")
    text_dir = out_dir / "pbc"
    index_path = out_dir / "index.jsonl"
    errors_path = out_dir / "errors.jsonl"
    stats: CrawlStats = {
        "source": "pbc",
        "pages": 0,
        "new": 0,
        "skipped": 0,
        "empty": 0,
        "errors": 0,
    }

    page_urls = [LIST_FIRST]
    page_index = 0
    while page_index < len(page_urls):
        if max_pages and page_index >= max_pages:
            break
        url = page_urls[page_index]
        page_index += 1
        try:
            html = decode_text(fetch(url, delay=delay))
        except FetchError as exc:
            append_jsonl(errors_path, {"source": "pbc", "url": url, "error": str(exc)})
            stats["errors"] += 1
            break
        items = _parse_list(html)
        if not items:
            break
        stats["pages"] += 1

        # 从第 1 页解析模块号与总页数，生成全部翻页 URL（模块号不写死）
        if page_index == 1:
            prefix_match = _NEXT_PAGE_RE.search(html)
            total_match = _TOTAL_PAGES_RE.search(html)
            total_pages = int(next(g for g in total_match.groups() if g)) if total_match else 1
            if prefix_match:
                module = prefix_match.group(1)
                for page_no in range(2, total_pages + 1):
                    page_urls.append(f"{BASE}{COLUMN_PATH}/{module}-{page_no}.html")

        page_all_seen = True
        for item in items:
            key = f"pbc:{item['slug']}"
            if key in state:
                stats["skipped"] += 1
                continue
            page_all_seen = False
            try:
                detail_html = decode_text(fetch(item["url"], delay=delay))
                text = _detail_text(detail_html)
            except FetchError as exc:
                append_jsonl(errors_path, {"source": "pbc", "url": item["url"], "error": str(exc)})
                stats["errors"] += 1
                continue
            if not text:
                stats["empty"] += 1
                state.add(key)
                continue
            txt_name = f"{item['slug']}.txt"
            write_text(text_dir / txt_name, text)
            append_jsonl(
                index_path,
                {
                    "source": "pbc",
                    "column_name": "人民银行行政处罚公示",
                    "slug": item["slug"],
                    "title": item["title"],
                    "publish_date": item["date"],
                    "source_url": item["url"],
                    "txt_path": str(text_dir / txt_name),
                    "sha1": sha1_text(text),
                },
            )
            state.add(key)
            stats["new"] += 1
        state.save()

        if not full and page_all_seen:
            break
    return stats
