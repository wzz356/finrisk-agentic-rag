"""官网 Word-HTML 正文清理；HTTP 访问统一由 polite_http 管理。"""

import re
from html import unescape

_TAG_COMMENT = re.compile(r"<!--.*?-->", re.S)
_TAG_XML = re.compile(r"<xml>.*?</xml>", re.S | re.I)
_TAG_STYLE = re.compile(r"<(style|script)[^>]*>.*?</\1>", re.S | re.I)
_TAG_BREAK = re.compile(r"</(?:p|tr|div|li|h\d)>|<br\s*/?>", re.I)
_TAG_CELL = re.compile(r"</t[dh]>", re.I)
_TAG_ANY = re.compile(r"<[^>]+>")
_WS_LINES = re.compile(r"\n{3,}")
_WS_SPACES = re.compile(r"[ \t\xa0　]+")


def html_to_text(html: str) -> str:
    """去掉脚本、样式与 Word 注释，保留表格、段落中的正文。"""
    text = _TAG_COMMENT.sub("", html)
    text = _TAG_XML.sub("", text)
    text = _TAG_STYLE.sub("", text)
    text = _TAG_BREAK.sub("\n", text)
    text = _TAG_CELL.sub(" ", text)
    text = _TAG_ANY.sub("", text)
    text = unescape(text)
    lines = [_WS_SPACES.sub(" ", line).strip() for line in text.splitlines()]
    return _WS_LINES.sub("\n\n", "\n".join(line for line in lines if line)).strip()
