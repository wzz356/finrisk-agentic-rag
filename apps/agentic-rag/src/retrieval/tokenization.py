"""确定性的中文 unigram/bigram 分词。

该实现取自赛题检索复现包的轻量 BM25 通道，不依赖在线模型、Jieba 或 GPU。
"""

from __future__ import annotations

import unicodedata


def _is_chinese(character: str) -> bool:
    if character == "〇":
        return True
    name = unicodedata.name(character, "")
    if name.startswith(("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")):
        return True
    return 0x31350 <= ord(character) <= 0x323AF


def _is_latin_letter(character: str) -> bool:
    return unicodedata.category(character).startswith("L") and "LATIN" in unicodedata.name(
        character, ""
    )


def _chinese_tokens(segment: str) -> tuple[str, ...]:
    unigrams = tuple(segment)
    bigrams = tuple(segment[index : index + 2] for index in range(len(segment) - 1))
    return unigrams + bigrams


def chinese_unigram_bigram(text: str) -> tuple[str, ...]:
    """返回规范化后的中文单字/双字以及拉丁字母数字词。"""

    if not isinstance(text, str):
        raise TypeError("text 必须是字符串")

    normalized = unicodedata.normalize("NFKC", text).lower()
    tokens: list[str] = []
    segment: list[str] = []
    segment_kind: str | None = None

    def flush() -> None:
        nonlocal segment_kind
        if not segment:
            return
        value = "".join(segment)
        tokens.extend(_chinese_tokens(value) if segment_kind == "chinese" else (value,))
        segment.clear()
        segment_kind = None

    for character in normalized:
        if _is_chinese(character):
            kind = "chinese"
        elif _is_latin_letter(character) or character.isdecimal():
            kind = "latin_digit"
        else:
            flush()
            continue
        if segment_kind != kind:
            flush()
            segment_kind = kind
        segment.append(character)
    flush()
    return tuple(tokens)
