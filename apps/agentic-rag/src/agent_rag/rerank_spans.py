"""Deterministic contiguous source references for the explicit span-ref-v2 protocol."""

from __future__ import annotations

import hashlib
from typing import Any

SPAN_VERSION = "finrisk-rerank-spans/1.0"
_BOUNDARIES = frozenset("\n。！？；!?;")
_MAX_CHARS = 96


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_spans(text: str) -> list[dict]:
    if not isinstance(text, str) or not text:
        raise ValueError("INVALID_SPAN_SOURCE")
    parts, start = [], 0
    while start < len(text):
        end = min(len(text), start + _MAX_CHARS)
        boundary = next((i + 1 for i in range(end - 1, start - 1, -1)
                         if text[i] in _BOUNDARIES), None)
        if boundary is not None:
            end = boundary
        chunk = text[start:end]
        parts.append({"ref_id": f"S{len(parts) + 1:04}", "start": start, "end": end,
                      "text": chunk, "text_sha256": _hash(chunk)})
        start = end
    return parts


def build_span_packet(pair_id: str, query_text: str, unit_text: str) -> dict:
    if not isinstance(pair_id, str) or not pair_id or not isinstance(query_text, str):
        raise ValueError("INVALID_SPAN_PACKET")
    parts = build_spans(unit_text)
    return {"pair_id": pair_id, "query_text": query_text, "unit_text": unit_text,
            "unit_text_sha256": _hash(unit_text), "span_version": SPAN_VERSION,
            "unit_parts": parts}


def _validate_packet(packet: dict) -> None:
    if not isinstance(packet, dict):
        raise ValueError("INVALID_SPAN_PACKET")
    source = packet.get("unit_text")
    parts = packet.get("unit_parts")
    if (not isinstance(source, str) or not source
            or packet.get("span_version") != SPAN_VERSION
            or packet.get("unit_text_sha256") != _hash(source)
            or not isinstance(parts, list)
            or any(not isinstance(p, dict) or type(p.get("start")) is not int
                   or type(p.get("end")) is not int for p in parts)
            or parts != build_spans(source)):
        raise ValueError("SPAN_SOURCE_BINDING_MISMATCH")


def span_wire(packet: dict) -> dict:
    _validate_packet(packet)
    return {"pair_id": packet["pair_id"], "query_text": packet["query_text"],
            "unit_parts": [{"ref_id": p["ref_id"], "text": p["text"]}
                           for p in packet["unit_parts"]]}


def resolve_span(packet: dict, selection: Any) -> dict:
    _validate_packet(packet)
    if selection is None:
        return {"source_quote": "", "start": None, "end": None,
                "unit_text_sha256": packet["unit_text_sha256"]}
    if (not isinstance(selection, dict) or set(selection) != {"start_ref", "end_ref"}
            or any(not isinstance(v, str) for v in selection.values())):
        raise ValueError("INVALID_SPAN_REFERENCE")
    parts = {p["ref_id"]: p for p in packet["unit_parts"]}
    first, last = parts.get(selection["start_ref"]), parts.get(selection["end_ref"])
    if first is None or last is None or first["start"] > last["start"]:
        raise ValueError("INVALID_SPAN_REFERENCE")
    start, end = first["start"], last["end"]
    return {"source_quote": packet["unit_text"][start:end], "start": start, "end": end,
            "unit_text_sha256": packet["unit_text_sha256"]}
