"""原文不可变，Unicode code point坐标；背景不冒充正文事实。"""

import hashlib
import re
from typing import Any

from agent_rag.contracts import canonical_json, load_contract, validate_schema

CONTEXT_KEYS = {
    "declared_scene",
    "author_role",
    "audience_role",
    "purpose_hint",
    "additional_notes",
}
SCENES = {
    "ST1_poster_sms_social",
    "ST2_live_short_video",
    "ST3_product_intro_detail",
    "ST4_meeting_material",
    "ST5_training_telemarketing_callback",
    "ST6_penalty_backend_fact",
}


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    # 保守字符估计，不冒充所选供应商的精确tokenizer。
    return 2 * len(text) + 32


class InputRegistry:
    def __init__(
        self, text: str, user_context: dict[str, str | None] | None = None, task_id: str = "test"
    ) -> None:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("EMPTY_INPUT")
        context = dict(user_context or {})
        if set(context) - CONTEXT_KEYS or any(
            value is not None and not isinstance(value, str) for value in context.values()
        ):
            raise ValueError("INVALID_USER_CONTEXT")
        if context.get("declared_scene") not in SCENES | {None}:
            raise ValueError("INVALID_SCENE")
        if sum(len(value or "") for value in context.values()) > 2000:
            raise ValueError("USER_CONTEXT_TOO_LONG")
        if estimate_tokens(text + canonical_json(context)) > 6000:
            raise ValueError("MATERIAL_TOO_LONG: conservative_character_estimate")
        self.text = text
        self.user_context = context
        self.task_id = task_id
        self.text_version = "registered-input/1.0"
        self.text_sha256 = text_hash(text)
        self.input_sha256 = text_hash(canonical_json({"text": text, "user_context": context}))
        self.source_id = f"INPUT:{task_id}"
        self.segments: list[dict[str, Any]] = []
        # 保留换行和空白，长段按实际Unicode长度拆分；不能丢掉未审的后半段。
        offset = 0
        for paragraph in re.findall(r"[^\r\n]*(?:\r\n|\r|\n|$)", text):
            if not paragraph:
                continue
            for start in range(0, len(paragraph), 2000):
                piece = paragraph[start : start + 2000]
                self.segments.append(
                    {
                        "segment_id": f"S{len(self.segments) + 1:04d}",
                        "source_id": self.source_id,
                        "start": offset + start,
                        "end": offset + start + len(piece),
                        "text": piece,
                    }
                )
            offset += len(paragraph)

    def resolve(self, ref: dict[str, Any]) -> dict[str, Any]:
        definition = load_contract("model-output.schema.json")["$defs"]["quoteRef"]
        try:
            validate_schema(ref, definition)
            segment = next(s for s in self.segments if s["segment_id"] == ref["segment_id"])
            position = -1
            for _ in range(ref["occurrence"]):
                position = segment["text"].find(ref["quote"], position + 1)
                if position == -1:
                    raise ValueError("quote not found")
        except (ValueError, StopIteration, KeyError, TypeError) as error:
            raise ValueError("INVALID_INPUT_REFERENCE") from error
        start = segment["start"] + position
        return {
            "source_id": self.source_id,
            "source_kind": "input",
            "segment_id": ref["segment_id"],
            "text_version": self.text_version,
            "text_sha256": self.text_sha256,
            "start": start,
            "end": start + len(ref["quote"]),
            "quote": ref["quote"],
        }

    def read_input(self, source_id: str) -> str:
        if source_id != self.source_id:
            raise ValueError("SOURCE_NOT_ALLOWED")
        return self.text

    def export(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "text": self.text,
            "user_context": self.user_context,
            "source_id": self.source_id,
            "text_version": self.text_version,
            "text_sha256": self.text_sha256,
            "input_sha256": self.input_sha256,
            "segments": self.segments,
            "token_count_method": "conservative_character_estimate",
        }
