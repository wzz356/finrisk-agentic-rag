"""Local-only official V4.1 prompt encoding with pinned, verified assets."""

import copy
import hashlib
import importlib.util
import json
import threading
from pathlib import Path
from typing import Any

from agent_rag.contracts import PROJECT_ROOT

COMMIT = "5254448103b6254fc646dacbca1cbbf20e9216c1"
ASSET_HASHES = {
    "tokenizer.json": "c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b",
    "encoding.py": "502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1",
    "tokenizer_config.json": "6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547",
    "chat_template.jinja": "d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3",
    "LICENSE": "f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040",
}


class OfficialPromptCounter:
    def __init__(self, assets: Path | None = None) -> None:
        self.assets = assets or PROJECT_ROOT / "resources/deepseek-v41"
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._encoder: Any = None

    def _verify(self) -> None:
        try:
            manifest = json.loads((self.assets / "manifest.json").read_text(encoding="utf-8"))
            if manifest["repo_commit"] != COMMIT or {
                row["file"]: row["sha256"] for row in manifest["assets"]
            } != ASSET_HASHES:
                raise ValueError("TOKEN_COUNTER_ASSET_MISMATCH")
            for filename, expected in ASSET_HASHES.items():
                if hashlib.sha256((self.assets / filename).read_bytes()).hexdigest() != expected:
                    raise ValueError("TOKEN_COUNTER_ASSET_MISMATCH")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("TOKEN_COUNTER_ASSET_MISMATCH") from error

    def count(self, payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("model", "deepseek-flash") != "deepseek-flash" or payload.get(
            "thinking_mode", "chat"
        ) != "chat":
            raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
        messages = copy.deepcopy(payload.get("messages"))
        if not isinstance(messages, list) or not messages:
            raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system", "user", "assistant", "tool"
            }:
                raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
            content = message.get("content")
            if isinstance(content, list):
                if any(not isinstance(block, dict) or block.get("type") != "text"
                       or not isinstance(block.get("text"), str) for block in content):
                    raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
            elif content is not None and not isinstance(content, str):
                raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
            if set(message) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}:
                raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
        tools = payload.get("tools", [])
        if not isinstance(tools, list) or any(
            not isinstance(tool, dict) or tool.get("type") != "function"
            or not isinstance(tool.get("function"), dict) for tool in tools
        ):
            raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
        if messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": ""})
        if tools:
            messages[0]["tools"] = tools
        if payload.get("response_format"):
            if payload["response_format"] != {"type": "json_object"}:
                raise ValueError("TOKEN_COUNTER_UNSUPPORTED")
            messages[0]["response_format"] = payload["response_format"]
        with self._lock:
            # Verify every invocation; a warmed singleton cannot hide changed assets.
            self._verify()
            if self._tokenizer is None:
                import tokenizers

                if tokenizers.__version__ != "0.23.1":
                    raise ValueError("TOKEN_COUNTER_DEPENDENCY_MISMATCH")
                spec = importlib.util.spec_from_file_location(
                    "finrisk_verified_v41_encoding", self.assets / "encoding.py"
                )
                if spec is None or spec.loader is None:
                    raise ValueError("TOKEN_COUNTER_ASSET_MISMATCH")
                self._encoder = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(self._encoder)
                self._tokenizer = tokenizers.Tokenizer.from_file(str(self.assets / "tokenizer.json"))
                self._tokenizer.no_truncation()
                self._tokenizer.no_padding()
            try:
                prompt = self._encoder.encode_messages(messages, thinking_mode="chat")
                count = len(self._tokenizer.encode(prompt, add_special_tokens=False).ids)
            except Exception as error:
                raise ValueError("TOKEN_COUNTER_ENCODING_FAILED") from error
        return {
            "prompt_tokens": count,
            "counter_kind": "deepseek_v41_official",
            "model_version": "DeepSeek-V4.1-Flash",
            "asset_commit": COMMIT,
            "asset_hashes": ASSET_HASHES,
        }


official_counter = OfficialPromptCounter()
