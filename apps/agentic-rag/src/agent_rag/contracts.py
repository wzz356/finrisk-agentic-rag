"""两侧使用同一JSON Schema，禁止自动替换风险标签或删除未知字段。"""

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = PROJECT_ROOT / "contracts" / "agent_rag"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@lru_cache
def load_contract(name: str) -> dict[str, Any]:
    return json.loads((CONTRACT_DIR / name).read_text(encoding="utf-8-sig"))


def validate_schema(value: Any, schema: dict[str, Any]) -> None:
    from jsonschema import Draft202012Validator

    errors = list(Draft202012Validator(schema).iter_errors(value))
    if errors:
        first = errors[0]
        path = ".".join(str(item) for item in first.absolute_path)
        raise ValueError(f"INVALID_SCHEMA: {path}: {first.message[:800]}")


def validate_model_output(value: Any, kind: str | None = None) -> None:
    validate_schema(value, load_contract("model-output.schema.json"))
    if kind is not None and value["kind"] != kind:
        raise ValueError(f"INVALID_OUTPUT_KIND: expected {kind}")


def tool_schemas_for_profile(profile: str) -> dict[str, Any]:
    rules = load_contract("tool-contract-profiles.json")["profiles"]
    if profile not in rules:
        raise ValueError("UNKNOWN_TOOL_CONTRACT_PROFILE")
    schemas = copy.deepcopy(load_contract("tool-parameters.schema.json"))
    schemas["search_cases"]["properties"]["queries"]["maxItems"] = rules[profile][
        "max_queries_per_search"
    ]
    return schemas


def tool_schema(name: str, profile: str | None = None) -> dict[str, Any]:
    return (
        tool_schemas_for_profile(profile)
        if profile
        else load_contract("tool-parameters.schema.json")
    )[name]
