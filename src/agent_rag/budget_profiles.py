"""Registered budget ceilings shared with Node; currency is integer micro-CNY."""

import hashlib
import json
import math
from typing import Any

from agent_rag.contracts import PROJECT_ROOT, load_contract, validate_schema

MAX_SAFE_INTEGER = 9007199254740991
PLANNED_PROFILE = "planned-evidence-loop-v1"
LEGACY_V2_PROFILE = "legacy-live-v2-v1"
ORCHESTRATION_MODES = {"staged_one_shot", "planned_fixed", "planned_adaptive"}


def _normalize(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) in (int, float):
        if abs(value) > MAX_SAFE_INTEGER or not math.isfinite(value) or value != int(value):
            raise ValueError("INVALID_BUDGET_PROFILE: unsafe integer")
        return int(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        # Match JavaScript's lexicographic UTF-16 object-key ordering.
        return {
            key: _normalize(value[key])
            for key in sorted(value, key=lambda key: key.encode("utf-16-be"))
        }
    raise ValueError("INVALID_BUDGET_PROFILE: non-JSON value")


def canonical_budget_json(value: Any) -> str:
    return json.dumps(_normalize(value), ensure_ascii=False, separators=(",", ":"))


def budget_profile_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_budget_json(value).encode("utf-8")).hexdigest()


def validate_budget_profile(value: Any) -> dict[str, Any]:
    normalized = _normalize(value)
    try:
        validate_schema(normalized, load_contract("budget-profile.schema.json"))
        for key, number in normalized.items():
            if key != "profile_id" and type(number) is not int:
                raise ValueError("numeric fields must be integers, not booleans")
        if normalized.get("initial_query_limit", 0) > normalized["max_search_query_attempts"]:
            raise ValueError("initial queries exceed task query budget")
        if normalized.get("feedback_query_reserve", 0) > normalized["max_search_query_attempts"]:
            raise ValueError("feedback reserve exceeds task query budget")
        if (
            normalized.get("initial_query_limit", 0) + normalized.get("feedback_query_reserve", 0)
            > normalized["max_search_query_attempts"]
        ):
            raise ValueError("initial queries consume feedback reserve")
        if (
            normalized["rerank_cap_micro_cny"] + normalized["model_trial_cap_micro_cny"]
            != normalized["task_total_cost_cap_micro_cny"]
        ):
            raise ValueError("task cost does not equal component costs")
        return normalized
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError(f"INVALID_BUDGET_PROFILE: {error}") from error


def load_budget_profile(profile_id: str) -> dict[str, Any]:
    catalog = json.loads(
        (PROJECT_ROOT / "configs/agent_rag/budget-profiles.json").read_text(encoding="utf-8")
    )
    if not isinstance(profile_id, str) or profile_id not in catalog["profiles"]:
        raise ValueError("UNKNOWN_BUDGET_PROFILE")
    profile = validate_budget_profile(catalog["profiles"][profile_id])
    if profile["profile_id"] != profile_id:
        raise ValueError("INVALID_BUDGET_PROFILE: catalog id mismatch")
    return profile


def validate_budget_config(config: dict[str, Any]) -> dict[str, Any] | None:
    profile_id = config.get("budget_profile_id")
    mode = config.get("orchestration_mode")
    fast = config.get("fast_path_enabled")
    if "budget_profile_id" not in config:
        if any(
            key in config
            for key in ("orchestration_mode", "fast_path_enabled", "budget_profile_sha256")
        ):
            raise ValueError("INVALID_BUDGET_CONFIG: orchestration requires a budget profile")
        return None
    profile = load_budget_profile(profile_id)
    if (
        profile_id != PLANNED_PROFILE
        or not isinstance(mode, str)
        or mode not in ORCHESTRATION_MODES
    ):
        raise ValueError("INVALID_BUDGET_CONFIG: orchestration/profile mismatch")
    if config.get("retrieval_profile") != "v2-hybrid-flash-v1":
        raise ValueError("INVALID_BUDGET_CONFIG: planned orchestration requires v2")
    if "fast_path_enabled" in config and type(fast) is not bool:
        raise ValueError("INVALID_BUDGET_CONFIG: fast_path_enabled must be boolean")
    if (
        not isinstance(config.get("engine"), str)
        or config["engine"] not in {"controlled_agent", "fixed_rag"}
        or (config.get("engine") == "fixed_rag" and mode != "planned_fixed")
    ):
        raise ValueError("INVALID_BUDGET_CONFIG: engine/mode mismatch")
    if config.get("budget_profile_sha256") != budget_profile_sha256(profile):
        raise ValueError("INVALID_BUDGET_CONFIG: budget profile hash mismatch")
    effective = {key: config.get(key) for key in profile}
    effective["profile_id"] = profile_id
    try:
        effective = validate_budget_profile(effective)
        if any(effective[key] > limit for key, limit in profile.items() if key != "profile_id"):
            raise ValueError("resource exceeds registered profile")
        for currency_key in ("rerank_cap", "model_trial_cap", "task_total_cost_cap"):
            alias = f"{currency_key}_cny"
            if alias in config and (
                type(config[alias]) not in (int, float)
                or config[alias] != effective[f"{currency_key}_micro_cny"] / 1_000_000
            ):
                raise ValueError("currency alias differs from integer cap")
        external = config.get("max_external_retrieval_requests", effective["rerank_max_requests"])
        if type(external) is not int or not 1 <= external <= effective["rerank_max_requests"]:
            raise ValueError("external requests exceed rerank budget")
    except ValueError as error:
        raise ValueError(f"INVALID_BUDGET_CONFIG: {error}") from error
    return profile


def budget_capacity_diagnostic(
    profile: dict[str, Any], effective_rerank_max_requests: int | None = None
) -> dict[str, Any]:
    profile = validate_budget_profile(profile)
    effective = (
        profile["rerank_max_requests"]
        if effective_rerank_max_requests is None
        else effective_rerank_max_requests
    )
    if type(effective) is not int or not 1 <= effective <= profile["rerank_max_requests"]:
        raise ValueError("INVALID_BUDGET_CONFIG: effective rerank limit")
    theoretical = (
        profile["max_search_query_attempts"]
        * profile["rerank_candidates_per_query"]
        * profile["rerank_attempts_per_candidate"]
    )
    return {
        "theoretical_rerank_requests": theoretical,
        "effective_rerank_max_requests": effective,
        "may_stop_before_all_queries": effective < theoretical,
    }
