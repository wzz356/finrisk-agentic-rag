"""Planned service admission contract, generated from the real Node exports."""

import json
import math
import re
from decimal import Decimal

from agent_rag.contracts import PROJECT_ROOT, canonical_json
from agent_rag.input_registry import text_hash

INVENTORY = PROJECT_ROOT / "contracts/agent_rag/planned-execution-contract.json"
PLANNED_PROFILE = "live-deepseek-flash-planned-v1"
FIXED_FIELDS = {
    "model_wire_sha256", "criteria_sha256", "prompt_sha256", "tool_contract_sha256",
    "request_admission_policy_sha256", "budget_profile_sha256", "agent_loop_schema_sha256",
    "agent_loop_protocol_sha256",
}
BINDING_FIELDS = (
    "profile_id", "retrieval_profile", "request_admission_policy", "budget_profile_id",
)


def _node_profile_hash(config):
    # Registered service configs are flat JSON scalars. Only CNY amounts are
    # floats: emit finite decimal amounts as Node does, including Python 1.0 -> 1.
    items = []
    for key, value in sorted(config.items()):
        if type(value) is float:
            if not math.isfinite(value) or (value and not 1e-6 <= abs(value) < 1e21):
                raise ValueError("PLANNED_SERVICE_CONFIG_NUMBER_UNSUPPORTED")
            token = format(Decimal(str(value)), "f")
            token = token.rstrip("0").rstrip(".") if "." in token else token
            token = "0" if value == 0 else token
        elif value is None or type(value) in (str, int, bool):
            token = canonical_json(value)
        else:
            raise ValueError("PLANNED_SERVICE_CONFIG_SHAPE_UNSUPPORTED")
        items.append(canonical_json(key) + ":" + token)
    return text_hash("{" + ",".join(items) + "}")


def planned_execution_contract(config, manifest):
    inventory = json.loads(INVENTORY.read_text(encoding="utf8"))
    fixed = inventory.get("execution_contract", {})
    if (
        inventory.get("schema_version") != "finrisk-planned-service-contract/1.0"
        or inventory.get("binding") != {key: config.get(key) for key in BINDING_FIELDS}
        or set(fixed) != FIXED_FIELDS
        or any(not isinstance(v, str) or not re.fullmatch(r"[0-9a-f]{64}", v)
               for v in fixed.values())
        or any(manifest.get(key) != fixed[key] for key in (
            "model_wire_sha256", "criteria_sha256", "request_admission_policy_sha256",
            "budget_profile_sha256",
        ))
    ):
        raise ValueError("PLANNED_SERVICE_CONTRACT_DRIFT")
    return {**fixed, "profile_sha256": _node_profile_hash(config)}
