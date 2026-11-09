"""Narrow current-text consistency guards, distinguished from model judgments."""

from __future__ import annotations

import re

from domain.risk_models import MechanismDecision


def apply_mechanism_guard(text: str, decision: MechanismDecision) -> MechanismDecision:
    """An explicit request to truthfully report non-understanding is not false guidance."""
    if decision.mechanism_id != "M01":
        return decision
    truthful = re.search(
        r"如实(?:说|回答|答复|反映|告知).{0,16}(?:没明白|没懂|不理解|未理解|不知道)", text
    )
    false_override = re.search(
        r"(?:一律|无论|即使|不懂也|没懂也|别提|别说).{0,20}(?:懂|明白|理解|回答)|"
        r"(?:未|没|不).{0,4}(?:懂|明白|理解).{0,35}(?:完全|充分|都已).{0,4}(?:懂|明白|理解)",
        text,
    )
    if truthful is None or false_override is not None:
        return decision
    quote = truthful.group()
    return decision.model_copy(
        update={
            "decision_state": "no_clear_risk",
            "risk_labels": [],
            "candidate_labels": [],
            "missing_facts": [],
            "evidence": [quote],
            "decision_origin": "deterministic_guard",
            "reasoning": "当前原句要求如实说明未理解，不满足不实答复指令条件",
            "condition_assessments": [
                {
                    "condition_id": "M01.false_answer_instruction",
                    "status": "not_met",
                    "evidence": [quote],
                }
            ],
        }
    )
