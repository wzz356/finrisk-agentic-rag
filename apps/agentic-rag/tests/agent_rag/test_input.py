import hashlib
import json
from pathlib import Path

import pytest

from agent_rag.contracts import validate_model_output
from agent_rag.input_registry import InputRegistry

FIXTURES = Path(__file__).resolve().parents[2] / "contracts/agent_rag/fixtures"


def test_unicode_and_second_occurrence():
    registry = InputRegistry("A😀é同句。同句。", task_id="t1")
    ref = {"segment_id": "S0001", "quote": "同句。", "occurrence": 2}
    span = registry.resolve(ref)
    assert (span["start"], span["end"]) == (7, 10)
    assert registry.text[span["start"] : span["end"]] == ref["quote"]
    assert span["text_sha256"] == hashlib.sha256(registry.text.encode()).hexdigest()


@pytest.mark.parametrize("quote,occurrence", [("同句！", 1), ("同句。", 0), ("同句。", 3)])
def test_invalid_reference_never_fuzzy_matches(quote, occurrence):
    registry = InputRegistry("同句。同句。")
    with pytest.raises(ValueError, match="INVALID_INPUT_REFERENCE"):
        registry.resolve({"segment_id": "S0001", "quote": quote, "occurrence": occurrence})


def test_context_bound_hash_and_immutable_segments():
    a = InputRegistry("第一段。\r\n第二段。", {"purpose_hint": "禁止性提醒"}, "t1")
    b = InputRegistry(a.text, {"purpose_hint": "广告"}, "t2")
    assert a.input_sha256 != b.input_sha256
    assert a.text_sha256 == b.text_sha256
    assert "".join(a.text[s["start"] : s["end"]] for s in a.segments) == a.text
    assert a.source_id != b.source_id
    with pytest.raises(ValueError, match="SOURCE_NOT_ALLOWED"):
        a.read_input(b.source_id)


@pytest.mark.parametrize(
    "filename", ["valid_analysis.json", "valid_decision.json", "valid_final.json"]
)
def test_positive_shared_contract(filename):
    value = json.loads((FIXTURES / filename).read_text(encoding="utf-8-sig"))
    validate_model_output(value, value["kind"])


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("invalid_*.json")), ids=lambda p: p.name)
def test_negative_shared_contract(path):
    with pytest.raises(ValueError):
        validate_model_output(json.loads(path.read_text(encoding="utf-8-sig")))


def test_empty_and_oversized_material_rejected():
    for text in ["   ", "保险" * 4000]:
        with pytest.raises(ValueError):
            InputRegistry(text)
