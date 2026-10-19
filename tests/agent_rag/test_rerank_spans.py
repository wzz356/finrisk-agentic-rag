import importlib
import importlib.util

import pytest


def module():
    assert importlib.util.find_spec("agent_rag.rerank_spans"), "尚无可验证的片段引用协议"
    return importlib.import_module("agent_rag.rerank_spans")


@pytest.mark.parametrize("text", ["赠送购物卡。以投保为条件。", "甲\r\n  乙。\n丙😀丁", "相同。相同。", "长" * 203, "  \n "])
def test_spans_preserve_every_unicode_character_and_offsets(text):
    parts = module().build_spans(text)
    assert "".join(p["text"] for p in parts) == text
    cursor = 0
    for i, part in enumerate(parts, 1):
        assert part["ref_id"] == f"S{i:04}"
        assert part["start"] == cursor
        assert 0 < part["end"] - cursor <= 96
        assert text[cursor:part["end"]] == part["text"]
        cursor = part["end"]
    assert cursor == len(text)


def test_span_wire_contains_one_copy_of_source_and_no_private_offsets():
    m = module()
    packet = m.build_span_packet("P1", "是否投保赠礼", "投保保险。\n赠购物卡😀")
    wire = m.span_wire(packet)
    assert set(wire) == {"pair_id", "query_text", "unit_parts"}
    assert wire["pair_id"] == "P1"
    assert "".join(p["text"] for p in wire["unit_parts"]) == packet["unit_text"]
    assert all(set(p) == {"ref_id", "text"} for p in wire["unit_parts"])


def test_contiguous_multi_part_selection_reconstructs_original_quote():
    m = module()
    text = "甲" * 95 + "😀" + "以购买保险为条件。赠送购物卡。\n保留限制说明"
    packet = m.build_span_packet("P1", "投保赠礼", text)
    result = m.resolve_span(packet, {"start_ref": "S0001", "end_ref": "S0002"})
    assert result["source_quote"] == text[:packet["unit_parts"][1]["end"]]
    assert result["start"] == 0
    assert result["end"] == packet["unit_parts"][1]["end"]
    assert result["unit_text_sha256"] == packet["unit_text_sha256"]


@pytest.mark.parametrize("selection", [
    {"start_ref": "S9999", "end_ref": "S9999"},
    {"start_ref": "S0002", "end_ref": "S0001"},
    {"start_ref": "S0001", "end_ref": "S0001", "source_quote": "伪造"},
    ["S0001"], {"start_ref": 1, "end_ref": "S0001"}, {},
])
def test_forged_or_reversed_selection_rejected(selection):
    m = module()
    packet = m.build_span_packet("P1", "查询", "甲" * 100)
    with pytest.raises(ValueError, match="SPAN"):
        m.resolve_span(packet, selection)


@pytest.mark.parametrize("mutation", ["text", "hash", "offset", "part_text", "version"])
def test_packet_tampering_is_rejected_before_quote_reconstruction(mutation):
    m = module()
    packet = m.build_span_packet("P1", "查询", "原始处罚材料😀" * 20)
    if mutation == "text":
        packet["unit_text"] += "伪造"
    elif mutation == "hash":
        packet["unit_text_sha256"] = "0" * 64
    elif mutation == "offset":
        packet["unit_parts"][0]["start"] = 1
    elif mutation == "part_text":
        packet["unit_parts"][0]["text"] = "其他案例"
    else:
        packet["span_version"] = "unknown"
    with pytest.raises(ValueError, match="SPAN"):
        m.resolve_span(packet, {"start_ref": "S0001", "end_ref": "S0001"})


def test_null_quote_is_explicit_and_empty_source_rejected():
    m = module()
    packet = m.build_span_packet("P1", "查询", "处罚材料")
    assert m.resolve_span(packet, None)["source_quote"] == ""
    with pytest.raises(ValueError, match="SPAN"):
        m.build_span_packet("P1", "查询", "")


@pytest.mark.parametrize("field,value", [("start", False), ("start", 0.0), ("end", 4.0)])
def test_non_integer_offsets_rejected_even_when_python_equality_matches(field, value):
    m = module()
    packet = m.build_span_packet("P1", "查询", "处罚材料")
    packet["unit_parts"][0][field] = value
    with pytest.raises(ValueError, match="SPAN"):
        m.span_wire(packet)


@pytest.mark.parametrize("source", [None, 1, [], b"source"])
def test_invalid_source_types_have_explicit_contract_error(source):
    with pytest.raises(ValueError, match="INVALID_SPAN_SOURCE"):
        module().build_span_packet("P1", "查询", source)
