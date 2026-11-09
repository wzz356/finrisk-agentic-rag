"""Semantic boundaries that must hold independently of any benchmark query ID."""

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from application.query_service import QueryProcessingService
from query.classification import QuerySceneClassifier
from query.context import has_adopted_instruction
from query.review_frame import carry_document_frame
from risk.detection import RuleRiskDetector
from risk.local_calibration import LocalRiskCalibration, train_model
from risk.pipeline import _explicit_content_labels


def test_rejected_proposal_does_not_clear_an_independent_implemented_fact():
    text = (
        "客户活动方案（待审）：拟赠送礼品。该方案已被否决，尚未执行。\n\n"
        "另已查实：业务员将客户电话替换成自己的号码，作为客户本人号码提交。"
    )
    q = asyncio.run(
        QueryProcessingService(classifier=QuerySceneClassifier(None)).process_text(text)
    )
    contexts = carry_document_frame(q, {c.sentence_id: c for c in q.sentence_contexts})
    fact = next(s for s in q.sentences if "另已查实" in s.text)
    assert contexts[fact.sentence_id].stance != "negative_example"
    assert not any("方案" in s.text for s in contexts[fact.sentence_id].source_sentences)
    proposal = q.sentences[0]
    assert contexts[proposal.sentence_id].stance == "negative_example"


@pytest.mark.parametrize(
    "text, absent",
    [
        ("主管账号用于提交，代理人的核验和处置职责尚未调查。", "代理人管理不到位"),
        ("会务公司负责产品讲解，其资质与实际销售职责待核。", "委托无资质机构销售"),
        ("家属可以代操作快递签收，未冒充客户本人确认合同。", "客户信息不真实"),
        ("保险去年收益率4%，销售承诺未来也达到4%。", "虚构收益率"),
        ("多份回访录音背景声相同，录制安排和客户真实答复待核。", "客户信息不真实"),
    ],
)
def test_incomplete_related_facts_do_not_generate_confirmed_extra_labels(text, absent):
    assert absent not in _explicit_content_labels(text)


def test_explicit_fabrication_and_unlicensed_actual_sale_remain_supported():
    assert "虚构收益率" in _explicit_content_labels("负责人编造历史收益率数据用于保险海报。")
    assert "委托无资质机构销售" in _explicit_content_labels(
        "乙公司未取得保险代理资质，仍受委托向客户销售保险。"
    )


def test_compliant_return_disclosure_cannot_clear_an_independent_calling_violation():
    root = Path(__file__).resolve().parents[2]
    detector = RuleRiskDetector(root / "data/risk/keyword_db/risk_keyword_db.json")
    result = detector.detect("保险收益不保证，但客户明确拒绝后仍换号码继续联系推销保险。")
    assert not result.safe_signal
    assert "电销违规" in result.risk_labels


def test_adopted_quote_cannot_be_cleared_by_a_prohibition_on_correction():
    text = (
        "培训讲师先展示错误例句“本保险的投资收益保证5%”，"
        "随后要求新人今天向客户就按这句讲，不得改成非保证。"
    )
    root = Path(__file__).resolve().parents[2]
    assert has_adopted_instruction(text)
    assert (
        not RuleRiskDetector(root / "data/risk/keyword_db/risk_keyword_db.json")
        .detect(text)
        .safe_signal
    )
    query = asyncio.run(
        QueryProcessingService(classifier=QuerySceneClassifier(None)).process_text(text)
    )
    assert any(context.requires_review for context in query.sentence_contexts)
    for prohibition in (
        "不得要求新人按这句讲。",
        "要求新人不得照讲错误话术。",
        "严禁指导学员照讲错误话术。",
    ):
        assert not has_adopted_instruction(prohibition)


def test_final_corrective_frame_explains_earlier_quotes_in_its_own_scope():
    text = "讲义摘录：“本保险投资收益保证5%。”以上是培训中出现的错误示范，实际展业不得照此表述。"
    query = asyncio.run(
        QueryProcessingService(classifier=QuerySceneClassifier(None)).process_text(text)
    )
    contexts = carry_document_frame(query, {c.sentence_id: c for c in query.sentence_contexts})
    assert all(c.stance == "negative_example" for c in contexts.values())
    assert all(c.requires_review for c in contexts.values())


def test_training_split_and_literal_annotations_are_auditable():
    root = Path(__file__).resolve().parents[2] / "data/eval/query_boundaries_20261024"
    train = [json.loads(line) for line in (root / "train.jsonl").read_text("utf-8").splitlines()]
    valid = [
        json.loads(line) for line in (root / "validation.jsonl").read_text("utf-8").splitlines()
    ]
    assert len(train) == 72 and len(valid) == 24
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in valid}
    for r in train + valid:
        assert (r["decision_state"] == "risk") == bool(r["risk_labels"])
        for e in r["evidence"]:
            assert r["text"][e["start"] : e["end"]] == e["quote"]
    assert len({label for r in train for label in r["risk_labels"]}) == 27


def test_candidate_model_rejects_corrupt_weights_and_uses_relevant_contrast(tmp_path):
    path = tmp_path / "model.npz"
    train_model(
        ["保险保证投资收益8%", "保险投资收益不保证", "保险收益性质待核"],
        ["risk", "no_clear_risk", "insufficient_information"],
        [["保证收益"], [], []],
        ["保证收益"],
        path,
    )
    model = LocalRiskCalibration(path)
    examples = model.few_shot_examples("保险保证投资收益8%")
    assert len(examples) == 2 and examples[0]["decision_state"] != examples[1]["decision_state"]
    np.savez(path, weights=np.full((2048, 4), np.nan), labels=np.array(["保证收益"]))
    with pytest.raises(ValueError, match="无效数值"):
        LocalRiskCalibration(path)
