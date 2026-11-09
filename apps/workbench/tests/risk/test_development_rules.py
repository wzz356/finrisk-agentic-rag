from pathlib import Path

from risk.detection import RuleRiskDetector
from risk.pipeline import _current_fact_gaps

ROOT = Path(__file__).resolve().parents[2]


def detector():
    return RuleRiskDetector(ROOT / "data/risk/keyword_db/risk_keyword_db.json")


def test_prohibited_quoted_list_does_not_become_guarantee():
    result = detector().detect(
        "介绍保险责任时须逐项对照条款原文，不得使用“什么都保”“肯定赔”等表述。"
    )
    assert result.risk_score < 0.2
    assert not result.risk_labels


def test_guarantee_regex_does_not_cross_disclosure_clause():
    result = detector().detect("明确标注保证部分与非保证部分，不做稳赚、保本高息一类表述。")
    assert result.risk_score < 0.2
    assert not result.risk_labels


def test_separate_affirmative_after_prohibition_is_still_found():
    result = detector().detect("不得保证收益，但这款产品保证每年收益8%。")
    assert result.risk_score >= 0.4
    assert "保证收益" in result.risk_labels


def test_private_premium_account_is_management_candidate_not_fake_expense():
    result = detector().detect("保费直接打我私人账户，佣金也进个人卡。")
    assert "其他" in result.risk_labels
    assert "虚列费用套取资金" not in result.risk_labels
    assert any(e.source == "rule_regex" for e in result.evidence)


def test_missing_business_documents_are_kept_separate_from_no_risk():
    facts = _current_fact_gaps("未附代理合同、执业登记和对应保单清单，无法判断业务是否真实发生。")
    assert "机构或人员资格与执业登记" in facts
    assert "业务明细与费用凭证" in facts
    assert not _current_fact_gaps("客户应当完整阅读合同条款。")


def test_unapproved_established_business_premises_are_an_explicit_other_risk():
    result = detector().detect("该职场在设立和使用过程中，未经监管部门批准。")
    assert result.strong_signal and result.confidence >= 0.9
    assert "其他" in result.risk_labels


def test_prohibition_and_approval_question_are_not_unapproved_premises_facts():
    assert not detector().detect("严禁未经监管部门批准擅自设立分支机构。").strong_signal
    assert not detector().detect("设立新的营业场所是否需要监管部门批准？").strong_signal
