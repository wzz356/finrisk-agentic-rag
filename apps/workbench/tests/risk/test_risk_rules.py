"""队员风险词库与修复后阈值的最小行为测试。"""

from pathlib import Path

import pytest

from domain.query_models import QueryScene
from domain.risk_models import RiskLevel
from risk.calibration import RiskScorer, RiskThresholder
from risk.detection import RuleRiskDetector, is_explicit_compliance_statement
from risk.pipeline import (
    _channel_label_supported,
    _explicit_content_labels,
    _looks_like_noncompliant_training_guidance,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "text",
    [
        "❤点击头像发送暗号【旅行险年卡】，告诉我你的目的地，快速获取1v1保障方案！",
        "发送口令【年卡】即可获取保障方案。",
        "发送关键词旅行险年卡，查看产品资料。",
        "推送旅行险年卡介绍和投保说明。",
        "递送保险卡和保单资料。",
    ],
)
def test_contact_instructions_are_not_gift_inducement(text: str) -> None:
    detector = RuleRiskDetector(
        PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
    )
    result = detector.detect(text)
    assert "赠送利益" not in result.risk_labels
    assert not any("gift_inducement" in item.description for item in result.evidence)


@pytest.mark.parametrize(
    "text",
    [
        "今天投保送购物卡。",
        "购买年卡赠送免费体检。",
        "购买保险奖励现金红包。",
        "投保后发送红包。",
        "投保成功发放加油卡。",
        "发送暗号【旅行险年卡】，投保再送购物卡。",
        "发送口令后投保送红包。",
    ],
)
def test_real_gifts_remain_detectable_after_contact_instructions(text: str) -> None:
    detector = RuleRiskDetector(
        PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
    )
    result = detector.detect(text)
    assert "赠送利益" in result.risk_labels
    assert any("gift_inducement" in item.description for item in result.evidence)


def test_real_keyword_database_detects_guaranteed_return() -> None:
    detector = RuleRiskDetector(
        PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
    )

    result = detector.detect("这款产品百分之百保证收益，不存在任何亏损风险。")

    assert result.risk_score >= 0.8
    assert "承诺收益" in result.risk_labels
    assert any(value.source == "rule_regex" for value in result.evidence)


def test_fallback_fusion_and_all_threshold_branches_are_reachable() -> None:
    scorer = RiskScorer()
    thresholder = RiskThresholder()

    assert scorer.fuse(1.0, 0.0, None) == 1.0
    assert thresholder.judge(0.90).level == RiskLevel.HIGH
    assert thresholder.judge(0.70).level == RiskLevel.MEDIUM
    assert thresholder.judge(0.50).level == RiskLevel.REVIEW
    assert thresholder.judge(0.25).level == RiskLevel.LOW
    assert thresholder.judge(0.10).level == RiskLevel.NONE


def test_explicit_content_labels_detect_fabricated_yield_without_overlabeling() -> None:
    positives = (
        "销售人员明知没有数据仍编造年化12%的收益率。",
        "负责人承认该收益率系伪造，并无历史业绩支持。",
    )

    assert all(_explicit_content_labels(text) == ["虚构收益率"] for text in positives)
    assert _explicit_content_labels("合同载明保证利率2.5%，非保证利益可能波动。") == []
    assert _explicit_content_labels("历史结算利率为3.0%，不代表未来收益。") == []
    assert _explicit_content_labels("复利滚存预计可达5.6%。") == []
    # A promise, an estimate and an unsupported bonus do not by themselves
    # establish fabrication of a historical or official return rate.
    for text in (
        "收益直接按12%写进合同，今天确认即可。",
        "过去三年平均收益8.8%，未来也会保持同样区间。",
        "年度补贴默认当成产品收益。",
        "内部加息名额实际上并不存在。",
        "5.6%由销售团队手工测算，并非产品历史结算利率。",
    ):
        assert "虚构收益率" not in _explicit_content_labels(text)


def test_training_channel_requires_noncompliant_guidance_evidence() -> None:
    risky = "话术不要直接说百分之八，可以改成几年后通常能多出一套家电钱。"
    compliant = "坐席必须使用已审批脚本，如实说明演示收益不代表未来结果。"

    assert _looks_like_noncompliant_training_guidance(risky)
    assert not _looks_like_noncompliant_training_guidance(compliant)


def test_compliance_guard_distinguishes_prohibition_from_evasion() -> None:
    detector = RuleRiskDetector(
        PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
    )
    compliant = "讲解人员不得承诺收益，必须逐项说明责任免除和退保损失。"
    risky = "退保损失无需介绍，红利不保证只是合同固定用语。"
    safe_live = "直播中不会展示收益预测，也不会承诺回本或固定领取结果。"

    assert is_explicit_compliance_statement(compliant)
    assert is_explicit_compliance_statement(safe_live)
    assert not is_explicit_compliance_statement(risky)
    safe_result = detector.detect(compliant)
    assert safe_result.safe_signal
    assert safe_result.risk_score == 0
    assert not detector.detect(risky).safe_signal


def test_strong_rules_cover_audit_and_sales_tail_risks() -> None:
    detector = RuleRiskDetector(
        PROJECT_ROOT / "data" / "risk" / "keyword_db" / "risk_keyword_db.json"
    )
    samples = {
        "家属可以代签投保文件。": "客户信息不真实",
        "电子签名时间早于验证码发送时间十二分钟。": "客户信息不真实",
        "七份回访录音均由同一员工手机上传。": "回访违规",
        "活动由没有保险销售资质的会务公司承办并讲解产品。": "委托无资质机构销售",
        "保障不限病史，门诊住院都能报，理赔不卡。": "销售误导",
        "客户可以按最高演示数值估算最终领取金额。": "夸大收益",
        "复利滚存预计可达5.6%，可覆盖退休补充需求。": "夸大收益",
        "由团队从服务费中补贴首年保费。": "回扣返佣",
        "A公司拒赔新闻、B公司演示收益和C公司投诉数量放在一起，不说明统计口径。": "不当比较",
    }

    for sample_text, expected_label in samples.items():
        result = detector.detect(sample_text)
        assert result.strong_signal, sample_text
        assert expected_label in result.risk_labels, sample_text


def test_channel_labels_require_violation_evidence_and_supported_scene() -> None:
    st5 = QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK
    st4 = QueryScene.ST4_MEETING_MATERIAL

    assert _channel_label_supported("回访违规", "回访时暂停录音，只保留第二次结果。", st5)
    assert not _channel_label_supported("回访违规", "回访前完整说明条款。", st5)
    assert not _channel_label_supported("回访违规", "回访时暂停录音，只保留第二次结果。", st4)
    assert _channel_label_supported("电销违规", "客户明确拒绝后仍换号码继续联系。", st5)
    assert not _channel_label_supported("电销违规", "坐席确认客户愿意后再介绍产品。", st5)
