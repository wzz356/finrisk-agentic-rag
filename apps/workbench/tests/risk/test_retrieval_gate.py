"""案例相似度不得单独把普通文本判成风险。"""

from risk.calibration import RiskScorer


def test_retrieval_only_signal_stays_below_risk_threshold() -> None:
    scorer = RiskScorer()

    assert scorer.fuse(rule_score=0.0, retrieval_score=1.0, llm_score=None) < 0.2
    assert scorer.fuse(rule_score=0.0, retrieval_score=1.0, llm_score=0.0) < 0.2
