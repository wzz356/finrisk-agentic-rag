"""多层证据融合与风险等级校准。"""

from __future__ import annotations

from dataclasses import dataclass

from domain.risk_models import RiskLevel


@dataclass(frozen=True, slots=True)
class RiskDecision:
    level: RiskLevel
    level_name: str
    action: str


class RiskScorer:
    """融合本地两阶段判断与语义复核；案例相似度不参与定性分数。"""

    def fuse(
        self,
        rule_score: float,
        retrieval_score: float = 0.0,
        llm_score: float | None = None,
        *,
        local_confidence: float = 0.0,
        strong_signal: bool = False,
    ) -> float:
        """保留 retrieval_score 参数兼容旧调用，但明确不把案例分用于风险定性。"""

        del retrieval_score
        if llm_score is None:
            score = rule_score
        elif strong_signal and local_confidence >= 0.85:
            score = rule_score * 0.55 + llm_score * 0.45
        else:
            score = rule_score * 0.25 + llm_score * 0.75
        if max(rule_score, llm_score or 0.0) < 0.2:
            score = min(score, 0.19)
        return round(max(0.0, min(1.0, score)), 4)


class RiskThresholder:
    """输出五档 Web 风险等级并修复原包低风险分支不可达的问题。"""

    def judge(self, score: float) -> RiskDecision:
        if score >= 0.85:
            return RiskDecision(RiskLevel.HIGH, "高风险", "拦截并人工复审")
        if score >= 0.60:
            return RiskDecision(RiskLevel.MEDIUM, "中风险", "标记并提示修改")
        if score >= 0.40:
            return RiskDecision(RiskLevel.REVIEW, "疑似风险", "建议人工复查")
        if score >= 0.20:
            return RiskDecision(RiskLevel.LOW, "低风险", "记录并观察")
        return RiskDecision(RiskLevel.NONE, "无明显风险", "放行")
