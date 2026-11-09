"""上下文只解释目标句，不把相邻句的独立风险复制到本句。"""

from dataclasses import replace

from domain.query_models import RiskQuerySentence, RiskSentenceContext
from domain.risk_models import RiskEvidence
from query.context import is_affirmative_answer, sentence_body
from risk.detection import RuleDetectionResult, RuleRiskDetector


def detect_with_context(
    detector: RuleRiskDetector,
    sentence: RiskQuerySentence,
    context: RiskSentenceContext,
) -> RuleDetectionResult:
    rules = detector.detect(sentence.text, sentence.scene)
    if context.stance == "negative_example":
        return RuleDetectionResult(
            risk_score=0.0,
            confidence=0.90,
            stage1_score=0.0,
            stage2_score=0.0,
            risk_labels=(),
            evidence=(
                RiskEvidence(
                    source="context_guard",
                    description="本句属于原文明确标示的反面或禁止话术，未发现独立实施表述",
                ),
            ),
            strong_signal=False,
            safe_signal=True,
        )
    if context.stance == "question":
        return replace(
            rules,
            risk_score=min(rules.risk_score, 0.19),
            confidence=min(rules.confidence, 0.45),
            strong_signal=False,
            safe_signal=False,
            evidence=(
                *rules.evidence,
                RiskEvidence(
                    source="context_role",
                    description="提问中的风险词不能直接归为销售承诺，需核对回答",
                ),
            ),
        )
    if rules.safe_signal:
        return rules
    if context.stance == "denial":
        # 不能把“不是……但保证……”等复合表述仅凭开头否定判为安全。
        body = sentence_body(sentence.text)
        if not any(word in body for word in ("但", "不过", "其实", "实际", "只是")):
            return replace(
                rules,
                risk_score=min(rules.risk_score, 0.19),
                confidence=min(rules.confidence, 0.60),
                strong_signal=False,
                evidence=(
                    *rules.evidence,
                    RiskEvidence(
                        source="context_role",
                        description="本句否定或纠正前述说法，需结合原问题理解",
                    ),
                ),
            )
    if is_affirmative_answer(sentence.text):
        previous = [
            source
            for source in context.source_sentences
            if not source.is_target
            and source.document_end_offset
            <= next(
                target.document_start_offset
                for target in context.source_sentences
                if target.is_target
            )
        ]
        if previous:
            question = previous[-1]
            if "？" in question.text or "?" in question.text or question.speaker_role == "question":
                question_rules = detector.detect(question.text, sentence.scene)
                if question_rules.risk_score >= 0.4 and not question_rules.safe_signal:
                    return replace(
                        question_rules,
                        risk_score=max(rules.risk_score, min(question_rules.risk_score, 0.72)),
                        confidence=0.60,
                        strong_signal=False,
                        safe_signal=False,
                        evidence=(
                            *rules.evidence,
                            RiskEvidence(
                                source="context_answer",
                                description=f"本句肯定回答前一问题，问题原文：{question.text}",
                                sentence_id=question.sentence_id,
                                document_start_offset=question.document_start_offset,
                                document_end_offset=question.document_end_offset,
                            ),
                        ),
                    )
    return rules
