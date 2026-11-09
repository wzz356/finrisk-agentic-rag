"""Query 两阶段本地初判、动态LLM复核、门控检索和文档聚合流水线。"""

from __future__ import annotations

import asyncio
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from application.evidence_service import KnowledgeEvidenceService
from application.retrieval_service import CaseRetrievalService
from domain.agent_models import AgentExecutionTrace
from domain.knowledge_models import MatchedCase
from domain.query_models import (
    ContextSourceSentence,
    QueryProcessingResult,
    QueryScene,
    RiskQuerySentence,
    RiskSentenceContext,
    SentenceSourceMapping,
)
from domain.risk_models import (
    MechanismDecision,
    QueryAnalysisResult,
    QueryRiskSummary,
    RiskEvidence,
    RiskLabelFinding,
    RiskLevel,
    SentenceRiskAssessment,
)
from llm import ChatMessage, JsonChatRequest, LLMClient, LLMProviderError
from query.context import build_retrieval_query, build_sentence_contexts, is_affirmative_answer
from query.review_frame import carry_document_frame
from risk.agent_tools import QueryToolBudget, RiskEvidenceTools
from risk.calibration import RiskScorer, RiskThresholder
from risk.context import detect_with_context
from risk.detection import RuleDetectionResult, RuleRiskDetector
from risk.label_guards import social_label_supported
from risk.longtext import screen_document
from risk.mechanism_guards import apply_mechanism_guard
from risk.prompt_context import compact_review_payload
from risk.recommendations import remediation_suggestions
from risk.taxonomy import taxonomy_prompt


@dataclass(frozen=True, slots=True)
class _PreliminaryAssessment:
    sentence: RiskQuerySentence
    mapping: SentenceSourceMapping
    context: RiskSentenceContext
    rules: RuleDetectionResult


@dataclass(frozen=True, slots=True)
class _PrefetchedRetrieval:
    """前置模式下、判定前就已取得的类案检索结果。

    ``matches`` 为空表示未过注入门槛或检索无结果；``failed`` 为真时只记告警，
    不影响后续判定（宁可少一个参考案例，也不能因检索失败中断分析）。

    ``used_tags`` 记录检索时实际使用的标签约束：前置检索用的是本地规则标签
    （可能为空，走纯语义召回），而判定后门控用的是融合后的最终标签。两者不同
    时不能复用结果，否则会漏掉标签过滤。
    """

    sentence_id: str
    matches: tuple[MatchedCase, ...]
    retrieval_query: str
    injected: bool
    failed: bool
    used_tags: frozenset[str] = frozenset()


class _LLMItem(BaseModel):
    sentence_id: str
    risk_score: float = Field(ge=0, le=1)
    risk_labels: list[str] = Field(default_factory=list, max_length=27)
    evidence: list[str] = Field(default_factory=list, max_length=12)
    reasoning: str = Field(min_length=1, max_length=500)
    decision_state: Literal["risk", "no_clear_risk", "insufficient_information"] | None = None
    candidate_labels: list[str] = Field(default_factory=list, max_length=27)
    missing_facts: list[str] = Field(default_factory=list, max_length=8)
    condition_assessments: list[dict[str, object]] = Field(default_factory=list, max_length=10)
    mechanism_assessments: list[MechanismDecision] = Field(default_factory=list, max_length=12)
    label_assessments: list[dict[str, object]] = Field(default_factory=list, max_length=27)
    checked_mechanism_ids: list[str] = Field(default_factory=list, max_length=12)
    review_depth: Literal["screen", "detail"] = "detail"


def _current_fact_gaps(text: str) -> list[str]:
    if not re.search(r"无法判断|无法确定|不能确定|不清楚|暂不能认定|尚不能认定|尚待核实", text):
        return []
    if re.search(r"反面|教学|错误示例|(?:先不|暂不|不得)(?:录制|发布|上线)|合规提醒", text):
        return []
    if not re.search(r"未(?:附|提供|见)|缺少|尚无|没有.{0,10}(?:依据|材料|凭证|合同|记录)", text):
        return []
    return [
        name
        for pattern, name in [
            (r"合同|条款|权益", "合同约定及权益范围"),
            (r"资质|资格|执业|登记", "机构或人员资格与执业登记"),
            (r"费用|列支|中介业务|凭证|保单清单", "业务明细与费用凭证"),
            (r"数据|比较|利率|演示", "数据来源、比较口径及演示依据"),
            (r"税|避债|身份", "适用身份、款项性质及依据"),
        ]
        if re.search(pattern, text)
    ]


class _LLMBatch(BaseModel):
    items: list[_LLMItem]


class RiskAssessmentPipeline:
    """先判断当前文本风险，再按阈值和标签检索案例。

    检索时机可配置：

    - ``post``（默认）：判定后门控检索，只有风险分达标且有标签的句子才查案例，
      案例仅作为佐证证据，不进入 LLM 提示词，也不参与打分；
    - ``pre``：判定前先按本地规则标签检索类案，过门槛的案例摘要注入 LLM 上下文
      供其参考。代价是每条待复核句都会触发云端向量与重排调用。

    两种模式下案例相似度都不进入 :meth:`RiskScorer.fuse`，避免"库里有个相似案例"
    把本来合规的文本推高风险档。
    """

    def __init__(
        self,
        *,
        rule_detector: RuleRiskDetector,
        retrieval_service: CaseRetrievalService,
        taxonomy: tuple[str, ...],
        llm_client: LLMClient | None,
        taxonomy_descriptions: dict[str, str] | None = None,
        risk_display_threshold: float = 0.20,
        retrieval_threshold: float = 0.40,
        local_direct_confidence: float = 0.90,
        whole_query_llm_chars: int = 6_000,
        llm_batch_chars: int = 6_000,
        llm_batch_items: int = 12,
        llm_concurrency: int = 4,
        retrieval_timing: str = "post",
        semantic_context_gate: float = 0.50,
        retrieval_top_k: int = 3,
        compact_context: bool = True,
        evidence_service: KnowledgeEvidenceService | None = None,
        agent_retrieval_enabled: bool = True,
        longtext_screening: bool = True,
    ) -> None:
        if not 0 <= risk_display_threshold <= retrieval_threshold <= 1:
            raise ValueError("风险展示阈值和检索阈值不合法")
        if retrieval_timing not in {"post", "pre", "hybrid", "adaptive"}:
            raise ValueError("retrieval_timing 只能是 post、pre、hybrid 或 adaptive")
        if not 0 <= semantic_context_gate <= 1:
            raise ValueError("语义上下文注入门槛必须在 0 到 1 之间")
        if not 1 <= retrieval_top_k <= 20:
            raise ValueError("retrieval_top_k 必须在 1 到 20 之间")
        self.rule_detector = rule_detector
        self.retrieval_service = retrieval_service
        self.taxonomy = tuple(label for label in taxonomy if label)
        self._taxonomy_set = set(self.taxonomy)
        self.taxonomy_descriptions = taxonomy_descriptions or {}
        self.llm_client = llm_client
        self.risk_display_threshold = risk_display_threshold
        self.retrieval_threshold = retrieval_threshold
        self.local_direct_confidence = local_direct_confidence
        self.whole_query_llm_chars = whole_query_llm_chars
        self.llm_batch_chars = llm_batch_chars
        self.llm_batch_items = llm_batch_items
        self.llm_concurrency = llm_concurrency
        self.adaptive = retrieval_timing == "adaptive"
        self.retrieval_timing = "hybrid" if self.adaptive else retrieval_timing
        self.semantic_context_gate = semantic_context_gate
        self.retrieval_top_k = retrieval_top_k
        self.compact_context = compact_context
        self.evidence_service = evidence_service
        self.agent_retrieval_enabled = agent_retrieval_enabled
        self.longtext_screening = longtext_screening
        self.scorer = RiskScorer()
        self.thresholder = RiskThresholder()

    async def analyze(self, query: QueryProcessingResult) -> QueryAnalysisResult:
        analysis_started = perf_counter()
        local_started = perf_counter()
        mappings = {mapping.sentence_id: mapping for mapping in query.source_mappings}
        contexts = {
            context.sentence_id: context
            for context in (
                query.sentence_contexts
                or build_sentence_contexts(
                    query.source_text, query.sentences, query.source_mappings
                )
            )
        }
        contexts = carry_document_frame(query, contexts)
        rule_contexts = contexts
        if self.adaptive:
            contexts = self._carry_scope_constraints(query, contexts)
        preliminary = tuple(
            _PreliminaryAssessment(
                sentence=sentence,
                mapping=mappings[sentence.sentence_id],
                context=contexts[sentence.sentence_id],
                rules=detect_with_context(
                    self.rule_detector, sentence, rule_contexts[sentence.sentence_id]
                ),
            )
            for sentence in query.sentences
        )
        local_rules_ms = (perf_counter() - local_started) * 1000
        review_ids = {
            item.sentence.sentence_id
            for item in preliminary
            if not self._local_direct(item.rules, item.context)
        }
        warnings = list(query.extraction.warnings)
        warnings.extend(
            dict.fromkeys(warning for context in contexts.values() for warning in context.warnings)
        )
        llm_outputs: dict[str, _LLMItem] = {}
        agent_runs: list[AgentExecutionTrace] = []
        tool_budget = QueryToolBudget()
        # 前置模式：先用本地规则标签检索类案，过门槛的摘要注入 LLM 上下文。
        prefetched: dict[str, _PrefetchedRetrieval] = {}
        pre_retrieval_started = perf_counter()
        if (
            self.retrieval_timing in {"pre", "hybrid"}
            and self.llm_client is not None
            and review_ids
        ):
            prefetched = await self._preretrieve(preliminary, contexts, review_ids)
            if any(item.failed for item in prefetched.values()):
                warnings.append("判定前的类案检索部分不可用，已按无参考案例继续判断")
        pre_retrieval_ms = (perf_counter() - pre_retrieval_started) * 1000
        llm_started = perf_counter()
        if self.llm_client is not None and review_ids:
            llm_outputs, llm_warnings, agent_runs = await self._llm_assess(
                preliminary, review_ids, prefetched, tool_budget
            )
            warnings.extend(llm_warnings)
        elif self.llm_client is None and review_ids:
            warnings.append(
                f"未配置大模型，{len(review_ids)}个中低置信句子已使用本地两阶段结果降级"
            )

        llm_review_ms = (perf_counter() - llm_started) * 1000
        finalize_started = perf_counter()
        assessments = [
            self._finalize(item, llm_outputs.get(item.sentence.sentence_id)) for item in preliminary
        ]
        assessments = self._apply_document_channel_labels(query, assessments)
        finalize_ms = (perf_counter() - finalize_started) * 1000
        retrieval_started = perf_counter()
        assessments = await self._retrieve_eligible(assessments, contexts, prefetched)
        assessments = [
            item.model_copy(
                update={
                    "remediation_suggestions": remediation_suggestions(
                        item.risk_labels, item.decision_state, item.missing_facts
                    )
                }
            )
            for item in assessments
        ]
        if any(item.retrieval_status == "unavailable" for item in assessments):
            warnings.append("部分语句的案例检索不可用，请独立核对风险；未返回类案不代表没有风险")
        retrieval_ms = (perf_counter() - retrieval_started) * 1000
        aggregate_started = perf_counter()
        summary = self._summarize(
            assessments,
            llm_outputs=llm_outputs,
            review_ids=review_ids,
            warnings=warnings,
        )
        aggregate_ms = (perf_counter() - aggregate_started) * 1000
        summary = summary.model_copy(
            update={
                "stage_timings_ms": {
                    "local_rules": round(local_rules_ms, 3),
                    "pre_retrieval": round(pre_retrieval_ms, 3),
                    "llm_review": round(llm_review_ms, 3),
                    "finalize": round(finalize_ms, 3),
                    "case_retrieval": round(retrieval_ms, 3),
                    "aggregate": round(aggregate_ms, 3),
                    "total": round((perf_counter() - analysis_started) * 1000, 3),
                },
                "agent_runs": agent_runs,
                "evidence_tool_events": tool_budget.events,
                "retrieval_strategy": "adaptive" if self.adaptive else self.retrieval_timing,
                "mechanism_coverage": self._coverage_report(preliminary, llm_outputs),
            }
        )
        return QueryAnalysisResult(
            **query.model_dump(exclude={"schema_version", "sentence_contexts"}),
            sentence_contexts=list(contexts.values()),
            schema_version="4.0",
            risk_summary=summary,
            risk_assessments=assessments,
        )

    def _local_direct(self, rules: RuleDetectionResult, context: RiskSentenceContext) -> bool:
        if rules.strong_signal and re.search(
            r"是否|无法判断|不能判断|尚待核实|未提供|真实性.{0,8}待核|一旦执行|如果执行",
            context.query_text,
        ):
            return False
        if self.adaptive and self.llm_client is not None and len(rules.risk_labels) > 1:
            return False  # A strong binary signal does not settle all related label boundaries.
        if self.adaptive and self.evidence_service is not None:
            mids = self.evidence_service.planner.reading_mechanisms(context.query_text)
            if (
                "M05" in mids
                and rules.strong_signal
                and any("合同" in source.text for source in context.source_sentences)
            ):
                return False
            if len(mids) > 1 and context.stance != "negative_example":
                return False
            if (
                mids
                and rules.strong_signal
                and any(
                    re.search(
                        r"未说明|不清楚|无法确定|不能确定|没(?:有|显示|拍到|写)|不明|不知|尚未",
                        source.text,
                    )
                    for source in context.source_sentences
                )
            ):
                return False
        if context.requires_review or self._hybrid_semantic_guard(rules, context):
            return False
        return rules.safe_signal or (
            rules.strong_signal and rules.confidence >= self.local_direct_confidence
        )

    def _hybrid_semantic_guard(
        self, rules: RuleDetectionResult, context: RiskSentenceContext
    ) -> bool:
        """A strong keyword is not enough to resolve negation or tax applicability."""
        if self.retrieval_timing != "hybrid" or self.evidence_service is None:
            return False
        if not rules.strong_signal:
            return False
        gift_labels = {"赠送利益", "合同外利益", "回扣返佣", "诱导投保"}
        gift_negation = bool(gift_labels.intersection(rules.risk_labels)) and bool(
            re.search(r"不是|并非|不|没有|未|不得|禁止", context.query_text)
        )
        return "避债避税" in rules.risk_labels or gift_negation

    def _candidate_mechanisms(self, item: _PreliminaryAssessment) -> list[str]:
        if self.evidence_service is None:
            return []
        mids = self.evidence_service.planner.reading_mechanisms(item.sentence.text)
        if re.search(r"这行|这页|该页|这一栏|本人确认件|已签文件夹|替您写", item.sentence.text):
            mids = list(
                dict.fromkeys(
                    [
                        *mids,
                        *self.evidence_service.planner.reading_mechanisms(
                            build_retrieval_query(item.context)
                        ),
                    ]
                )
            )
        return mids

    def _carry_scope_constraints(
        self,
        query: QueryProcessingResult,
        contexts: dict[str, RiskSentenceContext],
    ) -> dict[str, RiskSentenceContext]:
        """Carry a bounded set of explicit uncertainty/contract qualifiers within one scope."""
        if self.evidence_service is None:
            return contexts
        planner = self.evidence_service.planner
        by_scope: dict[str, list[RiskSentenceContext]] = defaultdict(list)
        for context in contexts.values():
            by_scope[context.scope_id].append(context)
        result = {}
        qualifier = re.compile(
            r"未说明|不清楚|无法确定|不能确定|没(?:有|显示|拍到|写)|不明|不知|尚未|如实|合同约定|合同列明"
        )
        for sid, context in contexts.items():
            mids = set(planner.reading_mechanisms(context.query_text))
            existing = {source.sentence_id for source in context.source_sentences}
            additions: list[ContextSourceSentence] = []
            used = 0
            for candidate in by_scope[context.scope_id]:
                if candidate.sentence_id in existing or not qualifier.search(candidate.query_text):
                    continue
                candidate_mids = set(planner.reading_mechanisms(candidate.query_text))
                if mids and candidate_mids and not mids.intersection(candidate_mids):
                    continue
                target = next(source for source in candidate.source_sentences if source.is_target)
                if used + len(target.text) > 600 or len(additions) >= 3:
                    continue
                additions.append(target.model_copy(update={"is_target": False}))
                used += len(target.text)
            result[sid] = (
                context.model_copy(
                    update={
                        "source_sentences": sorted(
                            [*context.source_sentences, *additions],
                            key=lambda source: source.document_start_offset,
                        ),
                    }
                )
                if additions
                else context
            )
        return result

    def _needs_pre(self, item: _PreliminaryAssessment) -> bool:
        """Use pre for concrete ambiguity; ordinary/explicit statements stay on post."""
        if not self._candidate_mechanisms(item) or item.rules.safe_signal:
            return False
        return (
            self._hybrid_semantic_guard(item.rules, item.context)
            or bool(
                re.search(
                    r"合同|授权|受托|停售|固定.{0,10}(?:领取|给付|金额)|免税|资质",
                    item.sentence.text,
                )
            )
            or (item.context.requires_review and 0 < item.rules.risk_score < 0.60)
        )

    def _coverage_report(
        self,
        preliminary: tuple[_PreliminaryAssessment, ...],
        outputs: dict[str, _LLMItem],
    ) -> dict[str, object]:
        if not self.adaptive:
            return {}
        candidates: dict[str, list[str]] = {}
        unchecked: dict[str, list[str]] = {}
        local_checked: dict[str, list[str]] = {}
        for item in preliminary:
            sid = item.sentence.sentence_id
            mids = self._candidate_mechanisms(item)
            if not mids:
                continue
            candidates[sid] = mids
            if self._local_direct(item.rules, item.context):
                local_checked[sid] = mids
                continue
            checked = (
                {m.mechanism_id for m in outputs[sid].mechanism_assessments}
                | set(outputs[sid].checked_mechanism_ids)
                if sid in outputs
                else set()
            )
            absent = [mid for mid in mids if mid not in checked]
            if absent:
                unchecked[sid] = absent
        return {
            "candidates": candidates,
            "unchecked": unchecked,
            "local_rule_checks": local_checked,
            "checking_origin": "model_reported_with_literal_quote_validation",
            "all_27_labels_covered": False,
            "whole_input_risk_free_proved": False,
        }

    async def _preretrieve(
        self,
        preliminary: tuple[_PreliminaryAssessment, ...],
        contexts: dict[str, RiskSentenceContext],
        review_ids: set[str],
    ) -> dict[str, _PrefetchedRetrieval]:
        """前置模式：在判定前为待复核句检索可参考的类案。

        与判定后的门控检索有本质区别：这里**不以本地标签为前提**，否则弱信号句
        （本地分为 0、无标签）将永远拿不到参考案例，前置注入形同虚设。本地规则
        给出标签时用它收窄候选，未给出时退化为无标签的语义召回——这正是独立版
        ``SemanticDetector`` 的行为，标签桥接仍留在判定之后。

        只有相关分过 ``semantic_context_gate`` 的结果才标记 ``injected``；其余
        仍保留在字典里，供判定后门控阶段复用，避免同一句重复调用云端。
        """

        targets = [item for item in preliminary if item.sentence.sentence_id in review_ids]
        if self.adaptive:
            targets = [item for item in targets if self._needs_pre(item)]
        grouped_targets: dict[tuple[str, ...], list[_PreliminaryAssessment]] = {}
        if self.retrieval_timing == "hybrid":
            for item in targets:
                query_text = (
                    item.sentence.text if self.adaptive else build_retrieval_query(item.context)
                )
                mids = (
                    (
                        self.evidence_service.planner.reading_mechanisms(query_text)
                        if self.adaptive
                        else self.evidence_service.planner.mechanisms(query_text)
                    )
                    if self.evidence_service
                    else []
                )
                if not (mids or item.rules.risk_labels):
                    continue
                key = (item.context.scope_id, *mids) if mids else (item.sentence.sentence_id,)
                grouped_targets.setdefault(key, []).append(item)
            targets = [
                max(group, key=lambda item: item.rules.risk_score)
                for group in grouped_targets.values()
            ]
            if self.adaptive:
                targets.sort(key=lambda item: (-item.rules.risk_score, item.sentence.sentence_id))
                targets = targets[:6]
                selected = {item.sentence.sentence_id for item in targets}
                grouped_targets = {
                    key: group
                    for key, group in grouped_targets.items()
                    if max(group, key=lambda item: item.rules.risk_score).sentence.sentence_id
                    in selected
                }
        semaphore = asyncio.Semaphore(self.llm_concurrency)

        async def retrieve(item: _PreliminaryAssessment) -> _PrefetchedRetrieval:
            sentence_id = item.sentence.sentence_id
            retrieval_query = (
                item.sentence.text
                if self.adaptive
                else build_retrieval_query(contexts[sentence_id])
            )
            # 有本地标签则收窄，无标签传 None 走纯语义召回。
            labels = (
                None if self.retrieval_timing == "hybrid" else set(item.rules.risk_labels) or None
            )
            async with semaphore:
                try:
                    matches = await asyncio.to_thread(
                        self.retrieval_service.search,
                        retrieval_query,
                        top_k=self.retrieval_top_k,
                        risk_tags=labels,
                    )
                except (OSError, ValueError, RuntimeError):
                    return _PrefetchedRetrieval(
                        sentence_id,
                        (),
                        retrieval_query,
                        injected=False,
                        failed=True,
                        used_tags=frozenset(labels or ()),
                    )
            injected = bool(matches) and matches[0].score >= self.semantic_context_gate
            return _PrefetchedRetrieval(
                sentence_id,
                tuple(matches),
                retrieval_query,
                injected=injected,
                failed=False,
                used_tags=frozenset(labels or ()),
            )

        results = await asyncio.gather(*(retrieve(item) for item in targets))
        indexed = {entry.sentence_id: entry for entry in results}
        if grouped_targets:
            for group in grouped_targets.values():
                representative = max(group, key=lambda item: item.rules.risk_score)
                value = indexed[representative.sentence.sentence_id]
                for item in group:
                    indexed[item.sentence.sentence_id] = replace(
                        value, sentence_id=item.sentence.sentence_id
                    )
        return indexed

    def _system_prompt(self, case_context_enabled: bool) -> str:
        """构造系统提示词。

        ``case_context_enabled`` 为假时与改造前逐字一致，保证 post 模式零行为变化；
        为真时把"不得使用历史案例"改为"可参考但不得据以改变分数或伪造证据"。
        """

        head = (
            "你是保险营销与经营合规审查员。参考历史案例仅用于理解违规类型，"
            "判定仍以待审文本本身为准，不得凭案例相似度改变风险分。"
            if case_context_enabled
            else "你是保险营销与经营合规审查员。只判断待审文本本身，不得使用或猜测历史案例。"
        )
        return (
            head + "逐句给出0到1风险分、风险标签、原文证据短语和简短理由。"
            "以context中的说话人、立场、作用域和逐字原文解释目标句。"
            "客户提问不等于销售承诺；简短回答必须结合问题判断。"
            "反面示例、否定纠正和禁止话术不能按肯定实施判断。"
            "不同产品和不同段落的承诺、限制不得互相借用；"
            "相邻独立风险不能复制到本句，普通免责声明也不能抵消已作出的承诺。"
            "context_truncated=true时保留未知。证据必须逐字来自该句"
            "context.source_sentences，不能来自同批另一段或另一个产品。"
            "逐标签核对成立条件，只输出当前事实直接支持的标签，不按上下位、近义或关联关系扩展。"
            "培训材料和处罚事实中已发生的违规同样要识别；后续整改不抹去已记录的违规事实。"
            "先确定本次待审对象和叙述立场：全文明确的第三方反例、被否决提案、定稿前纠错"
            "和如实历史处罚披露，不作为当前违规；本机构已核查事实、仍执行的错误要求仍须识别。"
            "全文立场引文仅解释目标句，不能把别段的动作和产品条件移到目标句。"
            "已明确成立的事实不因金额、责任人、覆盖范围或其他机制材料尚待核实而变成全部未知。"
            "正常流程没附额外凭证不自动成为信息不足；具体主张依赖缺失的资格、合同或版本才待核。"
            "无明显风险时risk_score应低于0.20且标签为空。"
            "带有明确条件、风险提示、合同限定、个人经历限定或正常流程描述时不要误判。"
            "社交文案中个人已获赔、投保成功、求助、性价比感受不是普遍赔付承诺；"
            "须区分‘我这次赔了’与‘任何病都赔、无需任何条件’。"
            "免健康告知、不记名雇主险、保证续保、0免赔、责任内100%赔付、家庭单折扣"
            "均可能是正常产品设计；不能凭缺少合同附件认定虚假、隐瞒或偏离备案费率，"
            "不能把不记名承保等同伪造客户信息，把免健告等同教唆不如实告知。"
            "只有目标句确有特定条款矛盾且依据不足时才列具体待核事实；普通产品说明不凭空待核。"
            "夸大收益只涉及投资回报，不能用于保额、报销比例、省保费、理赔款或医疗保障效果。"
            "家庭费率折扣、会员权益、合同服务不自动等于赠送利益或回扣；"
            "明确‘下单送礼包、投保赠门诊保额’仍须识别赠送利益，未证明合同外不得另加合同外利益。"
            "健康告知宽松、有机会承保、责任内限定不自动构成弱化风险提示或隐瞒重要信息；"
            "这两标签需要主动淡化风险或隐瞒条款的具体动作，不能从文案未列全条款推定。"
            "正常推荐、抓紧购买、适用年龄提醒不能单独证明不当施压；"
            "‘最、最高、100%’须看修饰对象，合同限额、比例、个人评价不自动绝对化。"
            "‘AI不会骗人、AI推荐就是最好保障’要核虚假背书和绝对化断言，不是贬低竞品；"
            "国家背书、全都能赔等明确宣传主张仍应按自身内容核查，文末免责不抵消。"
            "医疗险声称可用现金价值自动垫交保费并保证不断保，须核对该产品是否具有现金价值"
            "及垫交条款，不能直接套用其他险种功能；缺少依据应给具体待核事实。"
            "尤其注意：材料引用风险词来禁止该行为，不等于材料正在实施该行为。"
            "例如‘不得承诺收益、不能把演示当保证、客户拒绝后停止联系、必须完整披露’"
            "均属于合规要求，应判低于0.20且标签为空。相反，‘无需介绍免责、先投保后看条款、"
            "不区分保证和非保证、由家属代签’属于明确风险。"
            "结果对象字段为："
            '{"items":[{"sentence_id":"...","risk_score":0.0,'
            '"decision_state":"risk|no_clear_risk|insufficient_information",'
            '"risk_labels":[],"candidate_labels":[],"missing_facts":[],'
            '"evidence":[],"reasoning":"简短中文理由"}]}。'
            "明确风险填risk；正常内容填no_clear_risk；具体主张的必要依据缺失填"
            "insufficient_information，risk_labels留空、candidate_labels填写待核标签、"
            "missing_facts列出所需事实；不能把未核实直接写成虚假事实。\n"
            "允许的27类标签及定义：\n"
            f"{taxonomy_prompt(self.taxonomy, self.taxonomy_descriptions)}"
        )

    async def _llm_assess(
        self,
        preliminary: tuple[_PreliminaryAssessment, ...],
        review_ids: set[str],
        prefetched: dict[str, _PrefetchedRetrieval] | None = None,
        tool_budget: QueryToolBudget | None = None,
    ) -> tuple[dict[str, _LLMItem], list[str], list[AgentExecutionTrace]]:
        if self.llm_client is None:
            return {}, [], []
        prefetched = prefetched or {}
        screened: dict[str, _LLMItem] = {}
        screening_traces: list[AgentExecutionTrace] = []
        screening_warnings: list[str] = []
        screened_pass = False
        detailed = preliminary
        if (
            self.longtext_screening
            and len(preliminary) >= 20
            and sum(len(item.sentence.text) for item in preliminary) >= 1600
        ):
            try:
                screen = await screen_document(
                    self.llm_client, tuple(item.sentence for item in preliminary), self.taxonomy
                )
                # The model cannot discard local signals or unresolved context dependencies.
                forced = {
                    item.sentence.sentence_id
                    for item in preliminary
                    if item.rules.strong_signal
                    or item.rules.risk_labels
                    or item.rules.risk_score >= 0.20
                    or item.context.requires_review
                    or _current_fact_gaps(item.sentence.text)
                }
                normal_ids = (screen.normal_ids & review_ids) - forced
                screened = {
                    item.sentence.sentence_id: _LLMItem(
                        sentence_id=item.sentence.sentence_id,
                        risk_score=0.03,
                        decision_state="no_clear_risk",
                        evidence=[item.sentence.text],
                        reasoning="全文语义初筛为正常内容，保留原文定位。",
                        review_depth="screen",
                    )
                    for item in preliminary
                    if item.sentence.sentence_id in normal_ids
                }
                review_ids = review_ids - normal_ids
                detailed = tuple(
                    item for item in preliminary if item.sentence.sentence_id in review_ids
                )
                screening_traces.extend(screen.traces)
                screened_pass = True
            except LLMProviderError:
                screening_warnings.append("全文初筛未完成，已逐句完整复核")
        if not review_ids:
            return screened, screening_warnings, screening_traces
        batches = self._llm_batches(detailed, review_ids)
        semaphore = asyncio.Semaphore(self.llm_concurrency)

        async def run_batch(
            batch: tuple[_PreliminaryAssessment, ...],
            target_ids: set[str],
        ) -> tuple[dict[str, _LLMItem], AgentExecutionTrace | None, str | None]:
            async with semaphore:
                try:
                    outputs, trace = await self._llm_batch(
                        batch,
                        target_ids,
                        prefetched,
                        tool_budget,
                        longtext_review=screened_pass,
                    )
                    return outputs, trace, None
                except LLMProviderError as error:
                    return {}, None, str(error)

        responses = await asyncio.gather(
            *(run_batch(batch, target_ids) for batch, target_ids in batches)
        )
        outputs: dict[str, _LLMItem] = dict(screened)
        warnings: list[str] = list(screening_warnings)
        traces: list[AgentExecutionTrace] = list(screening_traces)
        for batch_outputs, trace, error in responses:
            outputs.update(batch_outputs)
            if trace is not None:
                traces.append(trace)
            if error:
                warnings.append(f"大模型批次复核失败，已对该批使用本地结果：{error}")
        if self.adaptive:
            # One targeted repair pass, retaining every valid first-pass result if repair fails.
            repair_ids = (review_ids - set(outputs)) | {
                item.sentence.sentence_id
                for item in preliminary
                if item.sentence.sentence_id in outputs
                and outputs[item.sentence.sentence_id].review_depth != "screen"
                and set(self._candidate_mechanisms(item))
                - {m.mechanism_id for m in outputs[item.sentence.sentence_id].mechanism_assessments}
                - set(outputs[item.sentence.sentence_id].checked_mechanism_ids)
            }
            if repair_ids:
                repairs = []
                # Repair only missing/unchecked targets, retaining each target's source context.
                # The whole-document shortcut in _llm_batches may also contain valid background
                # sentences, which must not be silently promoted to repair targets.
                for item in preliminary:
                    if item.sentence.sentence_id in repair_ids:
                        repairs.append(((item,), {item.sentence.sentence_id}))
                repaired = await asyncio.gather(*(run_batch(batch, ids) for batch, ids in repairs))
                for values, trace, error in repaired:
                    outputs.update(values)
                    if trace:
                        traces.append(trace)
                    if error:
                        warnings.append("一次定向覆盖补查未完成，保留首轮判断和未检查机制")
                remaining = self._coverage_report(preliminary, outputs).get("unchecked", {})
                if remaining:
                    warnings.append(
                        "部分候选机制尚未完成检查，详见机制覆盖清单；未检查不等于无风险"
                    )
        missing = review_ids - set(outputs)
        if missing:
            warnings.append(f"{len(missing)}个待复核句子未取得有效大模型输出")
        return outputs, warnings, traces

    def _llm_batches(
        self,
        preliminary: tuple[_PreliminaryAssessment, ...],
        review_ids: set[str],
    ) -> list[tuple[tuple[_PreliminaryAssessment, ...], set[str]]]:
        total_chars = sum(len(item.sentence.text) for item in preliminary)
        if (
            total_chars <= self.whole_query_llm_chars
            and len(review_ids) <= self.llm_batch_items
            and self._payload_chars(preliminary) <= self.llm_batch_chars
        ):
            return [(preliminary, set(review_ids))]

        batches: list[tuple[tuple[_PreliminaryAssessment, ...], set[str]]] = []
        current: list[_PreliminaryAssessment] = []
        for item in preliminary:
            if item.sentence.sentence_id not in review_ids:
                continue
            item_chars = self._payload_chars(tuple([*current, item]))
            if current and (
                len(current) >= self.llm_batch_items or item_chars > self.llm_batch_chars
            ):
                batch = tuple(current)
                batches.append((batch, {value.sentence.sentence_id for value in batch}))
                current = []
            current.append(item)
        if current:
            batch = tuple(current)
            batches.append((batch, {value.sentence.sentence_id for value in batch}))
        return batches

    def _payload_chars(self, batch: tuple[_PreliminaryAssessment, ...]) -> int:
        items = [_llm_input(item) for item in batch]
        payload = (
            compact_review_payload(items, lean=self.adaptive) if self.compact_context else items
        )
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))

    async def _llm_batch(
        self,
        batch: tuple[_PreliminaryAssessment, ...],
        target_ids: set[str],
        prefetched: dict[str, _PrefetchedRetrieval] | None = None,
        tool_budget: QueryToolBudget | None = None,
        *,
        longtext_review: bool = False,
    ) -> tuple[dict[str, _LLMItem], AgentExecutionTrace | None]:
        if self.llm_client is None:
            return {}, None
        prefetched = prefetched or {}
        # 只有确实注入了案例摘要时才放开"不得使用历史案例"的约束，
        # 否则保持原提示词，确保 post 模式行为与改造前完全一致。
        injected_ids = {
            sentence_id
            for sentence_id, item in prefetched.items()
            if item.injected and item.matches and sentence_id in target_ids
        }
        case_context_enabled = bool(injected_ids)
        sentences_payload = [
            _llm_input(item, _reference_cases(prefetched, item.sentence.sentence_id))
            for item in batch
        ]
        if self.adaptive:
            for preliminary_item, payload in zip(batch, sentences_payload, strict=True):
                payload["mechanism_candidates"] = self._candidate_mechanisms(preliminary_item)
        provider_payload = (
            compact_review_payload(sentences_payload, lean=self.adaptive)
            if self.compact_context
            else sentences_payload
        )
        batch_payload = json.dumps(provider_payload, ensure_ascii=False, separators=(",", ":"))
        knowledge = []
        if self.evidence_service is not None and self.retrieval_timing == "hybrid":
            knowledge = self.evidence_service.rule_context(
                "\n".join(item.sentence.text for item in batch)
            )
            if self.adaptive and self.evidence_service.p0 is not None:
                mids = list(
                    dict.fromkeys(mid for item in batch for mid in self._candidate_mechanisms(item))
                )
                knowledge = self.evidence_service.p0.guides(mids)
        extra_prompt = ""
        candidate_model = getattr(self.rule_detector, "candidate_model", None)
        if candidate_model is not None:
            examples = candidate_model.few_shot_examples(
                "\n".join(item.sentence.text for item in batch)
            )
            if examples:
                extra_prompt += "\n以下开发集示例仅校准标签口径，不是当前材料证据：" + json.dumps(
                    examples, ensure_ascii=False, separators=(",", ":")
                )
        if self.adaptive:
            extra_prompt += (
                "\n输出前逐标签独立核对：标签只来自当前目标的具体事实与其明确语境。"
                "收益承诺不自动证明收益率被编造；冒充客户或欺骗监管不自动等于欺骗投保人。"
                "公司、主管、账号或会务公司被提及，不证明代理人失管或机构没有销售资质。"
                "培训材料违规须是当前材料实际教授或要求违规；介绍、否决或引用旧案例不成立。"
                "每个标签分别需要行为、对象及必要条件，相关标签不得靠联想补齐。"
                "\n自适应审查：逐个检查目标句mechanism_candidates中的全部候选机制，不能只选前两个。"
                "触发词只提示核查，不证明风险；否定、禁止、合规解释要分别判no_clear_risk。"
                "除已明确成立风险，必要事实未知的机制判insufficient_information，列出具体缺口。"
                "若同一句有多个独立机制，分别判断：一个缺事实不能清除另一个已由当前原文支持的风险。"
                "合同保证给付金额不等于保证非保证收益；先核款项性质、是否超合同或混淆非保证利益。"
                "合同明确包含且不额外赠送的服务不是合同外利益。明确无条件保本或收益承诺、"
                "故意隐瞒免责、拒绝后仍外呼等表述，可以按其自身内容判断，不能泛化为一律等合同。"
                "每个item增加mechanism_assessments数组，格式为"
                '{"mechanism_id":"M01","decision_state":"risk|no_clear_risk|insufficient_information",'
                '"risk_labels":[],"candidate_labels":[],"evidence":["当前原文"],'
                '"missing_facts":[],"reasoning":"20字内"}。'
                "已成立机制risk_labels填全部有依据的27类标签；信息不足的标签只填candidate_labels。"
                "不从机制名自动映射标签，例如代确认不自动等于客户信息不真实，必须另有虚假信息事实。"
                "item.risk_labels合并已成立标签；只要有独立风险成立item.decision_state即risk，"
                "其他机制的未知事实仍保留。没有已成立风险且存在未知才判insufficient_information。"
                "纯背景、正常流程、未作出该机制主张的句子应判no_clear_risk；不能因为普通背景"
                "没写资质、合同或监管材料就凭空制造待核风险。只有实际主张依赖未明必要事实才保留未知。"
                "代操作的对象、环节不明时，不得推定为订立合同确认；必须区分合同确认、订单送达和回访。"
                "mechanism_assessments中判risk的机制还须提供condition_assessments，"
                "逐项覆盖解释卡中required_for_risk=true的condition_id，格式为"
                '{"condition_id":"卡中ID","status":"met|not_met|unknown","evidence":["当前原句"]}。'
                "任一必要条件unknown或not_met，该机制不能判risk。不要把核查条件按词面自动设为met。"
                "代确认本身若未有虚假客户信息记录，不得标客户信息不真实；可用有原文支持的其他标签。"
                "如实说没有理解是正常回访配合，不实辅导要求真实理解状态与所指令答复不一致；"
                "不能把任何回访准备都判违规。多个产品或表格中目标对象未明，不能借邻句出现的保险名补定。"
                "后续外呼目的不明，不能自动补成被拒绝的营销电话。"
                "十二机制以外另有独立风险时，补充label_assessments数组："
                '{"label":"字典标签","decision_state":"risk","evidence":["当前原句"],"missing_facts":[]}。'
                "主risk_labels须有已成立的机制或此额外标签核查支持，不能把待核嫌疑汇入已成立标签。"
                "若明确把未阅读或非本人确认的记录作为本人已阅读/本人确认件提交，"
                "这属于虚假确认记录事实，应核查客户信息不真实，不能仅报笼统欺骗或隐瞒。"
                "普通无风险句使用短格式：基本字段加checked_mechanism_ids列出已检查候选，"
                "decision_state=no_clear_risk、risk_labels为空、evidence只给本句20字以内短引文，"
                "reasoning不超过12字；不展开mechanism_assessments，不输出空缺事实或重复解释。"
                "风险或未知句才展开逐机制对象；未知机制也先核行为条件，任一必要条件not_met时"
                "该机制即no_clear_risk，不能再因为其他资料未知而列成信息不足。"
                "若混合风险句中某候选机制完全未出现，可给该机制no_clear_risk、空标签、"
                "空条件及空证据，表示未涉及；不能因此删除同句其他已成立风险。"
                "无需调用读工具重复读取已提供材料，直接用submit_result提交，理由简短。"
            )
        if self.retrieval_timing == "hybrid":
            guarded = [
                item for item in batch if self._hybrid_semantic_guard(item.rules, item.context)
            ]
            if guarded:
                extra_prompt += (
                    "\n本批含需要核对否定语义或适用条件的强规则候选。规则关键词置信度不等于结论成立。"
                    "请输出decision_state、candidate_labels和missing_facts。合同内权益不能当成合同外赠益，"
                    "明确否定额外赠送时标no_clear_risk、risk_labels为空，并引用支持该解释的当前原文。"
                )
            if any("避债避税" in item.rules.risk_labels for item in guarded):
                extra_prompt += (
                    "\n针对具体领取的税务结论，若当前输入缺少适用身份、款项性质或税收依据，"
                    "应保留待核事实，不能仅由免税一词认定结论不真实；资料缺失本身也不能补造成"
                    "当前已有避税误导或故意隐瞒。仅对同一税务主张关联该缺口，不影响其他独立风险。"
                )
        if knowledge:
            extra_prompt += (
                "\n以下为冻结版本的候选解释卡，不是当前事实或正式法律结论："
                + json.dumps(knowledge, ensure_ascii=False, separators=(",", ":"))
                + "\n必要合同、公告、资格、产品等当前事实缺失时，"
                "decision_state填insufficient_information，"
                "candidate_labels保留嫌疑，missing_facts列所缺材料。若有条件核查，condition_assessments"
                "逐项包含condition_id、status(met/not_met/unknown)、evidence(当前原句引文数组)。"
                "信息不足不能按风险已成立提交，也不能放行；正常结论仅限本次已核查范围。"
                "不得以历史案例、法规或检索分数证明当前事实。条件卡中的核查问题不等于全部为真即可违法。"
            )
        request = JsonChatRequest(
            task="risk_review",
            review_context={
                "target_ids": sorted(target_ids),
                "allowed_labels": list(self.taxonomy),
                "sentences": sentences_payload,
                "case_context_enabled": case_context_enabled,
                "compact_payload": provider_payload if self.compact_context else None,
                "review_policy": "tools"
                if self.retrieval_timing == "hybrid" and not self.adaptive
                else "direct",
                "evidence_mode": "source_ids" if self.compact_context else "literal",
                "required_predicates": {
                    guide["mechanism_id"]: [
                        condition["condition_id"]
                        for condition in guide.get("conditions", [])
                        if condition.get("required_for_risk")
                    ]
                    for guide in knowledge
                }
                if self.adaptive
                else {},
            },
            messages=[
                ChatMessage(
                    role="system",
                    content=self._system_prompt(case_context_enabled)
                    + extra_prompt
                    + (
                        "\nsource_sentences是共享原文表；targets中target_source指向目标原句，"
                        "source_ids限定本目标可引用的原文。保留角色、立场和作用域，"
                        "每条reasoning控制在60字以内；正常句给简短依据。"
                        "evidence数组直接填source_sentences中的原文编号（例如s1），"
                        "包括机制、标签和条件中的evidence也只填本目标source_ids内的编号，"
                        "由程序还原原文，无需复制句子。"
                        "最终直接调用submit_result提交，不要先输出一份普通聊天JSON。"
                        if self.compact_context
                        else ""
                    ),
                ),
                ChatMessage(
                    role="user",
                    content=(
                        "只需输出这些待复核句子的结果："
                        f"{json.dumps(sorted(target_ids), ensure_ascii=False)}\n"
                        "同批其余句子只用于理解全文上下文，不要输出。"
                        "本地规则是可质疑的初筛证据，不是最终结论。\n"
                        + (
                            "reference_cases 是同类历史处罚案例摘要，仅用于理解违规类型与手法；"
                            "它不是本句的原文证据，不得据此提高或降低 risk_score，"
                            "也不得把案例文字当作 evidence。\n"
                            if case_context_enabled
                            else ""
                        )
                        + f"材料句子：{batch_payload}"
                    ),
                ),
            ],
            temperature=0.0,
            max_tokens=max(
                4000 if self.adaptive or longtext_review else 2500,
                min(16000, (1200 if self.adaptive else 650) * len(target_ids)),
            ),
        )
        if (
            self.evidence_service is not None
            and self.agent_retrieval_enabled
            and not self.adaptive
            and self.retrieval_timing == "hybrid"
            and hasattr(self.llm_client, "complete_json_with_tools")
        ):
            request = request.model_copy(update={"available_tools": RiskEvidenceTools.names})
            dispatcher = RiskEvidenceTools(
                self.retrieval_service,
                self.evidence_service,
                sentences_payload,
                tool_budget or QueryToolBudget(),
            )
            response = await self.llm_client.complete_json_with_tools(request, dispatcher.dispatch)
        else:
            response = await self.llm_client.complete_json(request)
        try:
            parsed = _LLMBatch.model_validate(response.data)
        except ValidationError as error:
            raise LLMProviderError("大模型风险输出不符合批量字段契约") from error
        outputs: dict[str, _LLMItem] = {}
        for item in parsed.items:
            if item.sentence_id not in target_ids or item.sentence_id in outputs:
                continue
            source_item = next(
                value for value in batch if value.sentence.sentence_id == item.sentence_id
            )
            if any(
                not quote.strip()
                or not any(quote in source.text for source in source_item.context.source_sentences)
                for quote in item.evidence
            ):
                continue  # 不接受不能回引到该句作用域的模型证据。
            valid_labels = [label for label in item.risk_labels if label in self._taxonomy_set]
            checked_ids = (
                list(
                    dict.fromkeys(
                        mid
                        for mid in item.checked_mechanism_ids
                        if re.fullmatch(r"M(?:0[1-9]|1[0-2])", mid)
                        and mid in self._candidate_mechanisms(source_item)
                    )
                )
                if (item.decision_state == "no_clear_risk" and item.risk_score < 0.20)
                else []
            )
            for extra in item.label_assessments:
                quotes = extra.get("evidence", [])
                if (
                    extra.get("label") not in self._taxonomy_set
                    or extra.get("decision_state")
                    not in {"risk", "no_clear_risk", "insufficient_information"}
                    or not isinstance(quotes, list)
                    or any(
                        not isinstance(quote, str)
                        or not quote.strip()
                        or not any(
                            quote in source.text for source in source_item.context.source_sentences
                        )
                        for quote in quotes
                    )
                ):
                    raise LLMProviderError("额外风险标签核查或当前引文不合法")
                if extra["decision_state"] == "risk" and (not quotes or extra.get("missing_facts")):
                    raise LLMProviderError("额外独立风险必须有当前引文且不缺必要事实")
            for mechanism in item.mechanism_assessments:
                if any(
                    not quote.strip()
                    or not any(
                        quote in source.text for source in source_item.context.source_sentences
                    )
                    for quote in mechanism.evidence
                ):
                    raise LLMProviderError("机制判断证据不能回引当前输入")
                if mechanism.decision_state == "risk" and not mechanism.evidence:
                    raise LLMProviderError("已判断的机制缺少当前原文证据")
                if (
                    mechanism.decision_state == "no_clear_risk"
                    and not mechanism.evidence
                    and (
                        mechanism.risk_labels
                        or mechanism.missing_facts
                        or mechanism.condition_assessments
                    )
                ):
                    raise LLMProviderError("无引文的未涉及机制不能附加事实、标签或条件断言")
                if mechanism.decision_state == "risk" and mechanism.missing_facts:
                    raise LLMProviderError("同一机制事实缺失不能判已成立")
                if any(label not in self._taxonomy_set for label in mechanism.risk_labels):
                    raise LLMProviderError("机制标签超出27类字典")
                for condition in mechanism.condition_assessments:
                    if condition.get("status") not in {"met", "not_met", "unknown"}:
                        raise LLMProviderError("机制必要条件状态不合法")
                    quotes = condition.get("evidence", [])
                    if (
                        not isinstance(quotes, list)
                        or any(
                            not isinstance(quote, str)
                            or not quote.strip()
                            or not any(
                                quote in source.text
                                for source in source_item.context.source_sentences
                            )
                            for quote in quotes
                        )
                        or (condition.get("status") != "unknown" and not quotes)
                    ):
                        raise LLMProviderError("机制条件证据不能回引当前输入")
            for condition in item.condition_assessments:
                if condition.get("status") not in {"met", "not_met", "unknown"}:
                    raise LLMProviderError("条件核查状态不合法")
                quotes = condition.get("evidence", [])
                if not isinstance(quotes, list) or any(
                    not isinstance(quote, str)
                    or not quote
                    or not any(
                        quote in source.text for source in source_item.context.source_sentences
                    )
                    for quote in quotes
                ):
                    raise LLMProviderError("条件证据不能回引当前输入")
                if condition["status"] != "unknown" and not quotes:
                    raise LLMProviderError("已满足或未满足的条件缺少当前原文")
            guarded_mechanisms = [
                apply_mechanism_guard(
                    source_item.sentence.text,
                    mechanism.model_copy(update={"decision_origin": "model"}),
                )
                for mechanism in item.mechanism_assessments
            ]
            outputs[item.sentence_id] = item.model_copy(
                update={
                    "risk_labels": valid_labels,
                    "mechanism_assessments": guarded_mechanisms,
                    "checked_mechanism_ids": checked_ids,
                }
            )
            if source_item.sentence.scene == QueryScene.ST1_POSTER_SMS_SOCIAL:
                output = outputs[item.sentence_id]
                guard_text = (
                    build_retrieval_query(source_item.context)
                    if is_affirmative_answer(source_item.sentence.text)
                    else source_item.sentence.text
                )

                def supports(label: str) -> bool:
                    return social_label_supported(label, guard_text)

                guarded_labels = [label for label in output.risk_labels if supports(label)]
                guarded_extra = [
                    extra for extra in output.label_assessments if supports(str(extra["label"]))
                ]
                guarded_mechanisms = [
                    mechanism.model_copy(
                        update={
                            "risk_labels": [
                                label for label in mechanism.risk_labels if supports(label)
                            ],
                            "decision_state": "no_clear_risk"
                            if mechanism.decision_state == "risk"
                            and mechanism.risk_labels
                            and not any(supports(label) for label in mechanism.risk_labels)
                            else mechanism.decision_state,
                        }
                    )
                    for mechanism in output.mechanism_assessments
                ]
                removed_all = bool(output.risk_labels) and not guarded_labels
                outputs[item.sentence_id] = output.model_copy(
                    update={
                        "risk_labels": guarded_labels,
                        "label_assessments": guarded_extra,
                        "mechanism_assessments": guarded_mechanisms,
                        **(
                            {
                                "risk_score": min(output.risk_score, 0.19),
                                "decision_state": "no_clear_risk",
                                "reasoning": "本句未满足候选标签的必要事实；保留产品限定与原文。",
                            }
                            if removed_all and not output.missing_facts
                            else {}
                        ),
                    }
                )
        return outputs, response.agent_trace

    def _finalize(
        self,
        item: _PreliminaryAssessment,
        llm_output: _LLMItem | None,
    ) -> SentenceRiskAssessment:
        fact_gaps = _current_fact_gaps(item.sentence.text)
        fact_gap_guard = bool(
            llm_output and llm_output.decision_state == "no_clear_risk" and fact_gaps
        )
        if fact_gap_guard and llm_output:
            llm_output = llm_output.model_copy(
                update={
                    "decision_state": "insufficient_information",
                    "missing_facts": fact_gaps,
                    "risk_labels": [],
                    "reasoning": "当前业务依据尚待核实：" + "、".join(fact_gaps),
                }
            )
        final_score = self.scorer.fuse(
            item.rules.risk_score,
            llm_score=llm_output.risk_score if llm_output else None,
            local_confidence=item.rules.confidence,
            strong_signal=item.rules.strong_signal,
        )
        if item.context.stance == "negative_example" and item.rules.safe_signal:
            final_score = min(final_score, 0.19)
        if (
            llm_output is not None
            and llm_output.decision_state == "no_clear_risk"
            and (
                llm_output.evidence
                or any(m.evidence for m in llm_output.mechanism_assessments)
                or any(extra.get("evidence") for extra in llm_output.label_assessments)
            )
        ):
            # The host has already verified every quote against this target's source window.
            final_score = min(final_score, self.risk_display_threshold - 0.01)
        decision = self.thresholder.judge(final_score)
        labels = self._resolve_labels(item, llm_output, final_score)
        candidate_labels = list(labels)
        missing_facts = llm_output.missing_facts if llm_output else []
        conditions = llm_output.condition_assessments if llm_output else []
        unknown = bool(missing_facts) or any(c.get("status") == "unknown" for c in conditions)
        if llm_output:
            candidate_labels = list(
                dict.fromkeys(
                    [
                        *candidate_labels,
                        *(v for v in llm_output.candidate_labels if v in self._taxonomy_set),
                    ]
                )
            )
        mechanisms = llm_output.mechanism_assessments if llm_output else []
        extra_labels = llm_output.label_assessments if llm_output else []
        if self.adaptive and llm_output and (mechanisms or extra_labels):
            confirmed = [
                label
                for mechanism in mechanisms
                if mechanism.decision_state == "risk"
                for label in mechanism.risk_labels
                if label in self._taxonomy_set
            ]
            # Unmapped independent labels need their own current-input quotation assessment.
            confirmed.extend(
                str(extra["label"])
                for extra in llm_output.label_assessments
                if extra["decision_state"] == "risk"
            )
            labels = list(dict.fromkeys(confirmed))
            if labels:
                final_score = max(final_score, llm_output.risk_score, self.risk_display_threshold)
                decision = self.thresholder.judge(final_score)
            missing_facts = list(
                dict.fromkeys(
                    [
                        *missing_facts,
                        *(fact for mechanism in mechanisms for fact in mechanism.missing_facts),
                    ]
                )
            )
            candidate_labels = list(
                dict.fromkeys(
                    [
                        *candidate_labels,
                        *(
                            label
                            for mechanism in mechanisms
                            for label in mechanism.candidate_labels
                            if label in self._taxonomy_set
                        ),
                        *labels,
                    ]
                )
            )
            unknown = any(m.decision_state == "insufficient_information" for m in mechanisms)
            if not unknown and not any(
                extra["decision_state"] == "insufficient_information" for extra in extra_labels
            ):
                missing_facts = []
            if not labels and not unknown:
                final_score = min(final_score, self.risk_display_threshold - 0.01)
                decision = self.thresholder.judge(final_score)
        if self.adaptive and extra_labels:
            labels = list(
                dict.fromkeys(
                    [
                        *labels,
                        *(
                            str(extra["label"])
                            for extra in extra_labels
                            if extra["decision_state"] == "risk"
                        ),
                    ],
                )
            )
            missing_facts = list(
                dict.fromkeys(
                    [
                        *missing_facts,
                        *(
                            str(fact)
                            for extra in extra_labels
                            for fact in _string_list(extra.get("missing_facts"))
                        ),
                    ]
                )
            )
            candidate_labels = list(
                dict.fromkeys(
                    [
                        *candidate_labels,
                        *(
                            str(extra["label"])
                            for extra in extra_labels
                            if extra["decision_state"] == "insufficient_information"
                        ),
                    ]
                )
            )
            unknown = unknown or any(
                extra["decision_state"] == "insufficient_information" for extra in extra_labels
            )
        state: Literal["risk", "no_clear_risk", "insufficient_information"] = (
            "risk"
            if self.adaptive and (mechanisms or extra_labels) and labels
            else "insufficient_information"
            if unknown
            or (
                llm_output
                and not (self.adaptive and (mechanisms or extra_labels))
                and llm_output.decision_state == "insufficient_information"
            )
            else "risk"
            if final_score >= self.risk_display_threshold and labels
            else "no_clear_risk"
        )
        if state == "insufficient_information":
            labels = []
        evidence = list(item.rules.evidence)
        if llm_output:
            quotes = list(
                dict.fromkeys(
                    [
                        *llm_output.evidence,
                        *(quote for mechanism in mechanisms for quote in mechanism.evidence),
                        *(
                            quote
                            for extra in extra_labels
                            for quote in _string_list(extra.get("evidence"))
                        ),
                    ]
                )
            )
            for quote in quotes:
                source = next(
                    value
                    for value in sorted(
                        item.context.source_sentences, key=lambda value: not value.is_target
                    )
                    if quote in value.text
                )
                start = source.document_start_offset + source.text.index(quote)
                evidence.append(
                    RiskEvidence(
                        source="deterministic_current_input_guard"
                        if any(
                            mechanism.decision_origin == "deterministic_guard"
                            and quote in mechanism.evidence
                            for mechanism in mechanisms
                        )
                        else "llm_semantic_evidence",
                        description=f"原文证据：{quote}",
                        sentence_id=source.sentence_id,
                        document_start_offset=start,
                        document_end_offset=start + len(quote),
                    )
                )
            evidence.append(
                RiskEvidence(
                    source="llm_semantic_review",
                    description=llm_output.reasoning,
                )
            )
        reasoning = (
            llm_output.reasoning if llm_output else _fallback_reasoning(item.rules, decision.level)
        )
        return SentenceRiskAssessment(
            query_id=item.sentence.query_id,
            sentence_id=item.sentence.sentence_id,
            text=item.sentence.text,
            risk_score=final_score,
            risk_level=decision.level,
            risk_level_name="信息不足"
            if state == "insufficient_information"
            else decision.level_name,
            risk_labels=labels,
            decision_state=state,
            candidate_labels=candidate_labels,
            missing_facts=missing_facts,
            condition_assessments=conditions,
            mechanism_assessments=mechanisms,
            label_assessments=extra_labels,
            checked_mechanism_ids=llm_output.checked_mechanism_ids if llm_output else [],
            reasoning=reasoning,
            method=("local_two_stage+llm" if llm_output else "local_two_stage")
            + ("+fact_gap_guard" if fact_gap_guard else ""),
            local_score=item.rules.risk_score,
            local_confidence=item.rules.confidence,
            llm_reviewed=llm_output is not None,
            review_depth=llm_output.review_depth if llm_output else "rule",
            context_review_required=item.context.requires_review,
            review_status=(
                "needs_review"
                if item.context.context_truncated or state == "insufficient_information" or unknown
                else "reviewed"
                if llm_output
                else "local"
                if self._local_direct(item.rules, item.context)
                else "needs_review"
            ),
            evidence=evidence,
            action=(
                "保留已确认风险，补充其余业务事实复核"
                if state == "risk" and missing_facts
                else "补充当前业务事实后复核"
                if state == "insufficient_information"
                else "需结合上下文人工复核"
                if (llm_output is None or item.context.context_truncated)
                and item.context.requires_review
                and final_score < self.retrieval_threshold
                else decision.action
            ),
        )

    def _apply_document_channel_labels(
        self,
        query: QueryProcessingResult,
        assessments: list[SentenceRiskAssessment],
    ) -> list[SentenceRiskAssessment]:
        contexts = {context.sentence_id: context for context in query.sentence_contexts}
        marker_match = re.match(r"^【材料载体：[^】]+】", query.source_text)
        marker = marker_match.group(0) if marker_match else ""
        results: list[SentenceRiskAssessment] = []
        for item in assessments:
            if self.adaptive and item.llm_reviewed:
                # Carrier names and broad content patterns are candidates for the model,
                # not a second independent source of confirmed labels after adjudication.
                results.append(item)
                continue
            if (
                item.risk_score < self.risk_display_threshold
                or item.decision_state == "insufficient_information"
                or (self.adaptive and item.decision_state != "risk")
            ):
                results.append(item)
                continue
            content_labels = _explicit_content_labels(item.text)
            context = contexts.get(item.sentence_id)
            channel_text = (
                "\n".join(source.text for source in context.source_sentences)
                if context
                else item.text
            )
            channel_labels = _document_channel_labels(
                query.classification.scene, f"{marker}\n{channel_text}" if marker else channel_text
            )
            if (
                query.classification.scene is QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK
                and _looks_like_noncompliant_training_guidance(item.text)
            ):
                content_labels.append("培训材料违规")
            labels = list(dict.fromkeys([*channel_labels, *content_labels]))
            if not labels:
                results.append(item)
                continue
            merged_labels = list(dict.fromkeys([*item.risk_labels, *labels]))[
                : 27 if self.adaptive else 10
            ]
            evidence = [
                *item.evidence,
                RiskEvidence(
                    source="document_context",
                    description=f"材料载体或明确措辞补充标签：{'、'.join(labels)}",
                ),
            ]
            results.append(
                item.model_copy(
                    update={
                        "risk_labels": merged_labels,
                        "evidence": evidence,
                        "decision_state": "risk",
                        "candidate_labels": list(
                            dict.fromkeys([*item.candidate_labels, *merged_labels])
                        ),
                    }
                )
            )
        return results

    def _resolve_labels(
        self,
        item: _PreliminaryAssessment,
        llm_output: _LLMItem | None,
        final_score: float,
    ) -> list[str]:
        if final_score < self.risk_display_threshold:
            return []
        candidates: list[str] = []
        if llm_output:
            candidates.extend(llm_output.risk_labels)
        elif item.rules.risk_score >= self.risk_display_threshold or item.rules.strong_signal:
            candidates.extend(item.rules.risk_labels)
        candidates.extend(_scene_channel_labels(item.sentence, final_score))
        candidates = [
            label
            for label in candidates
            if _channel_label_supported(label, item.sentence.text, item.sentence.scene)
        ]
        if llm_output:
            return list(dict.fromkeys(candidates))
        # Rules contribute only their own observed labels. Related labels require
        # separate facts, and an independent 'other' risk may coexist with them.
        return list(dict.fromkeys(label for label in candidates if label in self._taxonomy_set))

    async def _retrieve_eligible(
        self,
        assessments: list[SentenceRiskAssessment],
        contexts: dict[str, RiskSentenceContext],
        prefetched: dict[str, _PrefetchedRetrieval] | None = None,
    ) -> list[SentenceRiskAssessment]:
        prefetched = prefetched or {}
        eligible = [
            item
            for item in assessments
            if item.risk_score >= self.retrieval_threshold and item.risk_labels
        ]
        if not eligible:
            return assessments
        if self.adaptive:
            return await self._retrieve_by_mechanism(assessments, contexts, prefetched)
        semaphore = asyncio.Semaphore(4)

        async def retrieve(
            item: SentenceRiskAssessment,
        ) -> tuple[str, list[MatchedCase], str, bool]:
            # 判定后的最终标签（含 LLM 与场景补充标签），与检索服务的过滤口径一致。
            final_tags = frozenset(tag for tag in item.risk_labels if tag and tag != "其他")
            async with semaphore:
                cached = prefetched.get(item.sentence_id)
                if cached is not None and not cached.failed:
                    # 仅当前置检索使用的标签约束与最终标签一致时才可复用；
                    # 否则前置结果缺少应有的标签过滤，必须重新检索。
                    if cached.used_tags == final_tags or self.retrieval_timing == "hybrid":
                        return (
                            item.sentence_id,
                            list(cached.matches),
                            cached.retrieval_query,
                            False,
                        )
                retrieval_query = (
                    cached.retrieval_query
                    if cached is not None
                    else build_retrieval_query(contexts[item.sentence_id])
                )
                try:
                    matches = await asyncio.to_thread(
                        self.retrieval_service.search,
                        retrieval_query,
                        top_k=self.retrieval_top_k,
                        risk_tags=None if self.retrieval_timing == "hybrid" else set(final_tags),
                    )
                    return item.sentence_id, matches, retrieval_query, False
                except (OSError, ValueError, RuntimeError):
                    return item.sentence_id, [], retrieval_query, True

        retrieved = {
            sentence_id: (matches, retrieval_query, failed)
            for sentence_id, matches, retrieval_query, failed in await asyncio.gather(
                *(retrieve(item) for item in eligible)
            )
        }
        results: list[SentenceRiskAssessment] = []
        for item in assessments:
            retrieved_item = retrieved.get(item.sentence_id)
            if retrieved_item is None:
                results.append(item)
                continue
            matches, retrieval_query, failed = retrieved_item
            evidence = list(item.evidence)
            if matches:
                top = matches[0]
                evidence.append(
                    RiskEvidence(
                        source="hybrid_case_retrieval",
                        description=(
                            f"类案参考 {top.case_id}，内部排序分 {top.score:.2f}；可比性待核验"
                        ),
                    )
                )
            results.append(
                item.model_copy(
                    update={
                        "method": item.method
                        + "+"
                        + (matches[0].retrieval_channel if matches else "case_retrieval"),
                        "retrieval_performed": True,
                        "evidence": evidence,
                        "matched_cases": matches,
                        "retrieval_query": retrieval_query,
                        "retrieval_status": (
                            "unavailable"
                            if failed
                            else "candidates"
                            if matches
                            else "no_sufficient_match"
                        ),
                    }
                )
            )
        return results

    async def _retrieve_by_mechanism(
        self,
        assessments: list[SentenceRiskAssessment],
        contexts: dict[str, RiskSentenceContext],
        prefetched: dict[str, _PrefetchedRetrieval],
    ) -> list[SentenceRiskAssessment]:
        """One case pool per scope/mechanism, fused for multi-mechanism targets."""
        groups: dict[tuple[str, str], list[SentenceRiskAssessment]] = defaultdict(list)
        target_groups: dict[str, list[tuple[str, str]]] = {}
        for item in assessments:
            if item.risk_score < self.retrieval_threshold or not item.risk_labels:
                continue
            mids = (
                [m.mechanism_id for m in item.mechanism_assessments if m.decision_state == "risk"]
                if item.mechanism_assessments
                else self.evidence_service.planner.reading_mechanisms(item.text)
                if self.evidence_service
                else []
            )
            keys = [(contexts[item.sentence_id].scope_id, mid) for mid in (mids or ["OTHER"])]
            target_groups[item.sentence_id] = keys
            for key in keys:
                groups[key].append(item)
        semaphore = asyncio.Semaphore(min(6, max(1, self.llm_concurrency)))

        async def retrieve_group(
            key: tuple[str, str],
            members: list[SentenceRiskAssessment],
        ) -> tuple[tuple[str, str], tuple[list[MatchedCase], str, bool]]:
            cached = next(
                (
                    prefetched[m.sentence_id]
                    for m in members
                    if m.sentence_id in prefetched
                    and not prefetched[m.sentence_id].failed
                    and prefetched[m.sentence_id].matches
                ),
                None,
            )
            if cached is not None:
                return key, (list(cached.matches), cached.retrieval_query, False)
            chosen = sorted(members, key=lambda item: (-item.risk_score, item.sentence_id))[:3]
            query = "\n".join(item.text for item in chosen)[:1200]
            async with semaphore:
                try:
                    matches = await asyncio.to_thread(
                        self.retrieval_service.search,
                        query,
                        top_k=self.retrieval_top_k,
                        risk_tags=None,
                    )
                    return key, (matches, query, False)
                except (OSError, ValueError, RuntimeError):
                    return key, ([], query, True)

        pools = dict(
            await asyncio.gather(*(retrieve_group(key, members) for key, members in groups.items()))
        )
        results = []
        for item in assessments:
            item_keys = target_groups.get(item.sentence_id)
            if not item_keys:
                results.append(item)
                continue
            scores: dict[str, float] = {}
            matches_by_id: dict[str, MatchedCase] = {}
            queries = []
            for key in item_keys:
                matches, query, _ = pools[key]
                queries.append(query)
                for rank, match in enumerate(matches, 1):
                    identity = match.file_id or match.case_id
                    scores[identity] = scores.get(identity, 0.0) + 1 / (60 + rank)
                    matches_by_id.setdefault(identity, match)
            selected = [
                matches_by_id[identity]
                for identity in sorted(scores, key=lambda identity: (-scores[identity], identity))[
                    : self.retrieval_top_k
                ]
            ]
            failed = any(pools[key][2] for key in item_keys)
            results.append(
                item.model_copy(
                    update={
                        "matched_cases": selected,
                        "retrieval_performed": True,
                        "retrieval_query": "\n---\n".join(dict.fromkeys(queries)),
                        "retrieval_status": "unavailable"
                        if failed
                        else "candidates"
                        if selected
                        else "no_sufficient_match",
                        "method": item.method + "+scope_mechanism_case_pool",
                        "evidence": [
                            *item.evidence,
                            *(
                                [
                                    RiskEvidence(
                                        source="hybrid_case_retrieval",
                                        description="同一作用域按机制复用历史类案，仅作参考，不证明当前事实",
                                    )
                                ]
                                if selected
                                else []
                            ),
                        ],
                    }
                )
            )
        return results

    def _summarize(
        self,
        assessments: list[SentenceRiskAssessment],
        *,
        llm_outputs: dict[str, _LLMItem],
        review_ids: set[str],
        warnings: list[str],
    ) -> QueryRiskSummary:
        scores = sorted((item.risk_score for item in assessments), reverse=True)
        maximum = scores[0]
        if maximum < self.risk_display_threshold:
            overall_score = maximum
        else:
            support = sum(scores[1:4])
            overall_score = min(1.0, maximum + (1 - maximum) * (1 - math.exp(-0.18 * support)))
        overall_score = round(overall_score, 4)
        decision = self.thresholder.judge(overall_score)

        label_sentences: dict[str, list[SentenceRiskAssessment]] = defaultdict(list)
        for item in assessments:
            if item.risk_score < self.risk_display_threshold:
                continue
            for label in item.risk_labels:
                label_sentences[label].append(item)
        findings = [
            RiskLabelFinding(
                label=label,
                max_score=max(item.risk_score for item in items),
                sentence_count=len(items),
                sentence_ids=[item.sentence_id for item in items],
            )
            for label, items in label_sentences.items()
        ]
        findings.sort(key=lambda item: (-item.max_score, -item.sentence_count, item.label))

        best_cases: dict[str, MatchedCase] = {}
        for item in assessments:
            for case in item.matched_cases:
                current = best_cases.get(case.case_id)
                if current is None or case.score > current.score:
                    best_cases[case.case_id] = case
        top_cases = sorted(best_cases.values(), key=lambda case: (-case.score, case.case_id))[:5]
        pending_count = sum(item.review_status == "needs_review" for item in assessments)
        document_state: Literal["risk", "no_clear_risk", "insufficient_information"] = (
            "risk"
            if any(item.decision_state == "risk" for item in assessments)
            else "insufficient_information"
            if any(item.decision_state == "insufficient_information" for item in assessments)
            else "no_clear_risk"
        )
        return QueryRiskSummary(
            decision_state=document_state,
            missing_facts=list(
                dict.fromkeys(fact for item in assessments for fact in item.missing_facts)
            ),
            overall_score=overall_score,
            overall_level=decision.level,
            overall_level_name="信息不足"
            if document_state == "insufficient_information"
            else decision.level_name,
            action=(
                "补充当前业务事实后确认"
                if document_state == "insufficient_information"
                else "补充语义或人工复核后确认"
                if pending_count and overall_score < self.retrieval_threshold
                else decision.action
            ),
            risk_labels=[finding.label for finding in findings],
            risk_findings=findings,
            top_matched_cases=top_cases,
            risk_display_threshold=self.risk_display_threshold,
            retrieval_threshold=self.retrieval_threshold,
            max_sentence_score=maximum,
            sentence_count=len(assessments),
            risky_sentence_count=sum(
                item.risk_score >= self.risk_display_threshold for item in assessments
            ),
            llm_used=bool(llm_outputs),
            llm_model=self.llm_client.model_name if llm_outputs and self.llm_client else None,
            llm_reviewed_sentence_count=len(llm_outputs),
            local_direct_sentence_count=len(assessments) - len(review_ids),
            retrieval_sentence_count=sum(item.retrieval_performed for item in assessments),
            warnings=warnings,
            pending_review_sentence_count=pending_count,
        )


def _string_list(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _llm_input(
    item: _PreliminaryAssessment,
    reference_cases: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "sentence_id": item.sentence.sentence_id,
        "scene": item.sentence.scene.value,
        "text": item.sentence.text,
        "context_before": item.mapping.context_before[-400:],
        "context_after": item.mapping.context_after[:400],
        "context": item.context.model_dump(),
        "local_stage1_score": item.rules.stage1_score,
        "local_stage2_score": item.rules.stage2_score,
        "local_labels": list(item.rules.risk_labels),
        "local_evidence": [value.description for value in item.rules.evidence[:5]],
    }
    if reference_cases:
        payload["reference_cases"] = reference_cases
    return payload


def _reference_cases(
    prefetched: dict[str, _PrefetchedRetrieval],
    sentence_id: str,
) -> list[dict[str, object]]:
    """把前置检索命中的类案压成短摘要，控制注入提示词的体积。

    只保留判断违规类型与手法所需的字段；不注入处罚金额、当事人与完整原文，
    既避免提示词膨胀，也降低模型把案例文字当作本句证据的风险。
    """

    entry = prefetched.get(sentence_id)
    if entry is None or not entry.injected or not entry.matches:
        return []
    summaries: list[dict[str, object]] = []
    for case in entry.matches[:3]:
        behavior = " ".join((case.violation_behavior or "").split())
        matched_text = " ".join((case.matched_text or "").split())
        summaries.append(
            {
                "case_id": case.case_id,
                "risk_tags": list(case.risk_tags)[:6],
                "matched_risk_tags": list(case.matched_risk_tags)[:4],
                "violation_behavior": behavior[:180],
                "evidence_excerpt": matched_text[:120],
                "relevance": round(case.score, 3),
                "channel": case.retrieval_channel,
            }
        )
    return summaries


def _fallback_reasoning(rules: RuleDetectionResult, level: RiskLevel) -> str:
    context_evidence = [value for value in rules.evidence if value.source.startswith("context_")]
    if context_evidence:
        return "；".join(value.description for value in context_evidence[:2])
    if rules.evidence:
        return "；".join(value.description for value in rules.evidence[:2])
    if level is RiskLevel.NONE:
        return "本地两阶段规则未发现明确风险信号；建议在完整模式下进行语义复核。"
    return "本地规则发现弱风险信号，当前为未取得大模型结果时的降级判断。"


def _document_channel_labels(scene: QueryScene, text: str) -> list[str]:
    if scene is QueryScene.ST4_MEETING_MATERIAL:
        return ["产品说明会违规"]
    if scene not in {
        QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        QueryScene.ST6_PENALTY_BACKEND_FACT,
    }:
        return []
    marker_match = re.match(r"^【材料载体：([^】]+)】\s*", text)
    marker = marker_match.group(1) if marker_match else ""
    if scene is QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK:
        if marker == "内部销售培训材料":
            return ["培训材料违规"]
        if marker == "电销话术材料":
            return ["电销违规"]
        if marker == "客户回访材料":
            return ["回访违规"]
    body = text[marker_match.end() :] if marker_match else text
    labels: list[str] = []
    if scene is QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK and any(
        keyword in body for keyword in ("培训材料", "培训课件", "内部培训", "讲师备注")
    ):
        labels.append("培训材料违规")

    if _channel_label_supported("电销违规", body, scene):
        labels.append("电销违规")
    if _channel_label_supported("回访违规", body, scene):
        labels.append("回访违规")
    return labels


def _looks_like_noncompliant_training_guidance(text: str) -> bool:
    """识别面向销售人员的规避式操作指引，而非普通通话或回访事实。"""
    if (
        "回访" in text
        and "录音" in text
        and not any(keyword in text for keyword in ("礼品", "赠送", "销售承诺"))
    ):
        return False
    patterns = (
        r"(?:话术|口径).{0,20}(?:不要|别|改成|统一|只说|就说)",
        r"(?:业务员|坐席|主管|团队|伙伴).{0,42}(?:垫付|垫一下|拆成|统一写|只保留|不要记录)",
        r"(?:职业|健康|住院|病史).{0,28}(?:先选|选择为否|未发现异常|按内勤录入)",
        r"(?:返现|返佣|佣金|保费).{0,28}(?:别说|不要写|语音回答|报销|补助)",
        r"(?:条款|现金价值表|演示利益|保证利益).{0,28}(?:先不|不必|只发|只读|后再|就说)",
        r"(?:名额|验证码|手机确认).{0,28}(?:先完成|先报|当天完成|自动释放)",
        r"(?:别说|不要说).{0,8}(?:返现|返佣|收益率|百分之)",
        r"(?:不必解释|不要说明).{0,16}(?:演示利益|保证利益|现金价值)",
        r"(?:检查|质检).{0,12}(?:抓到|发现).{0,16}(?:改成|换成|不要再说)?",
        r"主管.{0,160}(?:暗示|统一口径|不必解释|不要说明)",
        r"先从.{0,48}切入.{0,24}(?:暗示|不必解释|不要说明)",
        r"(?:礼品|赠送利益|销售承诺).{0,36}(?:不提交|暂停录音|重新回访|只上传)",
    )
    return any(re.search(pattern, text, re.DOTALL) for pattern in patterns)


def _explicit_content_labels(text: str) -> list[str]:
    """从明确措辞补充容易漏检的事实标签，不改变风险分值。"""
    labels: list[str] = []
    fabricated_yield_patterns = (
        r"(?:虚构|编造|伪造|捏造).{0,24}(?:收益率|年化|利率|收益数据|历史业绩)",
        r"(?:收益率|收益数据|年化|历史业绩).{0,24}(?:系编造|系伪造|凭空捏造)",
    )
    if any(re.search(pattern, text, re.DOTALL) for pattern in fabricated_yield_patterns):
        labels.append("虚构收益率")

    if re.search(r"(?:家属|亲属).{0,12}(?:代签|代填|代操作)", text) and re.search(
        r"冒充本人|伪造.{0,8}(?:签名|客户信息)|登记为本人操作", text
    ):
        labels.append("客户信息不真实")

    unlicensed = re.search(
        r"(?:无|没有|未取得|不具备).{0,8}(?:保险)?销售资质.{0,18}(?:会务|咨询|第三方|机构|公司)|"
        r"(?:机构|公司).{0,24}(?:未取得|不具备|没有).{0,8}(?:保险代理|保险销售)资质",
        text,
        re.DOTALL,
    )
    if unlicensed and re.search(r"委托|代理协议|实际销售|招揽保险|向客户销售", text):
        labels.append("委托无资质机构销售")

    if re.search(
        r"(?:负责人|主管).{0,80}(?:收到|已知).{0,20}(?:异常|提示|清单).{0,50}(?:未|没有)(?:停用|核查|处置)|"
        r"(?:系统|质检|机构).{0,24}(?:未|没有).{0,12}(?:拦截|覆盖|复核|重检).{0,18}(?:异常|相关业务)?",
        text,
        re.DOTALL,
    ) and re.search(r"代理人|营销员|销售人员", text):
        labels.append("代理人管理不到位")

    if re.search(
        r"(?:多份|大量|[一二三四五六七八九十百]+\s*份|\d+\s*(?:份|段|条)).{0,12}(?:回访)?录音.{0,36}(?:同一.{0,8}(?:手机|设备|ip)|时长.{0,8}(?:一致|相同)|背景声.{0,8}(?:一致|相同))|"
        r"(?:客户|投保人).{0,12}(?:表示|称).{0,8}(?:从未|没有|未曾).{0,8}(?:接到|接受|完成).{0,6}回访",
        text,
        re.DOTALL | re.IGNORECASE,
    ) and re.search(r"伪造|虚假|冒充|记录.{0,12}已完成|后台.{0,12}(?:勾选|记录)", text):
        labels.extend(("回访违规", "客户信息不真实"))

    if re.search(
        r"(?:资金|款项|费用).{0,45}(?:转入|流入|回流).{0,24}(?:营销|销售|员工).{0,12}(?:账户|成员).{0,80}(?:支付|返还|发放).{0,24}(?:新投保|投保)客户",
        text,
        re.DOTALL,
    ) and re.search(r"返佣|保费回扣|保费返现|合同约定以外", text):
        labels.append("回扣返佣")
        if re.search(r"合同约定以外|合同外利益", text):
            labels.append("合同外利益")

    return list(dict.fromkeys(labels))


def _channel_label_supported(label: str, text: str, scene: QueryScene) -> bool:
    """通道违规标签需要行为证据；仅出现“电话、回访”等载体词不构成违规。"""
    if label not in {"回访违规", "电销违规"}:
        return True
    if scene not in {
        QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        QueryScene.ST6_PENALTY_BACKEND_FACT,
    }:
        return False
    patterns: tuple[str, ...]
    if label == "回访违规":
        patterns = (
            r"回访.{0,28}(?:代接|冒充|暂停录音|重新录制|只保留|统一回答|答案预填|同事.{0,8}确认)",
            r"(?:暂不提交|结束当前).{0,18}(?:录音|回访).{0,30}(?:重新回访|重新开始|第二次)",
            r"(?:多份|大量|[一二三四五六七八九十百]+\s*份|\d+\s*(?:份|段|条)).{0,12}(?:回访)?录音.{0,40}(?:同一.{0,8}(?:手机|设备|ip)|时长.{0,8}(?:一致|相同)|背景声.{0,8}(?:一致|相同))",
            r"(?:客户|投保人).{0,18}(?:从未|没有|未曾|否认).{0,16}(?:接到回访|作出.{0,6}确认|完成回访)",
            r"录音.{0,20}客户.{0,12}(?:没看|未收到|不清楚).{0,24}后台.{0,12}(?:勾选|记录).{0,12}(?:知悉|完成)",
        )
    else:
        patterns = (
            r"(?:开场|通话).{0,24}(?:别提|不要|不报|不主动报).{0,12}(?:保险公司|保险|购买保险|公司全称)",
            r"(?:客户)?(?:不同意|明确拒绝|拒绝).{0,36}(?:继续呼叫|继续联系|换.{0,4}(?:号码|工号)|不同号码|另一工号)",
            r"(?:兼职|无资质|未取得资质).{0,40}(?:话术|拨打|呼叫|直接推荐具体保险)",
            r"(?:社区调查|健康调查).{0,32}(?:收集.{0,12}(?:身份证|银行卡|验证码)|推荐具体保险)",
        )
    return any(re.search(pattern, text, re.DOTALL | re.IGNORECASE) for pattern in patterns)


def _scene_channel_labels(sentence: RiskQuerySentence, score: float) -> list[str]:
    if score < 0.40:
        return []
    text = sentence.text
    if sentence.scene is QueryScene.ST4_MEETING_MATERIAL:
        return ["产品说明会违规"]
    if sentence.scene is QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK:
        labels: list[str] = []
        if any(keyword in text for keyword in ("培训", "课件", "内部话术")):
            labels.append("培训材料违规")
        if _channel_label_supported("回访违规", text, sentence.scene):
            labels.append("回访违规")
        if _channel_label_supported("电销违规", text, sentence.scene):
            labels.append("电销违规")
        return labels
    return []
