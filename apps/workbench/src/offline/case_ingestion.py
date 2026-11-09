"""新案例 PDF 的增量标准化、标签、切片和检索扩展流程。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from hashlib import sha256
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError, field_validator

from domain.file_models import FileExtractionResult
from llm import ChatMessage, JsonChatRequest, JsonChatResponse, LLMClient, LLMProviderError
from offline.insurance_relevance import evaluate_insurance_scope
from risk.detection import RuleRiskDetector


@dataclass(frozen=True, slots=True)
class CaseIngestionArtifacts:
    record: dict[str, object]
    chunks: list[dict[str, object]]
    expansions: list[dict[str, object]]
    llm_used: bool
    warnings: list[str]


class _RiskBasis(BaseModel):
    risk_tag: str
    reason: str = Field(min_length=1, max_length=300)
    quote: str = Field(min_length=1, max_length=500)


class _RelatedParty(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    role: str | None = Field(default=None, max_length=100)


class _LegalBasis(BaseModel):
    law_name: str = Field(min_length=1, max_length=200)
    article: str | None = Field(default=None, max_length=100)


class _CaseLLMOutput(BaseModel):
    is_insurance_related: bool | None = None
    fact_quotes: list[str] = Field(default_factory=list)
    case_title: str | None = Field(default=None, max_length=500)
    party_name: str | None = Field(default=None, max_length=300)
    institution_type: str | None = Field(default=None, max_length=100)
    penalty_doc_no: str | None = Field(default=None, max_length=200)
    violation_behavior: str = Field(min_length=1, max_length=6000)
    penalty_content: str | None = Field(default=None, max_length=3000)
    regulator: str | None = Field(default=None, max_length=300)
    penalty_date: str | None = None
    related_parties: list[_RelatedParty] = Field(default_factory=list, max_length=20)
    legal_basis_refs: list[_LegalBasis] = Field(default_factory=list, max_length=30)
    risk_tags: list[str] = Field(default_factory=list, max_length=27)
    risk_tag_basis: list[_RiskBasis] = Field(default_factory=list, max_length=27)
    case_summary: str = Field(min_length=1, max_length=1000)
    keywords: list[str] = Field(default_factory=list, max_length=12)
    marketing_expressions: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)

    @field_validator("legal_basis_refs", mode="before")
    @classmethod
    def _coerce_legal_basis_strings(cls, value: object) -> object:
        """模型常把法条输出成纯字符串，统一包成 {law_name, article} 结构。"""

        if not isinstance(value, list):
            return []
        return [
            {"law_name": item, "article": None} if isinstance(item, str) else item for item in value
        ]

    @field_validator("risk_tag_basis", mode="before")
    @classmethod
    def _coerce_risk_tag_basis(cls, value: object) -> object:
        """模型可能返回 null、单个对象或残缺项：单对象包成列表，缺 risk_tag/quote
        的项直接丢弃（后续 build 还会按 quote 是否见于原文再过滤一次）。"""

        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []
        items = []
        for item in value:
            if not isinstance(item, dict):
                continue
            if not item.get("risk_tag") or not item.get("quote"):
                continue
            if not item.get("reason"):
                item = {**item, "reason": "见引文"}
            items.append(item)
        return items

    @field_validator(
        "related_parties",
        "risk_tags",
        "keywords",
        "marketing_expressions",
        mode="before",
    )
    @classmethod
    def _none_list_to_empty(cls, value: object) -> object:
        """模型对空列表字段经常返回 null，按空列表处理而不是整单校验失败。"""

        if value is None:
            return []
        if isinstance(value, list):
            return [item for item in value if item]
        return value


class IncrementalCaseBuilder:
    """把已有 FileExtractionResult 转成案例库三表兼容记录。"""

    def __init__(
        self,
        *,
        taxonomy: tuple[str, ...],
        risk_domain_mapping: dict[str, str],
        rule_detector: RuleRiskDetector,
        llm_client: LLMClient | None,
        data_version: str,
    ) -> None:
        self.taxonomy = taxonomy
        self._taxonomy_set = set(taxonomy)
        self.risk_domain_mapping = risk_domain_mapping
        self.rule_detector = rule_detector
        self.llm_client = llm_client
        self.data_version = data_version

    async def build(
        self,
        extraction: FileExtractionResult,
        *,
        source_file: str,
        structured: _CaseLLMOutput | None = None,
        atomic_index: int = 1,
    ) -> CaseIngestionArtifacts:
        text = extraction.full_text.strip()
        if not text:
            raise ValueError("案例文件未抽取到文字，不能入库")
        warnings = list(extraction.warnings)
        llm_used = False
        output: _CaseLLMOutput
        if structured is not None:
            output = structured
            llm_used = True
        elif self.llm_client is not None:
            try:
                output = await self._call_llm(text)
                llm_used = True
            except LLMProviderError as error:
                warnings.append(f"DeepSeek 案例标准化失败，已使用规则降级：{error}")
                output = self._fallback(text)
        else:
            warnings.append("未配置 DeepSeek API Key，案例以规则降级模式入库")
            output = self._fallback(text)
        warnings.extend(_field_completeness_warnings(output))

        case_id = f"U{extraction.checksum_sha256[:15].upper()}"
        if atomic_index > 1:
            case_id += f"P{atomic_index:02}"
        file_id = f"F_{extraction.checksum_sha256[:16].upper()}"
        risk_tags = [label for label in output.risk_tags if label in self._taxonomy_set]
        if not risk_tags:
            risk_tags = [
                label
                for label in self.rule_detector.detect(text).risk_labels
                if label in self._taxonomy_set
            ]
        if not risk_tags:
            risk_tags = ["其他"] if "其他" in self._taxonomy_set else []
        valid_basis = [
            basis
            for basis in output.risk_tag_basis
            if basis.risk_tag in risk_tags and basis.quote in text
        ]
        risk_domains = list(
            dict.fromkeys(
                self.risk_domain_mapping[label]
                for label in risk_tags
                if label in self.risk_domain_mapping
            )
        )
        chunks = _build_chunks(
            case_id=case_id,
            file_id=file_id,
            extraction=extraction,
            violation_behavior=output.violation_behavior,
            penalty_content=output.penalty_content,
            legal_basis=output.legal_basis_refs,
            risk_tags=risk_tags,
            data_version=self.data_version,
        )
        for index, quote in enumerate(
            dict.fromkeys(output.fact_quotes + [b.quote for b in valid_basis]), 1
        ):
            if quote and quote in text:
                start = text.index(quote)
                chunks.append(
                    {
                        "chunk_id": f"CHK_{case_id}_EVIDENCE_{index:03}",
                        "case_id": case_id,
                        "file_id": file_id,
                        "chunk_type": "evidence",
                        "chunk_text": quote,
                        "supports_risk_tags": risk_tags,
                        "is_verbatim": True,
                        "retrieval_enabled": True,
                        "retrieval_weight": 1.0,
                        "source_span": {"start_char": start, "end_char": start + len(quote)},
                        "data_version": self.data_version,
                    }
                )
        expansions = _build_expansions(
            case_id=case_id,
            values=[*output.marketing_expressions, *output.keywords],
            risk_tags=risk_tags,
            risk_domains=risk_domains,
            data_version=self.data_version,
        )
        today = datetime.now(UTC).date().isoformat()
        title = output.case_title or extraction.source_filename
        field_confidence, field_sources = _field_quality(
            output,
            text=text,
            llm_used=llm_used,
            extraction_quality=extraction.quality_score,
        )
        entity_relations = _entity_relations(output, field_confidence)
        record: dict[str, object] = {
            "schema_version": "case_kb_v1",
            "case_id": case_id,
            "file_id": file_id,
            "case_origin": "web_ingestion",
            "atomic_case_index": atomic_index,
            "is_insurance_related": output.is_insurance_related
            if output.is_insurance_related is not None
            else bool(re.search(r"保险|投保人|保单|保费", text)),
            "party_name": output.party_name,
            "institution_type": output.institution_type,
            "penalty_doc_no": output.penalty_doc_no,
            "violation_behavior": output.violation_behavior,
            "penalty_content": output.penalty_content,
            "regulator": output.regulator,
            "case_summary": output.case_summary,
            "risk_tags": risk_tags,
            "risk_tag_schema": "official_27_v1",
            "risk_domains": risk_domains,
            "primary_evidence_chunk_ids": [chunks[0]["chunk_id"]],
            "parse_status": "ok" if llm_used else "degraded",
            "review_status": "auto_checked" if llm_used else "manual_review_required",
            "retrieval_status": "active",
            "evidence_status": "verified" if valid_basis else "limited",
            "official_retrieval_protected": False,
            "data_version": self.data_version,
            "case_title": title,
            "region": None,
            "penalty_date": _valid_date(output.penalty_date),
            "publish_date": None,
            "related_parties": [value.model_dump() for value in output.related_parties],
            "entity_relations": entity_relations,
            "penalty_amounts": [],
            "legal_basis_refs": [value.model_dump() for value in output.legal_basis_refs],
            "case_scene": None,
            "scene_subtype": None,
            "scene_source": None,
            "keywords": list(dict.fromkeys(output.keywords)),
            "marketing_relevance_score": 8 if output.marketing_expressions else 5,
            "marketing_relevance_reason": "Web 增量流程根据处罚事实与营销表达生成",
            "original_risk_tags": risk_tags,
            "risk_tag_basis": [
                {
                    **basis.model_dump(),
                    "link_source": "web_llm_evidence",
                    "evidence_chunk_ids": [chunks[0]["chunk_id"]],
                }
                for basis in valid_basis
            ],
            "risk_tag_confidence": output.confidence,
            "quality_flags": [
                "web_incremental",
                *("llm_structured" if llm_used else "rule_fallback",),
            ],
            "source_file": source_file,
            "source_type": extraction.media_type,
            "raw_text_path": None,
            "source_url": None,
            "source_sha256": extraction.checksum_sha256,
            "field_confidence": field_confidence,
            "field_sources": field_sources,
            "created_at": today,
            "updated_at": today,
        }
        return CaseIngestionArtifacts(record, chunks, expansions, llm_used, warnings)

    async def build_many(
        self, extraction: FileExtractionResult, *, source_file: str
    ) -> list[CaseIngestionArtifacts]:
        """One file may contain several separately penalized parties; retain their own facts."""
        if self.llm_client is None:
            artifact = await self.build(extraction, source_file=source_file)
            scope = evaluate_insurance_scope(
                extraction.full_text,
                party_name=str(artifact.record.get("party_name") or ""),
                violation_behavior=str(artifact.record.get("violation_behavior") or ""),
            )
            artifact.record["scope_assessment"] = scope
            artifact.record["is_insurance_related"] = scope["relevance"] == "insurance"
            if scope["route"] == "scope_review":
                artifact.record.update(
                    retrieval_status="inactive",
                    review_status="pending",
                    retrieval_disposition="scope_review",
                    risk_tags=[],
                    risk_tag_basis=[],
                )
            return [artifact]
        request = JsonChatRequest(
            task="case_ingestion",
            temperature=0,
            max_tokens=16000,
            messages=[
                ChatMessage(
                    role="system",
                    content=(
                        "从监管处罚材料逐一拆出每个被处罚主体，返回{cases:[...]}。银行、保险、个人等分别判断，"
                        "is_insurance_related必须布尔：处罚事实涉及保险经营、保险代理、保险责任人员才为true。"
                        "不能只因文尾保险字样纳入；责任人员可以通过明确机构关联纳入。"
                        "只拆出原文明确被单独处罚的主体；仅被提及的法定代表人、负责人、"
                        "保险公司或披露总公司不要单独建案例。"
                        "每项字段：is_insurance_related,party_name,institution_type,penalty_doc_no,violation_behavior,"
                        "penalty_content,regulator,penalty_date,related_parties,legal_basis_refs,risk_tags,risk_tag_basis,"
                        "case_summary,keywords,marketing_expressions,confidence,fact_quotes。"
                        "confidence为0到1；日期YYYY-MM-DD，不明字段null；列表为空用[]。"
                        "fact_quotes为该主体违法事实的原文连续引文数组，必须逐字复制，不能引用其他主体事实。"
                        "PDF内的换行可以保留；主体名不得因行中换行而截断。"
                        "violation_behavior和penalty_content用完整短句并以句号收尾。"
                        "risk_tag_basis项为{risk_tag,reason,quote}。risk_tags保留全部有证据的标签，最多27个。"
                        "处罚金额和处罚内容按主体保留；不合计不同主体；无保险相关案例返回空cases。"
                        "related_parties项为{name,role}，legal_basis_refs项为{law_name,article}。"
                    ),
                ),
                ChatMessage(
                    role="user",
                    content="允许标签："
                    + json.dumps(self.taxonomy, ensure_ascii=False)
                    + "\n原文：\n"
                    + extraction.full_text[:60000],
                ),
            ],
        )
        selected: list[_CaseLLMOutput] = []
        for attempt in range(2):
            response = await self.llm_client.complete_json(request)
            try:
                raw = response.data.get("cases")
                if not isinstance(raw, list):
                    raise ValueError("多主体抽取缺少cases数组")
                outputs = [_CaseLLMOutput.model_validate(r) for r in raw]
                selected = [
                    _bind_source_quotes(item, extraction.full_text)
                    for item in outputs
                    if item.is_insurance_related is True
                ]
                break
            except ValueError as error:
                if attempt:
                    raise LLMProviderError(
                        "案例抽取结果未能对应原文，请重新录入或核对文件文字"
                    ) from error
                request = request.model_copy(
                    update={
                        "messages": [
                            *request.messages,
                            ChatMessage(
                                role="assistant",
                                content=json.dumps(response.data, ensure_ascii=False),
                            ),
                            ChatMessage(
                                role="user",
                                content=(
                                    "上次结果未通过原文校验，请重新抽取。只接受原文中连续存在的主体和事实；"
                                    "可以忽略排版空白，不能改变数字、标点或用词。不合并不连续的引文。"
                                ),
                            ),
                        ]
                    }
                )
        artifacts = []
        for index, item in enumerate(selected, 1):
            artifacts.append(
                await self.build(
                    extraction, source_file=source_file, structured=item, atomic_index=index
                )
            )
        return artifacts

    async def build_imported(
        self,
        extraction: FileExtractionResult,
        *,
        source_file: str,
        payload: dict[str, object],
        atomic_index: int = 1,
    ) -> CaseIngestionArtifacts:
        """Validate existing structured fields and literal quotes without requesting a model."""
        output = _CaseLLMOutput.model_validate(payload)
        if output.is_insurance_related is not True:
            raise ValueError("导入案例未确认保险相关性")
        if not output.party_name or output.party_name not in extraction.full_text:
            raise ValueError("导入案例的主体未对应原文")
        if not output.fact_quotes or any(q not in extraction.full_text for q in output.fact_quotes):
            raise ValueError("导入案例的事实引文未对应原文")
        if set(output.risk_tags) - self._taxonomy_set:
            raise ValueError("导入案例包含27类以外的风险标签")
        if any(b.quote not in extraction.full_text for b in output.risk_tag_basis):
            raise ValueError("导入标签的引文未对应原文")
        for field in ("penalty_content", "penalty_doc_no", "regulator"):
            value = getattr(output, field)
            if value and value not in extraction.full_text:
                raise ValueError(f"导入案例的{field}未对应原文")
        import_kind = str(payload.get("import_kind") or "prestructured_release")
        if import_kind not in {"prestructured_release", "legacy_structured", "collected_fields"}:
            raise ValueError("未知的已有材料导入方式")
        built = await self.build(
            extraction, source_file=source_file, structured=output, atomic_index=atomic_index
        )
        record = built.record
        record.update(
            parse_status="ok",
            review_status="pending",
            risk_tag_confidence=None,
            quality_flags=[f"{import_kind}_import", "literal_evidence_checked"],
            field_confidence={},
            human_review_status="pending",
            human_approved=False,
            marketing_relevance_reason="从已采集原文导入并核对处罚事实与标签",
        )
        record["field_sources"] = {
            name: {"origin": import_kind, "check": "literal_source_validation"}
            for name in (
                "party_name",
                "violation_behavior",
                "penalty_content",
                "penalty_doc_no",
                "regulator",
                "penalty_date",
            )
            if record.get(name)
        }
        basis_rows = record.get("risk_tag_basis", [])
        if isinstance(basis_rows, list):
            for basis in basis_rows:
                if isinstance(basis, dict):
                    basis["link_source"] = import_kind
        if isinstance(payload.get("upstream_case_id"), str):
            record["upstream_case_id"] = payload["upstream_case_id"]
        if isinstance(payload.get("party_type"), str):
            record["party_type"] = payload["party_type"]
        if isinstance(payload.get("penalty_amounts"), list):
            record["penalty_amounts"] = payload["penalty_amounts"]
        return CaseIngestionArtifacts(record, built.chunks, built.expansions, False, built.warnings)

    async def _call_llm(self, text: str) -> _CaseLLMOutput:
        if self.llm_client is None:
            raise LLMProviderError("大模型未配置")
        excerpt = text if len(text) <= 40_000 else f"{text[:25_000]}\n[中间省略]\n{text[-15_000:]}"
        # 输出含完整案由/处罚内容/摘要/引文，3000 tokens 会在长文书上截断 JSON，
        # 因此放宽到 8000；若仍触发 length 上限，按 16000 重试一次，避免静默落入
        # 正则降级而产生"半句话"字段。
        last_error: LLMProviderError | None = None
        for token_budget in (8000, 16000):
            try:
                response = await self._request_structured_case(excerpt, token_budget)
            except LLMProviderError as error:
                last_error = error
                if "截断" in str(error):
                    continue
                raise
            if response.finish_reason == "length":
                last_error = LLMProviderError("大模型输出达到 max_tokens 上限被截断")
                continue
            try:
                return _CaseLLMOutput.model_validate(response.data)
            except ValidationError as error:
                raise LLMProviderError("DeepSeek 案例输出不符合字段契约") from error
        raise last_error or LLMProviderError("大模型服务返回了无法解析的响应")

    async def _request_structured_case(self, excerpt: str, max_tokens: int) -> JsonChatResponse:
        if self.llm_client is None:
            raise LLMProviderError("大模型未配置")
        return await self.llm_client.complete_json(
            JsonChatRequest(
                task="case_ingestion",
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "你是保险监管处罚案例结构化专家。只能提取原文明确存在的事实；"
                            "不确定字段必须为null。party_name只填主要被处罚方的完整名称"
                            '（机构全称或个人姓名），不得含"当事人"等标签字样、地址、'
                            "身份证号或职务说明。violation_behavior必须是违法违规事实的完整"
                            "表述、以句号收尾，禁止在半句话处截断；penalty_content必须是处罚"
                            "决定的完整表述、以句号收尾，不得混入当事人身份信息。"
                            "risk_tags只能从给定27标签选择全部明确成立的类别；"
                            "risk_tag_basis.quote必须是原文逐字片段。"
                            'legal_basis_refs每项必须是对象{"law_name":"法条名","article":"第X条或null"}；'
                            "所有列表字段没有内容时输出[]，禁止输出null。输出单个JSON对象，字段为："
                            "case_title,party_name,institution_type,penalty_doc_no,violation_behavior,"
                            "penalty_content,regulator,penalty_date(YYYY-MM-DD或null),related_parties,"
                            "legal_basis_refs,risk_tags,risk_tag_basis,case_summary,keywords,"
                            "marketing_expressions,confidence。"
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            f"风险标签：{json.dumps(self.taxonomy, ensure_ascii=False)}\n"
                            f"处罚材料原文：\n{excerpt}"
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        )

    def _fallback(self, text: str) -> _CaseLLMOutput:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        title = lines[0][:500] if lines else None
        document_number = _first_match(
            r"[^\s，。]{0,16}(?:罚决字|罚字)[〔\[]?\d{4}[〕\]]?\d+号", text
        )
        regulator = _first_match(r"[^\n，。]{2,40}(?:金融监管局|银保监局|银保监分局)", text)
        penalty_date = _extract_date(text)
        penalty = _first_match(
            r"[^。；\n]{0,100}(?:警告|罚款|没收|责令)[^。；\n]{0,200}[。；]?", text
        )
        facts = _extract_facts(text)
        rule_result = self.rule_detector.detect(text)
        labels = [label for label in rule_result.risk_labels if label in self._taxonomy_set]
        return _CaseLLMOutput(
            case_title=title,
            penalty_doc_no=document_number,
            violation_behavior=facts,
            penalty_content=penalty,
            regulator=regulator,
            penalty_date=penalty_date,
            risk_tags=labels,
            case_summary=facts[:900],
            keywords=labels,
            confidence=0.45,
        )


def _literal_quote(quote: str, text: str) -> str | None:
    """容忍 PDF 空白排版差异，返回真实连续原文；不容忍改写或拼接。"""
    if quote and quote in text:
        return quote
    compact_quote = "".join(quote.split())
    if not compact_quote:
        return None
    positions = [index for index, char in enumerate(text) if not char.isspace()]
    compact_text = "".join(text[index] for index in positions)
    start = compact_text.find(compact_quote)
    if start < 0:
        return None
    return text[positions[start] : positions[start + len(compact_quote) - 1] + 1]


def _bind_source_quotes(output: _CaseLLMOutput, text: str) -> _CaseLLMOutput:
    if not output.party_name or _literal_quote(output.party_name, text) is None:
        raise ValueError("被处罚主体需对应原文名称")
    quotes = [_literal_quote(quote, text) for quote in output.fact_quotes]
    if not quotes or any(quote is None for quote in quotes):
        raise ValueError("处罚事实引文未对应原文")
    basis = []
    for item in output.risk_tag_basis:
        quote = _literal_quote(item.quote, text)
        if quote is not None:
            basis.append(item.model_copy(update={"quote": quote}))
    return output.model_copy(update={"fact_quotes": quotes, "risk_tag_basis": basis})


def load_risk_domain_mapping(path: str) -> dict[str, str]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("风险归口映射必须是 JSON 对象")
    return {str(key): str(domain) for key, domain in value.items()}


_SENTENCE_END = tuple("。；;！？!?”』】)")
_MID_CLAUSE_END = tuple("对的和及与并或、，,；：:等为向把被从由按其于在到")
_PARTY_DIRTY_RE = re.compile(r"^(?:当事人|被处罚人)\s*[：:]|身份证|地址|住所|住址")


def _field_completeness_warnings(output: _CaseLLMOutput) -> list[str]:
    """抽取结果的事后完整性检查：截断/错位只产生警告，不阻断入库。"""

    warnings: list[str] = []
    violation = output.violation_behavior.rstrip()
    if violation and not violation.endswith(_SENTENCE_END):
        warnings.append("violation_behavior 未以完整句子收尾，可能存在截断，请对照原文复核")
    if violation and violation.endswith(_MID_CLAUSE_END):
        warnings.append("violation_behavior 在虚词处结尾，疑似半句截断，请对照原文复核")
    penalty = (output.penalty_content or "").rstrip()
    if penalty and penalty.endswith(_MID_CLAUSE_END):
        warnings.append("penalty_content 在虚词处结尾，疑似半句截断，请对照原文复核")
    if penalty and re.search(
        r"当事人[：:].{0,200}?(身份证|地址|住所|统一社会信用代码)",
        penalty,
        re.S,
    ):
        warnings.append("penalty_content 混入当事人身份信息，疑似字段错位，请对照原文复核")
    if output.party_name and _PARTY_DIRTY_RE.search(output.party_name):
        warnings.append("party_name 含标签前缀或地址/身份信息，请对照原文复核")
    return warnings


def _build_chunks(
    *,
    case_id: str,
    file_id: str,
    extraction: FileExtractionResult,
    violation_behavior: str,
    penalty_content: str | None,
    legal_basis: list[_LegalBasis],
    risk_tags: list[str],
    data_version: str,
) -> list[dict[str, object]]:
    values: list[tuple[str, str, float]] = [("fact", violation_behavior, 1.0)]
    if penalty_content:
        values.append(("penalty", penalty_content, 0.8))
    if legal_basis:
        legal_text = "；".join(f"{item.law_name}{item.article or ''}" for item in legal_basis)
        values.append(("legal_basis", legal_text, 0.65))
    chunks: list[dict[str, object]] = []
    for index, (chunk_type, chunk_text, weight) in enumerate(values, start=1):
        start = extraction.full_text.find(chunk_text)
        source_span = {
            "start_char": start if start >= 0 else None,
            "end_char": start + len(chunk_text) if start >= 0 else None,
            "page_no": None,
            "table_row": None,
            "section_title": None,
        }
        chunk_id = f"CHK_{case_id}_{chunk_type.upper()}_{index:03d}"
        chunks.append(
            {
                "schema_version": "case_chunk_v1",
                "chunk_id": chunk_id,
                "case_id": case_id,
                "file_id": file_id,
                "chunk_type": chunk_type,
                "chunk_text": chunk_text,
                "chunk_order": index,
                "retrieval_roles": [chunk_type, "evidence"]
                if chunk_type == "fact"
                else [chunk_type],
                "evidence_level": "direct" if start >= 0 else "normalized",
                "supports_risk_tags": risk_tags if chunk_type == "fact" else [],
                "is_verbatim": start >= 0,
                "source_span": source_span,
                "retrieval_enabled": True,
                "retrieval_modes": ["dense", "sparse"] if chunk_type == "fact" else ["sparse"],
                "retrieval_weight": weight,
                "extraction_method": extraction.extraction_method.value,
                "extraction_confidence": extraction.quality_score or 0.5,
                "quality_flags": [],
                "text_sha256": sha256(chunk_text.encode()).hexdigest(),
                "data_version": data_version,
            }
        )
    return chunks


def _build_expansions(
    *,
    case_id: str,
    values: list[str],
    risk_tags: list[str],
    risk_domains: list[str],
    data_version: str,
) -> list[dict[str, object]]:
    expansions: list[dict[str, object]] = []
    for value in dict.fromkeys(text.strip() for text in values if text.strip()):
        digest = sha256(f"{case_id}\0{value}".encode()).hexdigest().upper()
        expansions.append(
            {
                "schema_version": "case_expansion_v1",
                "expansion_id": f"EXP_{digest[:16]}",
                "expansion_text": value,
                "expansion_type": "marketing_expression",
                "case_links": [
                    {
                        "case_id": case_id,
                        "relevance": "high",
                        "link_source": "web_incremental",
                        "link_reason": "增量案例结构化流程生成",
                        "grounding_chunk_ids": [f"CHK_{case_id}_FACT_001"],
                    }
                ],
                "risk_tags": risk_tags,
                "risk_domains": risk_domains,
                "coarse_scene": None,
                "scene_subtype": None,
                "speaker_role": None,
                "communication_channel": None,
                "text_origin": "llm_or_rule",
                "fact_grounding": "case_fact",
                "synthetic_details": [],
                "prompt_version": "case-ingestion-v1",
                "source_ref": case_id,
                "quality_status": "generated_clean",
                "quality_flags": ["web_incremental"],
                "retrieval_enabled": True,
                "retrieval_modes": ["sparse"],
                "retrieval_weight": 0.35,
                "specificity_score": 0.7,
                "dataset_split": "none",
                "allowed_usage": "retrieval",
                "text_sha256": sha256(value.encode()).hexdigest(),
                "data_version": data_version,
            }
        )
    return expansions


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def _extract_date(text: str) -> str | None:
    match = re.search(r"(20\d{2})[年./-](\d{1,2})[月./-](\d{1,2})日?", text)
    if not match:
        return None
    try:
        return date(int(match[1]), int(match[2]), int(match[3])).isoformat()
    except ValueError:
        return None


def _valid_date(value: str | None) -> str | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return None


def _extract_facts(text: str) -> str:
    match = re.search(
        r"(?:主要违法违规事实|违法违规事实|违法事实|案由)[：:\s]*(.{20,5000}?)(?=处罚依据|处罚决定|行政处罚|$)",
        text,
        re.DOTALL,
    )
    return (match.group(1) if match else text[:5000]).strip()


def _field_quality(
    output: _CaseLLMOutput,
    *,
    text: str,
    llm_used: bool,
    extraction_quality: float | None,
) -> tuple[dict[str, float], dict[str, dict[str, str]]]:
    """逐字段估计置信度，并说明字段是原文直取还是模型/规则归纳。"""

    method = "deepseek_structured" if llm_used else "rule_fallback"
    base = output.confidence if llm_used else min(output.confidence, 0.65)
    if extraction_quality is not None:
        base *= 0.75 + 0.25 * extraction_quality
    values: dict[str, object] = {
        "case_title": output.case_title,
        "party_name": output.party_name,
        "institution_type": output.institution_type,
        "penalty_doc_no": output.penalty_doc_no,
        "violation_behavior": output.violation_behavior,
        "penalty_content": output.penalty_content,
        "regulator": output.regulator,
        "penalty_date": output.penalty_date,
        "related_parties": output.related_parties,
        "legal_basis_refs": output.legal_basis_refs,
        "risk_tags": output.risk_tags,
        "case_summary": output.case_summary,
    }
    confidence: dict[str, float] = {"overall": round(base, 4)}
    sources: dict[str, dict[str, str]] = {}
    for field_name, value in values.items():
        snippets = _field_snippets(value)
        if not snippets:
            confidence[field_name] = 0.0
            sources[field_name] = {"method": method, "grounding": "missing"}
            continue
        verbatim = any(snippet in text for snippet in snippets if len(snippet) >= 2)
        score = min(1.0, base + 0.1) if verbatim else max(0.0, base - 0.08)
        confidence[field_name] = round(score, 4)
        sources[field_name] = {
            "method": method,
            "grounding": "verbatim" if verbatim else "derived",
        }
    return confidence, sources


def _field_snippets(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, BaseModel):
        return _field_snippets(value.model_dump())
    if isinstance(value, dict):
        return [snippet for item in value.values() for snippet in _field_snippets(item)]
    if isinstance(value, list):
        return [snippet for item in value for snippet in _field_snippets(item)]
    return [str(value)]


def _entity_relations(
    output: _CaseLLMOutput,
    field_confidence: dict[str, float],
) -> list[dict[str, object]]:
    """构造主要被处罚方、监管机构和相关主体之间的可复核关系表。"""

    relations: list[dict[str, object]] = []
    party = output.party_name
    regulator = output.regulator
    if party and regulator:
        relations.append(
            {
                "subject": regulator,
                "predicate": "处罚",
                "object": party,
                "confidence": min(
                    field_confidence.get("regulator", 0.0),
                    field_confidence.get("party_name", 0.0),
                ),
            }
        )
    if party and output.institution_type:
        relations.append(
            {
                "subject": party,
                "predicate": "机构类型",
                "object": output.institution_type,
                "confidence": field_confidence.get("institution_type", 0.0),
            }
        )
    for related in output.related_parties:
        if not party:
            break
        relations.append(
            {
                "subject": party,
                "predicate": related.role or "关联主体",
                "object": related.name,
                "confidence": field_confidence.get("related_parties", 0.0),
            }
        )
    return relations
