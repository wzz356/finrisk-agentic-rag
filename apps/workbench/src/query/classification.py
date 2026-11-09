"""六类 Query 场景的规则优先、LLM 兜底分类。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field, ValidationError

from domain.query_models import (
    QueryClassificationMethod,
    QueryScene,
    QuerySceneClassification,
    SceneRuleCandidate,
)
from llm import ChatMessage, JsonChatRequest, LLMClient, LLMProviderError

TAXONOMY_VERSION = "query-scene-2.0"
CLASSIFIER_VERSION = "query-scene-rule-llm-3.3"
RULE_VERSION = "query-scene-rules-3.3"

SCENE_NAMES = {
    QueryScene.ST1_POSTER_SMS_SOCIAL: "海报/短信/朋友圈短文案",
    QueryScene.ST2_LIVE_SHORT_VIDEO: "直播/短视频/口播话术",
    QueryScene.ST3_PRODUCT_INTRO_DETAIL: "产品介绍/详情页",
    QueryScene.ST4_MEETING_MATERIAL: "产品说明会/客户答谢会材料",
    QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK: "培训/电销/回访流程话术",
    QueryScene.ST6_PENALTY_BACKEND_FACT: "监管处罚文书/后台业务事实",
}

_SCENE_DEFINITIONS = """
ST1_poster_sms_social：海报、营销短信、朋友圈或社交平台文案，包括小红书种草、求助、理赔经历和多行图文。表情符号、宝子/姐妹称呼、翻图、点击头像等是社交线索；含产品参数或较长篇幅仍按社交载体分类。
ST2_live_short_video：直播、短视频脚本、主播或口播话术，常含镜头、主播、直播间、小黄车等表达。
ST3_product_intro_detail：产品介绍、详情页、产品说明内容，重点描述产品要素、功能、期限、费用、申赎和风险等级；若文本只陈述保险或产品功能且没有其他渠道载体标识，归入本类。
ST4_meeting_material：产品说明会、客户答谢会等线下或线上会议材料，常含议程、主持、致辞、来宾和会议安排。
ST5_training_telemarketing_callback：内部培训、电销、客服回访流程或标准话术，常含坐席、客户问答、异议处理和回访步骤。
ST6_penalty_backend_fact：监管处罚决定、违法事实，或后台业务事实、系统记录、审批记录、交易流水、操作日志。
""".strip()


@dataclass(frozen=True, slots=True)
class QueryClassificationContext:
    """只包含文件格式特征，不使用不可确定的来源渠道。"""

    media_type: str | None = None
    page_count: int | None = None


@dataclass(frozen=True, slots=True)
class ClassifierStatus:
    """当前规则和 LLM 配置状态。"""

    taxonomy_version: str
    classifier_version: str
    rule_version: str
    llm_configured: bool
    llm_model: str | None


@dataclass(frozen=True, slots=True)
class _Rule:
    scene: QueryScene
    pattern: re.Pattern[str]
    weight: float
    description: str


@dataclass(frozen=True, slots=True)
class _RuleEvaluation:
    candidates: list[SceneRuleCandidate]
    high_confidence: bool


class _LLMSceneOutput(BaseModel):
    scene: QueryScene
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=1)


_RULES = (
    _Rule(
        QueryScene.ST1_POSTER_SMS_SOCIAL,
        re.compile(r"(?:海报|朋友圈|小红书|营销短信|短信文案|短文案|社交平台文案|回复TD退订)"),
        7.0,
        "出现海报、短信或社交短文案的明确标识",
    ),
    _Rule(
        QueryScene.ST1_POSTER_SMS_SOCIAL,
        re.compile(r"(?:扫码|长按识别|点击链接|立即领取|限时|转发|分享有礼|退订)"),
        2.2,
        "出现短营销文案的行动号召",
    ),
    _Rule(
        QueryScene.ST2_LIVE_SHORT_VIDEO,
        re.compile(
            r"(?:直播间|直播脚本|短视频脚本|口播稿|口播话术|主播|镜头[一二三四五六七八九十\d])"
        ),
        7.0,
        "出现直播、短视频或口播的明确标识",
    ),
    _Rule(
        QueryScene.ST2_LIVE_SHORT_VIDEO,
        re.compile(r"(?:家人们|宝宝们|老铁们|点个关注|小黄车|上链接|欢迎来到直播间|三二一)"),
        2.5,
        "出现直播或口播惯用表达",
    ),
    _Rule(
        QueryScene.ST2_LIVE_SHORT_VIDEO,
        re.compile(r"(?:我(?:来|跟你|给你|说)|听我说|发到直播间|评论区发来|平台负责|先把名额占住)"),
        2.4,
        "出现第一人称实时口播或直播互动表达",
    ),
    _Rule(
        QueryScene.ST3_PRODUCT_INTRO_DETAIL,
        re.compile(r"(?:产品介绍|产品详情|详情页|产品要素|产品说明书|产品概述)"),
        7.0,
        "出现产品介绍或详情页的明确标识",
    ),
    _Rule(
        QueryScene.ST3_PRODUCT_INTRO_DETAIL,
        re.compile(
            r"(?:产品名称|风险等级|投资期限|起购金额|业绩比较基准|申购|赎回|管理费|托管费|适合投资者)"
        ),
        1.7,
        "出现产品参数或交易要素",
    ),
    _Rule(
        QueryScene.ST4_MEETING_MATERIAL,
        re.compile(r"(?:产品说明会|客户答谢会|投资者交流会|说明会材料|答谢会材料)"),
        8.0,
        "出现产品说明会或客户答谢会的明确标识",
    ),
    _Rule(
        QueryScene.ST4_MEETING_MATERIAL,
        re.compile(r"(?:会议议程|主持词|开场致辞|各位来宾|签到|茶歇|会议时间|会议地点|嘉宾介绍)"),
        2.0,
        "出现会议组织或现场材料表达",
    ),
    _Rule(
        QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        re.compile(
            r"(?:培训材料|培训课件|说服语言|推销话术|销售话术|保险营销话术|促成话术|邀约话术|电销话术|回访话术|回访流程|异议处理|标准话术|电话销售)"
        ),
        7.0,
        "出现培训、推销、电销或回访话术的明确标识",
    ),
    _Rule(
        QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        re.compile(r"(?:坐席|客户|客服|销售|顾问)[：:]|(?:您好.{0,20}(?:致电|来电|回访))"),
        2.2,
        "出现坐席与客户的流程化对话",
    ),
    _Rule(
        QueryScene.ST6_PENALTY_BACKEND_FACT,
        re.compile(
            r"(?:行政处罚决定书|监管处罚|处罚决定|罚字|"
            r"(?:金融监管总局|银保监|人民银行).{0,16}(?:处罚|决定书|罚款))"
        ),
        8.0,
        "出现监管处罚文书的明确标识",
    ),
    _Rule(
        QueryScene.ST6_PENALTY_BACKEND_FACT,
        re.compile(r"(?:经查|违反.{0,30}规定|决定给予|罚款|没收违法所得|违法事实)"),
        2.0,
        "出现处罚事实和处罚结论表达",
    ),
    _Rule(
        QueryScene.ST6_PENALTY_BACKEND_FACT,
        re.compile(
            r"(?:后台业务事实|业务事实|后台记录|系统记录|审批记录|交易流水|操作日志|业务台账)"
        ),
        7.0,
        "出现后台业务事实或系统留痕",
    ),
    _Rule(
        QueryScene.ST2_LIVE_SHORT_VIDEO,
        re.compile(r"(?:短视频口播|短视频文案|主播台词|直播提纲|直播演示|镜头脚本)"),
        7.5,
        "出现短视频或直播内容载体标识",
    ),
    _Rule(
        QueryScene.ST3_PRODUCT_INTRO_DETAIL,
        re.compile(r"(?:保险详情|险种详情|产品宣传页|保障详情|对比宣传页|产品对比页|投保本产品)"),
        7.5,
        "出现产品详情或产品页面标识",
    ),
    _Rule(
        QueryScene.ST4_MEETING_MATERIAL,
        re.compile(
            r"(?:说明会|客户交流会|养老规划(?:会讲义|讲座)|社区养老讲座|讲师页|"
            r"讲师备注|会议讲义|活动执行单)"
        ),
        7.5,
        "出现说明会、讲座或会议讲师材料标识",
    ),
    _Rule(
        QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
        re.compile(
            r"(?:回访操作提示|回访质检|电销标准话术|电销脚本|电话促成脚本|电话销售安排|"
            r"销售培训|内部培训|合规异议处理|坐席整改|标准应答)"
        ),
        7.5,
        "出现培训、电销或回访流程标识",
    ),
    _Rule(
        QueryScene.ST6_PENALTY_BACKEND_FACT,
        re.compile(
            r"(?:财务凭证|系统抽查记录|后台审核记录|后台抽查|系统日志|业务后台|"
            r"稽核发现|审计发现|抽查发现|回访质检发现|质检记录显示|"
            r"佣金流水|承保记录|业务台账|审批留痕)"
        ),
        8.5,
        "出现财务、系统、审计或后台事实记录标识",
    ),
    _Rule(
        QueryScene.ST6_PENALTY_BACKEND_FACT,
        re.compile(
            r"(?:检查组|检查人员|稽核人员|监管人员).{0,24}(?:调取|抽取|核对|发现)|"
            r"(?:时间戳|上传ip|上传IP|门禁记录|签名时间|验证码发送时间).{0,28}(?:显示|早于|一致|异常)"
        ),
        7.8,
        "出现检查取证、时间戳或后台异常事实",
    ),
)


class RuleBasedSceneClassifier:
    """使用关键词、文体标识和文件格式特征计算六类规则分数。"""

    def evaluate(
        self,
        text: str,
        context: QueryClassificationContext | None = None,
    ) -> _RuleEvaluation:
        if not text.strip():
            raise ValueError("Query 文本不能为空")

        scores = {scene: 0.0 for scene in QueryScene}
        evidence: dict[QueryScene, list[str]] = {scene: [] for scene in QueryScene}
        for rule in _RULES:
            matches = rule.pattern.findall(text)
            if not matches:
                continue
            count = min(len(matches), 3)
            scores[rule.scene] += rule.weight * count
            evidence[rule.scene].append(f"{rule.description}（{len(matches)} 处）")

        self._add_structural_scores(text, context, scores, evidence)
        # An explicit material heading identifies the carrier. Mentions of other
        # carriers inside examples must not outvote it or trigger an API call.
        heading = text.lstrip().splitlines()[0][:110]
        heading_rules = (
            (
                QueryScene.ST1_POSTER_SMS_SOCIAL,
                r"小红书文案|朋友圈|短信文案|海报文案|朋友圈/短信|朋友圈/客户群",
            ),
            (
                QueryScene.ST2_LIVE_SHORT_VIDEO,
                r"直播口播|直播间|短视频|口播脚本|直播口播话术",
            ),
            (QueryScene.ST3_PRODUCT_INTRO_DETAIL, r"产品详情页|产品介绍页|保险公估服务说明页"),
            (QueryScene.ST4_MEETING_MATERIAL, r"客户答谢会|产品说明会|产说会宣讲材料"),
            (
                QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK,
                r"培训转述|培训话术|培训材料|培训现场|培训课件|电销坐席|电销及回访|回访话术培训",
            ),
            (
                QueryScene.ST6_PENALTY_BACKEND_FACT,
                r"后台业务|后台核查|后台归档|后台记录|监管处罚文书|行政处罚决定书",
            ),
        )
        explicit = [
            (match.start(), scene)
            for scene, pattern in heading_rules
            if (match := re.search(pattern, heading)) and match.start() <= 18
        ]
        if explicit and len({scene for _, scene in explicit}) == 1:
            _, heading_scene = min(explicit, key=lambda pair: pair[0])
            scores[heading_scene] += 32
            evidence[heading_scene].append("首行明确标识材料载体")
        total = sum(scores.values())
        candidates = [
            SceneRuleCandidate(
                scene=scene,
                score=round(score, 4),
                confidence=round(score / total, 6) if total else 0.0,
                evidence=evidence[scene],
            )
            for scene, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]
        top = candidates[0]
        second = candidates[1]
        high_confidence = (
            top.score >= 6.0 and top.score - second.score >= 2.5 and top.confidence >= 0.65
        )
        return _RuleEvaluation(candidates=candidates, high_confidence=high_confidence)

    @staticmethod
    def _add_structural_scores(
        text: str,
        context: QueryClassificationContext | None,
        scores: dict[QueryScene, float],
        evidence: dict[QueryScene, list[str]],
    ) -> None:
        compact_length = len("".join(text.split()))
        social_voice = re.search(
            r"宝子|姐妹|宝妈|谁懂|看过来|夸累了|种草|安利|卷王|内卷|闭眼入|闭眼冲|"
            r"翻图|点击头像|发送暗号|二编|求解答|家人们",
            text,
        )
        social_bullets = len(re.findall(r"(?m)^\s*[✅⭐👉🔥✔💡❤💁🔒🎁]", text))
        live_carrier = re.search(r"直播间|主播|镜头|小黄车|口播|短视频脚本", text)
        if social_voice and social_bullets >= 3 and not live_carrier:
            scores[QueryScene.ST1_POSTER_SMS_SOCIAL] += 9.0
            evidence[QueryScene.ST1_POSTER_SMS_SOCIAL].append(
                "社交称呼或种草表达与多行表情列表共同出现"
            )
        if context and context.media_type and context.media_type.startswith("image/"):
            if compact_length <= 600:
                scores[QueryScene.ST1_POSTER_SMS_SOCIAL] += 3.0
                evidence[QueryScene.ST1_POSTER_SMS_SOCIAL].append(
                    "输入是单页短文本图片，符合海报或社交短文案形态"
                )

        dialogue_roles = len(re.findall(r"(?:坐席|客户|客服|销售|顾问)[：:]", text))
        if dialogue_roles >= 2:
            scores[QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK] += 3.0
            evidence[QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK].append(
                "检测到多轮角色化客户沟通结构"
            )

        parameter_count = len(
            re.findall(r"(?:风险等级|投资期限|起购金额|申购|赎回|管理费|托管费)", text)
        )
        if parameter_count >= 4:
            scores[QueryScene.ST3_PRODUCT_INTRO_DETAIL] += 3.0
            evidence[QueryScene.ST3_PRODUCT_INTRO_DETAIL].append(
                "多项产品参数集中出现，符合详情页结构"
            )


class QuerySceneClassifier:
    """高置信规则直接输出，其余交给统一 LLM 接口。"""

    def __init__(
        self,
        llm_client: LLMClient | None,
        *,
        rule_classifier: RuleBasedSceneClassifier | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._rule_classifier = rule_classifier or RuleBasedSceneClassifier()

    @property
    def status(self) -> ClassifierStatus:
        return ClassifierStatus(
            taxonomy_version=TAXONOMY_VERSION,
            classifier_version=CLASSIFIER_VERSION,
            rule_version=RULE_VERSION,
            llm_configured=self._llm_client is not None,
            llm_model=self._llm_client.model_name if self._llm_client else None,
        )

    async def classify(
        self,
        text: str,
        context: QueryClassificationContext | None = None,
        *,
        scene_override: QueryScene | None = None,
    ) -> QuerySceneClassification:
        try:
            return await self._classify(text, context, scene_override=scene_override)
        except LLMProviderError:
            evaluation = self._rule_classifier.evaluate(text, context)
            top = evaluation.candidates[0]
            return QuerySceneClassification(
                scene=top.scene if top.score > 0 else _fallback_scene(text),
                confidence=top.confidence if top.score > 0 else 0.25,
                method=QueryClassificationMethod.RULE_FALLBACK,
                evidence=[*top.evidence, "场景模型暂不可用，按载体线索分类"],
                rule_candidates=evaluation.candidates,
                classifier_version=CLASSIFIER_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
            )

    async def _classify(
        self,
        text: str,
        context: QueryClassificationContext | None = None,
        *,
        scene_override: QueryScene | None = None,
    ) -> QuerySceneClassification:
        evaluation = self._rule_classifier.evaluate(text, context)
        top = evaluation.candidates[0]
        if scene_override is not None:
            return QuerySceneClassification(
                scene=scene_override,
                confidence=1.0,
                method=QueryClassificationMethod.USER,
                evidence=["用户指定材料场景"],
                rule_candidates=evaluation.candidates,
                classifier_version=CLASSIFIER_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
            )
        if evaluation.high_confidence:
            return QuerySceneClassification(
                scene=top.scene,
                confidence=top.confidence,
                method=QueryClassificationMethod.RULE,
                evidence=top.evidence,
                rule_candidates=evaluation.candidates,
                classifier_version=CLASSIFIER_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
            )

        if self._llm_client is None:
            fallback_scene = top.scene if top.score > 0 else _fallback_scene(text)
            fallback_confidence = top.confidence if top.score > 0 else 0.25
            return QuerySceneClassification(
                scene=fallback_scene,
                confidence=fallback_confidence,
                method=QueryClassificationMethod.RULE_FALLBACK,
                evidence=[*top.evidence, "未配置大模型，采用本地规则最高分候选"],
                rule_candidates=evaluation.candidates,
                classifier_version=CLASSIFIER_VERSION,
                taxonomy_version=TAXONOMY_VERSION,
            )

        llm_response = await self._llm_client.complete_json(
            JsonChatRequest(
                task="scene_classification",
                messages=[
                    ChatMessage(
                        role="system",
                        content=(
                            "你是金融材料场景分类器。只能从以下六类中选择一类。"
                            "请仅输出 json 对象，格式为 "
                            '{"scene":"六类代码之一","confidence":0到1,"reason":"简短依据"}。\n'
                            f"{_SCENE_DEFINITIONS}"
                        ),
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            "请对下面整份材料分类。规则候选仅供参考，若多个候选冲突，"
                            "请根据材料的主要用途作出唯一选择。\n"
                            f"规则候选：{_candidate_json(evaluation.candidates)}\n"
                            f"材料正文：\n{_classification_excerpt(text)}"
                        ),
                    ),
                ],
                temperature=0.0,
                max_tokens=300,
            )
        )
        try:
            output = _LLMSceneOutput.model_validate(llm_response.data)
        except ValidationError as error:
            raise LLMProviderError("大模型返回的场景分类不符合六类字段约定") from error

        return QuerySceneClassification(
            scene=output.scene,
            confidence=output.confidence,
            method=QueryClassificationMethod.LLM,
            evidence=[output.reason],
            rule_candidates=evaluation.candidates,
            classifier_version=CLASSIFIER_VERSION,
            taxonomy_version=TAXONOMY_VERSION,
            llm_model=llm_response.model,
        )


def _fallback_scene(text: str) -> QueryScene:
    """没有模型且规则无命中时，按文体给出可解释的保守路由。"""

    if re.search(r"(?:处罚|监管|检查|系统记录|业务台账|审计|违法事实)", text):
        return QueryScene.ST6_PENALTY_BACKEND_FACT
    if re.search(r"(?:坐席|回访|培训|销售话术|客户[：:])", text):
        return QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK
    if len("".join(text.split())) <= 120:
        return QueryScene.ST1_POSTER_SMS_SOCIAL
    return QueryScene.ST3_PRODUCT_INTRO_DETAIL


def scene_name(scene: QueryScene) -> str:
    """返回六类场景的中文名称。"""

    return SCENE_NAMES[scene]


def _candidate_json(candidates: list[SceneRuleCandidate]) -> str:
    values = [
        {
            "scene": candidate.scene.value,
            "score": candidate.score,
            "evidence": candidate.evidence,
        }
        for candidate in candidates
        if candidate.score > 0
    ]
    return json.dumps(values, ensure_ascii=False)


def _classification_excerpt(text: str, max_chars: int = 24_000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return f"{text[:half]}\n\n[中间内容已省略]\n\n{text[-half:]}"


__all__ = [
    "CLASSIFIER_VERSION",
    "ClassifierStatus",
    "QueryClassificationContext",
    "QuerySceneClassifier",
    "RULE_VERSION",
    "RuleBasedSceneClassifier",
    "SCENE_NAMES",
    "TAXONOMY_VERSION",
    "scene_name",
]
