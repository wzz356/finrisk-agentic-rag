"""场景感知的两阶段本地风险初判。"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from domain.query_models import QueryScene
from domain.risk_models import RiskEvidence
from query.context import has_adopted_instruction

REGEX_LABEL_MAP: dict[str, tuple[str, ...]] = {
    "unlicensed_premises": ("其他",),
    "absolute_guarantee": ("承诺收益", "保证收益"),
    "bank_comparison": ("不当比较",),
    "deposit_disguise": ("销售误导",),
    "no_risk_claim": ("绝对化表述",),
    "urgency_pressure": ("诱导投保",),
    "gift_inducement": ("赠送利益",),
    "tax_debt_evasion": ("避债避税",),
    "fake_endorsement": ("虚假宣传", "欺骗投保人"),
    "exaggerated_return": ("夸大收益",),
    "absolute_service_claim": ("绝对化表述",),
    "false_expense": ("虚列费用套取资金",),
    "fake_callback": ("回访违规", "客户信息不真实"),
    "unfiled_rate_terms": ("未按规定使用费率条款",),
    "unauthorized_sales": ("委托无资质机构销售",),
    "agent_mismanagement": ("代理人管理不到位",),
    "unqualified_individual": ("代理人管理不到位",),
    "private_premium_account": ("其他",),
    "false_financial_reporting": ("其他",),
    "customer_data_fabrication": ("客户信息不真实",),
    "kickback_rebate": ("回扣返佣",),
    "concealed_terms": ("弱化风险提示", "隐瞒重要信息"),
    "competitor_denigration": ("贬低竞品",),
    "unsolicited_telemarketing": ("电销违规",),
    "fixed_income_promise": ("承诺收益", "销售误导"),
    "broad_coverage_claim": ("销售误导", "绝对化表述"),
    "premium_offset_benefit": ("赠送利益",),
    "highest_projection_as_final": ("夸大收益", "销售误导"),
    "universal_suitability": ("绝对化表述", "销售误导"),
    "ai_absolute_endorsement": ("绝对化表述",),
    "purchase_gift": ("赠送利益",),
}

_STRONG_PATTERNS: dict[str, re.Pattern[str]] = {
    "unlicensed_premises": re.compile(
        r"(?:职场|营业场所|营销服务部|分支机构).{0,40}(?:设立|设置|开设|运营|使用)"
        r".{0,24}(?:未经|未获|未取得).{0,12}(?:监管部门批准|批准|设立许可)|"
        r"(?:未经|未获).{0,10}(?:监管部门批准|批准).{0,12}(?:擅自)?"
        r"(?:设立|开设|经营).{0,15}(?:职场|营业场所|营销服务部|分支机构)"
    ),
    "absolute_guarantee": re.compile(
        r"(?:保证|承诺|确保|保底)[^，,。；;！？!?\n]{0,12}(?:收益|年化|回报|利息|分红)|"
        r"稳赚不赔|保本保息|绝不亏损|零风险"
    ),
    "false_expense": re.compile(
        r"(?:虚构|虚假|伪造).{0,10}(?:发票|会务费|宣传费|业务费|费用|中介业务).{0,20}(?:列支|报销|套取|转入)|"
        r"虚列.{0,12}(?:费用|工资|薪酬|佣金|会议费|宣传费)|"
        r"(?:虚挂|虚构).{0,10}(?:代理|中介|人员).{0,20}(?:手续费|佣金|套取)|"
        r"(?:没有|并无|无).{0,8}真实(?:采购|服务|业务).{0,30}(?:报销|列支|套取|回流)"
    ),
    "fake_callback": re.compile(
        r"(?:业务员|代理人|销售人员).{0,12}(?:代接|冒充).{0,12}(?:客户|投保人)|"
        r"(?:代接回访|统一回答[‘'\"“]?(?:已了解|是|确认))|"
        r"(?:同一|相同).{0,8}(?:员工)?(?:手机|设备|ip).{0,18}(?:上传|生成).{0,18}(?:多份|\d+\s*(?:份|段|条)).{0,8}(?:回访)?录音|"
        r"(?:多份|大量|[一二三四五六七八九十百]+\s*份|\d+\s*(?:份|段|条)).{0,10}(?:回访)?录音.{0,24}(?:(?:均|全部)由.{0,10}(?:同一|相同).{0,8}(?:手机|设备|ip)|(?:时长|背景声|背景音).{0,8}(?:一致|相同))|"
        r"(?:客户|投保人).{0,12}(?:表示|称).{0,8}(?:从未|没有|未曾).{0,8}(?:接到|接受|完成).{0,6}回访"
    ),
    "unfiled_rate_terms": re.compile(
        r"(?:未经备案|未按备案|擅自|私自).{0,16}(?:费率|条款|特别约定)|"
        r"(?:费率|条款).{0,12}(?:未经备案|未按规定使用)"
    ),
    "unauthorized_sales": re.compile(
        r"(?:委托|交由).{0,12}(?:无资质|未取得资质|不具备资质).{0,12}(?:机构|公司|平台).{0,8}(?:销售|推介)|"
        r"(?:未取得|不具备|没有).{0,8}(?:保险)?(?:销售资质|代理业务许可).{0,30}(?:销售|出单|代收保费|讲解产品)"
    ),
    "unqualified_individual": re.compile(
        r"(?:委托|安排).{0,12}(?:未|无|没有).{0,8}(?:执业登记|资格证|展业证|执业证).{0,12}(?:人员|个人|业务员).{0,12}(?:销售|展业|营销)|"
        r"(?:人员|个人|业务员).{0,14}未.{0,6}(?:登记|注册).{0,10}(?:已|即|先).{0,8}(?:展业|销售)"
    ),
    "private_premium_account": re.compile(
        r"保费.{0,12}(?:打|进|转入|收至).{0,10}(?:私人|个人).{0,6}(?:账户|卡)"
    ),
    "false_financial_reporting": re.compile(
        r"财务(?:业务)?数据不真实|(?:编制|提供).{0,4}虚假(?:的)?财务(?:资料|报表)|"
        r"财务记录.{0,6}(?:与|和)实际.{0,8}不符"
    ),
    "agent_mismanagement": re.compile(
        r"(?:代理人|营销员).{0,12}(?:管理不到位|失管|违规销售)|"
        r"(?:未对|未能).{0,12}(?:代理人|营销员).{0,8}(?:培训|管理|监督)|"
        r"(?:主管|负责人|管理员).{0,8}(?:账号|权限).{0,24}(?:代录|批量|修改|确认|操作)|"
        r"(?:操作|登录|审批)?账号.{0,12}(?:属于|为|使用).{0,8}(?:团队)?(?:主管|负责人|管理员)|"
        r"(?:系统|机构|质检|部门).{0,16}(?:未|没有).{0,10}(?:拦截|核查|复核|处置|覆盖).{0,18}(?:异常|预警|夜间|高频|相关业务)?"
    ),
    "customer_data_fabrication": re.compile(
        r"(?:虚构|伪造|篡改|代填).{0,10}(?:客户|投保人).{0,8}(?:信息|签名|资料)|"
        r"(?:客户|投保人).{0,8}(?:信息|资料).{0,8}(?:不真实|虚假)|"
        r"(?:家属|亲属|员工|代理人|销售人员).{0,10}代签|"
        r"(?:电子签名|签署时间).{0,24}(?:早于|先于).{0,12}(?:验证码|身份核验|保单生成)|"
        r"(?:职业|病史|收入|联系方式).{0,48}(?:修改|改为|变更).{0,48}(?:未|没有|缺少).{0,10}(?:客户|投保人)?(?:验证码|确认|核实|知情)"
    ),
    "kickback_rebate": re.compile(
        r"(?:返还|返现|返佣|回扣).{0,8}(?:保费|佣金|现金|红包)|"
        r"(?:保费|佣金).{0,8}(?:返还|返现|回扣)|"
        r"(?:团队|经理|渠道|业务员).{0,18}(?:从.{0,10}服务费.{0,8})?补贴.{0,8}(?:首年|首期|保费)"
    ),
    "concealed_terms": re.compile(
        r"(?:不展示|未展示|不告知|未告知|不要告知|刻意隐瞒|故意隐瞒|无需|不必|不用|不需要|不提|不讨论|不解释|先不介绍|暂不说明)[^，,。；;！？!?\n]{0,22}"
        r"(?:免责|除外|等待期|退保|费用|风险|条款|费率|现金价值|红利|非保证)|"
        r"(?:不展示|未展示|不告知|未告知|刻意隐瞒|故意隐瞒|不解释|先不介绍|暂不说明|"
        r"无需介绍|不用说明).{0,12}(?:健康告知|核保)|"
        r"(?:免责|除外|等待期|退保|费用|风险|现金价值|红利|非保证|演示假设).{0,18}"
        r"(?:无需|不必|不用|不需要|不提|不讨论|不解释|只用于满足格式)"
    ),
    "competitor_denigration": re.compile(
        r"(?:其他|别家|竞品|同业).{0,10}(?:都是|全是|根本|一定).{0,8}(?:骗人|垃圾|不赔|不靠谱|差)|"
        r"(?:其他|别家|竞品|同业|(?<![a-z])[a-z](?:公司|保险)).{0,5}"
        r"(?:公司|产品|条款)?.{0,14}(?:理赔慢|条款坑|收益低|不赔|骗人|垃圾|不靠谱)"
    ),
    "unsolicited_telemarketing": re.compile(
        r"(?:未经同意|非法购买|违规获取).{0,16}(?:名单|手机号|联系方式).{0,16}"
        r"(?:拨打|呼叫|电销)|(?:明确拒绝|拒接).{0,16}"
        r"(?:更换号码|换号码|继续呼叫|继续联系|反复拨打)"
    ),
    "exaggerated_return": re.compile(
        r"(?:复利|收益|年化|回报).{0,12}(?:预计|预测|测算)?(?:可达|达到)\s*\d+(?:\.\d+)?\s*%"
    ),
    "absolute_service_claim": re.compile(r"(?:从不|绝不|保证不).{0,8}(?:拒赔|扯皮|拖延)"),
    "ai_absolute_endorsement": re.compile(
        r"(?<![a-z])ai.{0,4}(?:不会骗人|绝不骗人|不会出错|绝不出错)"
    ),
    "purchase_gift": re.compile(
        r"(?:投保|下单|签单|购买)[^。；;！？!?\n]{0,20}"
        r"(?:赠送|赠|(?<!发)送)[^，,。；;！？!?\n]{0,16}"
        r"(?:礼包|门诊保额|门诊报销金|保额)|"
        r"(?:投保|下单|签单|购买)[^。；;！？!?\n]{0,20}"
        r"(?:获得|获赠)[^，,。；;！？!?\n]{0,16}(?:礼包|赠品|礼品)"
    ),
    "fixed_income_promise": re.compile(
        r"(?:稳定|固定|确保|保证).{0,8}(?:到账|领取|收益|退休收入|退休工资)|"
        r"(?:按|作为|理解为).{0,8}(?:固定|稳定|确定).{0,8}(?:领取|收入|收益|工资)"
    ),
    "highest_projection_as_final": re.compile(
        r"(?:最高|高档).{0,8}(?:数值|演示|金额).{0,12}(?:估算|安排|作为).{0,10}(?:最终|固定|退休)"
    ),
    "broad_coverage_claim": re.compile(
        r"(?:保障|承保).{0,8}(?:不限|不问|无需).{0,8}(?:病史|健康告知|年龄)|"
        r"(?:门诊|住院).{0,8}(?:都|全部|一律).{0,5}(?:报|赔)|"
        r"(?:理赔|赔付).{0,10}(?:不卡|全额安排|都能通过|全部通过|不会受影响)|"
        r"(?:资料齐全|材料齐全).{0,10}(?:一定|都|不会).{0,8}(?:理赔|赔付|通过)"
    ),
    "premium_offset_benefit": re.compile(
        r"(?:赠品|礼品|福利|截图|优惠|补贴|红包|加油券).{0,12}(?:抵|冲|减免).{0,8}(?:首期|保费)|"
        r"(?:申购|投保|签单).{0,10}(?:额外加息|额外补贴|返现|送礼)"
    ),
    "universal_suitability": re.compile(
        r"(?:所有人|任何人|人人|谁).{0,8}(?:都适合|都可以买|都能投保)|"
        r"(?:不分|不限).{0,8}(?:年龄|职业|健康情况).{0,12}(?:都适合|均可投保)"
    ),
    "bank_comparison": re.compile(
        r"(?:别人家|别家公司|其他公司).{0,12}(?:都没有|没有这个|做不到)|"
        r"(?:只有|就).{0,4}(?:我们|本公司).{0,8}(?:才有|能做到|可以提供)|"
        r"(?:[a-z]公司).{0,140}(?:放在一起|混合比较|混在一起|直接对比).{0,36}(?:不说明|未说明|没有说明|不区分)"
    ),
    "urgency_pressure": re.compile(
        r"(?:强制|强行|必须).{0,12}(?:搭售|搭配购买|一起投保|自行购买|自保)|"
        r"(?:不买|未投保|不投保).{0,12}(?:不受理|不给办|扣绩效|扣工资)|"
        r"(?:虚构|捏造).{0,10}(?:限售|停售|限量|名额)"
    ),
}
_NEGATION_PREFIX = re.compile(r"(?:不|并非|不是|不得|严禁|禁止|不可|切勿|避免|杜绝)$")
_COMPLIANCE_QUALIFIERS = re.compile(
    r"(?:以(?:保险)?合同|条款为准|不构成(?:收益)?承诺|收益存在不确定|可能亏损|"
    r"须经审核|具体责任.{0,8}以条款为准|不得承诺|严禁宣传)"
)

_COMPLIANCE_DIRECTIVE = re.compile(
    r"(?:不得|严禁|禁止|不允许|不可|切勿|不能|不会).{0,30}"
    r"(?:承诺|保证|宣传|隐瞒|代签|代填|代接|代操作|诱导|夸大|比较|贬低|返佣|返现|赠礼|"
    r"虚构|伪造|篡改|收取.{0,6}验证码|设置.{0,6}(?:名额|时限)|继续联系|替代合同)|"
    r"(?:必须|应当|应该|应|须|需要).{0,35}"
    r"(?:如实(?:告知|填写|说明)?|完整(?:说明|披露|阅读|列示)|明确(?:说明|披露)|逐项(?:说明|披露)|区分|说明|披露|阅读|核验|核对|保留|停止|记录|自主办理)|"
    r"(?:演示|测算|历史(?:结算)?).{0,18}(?:不代表|并非|不能作为|不构成).{0,16}(?:保证|承诺|未来|合同)|"
    r"(?:红利|收益|分红|结算利率|非保证利益).{0,18}(?:不保证|可能为零|存在不确定|可能波动)|"
    r"(?:责任|费率|收益|保障|领取|计算方式).{0,18}(?:以合同|以条款|以保险单).{0,8}为准|"
    r"(?:不把|未把).{0,30}(?:称为|表述为|作为).{0,18}(?:固定|保证|承诺|确定)|"
    r"不代表.{0,24}(?:可以|能够|当然)?(?:规避|避债|避税|保证|承诺)|"
    r"(?:不评价|不贬低).{0,24}(?:其他|同业|竞品|公司)|"
    r"不使用.{0,24}(?:最好|唯一|第一|内部名额|保本|固定收益)|"
    r"(?:不得|严禁|禁止).{0,80}(?:抢购|倒计时|内部名额|固定收益|保本保息|避债|避税|返佣|赠礼)"
)
_NONCOMPLIANT_COUNTER = re.compile(
    r"(?:无需|不必|不用|不需要|不提|别提|不讨论|不解释|只展示|只保留|只读|先不介绍|暂不说明).{0,30}"
    r"(?:条款|免责|等待期|退保|费用|风险|现金价值|红利|非保证|核保|健康告知|演示)|"
    r"(?:不要|不向客户).{0,12}(?:告知|说明|解释|展示|披露)|"
    r"(?:不保证|风险提示).{0,12}(?:只是|仅是).{0,8}(?:固定用语|形式|话术)|"
    r"(?:不区分|不纠正|忽略|删除|伪造|虚构|冒充)|"
    r"先.{0,14}(?:投保|签名|签单|占名额|锁定额度|提交|确认).{0,18}(?:再|后)"
)
_CONTRAST_ASSERTION = re.compile(
    r"(?:却|但是|但|仍然|仍|实际上).{0,35}?(?P<assertion>承诺|保证|肯定|一定|固定拿)"
)
_ASSERTION_NEGATION = re.compile(
    r"(?:不|未|非|不是|不能|不得|严禁|禁止)(?:作出|宣称|使用|宣传|承诺|称|说|理解为)?[‘'\"“\s]*$"
)
_LABEL_HINTS: dict[str, re.Pattern[str]] = {
    "欺骗投保人": re.compile(r"骗|冒充|虚假|伪造|存单|储蓄"),
    "销售误导": re.compile(r"误导|存款|存单|稳赚|保本|无风险|不亏|隐瞒"),
    "夸大收益": re.compile(r"收益|回报|分红|翻倍|暴涨"),
    "虚假宣传": re.compile(r"虚假|唯一|首选|国家|监管|万能|全能"),
    "合同外利益": re.compile(r"送|赠|返|补贴|礼品|红包|返现|回扣"),
    "赠送利益": re.compile(r"送|赠|礼品|红包|补贴|体检|加油卡"),
    "回扣返佣": re.compile(r"返佣|回扣|返现|返还保费|佣金"),
    "产品说明会违规": re.compile(r"说明会|答谢会|现场签单"),
    "培训材料违规": re.compile(r"培训|课件|内部话术"),
    "回访违规": re.compile(r"回访|代接|冒充客户"),
    "电销违规": re.compile(r"电销|电话销售|坐席|致电"),
    "不当比较": re.compile(r"比较|超过|碾压|完胜|比银行|比存款"),
    "贬低竞品": re.compile(r"竞品|别家|其他公司|垃圾|骗人|不靠谱"),
    "绝对化表述": re.compile(r"绝对|唯一|百分之百|100%|任何|永远|从不|全部|都赔"),
    "承诺收益": re.compile(r"收益|回报|稳赚|保本|兑付"),
    "保证收益": re.compile(r"保证|确保|保底|最低收益"),
    "弱化风险提示": re.compile(r"零风险|无风险|不亏|不用看|忽略风险"),
    "隐瞒重要信息": re.compile(r"隐瞒|不告知|未告知|不展示|免责|等待期|退保损失"),
    "虚构收益率": re.compile(r"虚构|编造|年化|收益率|演示值"),
    "诱导投保": re.compile(r"限时|仅剩|马上|立即|闭眼|名额|停售|下架|签单"),
    "避债避税": re.compile(r"避债|避税|免税|资产隔离|法院|债权人"),
    "虚列费用套取资金": re.compile(r"发票|列支|套取|会务费|费用|资金"),
    "代理人管理不到位": re.compile(r"代理人|营销员|管理不到位|失管"),
    "委托无资质机构销售": re.compile(r"无资质|未取得资质|委托销售"),
    "未按规定使用费率条款": re.compile(r"备案|费率|条款|特别约定"),
    "客户信息不真实": re.compile(r"客户信息|投保人信息|代签|代填|冒充|虚假资料"),
}

_SCENE_CHANNEL_LABELS: dict[QueryScene, set[str]] = {
    QueryScene.ST1_POSTER_SMS_SOCIAL: {
        "虚假宣传",
        "绝对化表述",
        "诱导投保",
        "赠送利益",
        "合同外利益",
    },
    QueryScene.ST2_LIVE_SHORT_VIDEO: {"虚假宣传", "绝对化表述", "诱导投保", "夸大收益", "承诺收益"},
    QueryScene.ST3_PRODUCT_INTRO_DETAIL: {
        "销售误导",
        "弱化风险提示",
        "隐瞒重要信息",
        "夸大收益",
        "承诺收益",
    },
    QueryScene.ST4_MEETING_MATERIAL: {"产品说明会违规", "诱导投保", "赠送利益"},
    QueryScene.ST5_TRAINING_TELEMARKETING_CALLBACK: {
        "培训材料违规",
        "回访违规",
        "电销违规",
        "客户信息不真实",
    },
    QueryScene.ST6_PENALTY_BACKEND_FACT: {
        "虚列费用套取资金",
        "代理人管理不到位",
        "委托无资质机构销售",
        "未按规定使用费率条款",
        "客户信息不真实",
    },
}


@dataclass(frozen=True, slots=True)
class _KeywordRule:
    keyword: str
    source: str
    priority: int
    risk_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuleDetectionResult:
    risk_score: float
    confidence: float
    stage1_score: float
    stage2_score: float
    risk_labels: tuple[str, ...]
    evidence: tuple[RiskEvidence, ...]
    strong_signal: bool
    safe_signal: bool


class RuleRiskDetector:
    """第一阶段组合强规则，第二阶段场景加权关键词累积。"""

    def __init__(self, keyword_path: Path) -> None:
        if not keyword_path.is_file():
            raise FileNotFoundError(f"风险关键词库不存在: {keyword_path}")
        payload = json.loads(keyword_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("风险关键词库必须是 JSON 对象")
        self.version = str(payload.get("version") or "unknown")
        self._keywords = _parse_keyword_rules(payload.get("keywords"))
        self._patterns = _parse_patterns(payload.get("regex_patterns"))
        self._patterns.update(_STRONG_PATTERNS)
        self.candidate_model = None
        model_path = keyword_path.parent.parent / "calibration/competition500.npz"
        if (
            model_path.is_file()
            and os.getenv("FINRISK_LOCAL_CALIBRATION", "true").lower() != "false"
        ):
            from risk.local_calibration import LocalRiskCalibration

            self.candidate_model = LocalRiskCalibration(model_path)

    def detect(
        self,
        text: str,
        scene: QueryScene | None = None,
    ) -> RuleDetectionResult:
        result = self._detect_rules(text, scene)
        model = getattr(self, "candidate_model", None)
        if model is None or result.safe_signal:
            return result
        risk, candidates = model.predict(text)
        labels = [label for label, score in candidates if score >= 0.55]
        extra = [label for label in labels if label not in result.risk_labels]
        if risk < 0.7 or not extra:
            return result
        return RuleDetectionResult(
            risk_score=max(result.risk_score, 0.38),
            confidence=min(result.confidence, 0.65),
            stage1_score=result.stage1_score,
            stage2_score=max(result.stage2_score, 0.35),
            risk_labels=tuple([*result.risk_labels, *extra]),
            evidence=(
                *result.evidence,
                RiskEvidence(
                    source="local_calibration",
                    description="开发集训练的字词特征模型提示：" + "、".join(extra),
                ),
            ),
            strong_signal=result.strong_signal,
            safe_signal=False,
        )

    def _detect_rules(
        self,
        text: str,
        scene: QueryScene | None = None,
    ) -> RuleDetectionResult:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = re.sub(r"^【材料载体[:：][^】]+】\s*", "", normalized)
        if not normalized.strip():
            return RuleDetectionResult(0.0, 0.0, 0.0, 0.0, (), (), False, False)

        if is_explicit_compliance_statement(normalized):
            return RuleDetectionResult(
                risk_score=0.0,
                confidence=0.98,
                stage1_score=0.0,
                stage2_score=0.0,
                risk_labels=(),
                evidence=(
                    RiskEvidence(
                        source="compliance_guard",
                        description="检测到明确禁止性规范、披露义务或风险限定，按合规语境处理",
                    ),
                ),
                strong_signal=False,
                safe_signal=True,
            )

        label_scores: dict[str, float] = {}
        stage1_strengths: list[float] = []
        stage2_strengths: list[float] = []
        evidence: list[RiskEvidence] = []
        explicit_other = False
        for name, pattern in self._patterns.items():
            match = next(
                (m for m in pattern.finditer(normalized) if not _is_negated(normalized, m.start())),
                None,
            )
            if match is None:
                continue
            strength = 0.88 * _scene_factor(REGEX_LABEL_MAP.get(name, ()), scene)
            other_supported = "其他" in REGEX_LABEL_MAP.get(name, ())
            explicit_other = explicit_other or other_supported
            stage1_strengths.append(min(0.96, strength))
            for label in REGEX_LABEL_MAP.get(name, ("其他",)):
                if label != "其他" or other_supported:
                    label_scores[label] = label_scores.get(label, 0.0) + strength
            evidence.append(
                RiskEvidence(
                    source="rule_regex",
                    description=f"强规则“{name}”命中：{match.group(0)[:80]}",
                )
            )

        seen_keywords: set[str] = set()
        for rule in self._keywords:
            position = next(
                (
                    m.start()
                    for m in re.finditer(re.escape(rule.keyword), normalized)
                    if not _is_negated(normalized, m.start())
                ),
                -1,
            )
            if position < 0 or rule.keyword in seen_keywords:
                continue
            seen_keywords.add(rule.keyword)
            base = (
                0.16 + 0.06 * min(rule.priority, 3)
                if rule.source == "curated"
                else 0.06 + 0.025 * min(rule.priority, 3)
            )
            labels = _select_labels(rule)
            if not labels:
                continue
            strength = min(0.45, base * _scene_factor(labels, scene))
            stage2_strengths.append(strength)
            divisor = math.sqrt(len(labels))
            for label in labels:
                label_scores[label] = label_scores.get(label, 0.0) + strength / divisor
            evidence.append(
                RiskEvidence(
                    source="local_stage2",
                    description=(
                        f"场景加权词“{rule.keyword}”（{rule.source}，权重{strength:.2f}）"
                    ),
                )
            )

        stage1_score = _noisy_or(stage1_strengths)
        stage2_score = _noisy_or(sorted(stage2_strengths, reverse=True)[:6])
        qualifier_count = len(_COMPLIANCE_QUALIFIERS.findall(normalized))
        if qualifier_count and not stage1_strengths:
            stage2_score *= max(0.35, 1 - 0.25 * qualifier_count)
        combined = 1 - (1 - stage1_score) * (1 - 0.78 * stage2_score)
        labels = tuple(
            label
            for label, _ in sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))
            if label != "其他" or explicit_other
        )[:8]
        strong_signal = bool(stage1_strengths) or (
            stage2_score >= 0.68 and len(stage2_strengths) >= 3
        )
        confidence = _confidence(stage1_strengths, stage2_strengths, combined, qualifier_count)
        return RuleDetectionResult(
            risk_score=round(min(0.99, combined), 6),
            confidence=round(confidence, 6),
            stage1_score=round(stage1_score, 6),
            stage2_score=round(stage2_score, 6),
            risk_labels=labels,
            evidence=tuple(evidence[:10]),
            strong_signal=strong_signal,
            safe_signal=False,
        )


def is_explicit_compliance_statement(text: str) -> bool:
    """识别规范性禁止、正向披露和合同限定，避免把“不得违规”当作违规事实。"""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    if has_adopted_instruction(normalized):
        return False
    for match in _CONTRAST_ASSERTION.finditer(normalized):
        start = match.start("assertion")
        if not _ASSERTION_NEGATION.search(normalized[max(0, start - 14) : start]):
            return False  # A disclaimer cannot suppress a separate affirmative promise.
    # Keep the refusal and subsequent action together: generic keyword scanning
    # would mistake prohibited quotations for affirmative behavior.
    for match in _STRONG_PATTERNS["unsolicited_telemarketing"].finditer(normalized):
        if not _is_negated(normalized, match.start()):
            return False
    directive = _COMPLIANCE_DIRECTIVE.search(normalized) or re.search(
        r"(?:不得使用|禁止使用|严禁使用|不做|不使用)[^。；\n]{0,65}(?:什么都保|肯定赔|稳赚|保本|保证|承诺|返佣|代签)",
        normalized,
    )
    return bool(directive) and not bool(_NONCOMPLIANT_COUNTER.search(normalized))


def _confidence(
    stage1: list[float],
    stage2: list[float],
    combined: float,
    qualifier_count: int,
) -> float:
    if stage1:
        return min(0.99, 0.88 + 0.035 * len(stage1))
    if len(stage2) >= 3 and combined >= 0.65:
        return min(0.88, 0.72 + 0.035 * len(stage2))
    if not stage2 and qualifier_count:
        return min(0.88, 0.72 + 0.04 * qualifier_count)
    if not stage2:
        return 0.30
    return min(0.72, 0.45 + 0.06 * len(stage2))


def _scene_factor(labels: tuple[str, ...] | list[str], scene: QueryScene | None) -> float:
    if scene is None:
        return 1.0
    channel_labels = _SCENE_CHANNEL_LABELS[scene]
    return 1.18 if channel_labels.intersection(labels) else 0.92


def _is_negated(text: str, start: int) -> bool:
    prefix = text[max(0, start - 8) : start].rstrip()
    if _NEGATION_PREFIX.search(prefix) is not None:
        return True
    clause = re.split(r"[，,。；;！？!?\n]", text[max(0, start - 70) : start])[-1]
    clause = re.split(r"但是|但|却|仍然|实际", clause)[-1]
    return (
        re.search(
            r"(?:不得|严禁|禁止|不可|不允许|不能|不做|不使用|不应)[^，,。；;！？!?\n]{0,60}$",
            clause,
        )
        is not None
    )


def _noisy_or(strengths: list[float]) -> float:
    remaining = 1.0
    for strength in strengths:
        remaining *= 1 - max(0.0, min(0.99, strength))
    return 1 - remaining


def _select_labels(rule: _KeywordRule) -> tuple[str, ...]:
    if len(rule.risk_labels) <= 4:
        return tuple(label for label in rule.risk_labels if label != "其他")
    inferred = tuple(
        label
        for label in rule.risk_labels
        if label != "其他"
        and (pattern := _LABEL_HINTS.get(label)) is not None
        and pattern.search(rule.keyword)
    )
    if inferred:
        return inferred[:5]
    return rule.risk_labels[:2] if rule.source == "curated" else ()


def _parse_keyword_rules(value: object) -> tuple[_KeywordRule, ...]:
    if not isinstance(value, list):
        raise ValueError("风险关键词库缺少 keywords 数组")
    rules: list[_KeywordRule] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        keyword = unicodedata.normalize("NFKC", str(item.get("keyword") or "")).casefold()
        labels = item.get("risk_labels")
        if len(keyword) < 2 or not isinstance(labels, list):
            continue
        rules.append(
            _KeywordRule(
                keyword=keyword,
                source=str(item.get("source") or "unknown"),
                priority=int(item.get("priority") or 1),
                risk_labels=tuple(str(label) for label in labels if label),
            )
        )
    return tuple(sorted(rules, key=lambda rule: (-len(rule.keyword), rule.keyword)))


def _parse_patterns(value: object) -> dict[str, re.Pattern[str]]:
    if not isinstance(value, dict):
        return {}
    patterns: dict[str, re.Pattern[str]] = {}
    for name, expression in value.items():
        try:
            patterns[str(name)] = re.compile(str(expression), re.IGNORECASE)
        except re.error as error:
            raise ValueError(f"风险正则无法编译: {name}") from error
    return patterns
