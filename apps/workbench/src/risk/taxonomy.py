"""27类风险标签定义、关系扩展与提示词格式化。"""

from __future__ import annotations

import csv
from pathlib import Path

_RELATED_LABELS: dict[str, tuple[str, ...]] = {
    "欺骗投保人": ("销售误导",),
    "夸大收益": ("销售误导",),
    "承诺收益": ("销售误导",),
    "保证收益": ("销售误导",),
    "虚构收益率": ("夸大收益",),
    "弱化风险提示": ("隐瞒重要信息", "销售误导"),
    "隐瞒重要信息": ("销售误导",),
    "赠送利益": ("合同外利益", "诱导投保"),
    "回扣返佣": ("合同外利益",),
    "贬低竞品": ("不当比较",),
}

# Boundaries clarify the existing 27-label dictionary; they do not create labels or
# encode evaluation Query IDs. A related label always needs its own current fact.
LABEL_BOUNDARIES: dict[str, str] = {
    "欺骗投保人": "须有具体虚假事实、冒充身份或欺骗对象；不将所有收益话术再泛化为欺骗。",
    "销售误导": "产品性质、责任、条款或利益被实质误述；不要给纯财务、行政、后台问题加销售误导。",
    "夸大收益": "夸张演示或预期投资回报；正常保险金、保额、报销比例、省保费、理赔经历不成立。",
    "虚假宣传": "有具体不实的宣传内容或宣传依据；不因其他销售标签成立就同步追加。",
    "合同外利益": "当前事实明确提供合同约定外的额外利益；不从普通服务或正常合同给付推定。",
    "赠送利益": (
        "以投保或签单为条件额外送礼品、服务或保额；家庭单费率折扣、合同内服务、"
        "仅列会员权益不自动成立；不自动补合同外利益、诱导投保。"
    ),
    "回扣返佣": (
        "向投保人返保费、现金回扣或变相返点；"
        "正常中介佣金、套费返给团队、垫付后客户归还不算本标签。"
    ),
    "产品说明会违规": "说明会或客户活动中存在具体违规宣讲、数据或实施行为；会议背景本身不成立。",
    "培训材料违规": (
        "培训课件或培训管理指令实际教授、要求违规做法；"
        "培训档案缺失属于代理人管理，纠错或反面教学不成立。"
    ),
    "回访违规": "回访环节代接、伪造、诱导不实回答或故意规避；普通签到或销售电话不属于回访。",
    "电销违规": "电话营销环节有具体违规行为；仅出现电话或电销团队名称不成立。",
    "不当比较": "比较对象、口径、时间或条件不一致造成误导；与银行作正常客观比较不成立。",
    "贬低竞品": "明确贬损其他公司的产品或信誉；一般比较或自称优势不自动加本标签。",
    "绝对化表述": "有绝对、最高、唯一、全都等无条件断言；正常合同中确定给付不成立。",
    "承诺收益": "把不确定回报说成确定可得的承诺；正常解释已约定给付不成立。",
    "保证收益": "明确保底、兜底、保证收益率或保本回报；不从一般乐观预测自动追加。",
    "弱化风险提示": "淡化风险、缩小提示或用话术带过；一般收益夸张不自动加本标签。",
    "隐瞒重要信息": "明确隐瞒重要条款、费用、免责、损失等事实；不能因原文没写某字段就推定隐瞒。",
    "虚构收益率": "有编造、无依据数字、伪造收益数据的事实；承诺或夸大数字不等于已证明编造。",
    "诱导投保": "强制自保、施压、融资或其他明确不当促成投保手段；不把每个赠礼、宣传自动重复标注。",
    "避债避税": "明确将保险宣传成可不当规避债务或税务；正常有条件税务说明不成立。",
    "虚列费用套取资金": (
        "有虚假业务、支出、发票、薪酬或伪造报销事由套取资金；"
        "仅财务数据不实、真实费用科目错分或跨期属于其他，缺少凭证不能直接定本标签。"
    ),
    "代理人管理不到位": (
        "有机构对保险代理人培训、登记、监督、违规处置失职，或委托无资格个人销售的事实；"
        "销售人员学历是本标签；高管任职、许可证遗失、机构保证金、财务签批责任不属于本标签。"
    ),
    "委托无资质机构销售": (
        "机构无合法销售资格且实际承担保险销售；一般合作、会务支持或资料收集不自动成立。"
    ),
    "未按规定使用费率条款": "明确偏离备案或批准的费率、条款；不要混为一般宣传误导。",
    "客户信息不真实": (
        "客户身份、联系方式、健康收入、签名或本人确认记录被虚构、篡改或冒充真实；"
        "不把经营财务报表、代理人学历、关联方信息归到客户；客户如实口述后协助录单不直接成立。"
    ),
    "其他": (
        "有27类中未覆盖的独立明确违规，如未经批准设立经营场所；"
        "不是兜底的不确定标签，不与已归类的同一事实重复。"
    ),
}


def load_taxonomy_descriptions(path: Path) -> dict[str, str]:
    """读取风险字典说明；缺列时安全降级为标签名。"""

    if not path.is_file():
        return {}
    descriptions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            label = next(
                (
                    str(row[key]).strip()
                    for key in ("risk_tag", "risk_type", "label_name", "风险类型", "标签名称")
                    if row.get(key)
                ),
                "",
            )
            if not label:
                continue
            description = next(
                (
                    str(row[key]).strip()
                    for key in ("description", "definition", "说明", "定义")
                    if row.get(key)
                ),
                label,
            )
            descriptions[label] = description
    return descriptions


def taxonomy_prompt(
    labels: tuple[str, ...],
    descriptions: dict[str, str],
) -> str:
    return "\n".join(
        f"- {label}：{descriptions.get(label, label)}；{LABEL_BOUNDARIES.get(label, '')}"
        for label in labels
    )


def expand_related_labels(
    labels: list[str],
    allowed: set[str],
    *,
    limit: int = 10,
) -> list[str]:
    """补全明确的上下位或紧密相关标签，保留原始标签优先顺序。"""

    expanded = list(dict.fromkeys(label for label in labels if label in allowed))
    for label in tuple(expanded):
        expanded.extend(related for related in _RELATED_LABELS.get(label, ()) if related in allowed)
    expanded = list(dict.fromkeys(expanded))
    if len(expanded) > 1 and "其他" in expanded:
        expanded.remove("其他")
    return expanded[:limit]
