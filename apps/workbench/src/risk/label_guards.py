"""按当前原句核对标签的必要语义，避免把保障额度当投资收益。"""

import re

_INVESTMENT_RETURN = re.compile(r"收益|回报|复利|年化|利息|分红|结算利率|生息")
_DISCLOSURE_ACTION = re.compile(
    r"隐瞒|不告知|未告知|不要告知|不展示|未展示|不提|别提|"
    r"不(?:用|必|需要|要).{0,8}(?:说明|介绍|解释|阅读|看|考虑).{0,12}"
    r"(?:条款|免责|风险|告知|核保|等待期|退保)|"
    r"弱化|淡化|忽略|小字|删.{0,8}(?:风险|免责)|"
    r"(?:没有|不存在|无|零).{0,3}风险"
)
_QUALIFIED_REIMBURSEMENT = re.compile(
    r"(?:责任内|符合.{0,8}(?:合同|条款|条件)).{0,60}(?:100\s*%|百分之百).{0,4}(?:报销|赔付)|"
    r"(?:责任内|符合.{0,8}(?:合同|条款|条件)).{0,60}(?:报销|赔付).{0,8}(?:100\s*%|百分之百)"
)
_UNCONDITIONAL = re.compile(r"无条件|任何|全都|所有|永远|绝不|唯一|最好|全网|天花板|全市场")
_FEATURE_ONLY = re.compile(r"(?:(?:0|零|免|无需|不用)健康告知|免健告|既往症可赔)")
_UNNAMED_EMPLOYER = re.compile(
    r"(?:不记名|不实名制).{0,8}(?:投保|承保)|"
    r"不用提供员工名单.{0,80}(?:风险|承保)名额内"
)


def social_label_supported(label: str, text: str) -> bool:
    """只拦截违反明确标签边界的归因，不据此证明整句无其他风险。"""
    if label == "夸大收益":
        return bool(_INVESTMENT_RETURN.search(text))
    if label in {"隐瞒重要信息", "弱化风险提示"}:
        return bool(_DISCLOSURE_ACTION.search(text))
    if label == "绝对化表述" and _QUALIFIED_REIMBURSEMENT.search(text):
        return bool(_UNCONDITIONAL.search(text))
    if label == "销售误导" and _UNNAMED_EMPLOYER.search(text):
        return bool(re.search(r"无条件|任何职业|所有职业|超出名额|不限人数", text))
    if label in {"赠送利益", "合同外利益"} and re.search(r"会员|家庭单|多人.{0,8}折扣", text):
        return bool(re.search(r"投保|下单|签单|购买", text))
    if label == "避债避税":
        return bool(
            re.search(r"避税|避债|逃税|逃债|不(?:用|必|需).{0,4}(?:纳税|交税)|转移资产", text)
        )
    compact = re.sub(r"[^\w]", "", text)
    if _FEATURE_ONLY.fullmatch(compact):
        return False
    return True
