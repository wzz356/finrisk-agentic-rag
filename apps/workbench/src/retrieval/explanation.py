"""Business-level similarities grounded in the query and returned case fields."""

import re

_MECHANISMS = [
    ("合同外利益", r"合同.{0,5}以外|返佣|回扣|补贴|体检卡|加油卡|赠[送礼品]"),
    ("收益承诺", r"保证.{0,8}收益|承诺.{0,8}收益|保本|固定回报|不保证.{0,5}收益"),
    ("销售误导", r"欺骗投保人|误导|存款|储蓄|夸大|零风险|无风险"),
    ("费用及中介业务", r"虚列费用|套取|虚构.{0,8}中介|虚挂.{0,8}中介|虚假财务"),
    ("真实回访", r"回访|代接|代答|冒名"),
    ("资格与登记", r"无资质|无资格|执业登记|未登记|资格证|许可证"),
    ("重要信息披露", r"免责|等待期|退保|现金价值|隐瞒|披露"),
    ("客户信息与签署", r"客户信息|虚假.{0,6}信息|代签|签名|签字"),
    ("条款费率", r"费率|条款"),
    ("不实数据", r"数据不实|数据.{0,5}不真实|虚假.{0,5}资料|说明会"),
    ("营销外呼", r"外呼|电销|拒接|拒绝.{0,6}电话|禁呼"),
    ("税务与债务", r"避税|避债|免税"),
]


def business_match_reason(query: str, finding: str, quote: str, tags: list[str]) -> str:
    for name, pattern in _MECHANISMS:
        left = re.search(pattern, query)
        right = re.search(pattern, finding + " " + quote)
        if left and right:
            return f"当前表达“{left[0]}”与案例记载“{right[0]}”涉及相同的{name}问题。"
    subject = "、".join(tags[:3])
    excerpt = finding[:90].strip()
    return f"匹配{subject or '相似行为'}；案例违法事实：{excerpt}"
