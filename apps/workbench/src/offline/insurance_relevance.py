"""Evidence-backed scope triage; rule matches never grant publication approval."""

from __future__ import annotations

import re
from typing import Any, Literal, cast

RULE_VERSION = "insurance-scope/1"
InsuranceRelevance = Literal["insurance", "non_insurance", "uncertain"]
_ENTITY = re.compile(
    r"保险公司|保险集团|保险经纪|保险代理|保险公估|保险资产管理|保险资管"
    r"|人寿|财险|产险|寿险|再保险|再保|公估|财产保险|健康保险|养老保险"
)
_REGULATOR = re.compile(r"监管局|监管分局|监督管理|银保监|监管总局|保险业务监管|泛指")
_BANK = re.compile(r"银行|信用社|信用合作")
_OTHER_ENTITY = re.compile(r"银行|信用社|信用合作|证券|期货|信托|食品|餐饮|运输|商贸|科技")
_BUSINESS = re.compile(
    r"保险(?:产品|业务|合同|理赔|赔款|销售|营销|费率|条款|代理|经纪|资金)"
    r"|保单|保费|退保|投保人|被保险人|诱导投保|虚假投保|拒绝理赔|骗取保险金"
    r"|承保|核保|理赔|投保|银保专管员"
)
_NON_BUSINESS = re.compile(r"贷款|信贷|授信|存款|票据|信用卡|食品安全|食品抽检|证券交易|期货交易")
_STAFF = re.compile(r"经理|负责人|员工|业务员|代理人|董事|监事|职员|总裁|总监|代表人|管理岗")


def _signals(text: str) -> str:
    """Mask contextual names, without rejecting independent facts in the same paragraph."""
    return re.sub(r"《[^》]*》|保险业务监管|泛指保险业务", "", text)


def _employment(text: str) -> bool:
    explicit = re.search(r"(?:时任|担任|任职于)[^，,。；\n]+", text)
    if explicit:
        return not bool(re.search(r"(?:客户|投保人|被保险人|交易对手)[）)]?$", explicit.group()))
    relation = re.split(r"[，,。；\n]", text, maxsplit=1)[0]
    return bool("系" in relation and _STAFF.search(relation))


def _outside_parentheses(text: str) -> str:
    depth = 0
    result = []
    for char in text:
        if char in "（(":
            depth += 1
        elif char in "）)":
            depth = max(0, depth - 1)
        elif not depth:
            result.append(char)
    return "".join(result)


_HEADERS = {
    "party": r"被处罚当事人|主要被处罚方|被处罚单位|被处罚人|当事人|单位名称|机构名称",
    "facts": r"主要违法违规事实(?:[（(]案由[)）])?|违法违规事实|违法事实|案由",
    "other": (
        r"行政处罚决定书文号|行政处罚依据|处罚依据|法律依据|行政处罚内容"
        r"|行政处罚决定|处罚决定|作出处罚决定的\s*机关名称|处罚机关|监管机构"
        r"|作出处罚决定的日期|决定日期|来源公示|来源网址|来源链接"
    ),
}
_HEADER = re.compile(
    r"(?:^|[\r\n。；])[ \t]*(?:"
    + "|".join(f"(?P<{k}>{v})" for k, v in _HEADERS.items())
    + r")(?:[ \t]*[：:][ \t]*|[ \t]*(?=\r?\n|$))",
    re.MULTILINE,
)


def _fields(text: str) -> dict[str, tuple[str, int, int]]:
    matches = list(_HEADER.finditer(text))
    result = {}
    for i, match in enumerate(matches):
        key = match.lastgroup
        if key is None:
            continue
        if key == "other":
            continue
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        while start < end and text[start].isspace():
            start += 1
        if key == "party":
            boundary = re.search(r"[。\r\n]", text[start:end])
            if boundary:
                end = start + boundary.start()
        while end > start and text[end - 1].isspace():
            end -= 1
        if key in result:
            result["multiple"] = ("", 0, 0)
        result[key] = (text[start:end], start, end)
    return result


def extract_party_name(text: str) -> str | None:
    field = _fields(text).get("party")
    return field[0].strip("。；") if field else None


def evaluate_insurance_scope(
    original_text: str, *, party_name: str | None = None, violation_behavior: str = ""
) -> dict[str, Any]:
    fields = _fields(original_text)
    for key, fallback in (("party", party_name), ("facts", violation_behavior)):
        if key not in fields and fallback and fallback in original_text:
            start = original_text.index(fallback)
            fields[key] = (fallback, start, start + len(fallback))
    party = fields.get("party", ("", 0, 0))[0]
    facts = fields.get("facts", ("", 0, 0))[0]
    fact_signals = _signals(facts)
    evidence = []

    def cite(key: str, reason: str) -> None:
        quote, start, end = fields[key]
        evidence.append(dict(field=key, quote=quote, start=start, end=end, reason=reason))

    category, relevance = "uncertain", "uncertain"
    unknowns = []
    direct_role = _employment(party)
    entity_match = bool(
        _ENTITY.search(_outside_parentheses(party))
        or (direct_role and _ENTITY.search(party))
    ) and not _REGULATOR.search(party)
    institution = bool(
        re.search(
            r"(?:公司|集团|合作社|事务所|营销服务部)(?:及下辖分支机构)?"
            r"(?:(?:及)?(?:其|相关)?责任人员?)?$",
            re.split(r"[（(]", party)[0],
        )
    )
    fact_role = None
    if party and len(party) <= 6:
        fact_role = re.search(
            re.escape(party) + r"\s*[（(]?\s*(?:时任|担任|任职于|系)[^。；\n]{0,60}",
            fact_signals,
        )
    mixed = bool(re.search(r"[、；;]|及.*(?:公司|银行)", _outside_parentheses(party)))
    if "multiple" in fields or mixed:
        unknowns.append("多主体或多事实段，需逐案核对归属")
    elif entity_match and (institution or direct_role):
        category, relevance = "insurance_entity", "insurance"
        cite("party", "保险机构或明确任职关系线索")
    elif fact_role and _ENTITY.search(fact_role.group()) and _employment(fact_role.group()):
        category, relevance = "insurance_entity", "insurance"
        cite("party", "个人处罚主体")
        cite("facts", "事实段明确记载该主体的保险机构任职关系")
    elif fact_signals and _BUSINESS.search(fact_signals):
        category, relevance = "insurance_business", "insurance"
        cite("facts", "处罚事实存在保险业务线索")
        if party:
            cite("party", "业务涉及主体，需核对责任关系")
    elif (
        party
        and facts
        and _OTHER_ENTITY.search(party)
        and not entity_match
        and _NON_BUSINESS.search(fact_signals)
    ):
        category, relevance = "non_insurance_candidate", "non_insurance"
        cite("party", "其他行业主体线索")
        cite("facts", "明确非保险业务线索；仅建议排除")
    if not evidence:
        unknowns.append("缺少可确认的主体、任职或业务证据")
    if not facts:
        unknowns.append("未提取到独立违法事实段")
    non_penalty = bool(re.search(r"换领.{0,12}许可证|行政许可|许可证.{0,12}公告", original_text))
    if non_penalty:
        unknowns.append("许可或换证线索，须核实是否属于处罚材料")
    return {
        "rule_version": RULE_VERSION,
        "category": category,
        "relevance": relevance,
        "route": "candidate_review"
        if relevance == "insurance" and not non_penalty
        else "scope_review",
        "party_type": "bank"
        if _BANK.search(party) and not entity_match
        else "insurance"
        if entity_match
        else "unknown",
        "evidence": evidence,
        "unknowns": unknowns,
        "review_status": "pending",
        "contains_insurance_keyword": "保险" in original_text,
        "publication_allowed": False,
    }


def classify_insurance_relevance(
    *, party_name: str | None, violation_behavior: str, original_text: str
) -> InsuranceRelevance:
    return cast(
        InsuranceRelevance,
        evaluate_insurance_scope(
            original_text, party_name=party_name, violation_behavior=violation_behavior
        )["relevance"],
    )
