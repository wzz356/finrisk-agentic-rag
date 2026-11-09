"""Cheap, neutral mechanism expressions; they never assert missing business facts."""

from __future__ import annotations

import json
import re
from pathlib import Path

NEUTRAL = {
    "M01": "保险回访 真实知悉 指导答复 如实回答",
    "M02": "保险回访 身份核验 本人答复 第三人代接",
    "M03": "保险宣传 债务税务法律效果 适用条件",
    "M04": "保险销售 合同约定利益 返佣 礼品与投保关系",
    "M05": "保险分红 演示收益 保证与非保证利益 确定性表述",
    "M06": "保险产品性质 银行存款 储蓄 理财 产品区分",
    "M07": "保险退保 现金价值 犹豫期 费用与领取条件",
    "M08": "保险责任 理赔条件 免责范围 等待期 告知说明",
    "M09": "保险产品停售调价宣传 公司公告 产品及日期条件",
    "M10": "保险销售 受托机构资格 具体产品推介与销售行为",
    "M11": "电子投保文书 销售人员代操作 本人签署与确认",
    "M12": "保险电话营销 客户意愿 拒绝与继续外呼 禁呼名单",
}

# Reading hints from the P0 release; no trigger asserts a violation or assigns a label.
READING_TRIGGERS = {
    "M01": r"回访|(?:教|照着).{0,8}(?:回答|答复)",
    "M02": r"回访|冒充.{0,8}(?:客户|投保人)|代(?:接|答).{0,8}(?:电话|回访)",
    "M03": r"避债|免税|逃债|债权人|债务|税务|执行财产",
    "M04": (
        r"合同.{0,4}(?:之外|以外|外)|返还.{0,12}(?:千|百|元|现金)|返佣|回扣|赠送|返点|"
        r"(?:签单|投保|扣费|购买|买保险).{0,12}(?:领|送|返).{0,8}(?:元|礼|卡|额度)"
    ),
    "M05": (
        r"(?:保证|保底|固定).{0,12}(?:收益|回报|分红|利率|领取|到账|给付|金额)|万能|投连|投资连结|分红|"
        r"演示.{0,14}(?:肯定|保证)"
    ),
    "M06": r"银行存款|理财产品|存款|储蓄|保本理财",
    "M07": r"退保|退还保费|取回本金|中途退出",
    "M08": (
        r"免责|都能赔|全部赔|全额赔|不赔|保障范围|责任范围|理赔条件|"
        r"(?:全|都|基本|百分之百).{0,6}(?:报销|报|赔)"
    ),
    "M09": r"停售|下架|最后.{0,6}(?:一天|机会|名额)|限额|稀缺",
    "M10": r"无资格|没有资格|未取得.{0,8}资格|无牌|合作平台|销售资质|机构.{0,8}销售",
    "M11": (
        r"代签|替.{0,8}(?:签名|签字|确认)|(?:签名|确认).{0,8}(?:代为|替客户|代签|销售替)|"
        r"(?:代|替|帮).{0,12}(?:勾选|操作|点击|点).{0,12}(?:确认|提交|签署|风险提示)"
    ),
    "M12": r"退订|取消.{0,8}(?:营销|电话|来电)|别再.{0,8}(?:电话|联系)|拒接|电话营销|外呼",
}


class RetrievalQueryPlanner:
    def __init__(self, path: Path | None = None) -> None:
        self.entries = (
            json.loads(path.read_text(encoding="utf-8")) if path and path.is_file() else []
        )

    def mechanisms(self, query: str, *, limit: int = 12) -> list[str]:
        scores = []
        for entry in self.entries:
            count = sum(word in query for word in entry["trigger_any"])
            if count:
                scores.append((count, entry["mechanism_id"]))
        seen = {mid for _, mid in scores}
        scores.extend(
            (1, mid)
            for mid, pattern in READING_TRIGGERS.items()
            if mid not in seen and re.search(pattern, query)
        )
        scores.sort(key=lambda item: (-item[0], item[1]))
        return [item[1] for item in scores[:limit]]

    def variants(self, query: str) -> list[str]:
        return [NEUTRAL[mid] for mid in self.mechanisms(query, limit=3) if mid in NEUTRAL]

    def reading_mechanisms(self, text: str) -> list[str]:
        """Use specific behavior hints for coverage, separate from broad recall aliases."""
        return [mid for mid, pattern in READING_TRIGGERS.items() if re.search(pattern, text)]
