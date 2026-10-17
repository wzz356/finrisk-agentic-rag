"""汇总公示表拆行：NFRA 详情 docClob（Word-HTML 表格）→ 单案文本。

背景：金监总局现行公示口径以"信息公开表/公示列表"摘要表为主，
一张表含多案（序号/当事人/文号/违法违规行为/处罚内容/决定机关）。
本模块把每行拆成一条独立案例文本，字段标签对齐
src/query_rebuild/parser.py 的识别口径（主要违法违规事实/行政处罚内容/
作出处罚决定的机关名称/作出处罚决定的日期），保证拆分件直接过 parser。
"""

import re
from html import unescape

_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_CELL_FULL_RE = re.compile(r"<t[dh]([^>]*)>(.*?)</t[dh]>", re.S | re.I)
_ROWSPAN_RE = re.compile(r"rowspan\s*=\s*\"?(\d+)", re.I)
_TAG_ANY = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\xa0　]+")

# 列名关键词 → 输出字段标签（标签措辞对齐 parser.py 的 FACT_FIELD/PENALTY_FIELD）
# 顺序即优先级，长词/特异性词必须在前：
# - "作出处罚决定的日期"/"作出处罚决定的机关名称" 含 "处罚决定"/"名称"，
#   若处罚内容列或当事人列先匹配会被抢映射。
# - "行政处罚依据" 是独立列（法条），映射为独立标签，不能并进处罚内容：
#   parser.py 的 PENALTY_FIELD 含"依据"，并进去会污染 penalty_content 捕获。
_COLUMN_MAP = [
    (("决定书文号", "处罚文号", "文号"), "行政处罚决定书文号"),
    (("决定的日期", "决定日期", "处罚日期", "决定时间"), "作出处罚决定的日期"),
    (("决定的机关", "决定机关", "处罚机关"), "作出处罚决定的机关名称"),
    (("当事人", "名称"), "当事人"),
    (("违法违规行为", "违法违规事实", "违法行为类型", "案由"), "主要违法违规事实(案由)"),
    (("处罚依据",), "行政处罚依据"),
    (("处罚内容", "处罚决定"), "行政处罚内容"),
]


def _cell_text(cell_html: str) -> str:
    text = _TAG_ANY.sub("", cell_html)
    return _WS.sub(" ", unescape(text)).strip()


def _map_header(cell: str) -> str | None:
    for keywords, label in _COLUMN_MAP:
        if any(keyword in cell for keyword in keywords):
            return label
    return None


def _expand_rows(clob_html: str) -> list[list[str]]:
    """rowspan 感知的表格解析，返回补齐后的等宽行列表。

    汇总公示表大量使用纵向合并单元格（同一文号/机关/日期/依据覆盖机构行
    与其下的多个个人责任人行），被合并的格只在首行出现。不按 rowspan 补位
    会让后续行的单元格整体左移错位（案由装进文号、处罚装进案由等）。
    """
    expanded: list[list[str]] = []
    carry: dict[int, tuple[str, int]] = {}  # 列号 -> (跨行值, 剩余行数)
    for m in _ROW_RE.finditer(clob_html):
        raw_cells = _CELL_FULL_RE.findall(m.group(1))
        row: dict[int, str] = {}
        col = 0
        ci = 0
        while True:
            if col in carry and carry[col][1] > 0:
                row[col] = carry[col][0]
                carry[col] = (carry[col][0], carry[col][1] - 1)
                col += 1
                continue
            if ci < len(raw_cells):
                attrs, body = raw_cells[ci]
                ci += 1
                value = _cell_text(body)
                row[col] = value
                rs = _ROWSPAN_RE.search(attrs)
                if rs and int(rs.group(1)) > 1:
                    carry[col] = (value, int(rs.group(1)) - 1)
                col += 1
                continue
            break
        if row:
            expanded.append([row.get(c, "") for c in range(max(row) + 1)])
    return expanded


_ISO_DATE_RE = re.compile(r"(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})")


def _cn_date(value: str) -> str:
    """ISO 日期转中文格式：field_extraction.py 的 DATE_PATTERNS 只认 '2026年8月20日'。
    兼容带时间后缀（'2026-08-20 10:00:00'）和点/斜杠分隔；已是中文格式则原样返回。"""
    match = _ISO_DATE_RE.search(value.strip())
    if match:
        year, month, day = match.groups()
        return f"{year}年{int(month)}月{int(day)}日"
    return value


def extract_table_cases(
    clob_html: str,
    *,
    source_title: str = "",
    publish_date: str = "",
) -> list[str]:
    """从 Word-HTML 表格中拆出单案文本列表。

    返回 [] 表示不是可拆分的汇总表（如无表头、无数据行），调用方应回退整篇文本。
    """
    rows = [row for row in _expand_rows(clob_html) if any(row)]
    if not rows:
        return []

    # 找表头行：至少两个单元格能映射到已知列
    header_labels: list[str | None] | None = None
    header_index = -1
    for index, row in enumerate(rows[:5]):
        labels = [_map_header(cell) for cell in row]
        if sum(label is not None for label in labels) >= 2:
            header_labels = labels
            header_index = index
            break
    if header_labels is None:
        return []

    cases = []
    for row in rows[header_index + 1 :]:
        # 数据行首格为序号（数字）才是案件行；跳过续行/说明行
        if not row or not re.fullmatch(r"\d{1,3}", row[0] or ""):
            continue
        fields: dict[str, str] = {}
        for label, value in zip(header_labels, row):
            if label and value and label not in fields:
                if label == "作出处罚决定的日期":
                    value = _cn_date(value)
                fields[label] = value
        if not fields.get("当事人") or not (
            fields.get("主要违法违规事实(案由)") or fields.get("行政处罚内容")
        ):
            continue
        # 字段标签与值的排版对齐下游两个消费者的识别口径：
        # - parser.py 的 FACT_FIELD/PENALTY_FIELD 按 "标签\s*值" 捕获
        #   → 标签与值分行（同行冒号会污染捕获值）
        # - field_extraction.py 的当事人正则要求 "当事人[:：]值" 同行 → 当事人用同行冒号格式
        # - "行政处罚依据" 必须排在 机关/日期 之后：parser 的 PENALTY_FIELD 含"依据"，
        #   若夹在案由与处罚内容之间会被当成处罚段开头，把法条混进 penalty_content；
        #   排在处罚段结束后，LEGAL_PATTERN 仍能从全文中扫到《法》第X条填入 legal_basis。
        lines = []
        for label, value in fields.items():
            if label == "行政处罚依据":
                continue
            if label == "当事人":
                lines.append(f"当事人: {value}")
            else:
                lines.append(f"{label}\n{value}")
        if publish_date and "作出处罚决定的日期" not in fields:
            lines.append(f"作出处罚决定的日期\n{_cn_date(publish_date)}")
        if "行政处罚依据" in fields:
            lines.append(f"行政处罚依据\n{fields['行政处罚依据']}")
        if source_title:
            lines.append(f"来源公示\n{source_title}")
        cases.append("\n".join(lines))
    return cases
