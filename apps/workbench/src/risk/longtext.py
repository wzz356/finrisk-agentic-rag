"""A complete document screening pass before expensive sentence-level review."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from domain.agent_models import AgentExecutionTrace
from domain.query_models import RiskQuerySentence
from llm import ChatMessage, JsonChatRequest, LLMClient, LLMProviderError


@dataclass(frozen=True)
class DocumentScreen:
    normal_ids: frozenset[str]
    review_ids: frozenset[str]
    traces: tuple[AgentExecutionTrace, ...]


def parse_screen(data: dict[str, object], expected: set[int]) -> tuple[set[int], set[int]]:
    """Require an explicit, disjoint partition; omission never means normal."""
    review_raw = data.get("review_indexes")
    ranges = data.get("normal_ranges")
    if data.get("complete") is not True or not isinstance(review_raw, list):
        raise LLMProviderError("全文初筛未提交完整分区")
    if not isinstance(ranges, list) or any(type(v) is not int for v in review_raw):
        raise LLMProviderError("全文初筛编号格式错误")
    review = set(review_raw)
    if len(review) != len(review_raw) or not review <= expected:
        raise LLMProviderError("全文初筛待核查编号重复或越界")
    normal: set[int] = set()
    for interval in ranges:
        if (
            not isinstance(interval, list)
            or len(interval) != 2
            or any(type(v) is not int for v in interval)
        ):
            raise LLMProviderError("全文初筛正常区间格式错误")
        start, end = interval
        if start > end or start < 0 or end > max(expected):
            raise LLMProviderError("全文初筛正常区间越界")
        indexes = set(range(start, end + 1))
        if not indexes <= expected or indexes & (normal | review):
            raise LLMProviderError("全文初筛编号分区重叠或越界")
        normal.update(indexes)
    if normal | review != expected:
        raise LLMProviderError("全文初筛遗漏原句")
    return normal, review


async def screen_document(
    client: LLMClient,
    sentences: tuple[RiskQuerySentence, ...],
    taxonomy: tuple[str, ...],
) -> DocumentScreen:
    chunks: list[list[int]] = []
    current: list[int] = []
    size = 0
    for index, sentence in enumerate(sentences):
        if current and size + len(sentence.text) > 12_000:
            chunks.append(current)
            current, size = [], 0
        current.append(index)
        size += len(sentence.text)
    if current:
        chunks.append(current)
    semaphore = asyncio.Semaphore(3)

    async def one(indexes: list[int]) -> tuple[set[int], set[int], AgentExecutionTrace | None]:
        # Adjacent source sentences preserve questions, corrections and product boundaries.
        begin, end = max(0, indexes[0] - 2), min(len(sentences), indexes[-1] + 3)
        expected = set(indexes)
        payload = [
            {"index": i, "text": sentences[i].text, "target": i in expected}
            for i in range(begin, end)
        ]
        async with semaphore:
            response = await client.complete_json(
                JsonChatRequest(
                    task="risk_screen",
                    review_context={"expected_indexes": indexes},
                    messages=[
                        ChatMessage(
                            role="system",
                            content=(
                                "通读保险业务材料，做高召回全文初筛。当前只决定哪些句子需要详细核查。"
                                "所有可能违规、条件不明、数据资质待核、问答承接、跨句指代或难判断的句子"
                                "都放review_indexes，不限制数量，不因普通免责声明排除已作出的承诺。"
                                "后台财务、账户、人员资质、客户信息和业务操作异常同样需要核查。"
                                "只有明确正常的日程、目录、服务说明或完整合规要求可放normal_ranges。"
                                "禁止性培训与反面话术结合全文；有疑问仍放review_indexes。"
                                "每个target=true的编号恰好属于一个分区；normal_ranges用闭区间[start,end]"
                                "压缩连续正常句。不能遗漏、重叠或返回target=false编号。"
                                "直接submit_result提交complete=true、review_indexes和normal_ranges。"
                                "风险标签词表：" + "、".join(taxonomy)
                            ),
                        ),
                        ChatMessage(
                            role="user",
                            content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        ),
                    ],
                    temperature=0,
                    max_tokens=3000,
                )
            )
        normal, review = parse_screen(response.data, expected)
        return normal, review, response.agent_trace

    results = await asyncio.gather(*(one(indexes) for indexes in chunks))
    normal = {i for ns, _, _ in results for i in ns}
    review = {i for _, rs, _ in results for i in rs}
    return DocumentScreen(
        normal_ids=frozenset(sentences[i].sentence_id for i in normal),
        review_ids=frozenset(sentences[i].sentence_id for i in review),
        traces=tuple(trace for _, _, trace in results if trace is not None),
    )
