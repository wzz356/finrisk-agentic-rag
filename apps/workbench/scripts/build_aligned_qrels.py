"""Exhaustive closed-pool silver relevance with query/case evidence for every strong positive."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re

from build_competition_dataset import (
    DATA,
    OUT,
    ROOT,
    SEED,
    call,
    make_client,
    read_rows,
    save,
    write_rows,
)
from dotenv import load_dotenv

from settings import AppSettings

PROTOCOL = "paired-behavior-evidence-v2"


def literal_quote(quote, source):
    if not isinstance(quote, str) or len(quote) < 2:
        return None
    if quote in source:
        return quote
    # Restore whitespace/punctuation differences to an exact source interval; never paraphrase.
    positions = [i for i, char in enumerate(source) if char.isalnum()]
    normalized = "".join(source[i].casefold() for i in positions)
    needle = "".join(char.casefold() for char in quote if char.isalnum())
    start = normalized.find(needle)
    if len(needle) >= 4 and start >= 0:
        return source[positions[start] : positions[start + len(needle) - 1] + 1]
    return None


SYSTEM = (
    "你是保险合规类案检索的独立相关性标注员。必须比较目标query和每个案例的具体违法行为，不能按案例本身风险高低评分。"
    "严格区分：0=行为不相关，仅同属保险/销售/管理也必须为0；1=同一宽泛类型，但缺少具体行为或条件对应；"
    "2=具体行为机制相同，有可明确对应的语句；3=具体行为和关键条件高度一致。"
    "例如：送体检卡诱导投保与合同外赠礼可为2/3；与虚构中介套费、普通代理人管理、执业登记均为0。"
    "保证非保证收益与明确承诺收益为2/3；与虚列费用、代签、回访均为0。"
    "仅有‘内部管理不到位’等泛化表述，没有与query对应的具体行为，最多1。"
    "query可以是禁止性培训或正常材料；比较其讨论的具体行为，不把query是否违规当相关性。"
    "逐一标注全部输入index。返回{grades:{index字符串:值}}；0/1直接填整数，2/3必须填"
    "{grade:2或3,query_quote:query原文连续短语,case_quote:该案例fact原文连续短语}。"
    "双方引文必须包含实际行为，不能只选保险、销售、客户、管理、违规等通用词；不要用别的案例的引文。"
    "每个输入index恰好一个结果，不输出未提供的index。"
)


async def run_aligned_qrels(settings, split):
    inputs = read_rows(DATA / f"{split}.jsonl")
    if split == "development":
        inputs = inputs[:60]
    pool = read_rows(DATA / "cases.jsonl")
    ordered = list(pool)
    random.Random(SEED + 1).shuffle(ordered)
    table = [{"index": i, "fact": row["violation_behavior"][:450]} for i, row in enumerate(ordered)]
    client = make_client(
        "qwen", model_name=os.getenv("FINRISK_QRELS_MODEL", "qwen3.7-flash"), concurrency=12
    )
    semaphore = asyncio.Semaphore(12)

    async def one(row):
        path = OUT / f"qrels_aligned_{row['query_id']}.json"
        if path.exists():
            return
        async with semaphore:
            grades = []
            proofs = {}
            query_options = {
                f"q{i}": piece.strip()
                for i, piece in enumerate(re.split(r"(?<=[。；！？\n])", row["text"]))
                if piece.strip()
            }
            for offset in range(0, 500, 50):
                part_path = OUT / f"aligned_{row['query_id']}_{offset // 50}.json"
                if part_path.exists():
                    part = json.loads(part_path.read_text("utf-8"))
                else:
                    source = [
                        {
                            **r,
                            "case_quote_options": {
                                f"c{i}": piece.strip()
                                for i, piece in enumerate(re.split(r"(?<=[。；\n])", r["fact"]))
                                if piece.strip()
                            },
                        }
                        for r in table[offset : offset + 50]
                    ]
                    incomplete = OUT / f"aligned_incomplete_{row['query_id']}_{offset}.json"
                    collected = (
                        json.loads(incomplete.read_text("utf-8")) if incomplete.exists() else {}
                    )
                    for attempt in range(5):
                        pending = [r for r in source if str(r["index"]) not in collected]
                        if not pending:
                            break
                        try:
                            response = await call(
                                client,
                                SYSTEM + "原文可能含断行和错字，引用必须保留，不补字改写。"
                                "可在query_quote填q编号、case_quote填本案例c编号，程序还原原文；"
                                "选择对应同一具体行为的片段。",
                                {
                                    "paired_grades": True,
                                    "query": row["text"],
                                    "query_quote_options": query_options,
                                    "cases": pending,
                                },
                                tokens=6500,
                            )
                        except Exception:
                            continue
                        values = response.get("grades", {})
                        for candidate in pending:
                            key = str(candidate["index"])
                            value = values.get(key)
                            if type(value) is int and value in (0, 1):
                                collected[key] = value
                            elif isinstance(value, dict) and value.get("grade") in (2, 3):
                                raw_q = value.get("query_quote", "")
                                raw_c = value.get("case_quote", "")
                                q = literal_quote(query_options.get(raw_q, raw_q), row["text"])
                                c = literal_quote(
                                    candidate["case_quote_options"].get(raw_c, raw_c),
                                    candidate["fact"],
                                )
                                if (
                                    isinstance(q, str)
                                    and isinstance(c, str)
                                    and len(q) >= 2
                                    and len(c) >= 2
                                    and q in row["text"]
                                    and c in candidate["fact"]
                                ):
                                    collected[key] = {**value, "query_quote": q, "case_quote": c}
                        save(incomplete, collected)
                    if len(collected) != len(source):
                        save(OUT / f"aligned_incomplete_{row['query_id']}_{offset}.json", collected)
                        raise ValueError(
                            f"Incomplete pairwise evidence: {row['query_id']} batch {offset}"
                        )
                    part = {"values": collected}
                    save(part_path, part)
                for index in range(offset, offset + 50):
                    value = part["values"][str(index)]
                    grades.append(value if type(value) is int else value["grade"])
                    if isinstance(value, dict):
                        proofs[ordered[index]["case_id"]] = value
            save(
                path,
                {
                    "protocol": PROTOCOL,
                    "query_id": row["query_id"],
                    "grades": {r["case_id"]: g for r, g in zip(ordered, grades)},
                    "positive_evidence": proofs,
                },
            )
            print("aligned " + row["query_id"], flush=True)

    try:
        results = await asyncio.gather(*(one(r) for r in inputs), return_exceptions=True)
        failures = [str(r) for r in results if isinstance(r, BaseException)]
        if failures:
            save(OUT / f"aligned_failures_{split}.json", failures)
            raise RuntimeError(f"{len(failures)} annotation tasks require resume")
    finally:
        await client.aclose()
    write_rows(
        DATA / f"reference/qrels_{split}.jsonl",
        [
            json.loads((OUT / f"qrels_aligned_{r['query_id']}.json").read_text("utf-8"))
            for r in inputs
        ],
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["development", "test", "both"], default="both", nargs="?")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    for split in ["development", "test"] if args.split == "both" else [args.split]:
        await run_aligned_qrels(settings, split)


if __name__ == "__main__":
    asyncio.run(main())
