"""Measure synthetic 2k/6k-character documents through the real Pi pipeline."""

from __future__ import annotations

import asyncio
import json
import math
from collections import defaultdict
from dataclasses import replace
from statistics import median

from benchmark_competition500 import OUT, PoolRepository, freeze_runtime, run_risk
from build_competition_dataset import DATA, ROOT, read_rows, save, write_rows
from dotenv import load_dotenv

from application.retrieval_service import CaseRetrievalService
from settings import AppSettings


def samples():
    references = {r["query_id"]: r for r in read_rows(DATA / "reference/labels.jsonl")}
    selected = {}
    for row in read_rows(DATA / "development.jsonl"):
        reference = references[row["query_id"]]
        if reference["decision_state"] == "risk":
            selected.setdefault(reference["scene"], row)
    assert len(selected) == 6
    output = []
    for index, (scene, seed) in enumerate(sorted(selected.items())):
        for target in [2000, 6000]:
            paragraphs = []
            size = 0
            while size < target - len(seed["text"]):
                section = len(paragraphs) + 1
                paragraph = (
                    f"第{section}节材料编排：本节记录业务材料的目录位置和日常服务安排。"
                    f"资料编号为{index + 1:02d}-{section:03d}，归档时填写名称、日期和经办人。"
                    "客户可通过服务窗口了解办理步骤，工作人员按材料目录逐项讲解并保存服务记录。"
                    "本节另附页面检查清单、培训签到页和会议日程，目录按业务环节顺序排列。"
                )
                paragraphs.append(paragraph)
                size += len(paragraph) + 1
            insertion = (index % 3) * len(paragraphs) // 2
            paragraphs.insert(insertion, seed["text"])
            output.append(
                {
                    "query_id": f"LONG{index + 1:02d}_{target}",
                    "split": "longtext",
                    "text": "\n\n".join(paragraphs),
                    "seed_query_id": seed["query_id"],
                    "seed_scene": scene,
                    "target_chars": target,
                    "risk_position": ["head", "middle", "tail"][index % 3],
                }
            )
    return output


async def main():
    load_dotenv(ROOT / ".env")
    settings = replace(
        AppSettings.from_env(), type_bridge_enabled=False, query_rewrite_original_anchors=0
    )
    freeze_runtime()
    documents = samples()
    write_rows(OUT / "longtext_inputs.jsonl", documents)
    info = {r["query_id"]: r for r in documents}
    repository = PoolRepository(settings, [r["case_id"] for r in read_rows(DATA / "cases.jsonl")])
    retrieval = CaseRetrievalService(repository, settings)
    retrieval.rebuild()
    retrieval._ensure_dense_index(allow_build=True)
    reports = {}
    try:
        for mode in ["deepseek_post", "deepseek_adaptive"]:
            rows = await run_risk(mode, documents, settings, repository, retrieval)
            grouped = defaultdict(list)
            for row in rows:
                grouped[info[row["query_id"]]["target_chars"]].append(row)
            reports[mode] = {}
            for length, group in sorted(grouped.items()):
                times = sorted(r["seconds"] for r in group)
                reports[mode][str(length)] = {
                    "queries": len(group),
                    "actual_chars_min": min(r["chars"] for r in group),
                    "actual_chars_max": max(r["chars"] for r in group),
                    "median_seconds": median(times),
                    "p95_seconds": times[math.ceil(len(times) * 0.95) - 1],
                    "risk_queries_detected": sum(r["decision_state"] == "risk" for r in group),
                    "missing_model_targets": sum(r.get("missing_model_targets", 0) for r in group),
                    "errors": sum(bool(r.get("error")) for r in group),
                }
            save(
                OUT / "longtext_summary.json",
                {
                    "protocol": "synthetic-padding-from-development-only-v1",
                    "query_concurrency": 3,
                    "scope": "closed_500_cases_warm_index",
                    "test_set_used": False,
                    "risk_position": "head_middle_tail",
                    "results": reports,
                },
            )
            print(json.dumps(reports[mode], ensure_ascii=False), flush=True)
    finally:
        retrieval.close()


if __name__ == "__main__":
    asyncio.run(main())
