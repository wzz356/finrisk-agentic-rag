"""Build the new 500-case, 400-development, 200-test synthetic benchmark via real Pi APIs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm import ChatMessage, JsonChatRequest  # noqa: E402
from llm.client import LLMSettings, PiAgentLLMClient  # noqa: E402
from query.classification import SCENE_NAMES  # noqa: E402
from risk.taxonomy import load_taxonomy_descriptions  # noqa: E402
from settings import AppSettings  # noqa: E402

DATA = ROOT / "data/eval/competition500_v1"
OUT = ROOT / "outputs/competition_upgrade_20261019/dataset_build"
SEED = 20261019


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text("utf-8-sig").splitlines() if s.strip()]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    for attempt in range(6):
        try:
            temp.replace(path)
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.1 * (attempt + 1))


def write_rows(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values), "utf-8")
    temporary.replace(path)


def make_client(provider, *, model_name=None, concurrency=6):
    if provider == "qwen":
        key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("FINRISK_DASHSCOPE_API_KEY")
        model = os.getenv("FINRISK_EVAL_QWEN_MODEL", "qwen-plus")
        url = os.getenv(
            "FINRISK_EMBEDDING_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        key = os.getenv("FINRISK_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("FINRISK_LLM_MODEL", "deepseek-flash")
        url = os.getenv("FINRISK_LLM_BASE_URL", "https://api.deepseek.com")
    if not key:
        raise RuntimeError(f"Missing local {provider} credential")
    return PiAgentLLMClient(
        LLMSettings(
            api_key=key,
            model=model_name or model,
            base_url=url,
            thinking="disabled",
            read_timeout_seconds=180,
            max_concurrency=concurrency,
            max_retries=1,
            agent_max_turns=3,
        )
    )


async def call(client, system, data, *, tokens=7000):
    response = await client.complete_json(
        JsonChatRequest(
            task="benchmark_annotation",
            review_context={
                "annotation_kind": "grades"
                if "cases" in data
                else "generate"
                if data.get("inputs", [{}])[0].get("facts")
                else "labels",
                "annotation_ids": [r["query_id"] for r in data.get("inputs", [])],
                "grade_count": len(data.get("cases", [])),
                "grade_indexes": [r["index"] for r in data.get("cases", [])],
                "paired_grades": data.get("paired_grades", False),
                "allowed_labels": list(
                    load_taxonomy_descriptions(AppSettings.from_env().risk_dictionary_path)
                ),
            },
            messages=[
                ChatMessage(
                    role="system",
                    content=system + "直接调用submit_result原生工具提交对象，不输出普通聊天JSON。",
                ),
                ChatMessage(role="user", content=json.dumps(data, ensure_ascii=False)),
            ],
            max_tokens=tokens,
            temperature=0,
        )
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "api_usage.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "model": response.model,
                    "usage": response.usage,
                    "seconds": response.latency_seconds,
                    "task": "benchmark_annotation",
                }
            )
            + "\n"
        )
    return response.data


def select_cases(settings):
    rng = random.Random(SEED)
    records = read_rows(settings.cases_path)
    counts = Counter(t for r in records for t in r.get("risk_tags", []))
    eligible = [r for r in records if r.get("risk_tags") and r.get("behavior_ids")]
    rng.shuffle(eligible)
    selected = []
    used = set()
    # Guarantee rare labels are represented before filling the document-distinct pool.
    for label in sorted(counts, key=lambda t: (counts[t], t)):
        added = 0
        for r in eligible:
            if label not in r["risk_tags"] or r["file_id"] in used:
                continue
            selected.append(r)
            used.add(r["file_id"])
            added += 1
            if added == 8:
                break
    for r in eligible:
        if len(selected) >= 500:
            break
        if r["file_id"] not in used:
            selected.append(r)
            used.add(r["file_id"])
    assert len(selected) == 500 and len(used) == 500
    rng.shuffle(selected)
    con = sqlite3.connect(
        f"file:{(settings.case_kb_root / 'evidence_registry.sqlite3').as_posix()}?mode=ro", uri=True
    )
    pool = []
    for i, r in enumerate(selected):
        quotes = []
        for (raw,) in con.execute("SELECT data FROM behaviors WHERE case_id=?", (r["case_id"],)):
            b = json.loads(raw)
            for eid in b.get("fact_evidence_ids", [])[:4]:
                hit = con.execute("SELECT data FROM evidence WHERE id=?", (eid,)).fetchone()
                if hit:
                    quote = json.loads(hit[0]).get("verbatim", "")
                    if quote and quote not in quotes:
                        quotes.append(quote)
        fact = "；".join(quotes)[:1800]
        if not fact:
            source = con.execute(
                "SELECT text FROM sources WHERE file_id=?", (r["file_id"],)
            ).fetchone()
            fact = source[0][:1800]
        pool.append(
            {
                "case_id": r["case_id"],
                "file_id": r["file_id"],
                "source_sha256": r["source_sha256"],
                "risk_tags": r["risk_tags"],
                "violation_behavior": r["violation_behavior"],
                "fact_text": fact,
                "seed_split": "development" if i < 300 else "test",
            }
        )
    con.close()
    write_rows(DATA / "cases.jsonl", pool)
    scenes = list(SCENE_NAMES)
    plans = []
    for split, n, prefix in [("development", 400, "DEV"), ("test", 200, "TEST")]:
        seeds = [r for r in pool if r["seed_split"] == split]
        states = ["risk"] * int(n * 0.6) + ["no_clear_risk"] * int(n * 0.25)
        states += ["insufficient_information"] * (n - len(states))
        rng.shuffle(states)
        for i, state in enumerate(states):
            seed = seeds[i % len(seeds)]
            plans.append(
                {
                    "query_id": f"{prefix}{i + 1:03}",
                    "split": split,
                    "desired_state": state,
                    "scene": scenes[i % 6].value,
                    "scene_name": SCENE_NAMES[scenes[i % 6]],
                    "seed_case_id": seed["case_id"],
                    "facts": seed["fact_text"],
                    "source_tags": seed["risk_tags"],
                    "length": "long" if i % 10 == 0 else "medium" if i % 3 == 0 else "short",
                }
            )
    write_rows(DATA / "reference/generation_plan.jsonl", plans)
    return pool, plans


async def generate(settings):
    pool, plans = (
        (read_rows(DATA / "cases.jsonl"), read_rows(DATA / "reference/generation_plan.jsonl"))
        if (DATA / "cases.jsonl").exists()
        else select_cases(settings)
    )
    taxonomy = load_taxonomy_descriptions(settings.risk_dictionary_path)
    client = make_client("deepseek")
    sem = asyncio.Semaphore(5)
    system = (
        "你负责构造保险合规竞赛的合成材料，参考真实处罚事实，改写成业务端自然表达。"
        "返回JSON {items:[{query_id,text}]}，每个输入ID一个，不输出答案或编号到text。"
        "scene必须通过真实载体线索体现，不写ST编号或风险标签提示；不使用真实处罚主体名称、文号、处罚金额。"
        "不得连续复制处罚事实20字以上，不能在末尾加风险分析。"
        "risk：明确正在实施/要求实施违规事实；no_clear_risk：正常流程、完整条件披露、明确禁止或纠正反面话术；"
        "insufficient_information：关键合同约定、资质、税务身份或数据依据缺失，不能确定是否违规，不能同时加入明确成立风险。"
        "培训中转述错误示例必须明确其教学立场；独立产品分别写清。short约80-180字，medium约220-400字，"
        "long约900-1400字，长材料至少8段，混合正常流程与目标事实，不要重复灌水。"
        "标签固定为这27类，选择可自然表达的事实，不扩标签："
        + json.dumps(taxonomy, ensure_ascii=False)
    )

    async def batch(start):
        path = OUT / f"generated_{start:03}.json"
        if path.exists():
            return
        async with sem:
            for attempt in range(3):
                try:
                    result = await call(
                        client, system, {"inputs": plans[start : start + 5]}, tokens=14500
                    )
                    assert {r["query_id"] for r in result.get("items", [])} == {
                        r["query_id"] for r in plans[start : start + 5]
                    }
                    assert all(
                        isinstance(r.get("text"), str) and len(r["text"]) >= 30
                        for r in result["items"]
                    )
                    break
                except Exception:
                    if attempt == 2:
                        raise
            items = result.get("items", [])
            assert {r["query_id"] for r in items} == {
                r["query_id"] for r in plans[start : start + 5]
            }
            assert all(isinstance(r.get("text"), str) and len(r["text"]) >= 30 for r in items)
            save(path, result)
            print(f"generated {start + len(items)}/600", flush=True)

    try:
        await asyncio.gather(*(batch(i) for i in range(0, len(plans), 5)))
    finally:
        await client.aclose()
    generated = {
        r["query_id"]: r
        for p in sorted(OUT.glob("generated_*.json"))
        for r in json.loads(p.read_text("utf-8"))["items"]
    }
    assert len(generated) == 600
    write_rows(
        DATA / "model_inputs.jsonl",
        [
            {
                "query_id": p["query_id"],
                "text": generated[p["query_id"]]["text"],
                "split": p["split"],
            }
            for p in plans
        ],
    )


async def annotate(settings):
    inputs = read_rows(DATA / "model_inputs.jsonl")
    taxonomy = load_taxonomy_descriptions(settings.risk_dictionary_path)
    client = make_client("qwen")
    sem = asyncio.Semaphore(5)
    system = (
        "独立审查保险业务材料，不参考任何生成预设或系统预测。仅依据输入文本，区分风险已成立risk、"
        "正常或禁止纠错no_clear_risk、必要业务事实缺失insufficient_information。正常与信息不足的risk_labels为空。"
        "同一材料只要有一项明确成立风险，整篇risk；保留所有语义明确的27类标签，不机械扩展同义标签。"
        "给出Query场景scene，必须在"
        + json.dumps({k.value: v for k, v in SCENE_NAMES.items()}, ensure_ascii=False)
        + "中选择。"
        "给出连续原文evidence短语和简短reason。返回{items:[{query_id,scene,decision_state,risk_labels,evidence,reason}]}。"
        "这是模型辅助银标，不能伪装人工审核。固定标签与定义："
        + json.dumps(taxonomy, ensure_ascii=False)
    )

    async def batch(start):
        path = OUT / f"labels_{start:03}.json"
        if path.exists():
            return
        async with sem:
            source = inputs[start : start + 6]
            collected = {}
            for attempt in range(6):
                pending = [r for r in source if r["query_id"] not in collected]
                if not pending:
                    break
                response = await call(
                    client,
                    system,
                    {"inputs": [{"query_id": r["query_id"], "text": r["text"]} for r in pending]},
                    tokens=10000,
                )
                allowed = {r["query_id"] for r in pending}
                for item in response.get("items", []):
                    if item.get("query_id") in allowed:
                        collected[item["query_id"]] = item
            assert set(collected) == {r["query_id"] for r in source}, (start, list(collected))
            response = {"items": list(collected.values())}
            byid = {r["query_id"]: r for r in source}
            for r in response["items"]:
                assert r["scene"] in {s.value for s in SCENE_NAMES}
                assert r["decision_state"] in {"risk", "no_clear_risk", "insufficient_information"}
                assert set(r["risk_labels"]) <= set(taxonomy)
                literal = [
                    q
                    for q in r["evidence"]
                    if isinstance(q, str) and q in byid[r["query_id"]]["text"]
                ]
                r["nonliteral_evidence_count"] = len(r["evidence"]) - len(literal)
                r["evidence"] = literal
            save(path, response)
            print(f"labeled {start + len(source)}/600", flush=True)

    try:
        await asyncio.gather(*(batch(i) for i in range(0, len(inputs), 6)))
    finally:
        await client.aclose()
    labels = {
        r["query_id"]: r
        for p in OUT.glob("labels_*.json")
        for r in json.loads(p.read_text("utf-8"))["items"]
    }
    assert len(labels) == 600
    write_rows(DATA / "reference/labels.jsonl", [labels[r["query_id"]] for r in inputs])
    for split in ["development", "test"]:
        write_rows(DATA / f"{split}.jsonl", [r for r in inputs if r["split"] == split])
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            DATA / "cases.jsonl",
            DATA / "model_inputs.jsonl",
            DATA / "reference/labels.jsonl",
        ]
    }
    save(
        DATA / "manifest.json",
        {
            "version": "competition500-v1-20261019",
            "cases": 500,
            "source_documents": 500,
            "development_queries": 400,
            "test_queries": 200,
            "risk_labels": 27,
            "query_scenes": 6,
            "query_generator": "Pi/DeepSeek",
            "label_judge": "Pi/Qwen",
            "annotation_status": "model_silver_not_human_gold",
            "split_by": "seed_source_document",
            "old_test_sets_used": False,
            "hashes": hashes,
            "label_distribution": dict(
                Counter(t for r in labels.values() for t in r["risk_labels"])
            ),
            "state_distribution": dict(Counter(r["decision_state"] for r in labels.values())),
        },
    )


async def qrels(settings, split):
    from build_aligned_qrels import run_aligned_qrels

    return await run_aligned_qrels(settings, split)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=["select", "generate", "annotate", "qrels-dev", "qrels-test", "all"]
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    settings = AppSettings.from_env()
    if args.phase == "select":
        select_cases(settings)
        return
    if args.phase in {"generate", "all"}:
        await generate(settings)
    if args.phase in {"annotate", "all"}:
        await annotate(settings)
    if args.phase in {"qrels-dev", "all"}:
        await qrels(settings, "development")
    if args.phase in {"qrels-test", "all"}:
        await qrels(settings, "test")


if __name__ == "__main__":
    asyncio.run(main())
