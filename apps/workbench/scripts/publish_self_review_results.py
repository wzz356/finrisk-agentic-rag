"""Write local public-format reports for the directly reviewed reference version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter

from benchmark_competition500 import ranking_metrics, risk_metrics
from build_competition_dataset import DATA, ROOT, read_rows, save, write_rows

from risk.taxonomy import LABEL_BOUNDARIES

PUBLIC = ROOT / "reports/competition500"
RUNS = ROOT / "outputs/competition_upgrade_20261019/runs"
MODE = "deepseek_adaptive"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = RUNS / args.run
    samples = read_rows(DATA / "test.jsonl")
    gold = {r["query_id"]: r for r in read_rows(DATA / "reference/labels.jsonl")}
    predictions = json.loads((run / f"test_{MODE}.json").read_text("utf-8"))
    rankings = json.loads((run / "retrieval_test_deepseek.json").read_text("utf-8"))
    expected = {r["query_id"] for r in samples}
    assert len(predictions) == 200 and {r["query_id"] for r in predictions} == expected
    assert len(rankings) == 200 and {r["query_id"] for r in rankings} == expected
    current = risk_metrics(predictions, gold, list(LABEL_BOUNDARIES))
    counts = {key: sum(r[key] for r in current["per_label"].values()) for key in ("tp", "fp", "fn")}
    current["label_micro_precision"] = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    current["label_micro_recall"] = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    current["label_counts"] = counts
    current["exact_label_set_accuracy"] = (
        sum(set(r["risk_labels"]) == set(gold[r["query_id"]]["risk_labels"]) for r in predictions)
        / 200
    )
    bands = {}
    for name, lower, upper in (
        ("under500", 0, 499),
        ("500to899", 500, 899),
        ("900plus", 900, 1000000),
    ):
        subset = [r for r in predictions if lower <= r["chars"] <= upper]
        if subset:
            metrics = risk_metrics(subset, gold, list(LABEL_BOUNDARIES))
            bands[name] = {
                key: metrics[key]
                for key in (
                    "queries",
                    "risk_binary_accuracy",
                    "risk_recall",
                    "label_micro_f1",
                    "median_seconds",
                    "p95_seconds",
                    "errors",
                    "missing_model_targets",
                )
            }
            bands[name]["gold_risk_queries"] = sum(
                gold[r["query_id"]]["decision_state"] == "risk" for r in subset
            )
            bands[name]["label_support"] = sum(
                len(gold[r["query_id"]]["risk_labels"]) for r in subset
            )
            bands[name]["false_positive_queries"] = sum(
                gold[r["query_id"]]["decision_state"] != "risk" and r["decision_state"] == "risk"
                for r in subset
            )
    qrels = {r["query_id"]: r["grades"] for r in read_rows(DATA / "reference/qrels_test.jsonl")}
    retrieval = ranking_metrics(rankings, qrels)
    manifest = json.loads((DATA / "manifest.json").read_text("utf-8"))
    runtime = json.loads((run / "run_manifest.json").read_text("utf-8"))
    assert runtime["source_hashes"][str((DATA / "manifest.json").relative_to(ROOT))] == fingerprint(
        DATA / "manifest.json"
    )
    api = json.loads((run / f"test_{MODE}_api.json").read_text("utf-8"))
    report = {
        "available": True,
        "version": "competition500-direct-review-evaluation-v3-20261019",
        "scope": "closed_500_cases_directly_reviewed_reference",
        "selected_mode": MODE,
        "default_config": {
            "rerank_provider": "deepseek",
            "retrieval_timing": "adaptive",
            "query_anchors": 0,
            "retrieval_top_k": 5,
            "local_calibration": True,
            "longtext_screening": True,
        },
        "dataset": {
            "version": manifest["version"],
            "cases": 500,
            "source_documents": 500,
            "development_queries": 400,
            "development_validation_queries": 100,
            "test_queries": 200,
            "label_count": 27,
            "scene_count": 6,
            "annotation": "Codex逐条自复核",
            "human_expert_gold": False,
            "qrels_protocol": manifest["relevance_protocol"],
            "explicit_test_query_case_pairs": 100000,
            "test_used_for_training_or_selection": False,
            "external_model_review_calls": 0,
        },
        "risk": {MODE: current},
        "risk_names": {MODE: "DeepSeek · 自适应融合"},
        "retrieval": {"deepseek": retrieval},
        "retrieval_measurement_note": "按逐条复核的具体行为目录与来源证据重新评测。",
        "review": manifest["review_summary"],
        "length_bands": bands,
        "provenance": {
            "run": args.run,
            "label_reference_sha256": fingerprint(DATA / "reference/labels.jsonl"),
            "test_input_sha256": fingerprint(DATA / "test.jsonl"),
            "test_references_modified_for_this_upgrade": True,
            "labels_modified_using_predictions": False,
            "external_model_review_calls": 0,
            "comparison_note": (
                "Reference changed; old silver-reference numbers are not comparable."
            ),
            "risk_api_tasks": dict(Counter(r["task"] for r in api)),
            "risk_api_errors": sum("error" in r for r in api),
            "all_ranked_evidence_verified": all(r["literal_verified"] for r in rankings),
        },
    }
    previous_path = (
        ROOT / "outputs/dataset_self_review_20261019/old_model_rescored_new_reference.json"
    )
    if previous_path.exists():
        report["previous_predictions_rescored"] = {
            "source_run": "label_precision_v2",
            "reference": manifest["version"],
            "new_api_run_of_previous_model": False,
            "metrics": json.loads(previous_path.read_text("utf-8")),
        }
    PUBLIC.mkdir(parents=True, exist_ok=True)
    save(PUBLIC / "evaluation.json", report)
    save(run / "test_summary.json", {MODE: current})
    save(
        PUBLIC / "dataset_audit.json",
        json.loads(
            (ROOT / "outputs/competition_upgrade_20261019/dataset_audit.json").read_text("utf-8")
        ),
    )
    save(PUBLIC / "dataset_hashes.json", manifest["hashes"])
    write_rows(
        PUBLIC / "predictions_test200.jsonl",
        [
            {
                key: r.get(key)
                for key in (
                    "query_id",
                    "scene",
                    "decision_state",
                    "risk_labels",
                    "seconds",
                    "missing_model_targets",
                    "error",
                )
            }
            for r in sorted(predictions, key=lambda r: r["query_id"])
        ],
    )
    write_rows(PUBLIC / "retrieval_test200.jsonl", rankings)
    write_csv(
        PUBLIC / "27标签复核后测试指标.csv",
        [
            {
                "标签": label,
                "支持数": r["support"],
                "TP": r["tp"],
                "FP": r["fp"],
                "FN": r["fn"],
                "F1": r["f1"],
            }
            for label, r in current["per_label"].items()
        ],
    )
    by_id = {r["query_id"]: r for r in predictions}
    write_csv(
        PUBLIC / "200Query复核答案与预测.csv",
        [
            {
                "Query": row["query_id"],
                "参考三态": gold[row["query_id"]]["decision_state"],
                "预测三态": row["decision_state"],
                "参考标签": "；".join(gold[row["query_id"]]["risk_labels"]),
                "预测标签": "；".join(row["risk_labels"]),
                "复核理由": gold[row["query_id"]]["reason"],
                "耗时秒": row["seconds"],
            }
            for row in sorted(predictions, key=lambda r: r["query_id"])
        ],
    )
    submission = []
    for sample in samples:
        row = by_id[sample["query_id"]]
        result = row.get("result", {})
        cases = result.get("risk_summary", {}).get("top_matched_cases", [])
        suggestions = list(
            dict.fromkeys(
                s
                for a in result.get("risk_assessments", [])
                for s in a.get("remediation_suggestions", [])
            )
        )
        submission.append(
            {
                "question_id": sample["query_id"],
                "risk_type": "；".join(row["risk_labels"]),
                "retrieved_cases": [
                    {"case_id": c["case_id"], "rank": i + 1, "reason": c["match_reason"]}
                    for i, c in enumerate(cases)
                ],
                "suggestion": "；".join(suggestions),
            }
        )
    write_rows(PUBLIC / "submission.jsonl", submission)
    lines = [
        "# FinRisk 数据自复核与真实API测试（2026-10-19）",
        "",
        "本次由Codex逐条阅读400条开发Query、200条测试Query和500个候选案例。"
        "标注复核未调用其他模型；真实API只运行项目推理和检索。",
        "",
        "357条Query修正状态、标签或场景；72个案例修正标签或补充摘录；33个条目回查来源全文。"
        "600条Query引文与500个案例来源校验通过，来源SHA和输入/参考指纹随报告保存。",
        "",
        "修改前后及理由见data/eval/competition500_v1/reference/review_changes.json。"
        "新参考是Codex自复核结果，未作为人类专家金标；没有依据系统预测修改答案。",
        "",
        "相关性从逐条核查的行为目录展开完整600×500矩阵，"
        "保留双方证据、机制和来源，不将30万对称为独立人工逐对打分。",
        "",
        "本地规则收窄财务不实/套费、个人/机构资质、返佣/垫款和普通限时宣传的边界；"
        "跨段落反例说明加入共享原文上下文，避免分批时丢失立场。"
        "训练及few-shot仅使用修正后的400条开发Query。"
        "Pi、三种重排选择、Pre/Post/自适应选项均保留；默认DeepSeek重排和自适应融合。",
        "",
        "## 200Query、500案例闭集测试",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
    ]
    fields = [
        ("风险二分类准确率", "risk_binary_accuracy"),
        ("风险Precision", "risk_precision"),
        ("风险Recall", "risk_recall"),
        ("三态准确率", "state_accuracy"),
        ("27标签Micro-F1", "label_micro_f1"),
        ("27标签Macro-F1", "label_macro_f1"),
        ("标签Precision", "label_micro_precision"),
        ("标签Recall", "label_micro_recall"),
        ("六场景准确率", "scene_accuracy"),
    ]
    lines += [f"| {name} | {current[key]:.2%} |" for name, key in fields]
    lines += [
        f"| 中位耗时 | {current['median_seconds']:.2f}s |",
        f"| P95耗时 | {current['p95_seconds']:.2f}s |",
        f"| 未完成模型复核句 | {current['missing_model_targets']} |",
        f"| 请求失败Query | {current['errors']} |",
        f"| 检索Hit@5 | {retrieval['hit5']:.2%} |",
        f"| 检索Recall@10 | {retrieval['recall_at_10']:.2%} |",
        f"| MRR@20 | {retrieval['mrr']:.3f} |",
        f"| NDCG@10 | {retrieval['ndcg_at_10']:.3f} |",
        "",
        f"检索有强相关案例的Query为{retrieval['positive_queries']}/200，"
        f"无强相关案例{retrieval['no_positive_queries']}条单列。"
        "Hit表示至少找到一条；Recall表示整个相关案例集合的覆盖率。",
        "",
        "新参考修正了旧银标，不能将新旧分数直接作为同口径提升。"
        "测试集仍是自建合成材料与500案例闭集，不等于全库或赛事隐藏测试效果。",
        "",
        "本次只修改本地，未上传GitHub、未部署服务器。",
    ]
    lines += [
        "",
        f"最终标签TP={counts['tp']}、FP={counts['fp']}、FN={counts['fn']}。"
        f"内部API调用异常{sum('error' in r for r in api)}次，"
        f"最终未完成模型核查句{current['missing_model_targets']}、整条Query失败{current['errors']}。"
        "风险识别已完成，不表示27标签没有剩余漏标；逐标签支持数和错误计数见CSV。",
    ]
    if previous_path.exists():
        previous = report["previous_predictions_rescored"]["metrics"]
        old_counts = {
            key: sum(r[key] for r in previous["per_label"].values()) for key in ("tp", "fp", "fn")
        }
        lines += [
            "",
            "将上版冻结预测按同一新参考重评分：风险准确率"
            f"{previous['risk_binary_accuracy']:.1%}→{current['risk_binary_accuracy']:.1%}，"
            f"标签Micro-F1 {previous['label_micro_f1']:.2%}→{current['label_micro_f1']:.2%}。"
            f"标签FP {old_counts['fp']}→{counts['fp']}，FN {old_counts['fn']}→{counts['fn']}。"
            "改善主要来自误标减少，漏标尚未全面改善；旧版没有重新调用API。",
        ]
    lines += [
        "",
        "## 本轮按实际字数分层",
        "",
        "| 字数 | Query数 | 风险样本 | 风险Recall | 标签Micro-F1 | 误报Query | 中位秒 | P95秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, value in bands.items():
        display = {"under500": "小于500", "500to899": "500–899", "900plus": "至少900"}[name]
        recall_text = f"{value['risk_recall']:.2%}" if value["gold_risk_queries"] else "—"
        f1_text = f"{value['label_micro_f1']:.2%}" if value["label_support"] else "—"
        lines.append(
            f"| {display} | {value['queries']} | {value['gold_risk_queries']} | {recall_text} | "
            f"{f1_text} | {value['false_positive_queries']} | {value['median_seconds']:.2f} | "
            f"{value['p95_seconds']:.2f} |"
        )
    lines += [
        "",
        "字数为Python字符计数。本轮最长材料为"
        + str(max(r["chars"] for r in predictions))
        + "字；这些分层结果不替代数千至上万字的独立长文压测。",
        "900字以上本轮只有1条正常材料且被误报，不据此声称长文风险召回已经验证通过。",
    ]
    (ROOT / "docs/dataset_review_20261019.md").write_text("\n".join(lines) + "\n", "utf-8")
    print(json.dumps({"risk": current, "retrieval": retrieval}, ensure_ascii=False))


if __name__ == "__main__":
    main()
