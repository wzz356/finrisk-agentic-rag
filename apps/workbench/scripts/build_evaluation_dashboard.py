"""Build the public evaluation snapshot from frozen runs, without calling any model."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from benchmark_competition500 import ranking_metrics, risk_metrics  # noqa: E402

RUNS = ROOT / "outputs/competition_upgrade_20261019/runs"
DEST = ROOT / "reports/evaluation_dashboard/evaluation.json"
SELECTED = "query_20261024_candidate_final"
STATES = ("risk", "no_clear_risk", "insufficient_information")


def read(path):
    return json.loads(path.read_text("utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text("utf-8-sig").splitlines() if line]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    data = ROOT / "data/eval/competition500_v1"
    inputs = {row["query_id"]: row for row in rows(data / "test.jsonl")}
    gold = {row["query_id"]: row for row in rows(data / "reference/labels.jsonl")}
    qrels = {row["query_id"]: row["grades"] for row in rows(data / "reference/qrels_test.jsonl")}
    comparison_path = ROOT / "reports/query_optimization_20261024/comparison.json"
    comparison = read(comparison_path)
    original = comparison["runs"]["candidate"]
    assert original["run"] == SELECTED
    labels = list(original["risk"]["per_label"])
    sources = {comparison_path.relative_to(ROOT).as_posix(): digest(comparison_path.read_bytes())}
    run_dir = RUNS / original["run"]
    summaries = read(run_dir / "test_summary.json")
    assert summaries["deepseek_adaptive"] == original["risk"]
    rankings = read(run_dir / "retrieval_test_deepseek.json")
    assert ranking_metrics(rankings, qrels) == original["retrieval"]
    assert len(rankings) == 200 and {r["query_id"] for r in rankings} == set(inputs)
    ranked = {}
    for row in rankings:
        relevant = {cid for cid, grade in qrels[row["query_id"]].items() if grade >= 2}
        ranked[row["query_id"]] = {
            **row,
            "relevant_count": len(relevant),
            "grades": [qrels[row["query_id"]][cid] for cid in row["case_ids"]],
            "metrics": ranking_metrics([row], qrels) if relevant else None,
        }
    run = {
        "id": original["run"],
        "name": "当前启用版本",
        "status": "current",
        "evaluated_on": "2026-10-24",
        "risk": summaries,
        "retrieval": original["retrieval"],
        "retrieval_verified": sum(r["literal_verified"] for r in rankings),
        "retrieval_warnings": sum(bool(r["warnings"]) for r in rankings),
        "rankings": ranked,
        "modes": {},
    }
    for mode, expected in summaries.items():
        predictions = read(run_dir / f"test_{mode}.json")
        assert len(predictions) == 200 and {r["query_id"] for r in predictions} == set(inputs)
        assert risk_metrics(predictions, gold, labels) == expected
        counts = Counter(
            (gold[r["query_id"]]["decision_state"], r["decision_state"]) for r in predictions
        )
        groups = {
            "scenes": {
                s: [r for r in predictions if gold[r["query_id"]]["scene"] == s]
                for s in sorted({gold[q]["scene"] for q in inputs})
            },
            "lengths": {
                "少于 500 字": [r for r in predictions if r["chars"] < 500],
                "500–899 字": [r for r in predictions if 500 <= r["chars"] < 900],
                "900 字及以上": [r for r in predictions if r["chars"] >= 900],
            },
        }
        run["modes"][mode] = {
            "confusion": {a: {p: counts[a, p] for p in STATES} for a in STATES},
            **{
                group: {
                    key: risk_metrics(value, gold, labels) for key, value in values.items() if value
                }
                for group, values in groups.items()
            },
            "predictions": [
                {
                    key: r.get(key)
                    for key in (
                        "query_id",
                        "scene",
                        "decision_state",
                        "risk_labels",
                        "seconds",
                        "chars",
                        "missing_model_targets",
                        "unchecked_mechanisms",
                        "error",
                        "warnings",
                    )
                }
                for r in sorted(predictions, key=lambda r: r["query_id"])
            ],
        }

    # Bind to the recorded evaluated bytes, never bless the current checkout as a new test.
    manifest = read(RUNS / SELECTED / "run_manifest.json")
    bindings = {}
    with ZipFile(RUNS / SELECTED / "source_snapshot.zip") as archive:
        for original, expected in manifest["source_hashes"].items():
            name = original.replace("\\", "/")
            if name.startswith("src/api/"):
                continue  # HTTP/report presentation is changed in this release, not the algorithm.
            frozen = (
                archive.read(name) if name in archive.namelist() else (ROOT / name).read_bytes()
            )
            assert digest(frozen) == expected, name
            bindings[name] = digest(
                frozen if name.endswith(".npz") else frozen.replace(b"\r\n", b"\n")
            )
    context = read(ROOT / "reports/information_sufficiency_20261024/comparison.json")[
        "context_bindings"
    ]
    payload = {
        "available": True,
        "version": "finrisk-evaluation-acceptance-20261024-v3",
        "evaluated_on": "2026-10-24",
        "selected_run": SELECTED,
        "selected_mode": "deepseek_adaptive",
        "measurement": {"kind": "release_acceptance", "label": "发布验收", "run": SELECTED},
        "algorithm_revision": "41f2765780c71d740eed57e17fe8b64885cfb9b4",
        "knowledge_version": "finrisk-full-20261024",
        "dataset": {
            "version": "competition500-codex-reviewed-v2-20261019",
            "cases": 500,
            "test_queries": 200,
            "label_count": 27,
            "scene_count": 6,
            "annotation": "Codex 逐条自复核",
            "human_expert_gold": False,
        },
        "default_config": {
            "rerank_provider": "deepseek",
            "retrieval_timing": "adaptive",
            "local_calibration": True,
            "longtext_screening": True,
        },
        "notes": [
            "采用当前启用算法的 2026-10-24 发布验收记录，使用同一批 200 条 Query；"
            "参考由 Codex 自复核，非专家金标。",
            "无大模型对照采用本地规则与轻量模型，仅比较风险识别；检索结果采用当前启用配置单独展示。",
            "检索在 500 案例闭池内评测，187 条有强相关案例，13 条无强相关案例。",
            "本次为单次真实 API 工程回归，模型输出存在波动；900 字及以上仅 1 条样本。",
        ],
        "runs": [run],
        "inputs": {qid: {"text": inputs[qid]["text"], "reference": gold[qid]} for qid in inputs},
        "source_bindings": bindings,
        "context_bindings": context,
        "source_reports": sources,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(f"Published current version, 2 risk configurations, 200 rankings: {DEST}")


if __name__ == "__main__":
    main()
