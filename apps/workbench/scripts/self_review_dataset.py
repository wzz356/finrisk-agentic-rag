"""Display and record Codex's direct source review; this script never calls an LLM."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from build_competition_dataset import DATA, ROOT, read_rows, save, write_rows

OUT = ROOT / "outputs/dataset_self_review_20261019"
LABELS = [
    "欺骗投保人",
    "销售误导",
    "夸大收益",
    "虚假宣传",
    "合同外利益",
    "赠送利益",
    "回扣返佣",
    "产品说明会违规",
    "培训材料违规",
    "回访违规",
    "电销违规",
    "不当比较",
    "贬低竞品",
    "绝对化表述",
    "承诺收益",
    "保证收益",
    "弱化风险提示",
    "隐瞒重要信息",
    "虚构收益率",
    "诱导投保",
    "避债避税",
    "虚列费用套取资金",
    "代理人管理不到位",
    "委托无资质机构销售",
    "未按规定使用费率条款",
    "客户信息不真实",
    "其他",
]
SCENES = [
    "ST1_poster_sms_social",
    "ST2_live_short_video",
    "ST3_product_intro_detail",
    "ST4_meeting_material",
    "ST5_training_telemarketing_callback",
    "ST6_penalty_backend_fact",
]


def parse_review_lines(text, labels):
    result = []
    for line in text.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("|")
        if fields[0].startswith("C"):
            assert len(fields) in (2, 3), line
            result.append(
                {
                    "id": fields[0],
                    "atoms": fields[1].split(",") if fields[1] else [],
                    "note": fields[2] if len(fields) == 3 else "已核对源事实与机制",
                }
            )
            continue
        assert len(fields) in (6, 7), line
        qid, state, tags, atoms, evidence, note = fields[:6]
        item = {
            "id": qid,
            "state": {"r": "risk", "n": "no_clear_risk", "u": "insufficient_information"}[state],
            "labels": labels[qid]["risk_labels"]
            if tags == "="
            else [LABELS[int(i)] for i in tags.split(",") if i],
            "atoms": [a for a in atoms.split(",") if a],
            "evidence": [e for e in evidence.split("~") if e],
            "note": note,
        }
        if len(fields) == 7 and fields[6]:
            item["scene"] = SCENES[int(fields[6]) - 1]
        result.append(item)
    return result


def snapshot() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "before_review.zip"
    if target.exists():
        return
    with ZipFile(target, "w", ZIP_DEFLATED) as archive:
        for path in DATA.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))
        for path in (ROOT / "data/risk/calibration").glob("competition500.*"):
            archive.write(path, path.relative_to(ROOT))
        for path in (ROOT / "reports/competition500").glob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "kind", choices=["queries", "cases", "contexts", "record", "context-record"]
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--batch", type=Path)
    args = parser.parse_args()
    snapshot()
    queries = read_rows(DATA / "model_inputs.jsonl")
    labels = {r["query_id"]: r for r in read_rows(DATA / "reference/labels.jsonl")}
    pool = read_rows(DATA / "cases.jsonl")
    if args.kind == "queries":
        for row in queries[args.start : args.start + args.count]:
            label = labels[row["query_id"]]
            print(
                f"\n{row['query_id']} | {label['scene']} | {label['decision_state']} | "
                f"{'/'.join(label['risk_labels'])}"
            )
            print(row["text"])
    elif args.kind == "cases":
        for index in range(args.start, min(args.start + args.count, len(pool))):
            row = pool[index]
            print(f"C{index:03} | {'/'.join(row['risk_tags'])} | {row['case_id']}")
            print("SOURCE: " + row["fact_text"])
    elif args.kind == "contexts":
        from settings import AppSettings

        flagged = [
            r for r in read_rows(OUT / "case_reviews.jsonl") if "source_context_needed" in r["note"]
        ]
        database = AppSettings.from_env().case_kb_root / "evidence_registry.sqlite3"
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for reviewed in flagged[args.start : args.start + args.count]:
                row = pool[int(reviewed["id"][1:])]
                source = connection.execute(
                    "SELECT text FROM sources WHERE file_id=?", (row["file_id"],)
                ).fetchone()
                print(f"\n{reviewed['id']} | {reviewed['atoms']} | {row['violation_behavior']}")
                print(source[0] if source else "SOURCE_NOT_FOUND")
    elif args.kind == "context-record":
        from settings import AppSettings

        target = OUT / "case_reviews.jsonl"
        reviewed = {r["id"]: r for r in read_rows(target)}
        database = AppSettings.from_env().case_kb_root / "evidence_registry.sqlite3"
        assert args.batch is not None
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for line in args.batch.read_text("utf-8").splitlines():
                cid, atoms, quotes, note, *tags = line.split("|")
                row = pool[int(cid[1:])]
                source, sha = connection.execute(
                    "SELECT text,sha256 FROM sources WHERE file_id=?", (row["file_id"],)
                ).fetchone()
                quoted = [q.replace("\\n", "\n") for q in quotes.split("~")]
                assert all(q in source for q in quoted), (cid, quoted)
                reviewed[cid].update(
                    atoms=atoms.split(","),
                    note=note,
                    source_text="\n".join(quoted),
                    context_source_sha256=sha,
                    context_quotes=quoted,
                )
                if tags:
                    reviewed[cid]["corrected_labels"] = [LABELS[int(i)] for i in tags[0].split(",")]
        write_rows(target, list(reviewed.values()))
        print(
            json.dumps(
                {"context_corrections": sum("context_quotes" in r for r in reviewed.values())}
            )
        )
    else:
        assert args.batch is not None
        raw = args.batch.read_text("utf-8")
        batch = json.loads(raw) if args.batch.suffix == ".json" else parse_review_lines(raw, labels)
        kind = "case" if batch[0]["id"].startswith("C") else "query"
        target = OUT / f"{kind}_reviews.jsonl"
        existing = {r["id"]: r for r in read_rows(target)} if target.exists() else {}
        by_id = {r["query_id"]: r for r in queries}
        for item in batch:
            source = (
                pool[int(item["id"][1:])]["fact_text"]
                if kind == "case"
                else by_id[item["id"]]["text"]
            )
            item.setdefault("reviewer", "Codex direct source review")
            item.setdefault("reviewed_at", datetime.now().astimezone().isoformat())
            item.setdefault("scene", labels[item["id"]]["scene"] if kind == "query" else None)
            if kind == "query":
                assert item["state"] in {"risk", "no_clear_risk", "insufficient_information"}
                assert not item["labels"] or item["state"] == "risk"
                for quote in item.get("evidence", []):
                    assert quote in source, (item["id"], quote)
            item["source_text"] = source
            existing[item["id"]] = item
        write_rows(target, list(existing.values()))
        save(
            OUT / "progress.json",
            {
                "kind": kind,
                "reviewed": len(existing),
                "total": 500 if kind == "case" else 600,
                "model_review_api_calls": 0,
            },
        )
        print(json.dumps({"kind": kind, "reviewed": len(existing), "model_review_api_calls": 0}))


if __name__ == "__main__":
    main()
