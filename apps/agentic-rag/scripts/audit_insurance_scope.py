"""Read-only replay of a frozen inventory; never writes a database or calls a model."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from offline.insurance_relevance import RULE_VERSION, evaluate_insurance_scope  # noqa: E402


def audit(inventory: Path, blob_root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("Use a fresh output directory to preserve earlier reports")
    root = blob_root.resolve()
    items = []
    for line in inventory.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        path = Path(row["blob_path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("BLOB_OUTSIDE_APPROVED_ROOT")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["raw_sha256"]:
            raise ValueError("RAW_HASH_MISMATCH")
        text = raw.decode("utf-8-sig")
        if hashlib.sha256(text.encode()).hexdigest() != row["digest"]:
            raise ValueError("TEXT_HASH_MISMATCH")
        scope = evaluate_insurance_scope(text)
        assert all(text[e["start"] : e["end"]] == e["quote"] for e in scope["evidence"])
        items.append(
            {
                "digest": row["digest"],
                "raw_sha256": row["raw_sha256"],
                "party": row["template_fields"].get("party", ""),
                "facts": row["template_fields"].get("finding", ""),
                "old_keyword_gate": row["contains_insurance_keyword"],
                "old_triage": row["scope_triage"],
                "scope_assessment": scope,
            }
        )
    restored = [r for r in items if not r["old_keyword_gate"]]
    prior_misses = [r for r in restored if r["old_triage"] == "insurance_institution_or_role"]
    summary = {
        "rule_version": RULE_VERSION,
        "documents": len(items),
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "old_keyword_eligible": sum(r["old_keyword_gate"] for r in items),
        "new_reviewable": len(items),
        "restored_to_review": len(restored),
        "categories": dict(Counter(r["scope_assessment"]["category"] for r in items)),
        "routes": dict(Counter(r["scope_assessment"]["route"] for r in items)),
        "prior_abbreviation_misses": len(prior_misses),
        "prior_misses_recognized": sum(
            r["scope_assessment"]["relevance"] == "insurance" for r in prior_misses
        ),
        "raw_hashes_checked": len(items),
        "model_requests": 0,
        "production_modified": False,
        "limitations": "规则分流对照，不是准确率；恢复待审不等于批准或入库。原件只读。",
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "comparison.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in items), encoding="utf-8"
    )
    labels = {
        "insurance_entity": "保险机构/任职线索",
        "insurance_business": "保险业务线索",
        "non_insurance_candidate": "非保险建议，待核",
        "uncertain": "范围待核",
    }
    escape = html.escape

    def rows(records):
        return "".join(
            "<tr><td>"
            + escape(r["party"])
            + "</td><td>"
            + escape(r["facts"])
            + "</td><td>"
            + ("含保险" if r["old_keyword_gate"] else "无保险，旧入口跳过")
            + "</td><td>"
            + labels[r["scope_assessment"]["category"]]
            + "</td></tr>"
            for r in records
        )

    header = (
        "<tr><th>主体</th><th>原文事实</th><th>旧关键词条件</th><th>新范围建议（均待审）</th></tr>"
    )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>保险相关性离线对照</title>'
        "<style>body{font:16px/1.7 system-ui;max-width:1200px;margin:40px auto;color:#213747}"
        "table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccd7df;padding:10px;text-align:left}"
        "th{background:#eef4f7}pre{white-space:pre-wrap;background:#eef4f7;padding:20px}</style>"
        "<h1>保险相关性统一分流：库存离线对照</h1><p>全部原件只读，未调用模型，未正式入库。"
        "非保险建议与未知材料继续保留待核；结果不代表分类准确率。</p><pre>"
        + escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>"
        "<h2>此前发现的简称潜在漏筛材料</h2><table>" + header + rows(prior_misses) + "</table>"
        "<details><summary>展开全量逐件结果</summary><table>"
        + header
        + rows(items)
        + "</table></details></html>"
    )
    (output / "保险相关性离线对照.html").write_text(page, encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--blob-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(audit(args.inventory, args.blob_root, args.output), ensure_ascii=False, indent=2)
    )
