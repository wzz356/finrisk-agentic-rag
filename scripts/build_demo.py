"""Build a synthetic BM25 snapshot. No network, secrets, or model calls."""
import argparse
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_rag.contracts import canonical_json
from agent_rag.input_registry import text_hash
from agent_rag.knowledge_snapshot import KnowledgeSnapshot, make_evidence


def build(destination: Path) -> KnowledgeSnapshot:
    snapshot = KnowledgeSnapshot.build(
        ROOT / "examples/cases", ROOT / "contracts/agent_rag/risk_type_dictionary.csv", destination
    )
    with gzip.open(destination / "index.json.gz", "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    data["chunks"] = []
    for source in data["sources"]:
        source["text_version"] = "synthetic-demo/1.0"
        source["annotation_status"] = "synthetic_not_a_penalty"
        source["synthetic"] = True
        text = (destination / source["relative_path"]).read_text(encoding="utf-8")
        chunk = make_evidence(source, 0, len(text), text)
        chunk["text_quality"] = "synthetic_demo_not_real_penalty"
        data["chunks"].append(chunk)
    with gzip.open(destination / "index.json.gz", "wt", encoding="utf-8") as stream:
        stream.write(canonical_json(data))
    manifest = snapshot.manifest
    manifest.update(
        snapshot_id="synthetic-demo-" + text_hash(canonical_json(data))[:24],
        index_sha256=text_hash(canonical_json(data)),
        chunk_count=len(data["chunks"]),
        synthetic=True,
        warning="Synthetic engineering demo; not real penalty decisions or evaluation data."
    )
    (destination / "manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
    return KnowledgeSnapshot(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runtime/agent_rag/snapshot-v1")
    args = parser.parse_args()
    print(json.dumps(build(args.output).manifest, ensure_ascii=False, indent=2))
