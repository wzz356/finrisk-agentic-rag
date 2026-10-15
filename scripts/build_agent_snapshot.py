"""一次性构建只读TXT快照和本地索引，不调用大模型。"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from agent_rag.knowledge_snapshot import KnowledgeSnapshot

parser = argparse.ArgumentParser()
parser.add_argument("--source-root", type=Path, required=True)
parser.add_argument("--dictionary", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
snapshot = KnowledgeSnapshot.build(args.source_root, args.dictionary, args.output)
print(json.dumps(snapshot.manifest, ensure_ascii=False))
