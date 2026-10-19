"""Node跨层回归用真实Python任务/工具；仅评分后端为合成fixture，无网络。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests/agent_rag")]
from test_v2_snapshot import create_snapshot, fixture_assets  # noqa: E402
from test_v2_tools import Backend  # noqa: E402

from agent_rag.task_service import TaskService  # noqa: E402
from agent_rag.token_count import official_counter  # noqa: E402

directory = Path(sys.argv[1])
sources, units, dictionary, _, _ = fixture_assets(directory)
snapshot = create_snapshot(sources, units, dictionary, directory / "snapshot")
snapshot.manifest["rerank_mode"] = "replay"


class LocalBackend(Backend):
    def search(self, *args, **kwargs):
        result = super().search(*args, **kwargs)
        result["meta"]["external_retrieval_requests"] = 0
        return result


snapshot.backend = LocalBackend(snapshot)
service = TaskService(directory / "tasks", snapshot)
record, _ = service.admit(
    {
        "text": "购买这份保险就送购物卡。",
        "user_context": {},
        "engine": "controlled_agent",
        "mock_scenario": "gift-example",
    },
    None,
)
task_id = record["task_id"]
print(json.dumps(service.bundle(task_id), ensure_ascii=False), flush=True)
for line in sys.stdin:
    try:
        command = json.loads(line)
        if command["op"] == "call":
            result = service.tool(task_id).call(command["name"], command["args"], command["limit"])
        elif command["op"] == "preview":
            result = service.tool(task_id).preview_read(command["args"])
        elif command["op"] == "count":
            result = official_counter.count(command["payload"])
        elif command["op"] == "verify":
            result = service.verify(task_id, command["ids"])
        else:
            raise ValueError("UNEXPECTED_TEST_COMMAND")
        print(json.dumps(result, ensure_ascii=False), flush=True)
    except Exception as error:
        print(
            json.dumps({"test_worker_error": type(error).__name__ + ":" + str(error)}), flush=True
        )
