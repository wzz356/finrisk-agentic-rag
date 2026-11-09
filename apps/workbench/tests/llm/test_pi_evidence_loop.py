"""A real local HTTP → Pi → Python tool → Pi loop; no external provider is contacted."""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llm import ChatMessage, JsonChatRequest, PiAgentLLMClient
from llm.client import LLMSettings


def test_real_pi_worker_roundtrip_reads_host_evidence_and_submits():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(body)
            has_tool_result = any(m.get("role") == "tool" for m in body["messages"])
            if has_tool_result:
                name = "submit_result"
                args = {
                    "result_json": json.dumps(
                        {
                            "items": [
                                {
                                    "sentence_id": "S1",
                                    "risk_score": 0.2,
                                    "risk_labels": [],
                                    "evidence": ["资格正在申请"],
                                    "reasoning": "资格状态需核对。",
                                    "decision_state": "insufficient_information",
                                    "candidate_labels": ["销售误导"],
                                    "missing_facts": ["当前资格证明"],
                                }
                            ]
                        }
                    )
                }
            else:
                name, args = "search_regulations", {"query": "销售资格", "top_k": 1}
            chunk = {
                "id": "local",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deepseek-flash",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-" + name,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            done = {
                **chunk,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }
            stream = (
                "data: "
                + json.dumps(chunk)
                + "\n\ndata: "
                + json.dumps(done)
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(stream)))
            self.end_headers()
            self.wfile.write(stream)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def execute():
        calls = []

        async def tool(name, arguments):
            calls.append((name, arguments))
            return {"id": "RULE-1", "text": "资格需核验", "current_facts_proved": False}

        client = PiAgentLLMClient(
            LLMSettings(api_key="local-dummy", base_url=f"http://127.0.0.1:{server.server_port}/v1")
        )
        try:
            request = JsonChatRequest(
                task="risk_review",
                messages=[ChatMessage(role="user", content="资格正在申请。")],
                available_tools=["search_regulations"],
                review_context={
                    "target_ids": ["S1"],
                    "allowed_labels": ["销售误导"],
                    "sentences": [{"sentence_id": "S1", "text": "资格正在申请。"}],
                },
            )
            result = await client.complete_json_with_tools(request, tool)
            assert calls == [("search_regulations", {"query": "销售资格", "top_k": 1})]
            assert result.agent_trace.turns == 2
            assert result.agent_trace.tool_calls == 2
            assert result.data["items"][0]["decision_state"] == "insufficient_information"
            assert len(received) == 2
            assert any(
                m.get("role") == "tool" and "RULE-1" in str(m["content"])
                for m in received[1]["messages"]
            )
        finally:
            await client.aclose()

    try:
        asyncio.run(execute())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
