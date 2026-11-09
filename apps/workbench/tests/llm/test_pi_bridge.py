"""通过本地模拟供应商验证 Python → Node → Pi 的真实通信链。"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm import ChatMessage, JsonChatRequest, PiAgentLLMClient, create_llm_client_from_env
from llm.client import LLMConfigurationError, LLMSettings


@pytest.fixture
def provider_url():
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert body["model"] == "deepseek-flash"
            assert body["thinking"] == {"type": "disabled"}
            echo = body["messages"][-1]["content"]
            if isinstance(echo, list):
                echo = "".join(value["text"] for value in echo if value["type"] == "text")
            chunk = {
                "id": "test",
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
                                    "id": "result",
                                    "type": "function",
                                    "function": {
                                        "name": "submit_result",
                                        "arguments": json.dumps(
                                            {"result_json": json.dumps({"echo": echo})}
                                        ),
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            done = {
                **chunk,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls",
                    }
                ],
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

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)


def request(text: str) -> JsonChatRequest:
    return JsonChatRequest(messages=[ChatMessage(role="user", content=text)])


def test_shared_worker_keeps_concurrent_agent_contexts_isolated_and_closes(provider_url) -> None:
    async def execute() -> None:
        client = PiAgentLLMClient(LLMSettings(api_key="local-dummy", base_url=provider_url))
        try:
            await client.start()
            assert client.worker_running
            results = await asyncio.gather(
                client.complete_json(request("first-document")),
                client.complete_json(request("second-document")),
            )
            assert [value.data["echo"] for value in results] == [
                "first-document",
                "second-document",
            ]
            assert all(value.agent_trace is not None for value in results)
            assert all(value.agent_trace.runtime == "pi-agent-core" for value in results)
            assert client._process is not None
        finally:
            await client.aclose()
        assert client._process is None
        assert not client.worker_running
        assert not client._pending

    asyncio.run(execute())


def test_worker_can_restart_after_interruption(provider_url) -> None:
    async def execute() -> None:
        client = PiAgentLLMClient(LLMSettings(api_key="local-dummy", base_url=provider_url))
        try:
            await client.complete_json(request("before"))
            process = client._process
            assert process is not None
            previous_pid = process.pid
            process.kill()
            await process.wait()
            response = await client.complete_json(request("after"))
            assert response.data["echo"] == "after"
            assert client._process is not None and client._process.pid != previous_pid
        finally:
            await client.aclose()

    asyncio.run(execute())


def test_deepseek_legacy_name_migrates_only_for_official_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "local-dummy")
    monkeypatch.delenv("FINRISK_LLM_API_KEY", raising=False)
    monkeypatch.setenv("FINRISK_LLM_MODEL", "deepseek-v4-flash")
    monkeypatch.setenv("FINRISK_LLM_BASE_URL", "https://api.deepseek.com/v1")
    client = create_llm_client_from_env()
    assert isinstance(client, PiAgentLLMClient)
    assert client.model_name == "deepseek-flash"
    monkeypatch.setenv("FINRISK_LLM_BASE_URL", "http://127.0.0.1:1234/v1")
    custom = create_llm_client_from_env()
    assert custom is not None and custom.model_name == "deepseek-v4-flash"


def test_agent_loop_budget_rejects_unbounded_configuration() -> None:
    with pytest.raises(LLMConfigurationError, match="1 到 8"):
        PiAgentLLMClient(LLMSettings(api_key="local-dummy", agent_max_turns=100))
