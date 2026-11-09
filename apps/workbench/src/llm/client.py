"""全项目统一的 Pi Agent 适配器；Python 不直接请求模型供应商。"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from llm.types import JsonChatRequest, JsonChatResponse

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-flash"
PI_RUNTIME_DIR = Path(__file__).resolve().parents[2] / "agent_runtime"


class LLMConfigurationError(RuntimeError):
    """模型或 Agent 运行参数不合法。"""


class LLMProviderError(RuntimeError):
    """Agent、模型供应商或结构化提交失败。"""


class LLMClient(Protocol):
    """Offline、Query、Risk 共用的最小结构化审查接口。"""

    @property
    def model_name(self) -> str: ...

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse: ...


@dataclass(frozen=True, slots=True)
class LLMSettings:
    api_key: str = field(repr=False)
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    thinking: str = "disabled"
    connect_timeout_seconds: float = 10.0
    read_timeout_seconds: float = 90.0
    max_retries: int = 2
    max_concurrency: int = 6
    retry_base_seconds: float = 0.6
    agent_max_turns: int = 4
    node_binary: str = "node"


class _BridgeError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class _BridgeReply(BaseModel):
    id: str | None = None
    result: JsonChatResponse | None = None
    error: _BridgeError | None = None
    type: str | None = None
    runtime: str | None = None
    version: str | None = None


class PiAgentLLMClient:
    """复用本地 Node 工作进程，由 Pi 管理状态、工具循环和事件。"""

    runtime_name = "pi-agent-core"
    runtime_version = "1.2.0"

    def __init__(self, settings: LLMSettings) -> None:
        if settings.thinking not in {"enabled", "disabled"}:
            raise LLMConfigurationError("FINRISK_LLM_THINKING 只能是 enabled 或 disabled")
        if settings.max_retries < 0 or settings.max_concurrency < 1:
            raise LLMConfigurationError("LLM 重试次数和最大并发配置不合法")
        if not 1 <= settings.agent_max_turns <= 8:
            raise LLMConfigurationError("FINRISK_AGENT_MAX_TURNS 必须在 1 到 8 之间")
        if min(settings.connect_timeout_seconds, settings.read_timeout_seconds) <= 0:
            raise LLMConfigurationError("Agent 超时必须大于零")
        if settings.retry_base_seconds < 0:
            raise LLMConfigurationError("Agent 重试等待时间不得为负数")
        self._settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrency)
        self._start_lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._process: asyncio.subprocess.Process | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._ready: asyncio.Future[None] | None = None
        self._pending: dict[str, asyncio.Future[_BridgeReply]] = {}
        self._closed = False
        self._tool_handlers: dict[str, Callable[[str, dict[str, object]], Awaitable[object]]] = {}
        self._tool_tasks: set[asyncio.Task[None]] = set()

    @property
    def model_name(self) -> str:
        return self._settings.model

    @property
    def runtime_available(self) -> bool:
        return bool(
            shutil.which(self._settings.node_binary)
            and (PI_RUNTIME_DIR / "node_modules" / "@earendil-works" / "pi-agent-core").is_dir()
        )

    @property
    def worker_running(self) -> bool:
        return bool(
            self._process is not None
            and self._process.returncode is None
            and self._reader_task is not None
            and not self._reader_task.done()
        )

    async def start(self) -> None:
        """预热本地工作进程；不请求模型、不消耗模型额度。"""
        await self._ensure_worker()

    async def complete_json(self, request: JsonChatRequest) -> JsonChatResponse:
        return await self._complete_json(request)

    async def complete_json_with_tools(
        self,
        request: JsonChatRequest,
        handler: Callable[[str, dict[str, object]], Awaitable[object]],
    ) -> JsonChatResponse:
        return await self._complete_json(request, handler)

    async def _complete_json(
        self,
        request: JsonChatRequest,
        handler: Callable[[str, dict[str, object]], Awaitable[object]] | None = None,
    ) -> JsonChatResponse:
        started = perf_counter()
        async with self._semaphore:
            for attempt in range(1, self._settings.max_retries + 2):
                await self._ensure_worker()
                request_id = uuid4().hex
                future: asyncio.Future[_BridgeReply] = asyncio.get_running_loop().create_future()
                self._pending[request_id] = future
                if handler is not None:
                    self._tool_handlers[request_id] = handler
                try:
                    await self._send(
                        {
                            "type": "run",
                            "id": request_id,
                            "request": request.model_dump(),
                            "settings": {
                                "model": self._settings.model,
                                "base_url": self._settings.base_url,
                                "thinking": self._settings.thinking,
                                "read_timeout_seconds": self._settings.read_timeout_seconds,
                                "agent_max_turns": self._settings.agent_max_turns,
                            },
                        }
                    )
                    reply = await asyncio.wait_for(
                        future, timeout=self._settings.read_timeout_seconds + 5
                    )
                    if reply.result is not None:
                        return reply.result.model_copy(
                            update={
                                "attempts": attempt,
                                "latency_seconds": round(perf_counter() - started, 6),
                            }
                        )
                    if reply.error is not None:
                        if reply.error.retryable and attempt <= self._settings.max_retries:
                            await asyncio.sleep(
                                self._settings.retry_base_seconds * 2 ** (attempt - 1)
                            )
                            continue
                        raise LLMProviderError(reply.error.message)
                    raise LLMProviderError("Pi Agent 返回了无效的结构化响应")
                except (TimeoutError, asyncio.CancelledError):
                    await self._cancel(request_id)
                    task = asyncio.current_task()
                    if task is not None and task.cancelling():
                        raise
                    raise LLMProviderError("Pi Agent 审查超时") from None
                finally:
                    self._pending.pop(request_id, None)
                    self._tool_handlers.pop(request_id, None)
        raise LLMProviderError("Pi Agent 未返回结果")  # pragma: no cover

    async def _ensure_worker(self) -> None:
        if self._closed:
            raise LLMProviderError("Pi Agent 客户端已关闭")
        async with self._start_lock:
            if (
                self._process is not None
                and self._process.returncode is None
                and self._reader_task is not None
                and not self._reader_task.done()
            ):
                return
            if self._process is not None:
                await self._stop_worker()
            if not self.runtime_available:
                raise LLMProviderError(
                    "Pi Agent 依赖未就绪；需 Node.js 22.19+，并在 agent_runtime 执行 npm ci"
                )
            for startup_attempt in range(2):
                self._ready = asyncio.get_running_loop().create_future()
                worker_env = os.environ.copy()
                worker_env["FINRISK_PI_API_KEY"] = self._settings.api_key
                try:
                    self._process = await asyncio.create_subprocess_exec(
                        self._settings.node_binary,
                        str(PI_RUNTIME_DIR / "worker.mjs"),
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=PI_RUNTIME_DIR,
                        env=worker_env,
                        limit=8 * 1024 * 1024,
                        creationflags=0x08000000 if os.name == "nt" else 0,
                    )
                    self._reader_task = asyncio.create_task(self._read_replies(self._process))
                    self._stderr_task = asyncio.create_task(self._drain_stderr(self._process))
                    await asyncio.wait_for(
                        self._ready, self._settings.connect_timeout_seconds * (startup_attempt + 1)
                    )
                    return
                except (OSError, TimeoutError, LLMProviderError) as error:
                    reason = (
                        "启动超时"
                        if isinstance(error, TimeoutError)
                        else f"系统错误 {error.errno}"
                        if isinstance(error, OSError)
                        else "进程未就绪"
                    )
                    await self._stop_worker()
                    if isinstance(error, TimeoutError) and startup_attempt == 0:
                        continue
                    raise LLMProviderError(
                        f"无法启动本地 Pi Agent（{reason}），请检查 Node.js 和 npm 依赖"
                    ) from None

    async def _send(self, payload: dict[str, object]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.returncode is not None:
            raise LLMProviderError("Pi Agent 工作进程不可用")
        encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        async with self._write_lock:
            try:
                process.stdin.write(encoded)
                await process.stdin.drain()
            except (OSError, ConnectionError):
                raise LLMProviderError("Pi Agent 工作进程连接中断") from None

    async def _cancel(self, request_id: str) -> None:
        try:
            await self._send({"type": "cancel", "id": request_id})
        except LLMProviderError:
            pass

    async def _read_replies(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        try:
            while raw := await process.stdout.readline():
                decoded = json.loads(raw)
                if decoded.get("type") == "tool_call":
                    task = asyncio.create_task(self._dispatch_tool(decoded))
                    self._tool_tasks.add(task)
                    task.add_done_callback(self._tool_tasks.discard)
                    continue
                reply = _BridgeReply.model_validate(decoded)
                if reply.type == "ready":
                    if self._ready is not None and not self._ready.done():
                        self._ready.set_result(None)
                elif reply.id is not None and reply.id in self._pending:
                    pending = self._pending[reply.id]
                    if not pending.done():
                        pending.set_result(reply)
        except (ValueError, ValidationError, OSError):
            pass
        finally:
            error = LLMProviderError("Pi Agent 工作进程中断，当前审查未完成")
            if self._ready is not None and not self._ready.done():
                self._ready.set_exception(error)
            for pending in self._pending.values():
                if not pending.done():
                    pending.set_exception(error)

    async def _dispatch_tool(self, message: dict[str, object]) -> None:
        reply: dict[str, object] = {
            "type": "tool_result",
            "id": message.get("id"),
            "tool_call_id": message.get("tool_call_id"),
        }
        handler = self._tool_handlers.get(str(message.get("id")))
        try:
            if handler is None or not isinstance(message.get("args"), dict):
                raise ValueError("当前审查未注册该工具请求")
            reply["result"] = await asyncio.wait_for(
                handler(str(message.get("name")), cast(dict[str, object], message["args"])),
                timeout=30,
            )
        except Exception:  # noqa: BLE001 - tool failure must return to the bounded Pi loop
            reply["error"] = "只读证据工具不可用、参数不合法或预算已用尽；请保留待核状态。"
        try:
            await self._send(reply)
        except LLMProviderError:
            pass

    @staticmethod
    async def _drain_stderr(process: asyncio.subprocess.Process) -> None:
        assert process.stderr is not None
        while await process.stderr.read(4096):
            pass  # 不把可能含供应商请求内容的异常原样写入日志。

    async def _stop_worker(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.returncode is None:
            if process.stdin is not None:
                process.stdin.close()
                try:
                    await asyncio.wait_for(process.stdin.wait_closed(), 1)
                except (OSError, TimeoutError):
                    pass
            try:
                await asyncio.wait_for(process.wait(), 3)
            except TimeoutError:
                process.kill()
                await process.wait()
        for task in (self._reader_task, self._stderr_task):
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self._reader_task = None
        self._stderr_task = None

    async def aclose(self) -> None:
        self._closed = True
        tasks = tuple(self._tool_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._stop_worker()


def create_llm_client_from_env() -> PiAgentLLMClient | None:
    api_key = os.getenv("FINRISK_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        return None
    base_url = os.getenv("FINRISK_LLM_BASE_URL", DEFAULT_BASE_URL)
    model = os.getenv("FINRISK_LLM_MODEL", DEFAULT_MODEL)
    if urlparse(base_url).hostname == "api.deepseek.com" and model in {
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
    }:
        model = DEFAULT_MODEL
    return PiAgentLLMClient(
        LLMSettings(
            api_key=api_key,
            base_url=base_url,
            model=model,
            thinking=os.getenv("FINRISK_LLM_THINKING", "disabled"),
            connect_timeout_seconds=float(os.getenv("FINRISK_LLM_CONNECT_TIMEOUT", "10")),
            read_timeout_seconds=float(os.getenv("FINRISK_LLM_READ_TIMEOUT", "90")),
            max_retries=int(os.getenv("FINRISK_LLM_MAX_RETRIES", "2")),
            max_concurrency=int(os.getenv("FINRISK_LLM_MAX_CONCURRENCY", "6")),
            retry_base_seconds=float(os.getenv("FINRISK_LLM_RETRY_BASE_SECONDS", "0.6")),
            agent_max_turns=int(os.getenv("FINRISK_AGENT_MAX_TURNS", "4")),
            node_binary=os.getenv("FINRISK_NODE_BINARY", "node"),
        )
    )
