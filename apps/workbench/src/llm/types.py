"""与具体供应商无关的大模型请求和响应字段。"""

from typing import Literal

from pydantic import Field

from domain.agent_models import AgentExecutionTrace
from domain.common import ContractModel


class ChatMessage(ContractModel):
    """一条通用聊天消息。"""

    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class JsonChatRequest(ContractModel):
    """要求供应商返回 JSON 对象的通用聊天请求。"""

    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1)
    extra_body: dict[str, object] = Field(default_factory=dict)
    task: str = "json_review"
    review_context: dict[str, object] = Field(default_factory=dict)
    available_tools: list[str] = Field(default_factory=list)


class JsonChatResponse(ContractModel):
    """已经解析并验证为 JSON 对象的大模型响应。"""

    data: dict[str, object]
    model: str
    usage: dict[str, object] = Field(default_factory=dict)
    finish_reason: str | None = None
    latency_seconds: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    agent_trace: AgentExecutionTrace | None = None
