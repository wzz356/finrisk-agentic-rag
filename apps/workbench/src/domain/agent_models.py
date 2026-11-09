"""不含原文和密钥的 Agent 执行元数据。"""

from pydantic import Field

from domain.common import ContractModel


class AgentExecutionTrace(ContractModel):
    runtime: str
    version: str
    task: str
    turns: int = Field(ge=1)
    tool_calls: int = Field(ge=0)
    tool_failures: int = Field(default=0, ge=0)
    events: list[dict[str, str]] = Field(default_factory=list)
    validation_failures: list[str] = Field(default_factory=list)
