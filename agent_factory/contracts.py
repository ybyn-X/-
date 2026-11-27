"""Agent 的公开契约和统一配置定义。"""

from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypeAlias

AgentConfig: TypeAlias = Mapping[str, Any]
ToolSpec: TypeAlias = str | dict[str, Any]


class AgentConfigurationError(ValueError):
    """表示 Agent 配置或组件解析失败。"""


@dataclass(frozen=True)
class AgentDefinition:
    """描述一次 Agent 组装所需的全部声明式组件。"""

    name: str
    model: dict[str, Any]
    prompt_path: Path
    tools: dict[str, ToolSpec]
    middleware: tuple[dict[str, Any], ...]
    skill_paths: tuple[Path, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentCapabilities:
    """不实例化模型与图即可读取的 Agent 静态能力边界摘要。"""

    name: str
    model_ref: str
    tools: tuple[str, ...]
    deferred_tools: tuple[str, ...]
    middleware: tuple[str, ...]
    skills: tuple[str, ...]
    metadata: dict[str, Any]


class Agent(Protocol):
    """对外隐藏底层框架实现的统一 Agent 调用契约。"""

    @property
    def name(self) -> str:
        """返回当前 Agent 的注册名称。"""
        ...

    def invoke(self, input: dict[str, Any], config: AgentConfig | None = None) -> dict[str, Any]:
        """同步执行一次 Agent 交互。"""
        ...

    async def ainvoke(self, input: dict[str, Any], config: AgentConfig | None = None) -> dict[str, Any]:
        """异步执行一次 Agent 交互，用于并发工作流调度。"""
        ...

    def stream(self, input: dict[str, Any], config: AgentConfig | None = None) -> Iterator[Any]:
        """同步流式输出 Agent 执行事件。"""
        ...

    def astream(self, input: dict[str, Any], config: AgentConfig | None = None) -> AsyncIterator[Any]:
        """异步流式输出 Agent 执行事件。"""
        ...
