"""对外运行接口的 LangGraph 实现。"""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from .contracts import AgentConfig


class LangChainAgent:
    def __init__(self, graph: CompiledStateGraph, *, name: str = ""):
        """保存官方工厂编译出的 LangGraph，并对外屏蔽底层类型。"""
        self._graph = graph
        self._name = name or getattr(graph, "name", "") or "agent"

    @property
    def name(self) -> str:
        return self._name

    def _prepare_config(self, config: AgentConfig | None) -> dict[str, Any]:
        merged: dict[str, Any] = dict(config) if config is not None else {}
        metadata = dict(merged.get("metadata") or {})
        metadata.setdefault("agent_name", self._name)
        merged["metadata"] = metadata
        return merged

    def invoke(
        self,
        input: dict[str, Any],
        config: AgentConfig | None = None,
    ) -> dict[str, Any]:
        """同步执行一次 Agent 图，并返回包含完整消息历史的状态。"""
        return self._graph.invoke(input, config=self._prepare_config(config))

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: AgentConfig | None = None,
    ) -> dict[str, Any]:
        """异步执行一次 Agent 图，并返回包含完整消息历史的状态。"""
        return await self._graph.ainvoke(input, config=self._prepare_config(config))

    def stream(
        self,
        input: dict[str, Any],
        config: AgentConfig | None = None,
    ) -> Iterator[Any]:
        """同步流式返回 Agent 执行过程。"""
        return self._graph.stream(input, config=self._prepare_config(config))

    async def astream(
        self,
        input: dict[str, Any],
        config: AgentConfig | None = None,
    ) -> AsyncIterator[Any]:
        """异步流式返回 Agent 执行过程。"""
        async for chunk in self._graph.astream(input, config=self._prepare_config(config)):
            yield chunk
