"""Filter deferred tools using discovery state belonging to each conversation."""

from collections.abc import Awaitable, Callable, Iterable
from typing import Annotated, Any, NotRequired

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


def merge_promoted(left: list[str], right: list[str]) -> list[str]:
    """Union discoveries, including updates from parallel tool calls."""
    return list(dict.fromkeys([*left, *right]))


class DiscoveryAgentState(AgentState):
    promoted_tools: NotRequired[Annotated[list[str], merge_promoted]]


class DeferredMiddleware(AgentMiddleware[DiscoveryAgentState]):
    """控制延迟工具的 schema 可见性和实际调用。"""

    state_schema = DiscoveryAgentState

    def __init__(self, deferred_names: Iterable[str]):
        names = tuple(deferred_names)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate deferred tool names")
        self.deferred_names = frozenset(names)

    def _hidden(self, state: Any) -> set[str]:
        return self.deferred_names - set((state or {}).get("promoted_tools", []))

    def _prepare_request(self, request: ModelRequest) -> ModelRequest:
        hidden = self._hidden(request.state)
        return request.override(
            tools=[item for item in request.tools if getattr(item, "name", None) not in hidden],
        )

    def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelCallResult:
        return handler(self._prepare_request(request))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        name = str(request.tool_call.get("name") or "")
        if name in self._hidden(request.state):
            return ToolMessage(content=f"Tool '{name}' has not been promoted; call tool_research first.", tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"), name=name, status="error")
        return handler(request)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], Awaitable[ModelResponse]]) -> ModelCallResult:
        return await handler(self._prepare_request(request))

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]) -> ToolMessage | Command:
        name = str(request.tool_call.get("name") or "")
        if name in self._hidden(request.state):
            return ToolMessage(content=f"Tool '{name}' has not been promoted; call tool_research first.", tool_call_id=str(request.tool_call.get("id") or "missing_tool_call_id"), name=name, status="error")
        return await handler(request)
