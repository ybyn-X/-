"""Expose recoverable tool failures without swallowing graph control flow."""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp


class ToolErrorMiddleware(AgentMiddleware):
    """Return operational errors to the model; never retry writes automatically."""

    @staticmethod
    def _error(request, exc):
        return ToolMessage(
            content=f"{type(exc).__name__}: {exc}. Correct the input or choose another approach.",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except GraphBubbleUp:
            raise
        except (OSError, ValueError, UnicodeError) as exc:
            return self._error(request, exc)

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except GraphBubbleUp:
            raise
        except (OSError, ValueError, UnicodeError) as exc:
            return self._error(request, exc)
