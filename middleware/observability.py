"""结构化可观测性中间件，为并发调度与故障诊断提供模型/工具遥测事件。"""

from collections.abc import Awaitable, Callable
import logging
from time import perf_counter
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from agent_factory._loading import load_symbol

EventSink = Callable[[dict[str, Any]], Any]


class ObservabilityMiddleware(AgentMiddleware):
    """记录模型与工具调用的耗时、状态和异常事件（同时支持同步与异步）。"""

    def __init__(
        self,
        *,
        logger_name: str = "middleware.observability",
        sink: str | EventSink | None = None,
    ):
        self._logger = logging.getLogger(logger_name)
        if isinstance(sink, str):
            loaded = load_symbol(sink)
            if not callable(loaded):
                raise TypeError(f"Observability sink {sink!r} must be callable")
            self._sink: EventSink | None = loaded
        elif sink is not None and not callable(sink):
            raise TypeError("Observability sink must be a callable or module:attribute reference")
        else:
            self._sink = sink

    def _emit(self, event: dict[str, Any]) -> None:
        level = logging.ERROR if event.get("status") == "error" else logging.INFO
        self._logger.log(
            level,
            "observability event=%s status=%s duration_ms=%s",
            event.get("event"),
            event.get("status", "ok"),
            event.get("duration_ms"),
            extra={"observability": event},
        )
        if self._sink is not None:
            self._sink(event)

    @staticmethod
    def _model_event_base(request: ModelRequest) -> dict[str, Any]:
        return {
            "message_count": len(request.messages),
            "tool_count": len(request.tools),
        }

    @staticmethod
    def _tool_event_base(request: ToolCallRequest) -> dict[str, Any]:
        return {
            "tool_name": str(request.tool_call.get("name") or ""),
            "tool_call_id": str(request.tool_call.get("id") or ""),
        }

    def _finalize_tool_event(
        self,
        base: dict[str, Any],
        started_at: float,
        result: ToolMessage | Command,
    ) -> None:
        duration_ms = round((perf_counter() - started_at) * 1000, 3)
        if isinstance(result, ToolMessage) and getattr(result, "status", None) == "error":
            self._emit({
                **base,
                "event": "tool.error",
                "status": "error",
                "duration_ms": duration_ms,
                "error": str(result.content),
            })
        else:
            self._emit({
                **base,
                "event": "tool.end",
                "status": "ok",
                "duration_ms": duration_ms,
            })

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        started_at = perf_counter()
        base = self._model_event_base(request)
        try:
            result = handler(request)
        except Exception as exc:
            self._emit({
                **base,
                "event": "model.error",
                "status": "error",
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        self._emit({
            **base,
            "event": "model.end",
            "status": "ok",
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
        })
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        started_at = perf_counter()
        base = self._model_event_base(request)
        try:
            result = await handler(request)
        except Exception as exc:
            self._emit({
                **base,
                "event": "model.error",
                "status": "error",
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        self._emit({
            **base,
            "event": "model.end",
            "status": "ok",
            "duration_ms": round((perf_counter() - started_at) * 1000, 3),
        })
        return result

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        started_at = perf_counter()
        base = self._tool_event_base(request)
        try:
            result = handler(request)
        except Exception as exc:
            self._emit({
                **base,
                "event": "tool.error",
                "status": "error",
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        self._finalize_tool_event(base, started_at, result)
        return result

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        started_at = perf_counter()
        base = self._tool_event_base(request)
        try:
            result = await handler(request)
        except Exception as exc:
            self._emit({
                **base,
                "event": "tool.error",
                "status": "error",
                "duration_ms": round((perf_counter() - started_at) * 1000, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
            raise
        self._finalize_tool_event(base, started_at, result)
        return result
