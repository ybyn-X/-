"""Ordered extension points. Hooks return a patch or None and receive private snapshots."""
from copy import deepcopy
from dataclasses import dataclass, replace
from inspect import isawaitable
from typing import Callable, Literal

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, convert_to_messages
from langgraph.config import get_config


@dataclass(frozen=True)
class Hook:
    """声明一个回调及其触发阶段。回调读取上下文快照，可以返回补丁供同阶段后续回调和执行图使用。"""
    stage: Literal["before_model", "after_model", "before_tool", "after_tool"]
    callback: Callable

    def __post_init__(self):
        if self.stage not in {"before_model", "after_model", "before_tool", "after_tool"} or not callable(self.callback):
            raise ValueError("invalid hook stage or callback")


class Dispatch(AgentMiddleware):
    """LangChain 中间件。它包住模型和工具调用，按注册顺序执行对应阶段的 Hook，校验补丁并记录执行尝试。"""
    def __init__(self, runtime, registered):
        self.runtime = runtime
        self.registered = [(name,h) for name,h in registered if isinstance(h, Hook)]

    async def apply(self, stage, request, values, allowed, validate):
        sid = get_config()["configurable"]["thread_id"]
        for name, hook in self.registered:
            if hook.stage != stage:
                continue
            attempt = self.runtime.begin(sid, stage, name)
            try:
                # 每个 hook 读取独立快照；返回的补丁通过校验后，下一个同阶段 hook 才会看到修改。
                context = deepcopy(dict(values, state=request.state, session_id=sid))
                patch = hook.callback(context)
                if isawaitable(patch):
                    patch = await patch
                if patch is not None:
                    if not isinstance(patch, dict) or patch.keys() - allowed:
                        raise ValueError("invalid hook patch fields")
                    candidate = {**values, **deepcopy(patch)}
                    validate(values, candidate)
                    values = candidate
            except Exception as exc:
                self.runtime.end(sid, attempt, type(exc).__name__)
                raise
            self.runtime.end(sid, attempt)
        return values

    async def observed(self, step, target, call, call_id=None):
        sid = get_config()["configurable"]["thread_id"]
        attempt = self.runtime.begin(sid, step, target, call_id)
        try:
            result = await call()
        except Exception as exc:
            self.runtime.end(sid, attempt, type(exc).__name__)
            raise
        self.runtime.end(sid, attempt)
        return result

    async def awrap_model_call(self, request, handler):
        tools = {getattr(t, "name", None): t for t in request.tools}
        values = dict(messages=request.messages, tools=list(tools), system_prompt=request.system_message.content if request.system_message else None)
        def validate(old, new):
            if not set(new["tools"]) <= set(old["tools"]):
                raise ValueError("hook expanded tool access")
            new["messages"] = convert_to_messages(new["messages"])
            if new["system_prompt"] is not None and not isinstance(new["system_prompt"], str):
                raise ValueError("invalid system_prompt")

        values = await self.apply("before_model", request, values, set(values), validate)
        modified = request.override(messages=values["messages"], tools=[tools[n] for n in values["tools"]],
                                    system_message=SystemMessage(content=values["system_prompt"]) if values["system_prompt"] else None)
        # handler 执行本次模型调用。返回的工具请求还会经过 after_model，再交给图决定下一步。
        response = await self.observed("model", "model", lambda: handler(modified))
        messages = list(response.result)
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage):
                continue
            def check(old, new):
                # 工具结果按调用 ID 回填；修改请求内容时必须保留 ID 及顺序。
                if [c["id"] for c in old["tool_calls"]] != [c["id"] for c in new["tool_calls"]]:
                    raise ValueError("tool call identity changed")
                for c in new["tool_calls"]:
                    if c["name"] not in values["tools"]:
                        raise ValueError("unavailable tool")
                    tools[c["name"]].tool_call_schema.model_validate(c["args"])
                AIMessage(content=new["content"], tool_calls=new["tool_calls"])
            reply = dict(content=msg.content, tool_calls=msg.tool_calls)
            updated = await self.apply("after_model", request, reply, set(reply), check)
            if any(c["name"] not in values["tools"] for c in updated["tool_calls"]):
                raise ValueError("model selected unavailable tool")
            messages[i] = msg.model_copy(update=updated)
        return replace(response, result=messages)

    async def awrap_tool_call(self, request, handler):
        call = request.tool_call
        def check(old, new):
            if old["deny"] and not new["deny"]:
                raise ValueError("cannot undo tool denial")
            if not isinstance(new["deny"], bool) or not isinstance(new["args"], dict):
                raise ValueError("invalid tool patch")
            schema = request.tool.tool_call_schema
            if set(new["args"]) - set(schema.model_fields):
                raise ValueError("unknown argument")
            schema.model_validate(new["args"])

        values = await self.apply("before_tool", request, dict(name=call["name"], call_id=call["id"],
                                  args=call["args"], deny=False), {"args","deny"}, check)
        # 拒绝也要返回带原调用 ID 的 ToolMessage，让模型能收到拒绝结果。
        if values["deny"]:
            sid = get_config()["configurable"]["thread_id"]
            with self.runtime.db:
                self.runtime.event(sid, "tool_denied", step="tool", target=call["name"], call_id=call["id"])
            return ToolMessage(content="Tool call denied", tool_call_id=call["id"], status="error")
        modified = request.override(tool_call={**call, "args": values["args"]})
        result = await self.observed("tool", call["name"], lambda: handler(modified), call["id"])
        if not isinstance(result, ToolMessage):
            return result
        def validate_content(old, new):
            ToolMessage(content=new["content"], tool_call_id=call["id"])
        changed = await self.apply("after_tool", request, dict(content=result.content,
                                   name=call["name"], call_id=call["id"]), {"content"}, validate_content)
        return result.model_copy(update={"content": changed["content"]})
