from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import get_type_hints
from typing_extensions import TypedDict

from langchain.agents import AgentState, create_agent as compile_agent
from langchain.agents.middleware import AgentMiddleware


from .hooks import Hook, Dispatch


class InvalidDefinition(ValueError):
    """Agent 定义无法组装。例如工具名没有注册，或自定义状态字段与框架字段重名。"""


@dataclass(frozen=True)
class Component:
    """注册表中的一个组件条目。value 是模型、工具或中间件实例，version 参与计算 Agent 指纹。"""
    version: str
    value: object


@dataclass(frozen=True)
class AgentDefinition:
    """描述要构建的 Agent。工厂按名称解析模型、工具和 hook，并读取系统提示词与扩展状态字段。"""
    name: str
    revision: str
    model_ref: str
    system_prompt: str
    tool_refs: tuple[str, ...] = ()
    hook_refs: tuple[str, ...] = ()
    state_fields: tuple[str, ...] = ()


@dataclass
class ExecutableAgent:
    """组装完成的 Agent，持有执行图、运行时和定义指纹。调用方通过 start、send、resume 操作指定 session。"""
    graph: object
    runtime: object
    fingerprint: str

    async def start(self, session_id, input):
        return self.runtime.start(self, session_id, input)

    async def send(self, session_id, message):
        return await self.runtime.send(self, session_id, message)

    async def resume(self, session_id):
        return await self.runtime.resume(self, session_id)


def create_agent(definition, runtime):
    versions = {}
    def resolve(group, field, key):
        try:
            item = runtime.registry[group][key]
        except KeyError:
            raise InvalidDefinition(f"{field}: unknown {key!r}") from None
        versions[f"{group}/{key}"] = item.version
        return item.value

    # 按定义中的名称取出组件，并记下它们的声明版本，供续聊和恢复时核对。
    model = resolve("models", "model_ref", definition.model_ref)
    tools = [resolve("tools", f"tool_refs[{i}]", key) for i, key in enumerate(definition.tool_refs)]
    hooks = [resolve("hooks", f"hook_refs[{i}]", key) for i, key in enumerate(definition.hook_refs)]
    for i, hook in enumerate(hooks):
        if not isinstance(hook, (Hook, AgentMiddleware)):
            raise InvalidDefinition(f"hook_refs[{i}]: expected Hook or AgentMiddleware")
    reserved = set(get_type_hints(AgentState)) | {"session_id", "thread_id", "checkpoint_id",
                                                  "attempt_id", "last_committed", "current_attempt"}
    fields = {}
    for i, key in enumerate(definition.state_fields):
        if not key.isidentifier() or key.startswith("_") or key in reserved or key in fields:
            raise InvalidDefinition(f"state_fields[{i}]: invalid field {key!r}")
        fields[key] = str
    # 把自定义字段作为可选字段并入图状态，旧检查点可以缺少这些字段。
    extra = TypedDict("Extra", fields, total=False)
    class State(AgentState, extra):
        """本次执行图的状态结构，合并 LangChain AgentState 的消息历史与当前定义声明的可选字段。"""

    # LangChain 构建模型与工具之间的循环。Dispatch 执行框架的 Hook，原生中间件交给 LangChain。
    graph = compile_agent(model, tools, middleware=[Dispatch(runtime, list(zip(definition.hook_refs, hooks))),
                                      *(h for h in hooks if isinstance(h, AgentMiddleware))],
                          state_schema=State, system_prompt=definition.system_prompt,
                          checkpointer=runtime.saver)
    # 指纹覆盖定义、组件选择和声明版本。更换组件实现时需要同步更新版本。
    payload = json.dumps([asdict(definition), versions], sort_keys=True).encode()
    return ExecutableAgent(graph, runtime, sha256(payload).hexdigest())
