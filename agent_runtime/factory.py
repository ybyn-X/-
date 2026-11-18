from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import get_type_hints
from typing_extensions import TypedDict

from langchain.agents import AgentState, create_agent as compile_agent
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config


class InvalidDefinition(ValueError):
    pass


@dataclass(frozen=True)
class Component:
    version: str
    value: object


@dataclass(frozen=True)
class AgentDefinition:
    name: str
    revision: str
    model_ref: str
    system_prompt: str
    tool_refs: tuple[str, ...] = ()
    hook_refs: tuple[str, ...] = ()
    state_fields: tuple[str, ...] = ()


class Observe(AgentMiddleware):
    def __init__(self, runtime):
        self.runtime = runtime

    async def observe(self, step, target, call):
        sid = get_config()["configurable"]["thread_id"]
        attempt = self.runtime.begin(sid, step, target)
        try:
            result = await call()
        except Exception as exc:
            self.runtime.end(sid, attempt, type(exc).__name__)
            raise
        self.runtime.end(sid, attempt)
        return result

    async def awrap_model_call(self, request, handler):
        return await self.observe("model", "model", lambda: handler(request))

    async def awrap_tool_call(self, request, handler):
        return await self.observe("tool", request.tool_call["name"], lambda: handler(request))


@dataclass
class ExecutableAgent:
    graph: object
    runtime: object
    fingerprint: str

    async def start(self, session_id, input):
        return self.runtime.start(self, session_id, input)

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

    model = resolve("models", "model_ref", definition.model_ref)
    tools = [resolve("tools", f"tool_refs[{i}]", key) for i, key in enumerate(definition.tool_refs)]
    hooks = [resolve("hooks", f"hook_refs[{i}]", key) for i, key in enumerate(definition.hook_refs)]
    reserved = set(get_type_hints(AgentState)) | {"session_id", "thread_id", "checkpoint_id",
                                                  "attempt_id", "last_committed", "current_attempt"}
    fields = {}
    for i, key in enumerate(definition.state_fields):
        if not key.isidentifier() or key.startswith("_") or key in reserved or key in fields:
            raise InvalidDefinition(f"state_fields[{i}]: invalid field {key!r}")
        fields[key] = str
    extra = TypedDict("Extra", fields, total=False)
    class State(AgentState, extra):
        pass

    graph = compile_agent(model, tools, middleware=[Observe(runtime), *hooks],
                          state_schema=State, system_prompt=definition.system_prompt,
                          checkpointer=runtime.saver)
    payload = json.dumps([asdict(definition), versions], sort_keys=True).encode()
    return ExecutableAgent(graph, runtime, sha256(payload).hexdigest())
