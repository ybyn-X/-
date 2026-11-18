import asyncio
import os
import sys
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from agent_runtime import AgentDefinition, Component, Runtime, create_agent


class Model(BaseChatModel):
    directory: str
    mode: str

    @property
    def _llm_type(self):
        return "controlled"

    def bind_tools(self, tools, **kwargs):
        assert [t.name for t in tools] == ["lookup"]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        path = Path(self.directory)
        assert messages[0].content == "configured prompt"
        with (path / "models").open("a") as f:
            f.write("call\n")
        if self.mode == "model_error":
            raise ValueError("test-secret")
        if isinstance(messages[-1], ToolMessage):
            (path / "B").write_text(messages[-1].content)
            msg = AIMessage(content=messages[-1].content)
        else:
            msg = AIMessage(content="", tool_calls=[dict(name="lookup", args={}, id="call-1")])
        return ChatResult(generations=[ChatGeneration(message=msg)])


async def build(path, mode="normal", version="1"):
    @tool
    async def lookup() -> str:
        """Return a unique durable receipt."""
        if mode == "tool_error":
            raise RuntimeError("test-secret")
        with (path / "A").open("a") as f:
            f.write("called\n")
        if mode == "waiting":
            (path / "ready").touch()
            await asyncio.Event().wait()
        return (path / "value").read_text()

    registry = {"models": {"model": Component(version, Model(directory=str(path), mode=mode))},
                "tools": {"lookup": Component("1", lookup)}, "hooks": {}}
    rt = await Runtime.open(path / "sessions.db", registry)
    definition = AgentDefinition("agent", "1", "model", "configured prompt", ("lookup",))
    try:
        return rt, create_agent(definition, rt), definition
    except BaseException:
        await rt.close()
        raise


async def worker(path, mode):
    (path / 'worker.pid').write_text(str(os.getpid()))
    rt, agent, _ = await build(path, mode)
    if mode == "committed":
        put = rt.saver.aput
        async def persist(config, checkpoint, metadata, versions):
            result = await put(config, checkpoint, metadata, versions)
            msgs = checkpoint["channel_values"].get("messages", [])
            if msgs and isinstance(msgs[-1], ToolMessage):
                (path / "ready").touch()
                await asyncio.Event().wait()
            return result
        rt.saver.aput = persist
    handle = await agent.start("original", "initial")
    if mode == "queued":
        os._exit(0)
    await handle.wait()


if __name__ == "__main__":
    asyncio.run(worker(Path(sys.argv[1]), sys.argv[2]))
