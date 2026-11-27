"""Exercise registered components through the actual LangChain graph."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.errors import GraphInterrupt
from pydantic import Field

from agent_factory import AgentFactory
from agent_factory.tool import load_tool
from middleware.tool_errors import ToolErrorMiddleware
from tests.test_agent_factory import ScriptedModel
from tools.filesystem import read_file, write_file, str_replace


class GeneralModel(ScriptedModel):
    steps: list[tuple[str, dict]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        index = len(self.calls)
        self.calls.append(list(messages))
        if index < len(self.steps):
            name, args = self.steps[index]
            reply = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call-{index}"}])
        else:
            reply = AIMessage(content="完成并验证。")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_general_agent_plan_discover_write_edit_read(tmp_path):
    model = GeneralModel(steps=[
        ("write_todos", {"todos": [{"content": "Write and verify note", "status": "in_progress"}]}),
        ("tool_research", {"query": "select:write_file,str_replace"}),
        ("write_file", {"path": "note.txt", "content": "hello"}),
        ("str_replace", {"path": "note.txt", "old": "hello", "new": "你好"}),
        ("read_file", {"path": "note.txt"}),
        ("write_todos", {"todos": [{"content": "Write and verify note", "status": "completed"}]}),
    ])
    agent = AgentFactory.create_by_name("general", model_name=model)
    result = agent.invoke({"messages": [("user", "创建并验证笔记")]}, {"configurable": {"workspace_root": str(tmp_path)}})
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "你好"
    assert result["todos"][0]["status"] == "completed"
    assert set(result["promoted_tools"]) == {"write_file", "str_replace"}
    assert set(model.bound_tools[0]) == {"list_directory", "read_file", "tool_research", "write_todos"}
    assert {"write_file", "str_replace"} <= set(model.bound_tools[2])
    assert "通用任务助手" in str(model.calls[0][0].content)
    assert "shipping_fee" not in str(model.bound_tools)
    assert all(m.status == "success" for m in result["messages"] if isinstance(m, ToolMessage))


def test_error_recovery_and_unpromoted_call(tmp_path):
    model = GeneralModel(steps=[
        ("write_file", {"path": "blocked.txt", "content": "must not write"}),
        ("read_file", {"path": "missing.txt"}),
        ("list_directory", {"path": "."}),
    ])
    result = AgentFactory.create_by_name("general", model_name=model).invoke(
        {"messages": [("user", "检查文件")]}, {"configurable": {"workspace_root": str(tmp_path)}}
    )
    responses = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert [m.status for m in responses] == ["error", "error", "success"]
    assert not (tmp_path / "blocked.txt").exists()


def test_workspace_contract_and_write_safety(tmp_path):
    config = {"configurable": {"workspace_root": str(tmp_path)}}
    with pytest.raises(ValueError, match="workspace_root"):
        read_file("note.txt", {})
    for path in ["../outside.txt", str(tmp_path / "absolute.txt"), "note.txt:stream"]:
        with pytest.raises(ValueError):
            write_file(path, "bad", config)
    write_file("note.txt", "one one", config)
    with pytest.raises(FileExistsError):
        write_file("note.txt", "clobber", config)
    with pytest.raises(ValueError, match="exactly once"):
        str_replace("note.txt", "one", "two", config)
    assert read_file("note.txt", config) == "one one"
    with pytest.raises(ValueError, match="1 MB"):
        write_file("large.txt", "x" * 1_000_001, config)
    assert not (tmp_path / "large.txt").exists()


def test_workspace_is_injected_not_model_controlled():
    tool = load_tool("write_file", "tools.filesystem:write_file")
    properties = tool.tool_call_schema.model_json_schema()["properties"]
    assert set(properties) == {"path", "content", "overwrite"}


def test_unknown_agent_and_unknown_tool_fail(build_agent):
    with pytest.raises(ValueError, match="Unknown agent"):
        AgentFactory.create_by_name("missing", model_name=GeneralModel())
    with pytest.raises(ValueError, match="missing_tool"):
        build_agent(GeneralModel(), "prompt", {"missing_tool": "tools.shipping:missing_tool"})


def test_error_middleware_preserves_interrupts_and_programming_errors():
    middleware = ToolErrorMiddleware()
    request = SimpleNamespace(tool_call={"name": "read_file", "id": "1"})
    for error in [GraphInterrupt(), RuntimeError("bug")]:
        def handler(_):
            raise error
        with pytest.raises(type(error)):
            middleware.wrap_tool_call(request, handler)

    async def missing(_):
        raise FileNotFoundError("missing")
    assert asyncio.run(middleware.awrap_tool_call(request, missing)).status == "error"


def test_separate_agents_do_not_share_promotions(tmp_path):
    first = GeneralModel(steps=[("tool_research", {"query": "select:write_file"})])
    second = GeneralModel()
    for model in [first, second]:
        AgentFactory.create_by_name("general", model_name=model).invoke({"messages": [("user", "hello")]})
    assert "write_file" in first.bound_tools[-1]
    assert "write_file" not in second.bound_tools[0]
