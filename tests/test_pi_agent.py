"""验证仿 Pi 最小 Agent（仅 4 个核心工具：read, write, edit, bash）。"""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from agent_factory import AgentFactory


class PiScriptedModel(BaseChatModel):
    """按 write -> edit -> read -> bash 顺序驱动 4 个工具的离线脚本模型。"""

    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted-pi"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        last = messages[-1]
        if not isinstance(last, ToolMessage):
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write",
                        "args": {"path": "hello.txt", "content": "hello world\n"},
                        "id": "call-write",
                    }
                ],
            )
        elif last.name == "write":
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit",
                        "args": {"path": "hello.txt", "old": "world", "new": "pi"},
                        "id": "call-edit",
                    }
                ],
            )
        elif last.name == "edit":
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read",
                        "args": {"path": "hello.txt"},
                        "id": "call-read",
                    }
                ],
            )
        elif last.name == "read":
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "bash",
                        "args": {"command": "python -c \"print('ok')\""},
                        "id": "call-bash",
                    }
                ],
            )
        else:
            reply = AIMessage(content=f"done: {last.content}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_pi_agent_exposes_exact_four_tools(tmp_path):
    factory = AgentFactory()
    assert "pi" in factory.list_agents()

    caps = factory.describe("pi")
    assert caps.tools == ("read", "write", "edit", "bash")
    assert caps.deferred_tools == ()

    model = PiScriptedModel()
    agent = factory.create("pi", model_name=model)
    result = agent.invoke(
        {"messages": [("user", "创建并验证 hello.txt")]},
        config={"configurable": {"workspace_root": str(tmp_path)}},
    )

    assert set(model.bound_tools[0]) == {"read", "write", "edit", "bash"}
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello pi\n"
    assert "Exit code: 0" in result["messages"][-1].content
    assert "ok" in result["messages"][-1].content


def test_bash_tool_requires_workspace_and_rejects_empty_command(tmp_path):
    from tools.shell import bash

    try:
        bash("echo hi", config={})
        assert False, "Expected ValueError when workspace_root is missing"
    except ValueError as exc:
        assert "workspace_root" in str(exc)

    try:
        bash("   ", config={"configurable": {"workspace_root": str(tmp_path)}})
        assert False, "Expected ValueError for empty command"
    except ValueError as exc:
        assert "non-empty" in str(exc)
