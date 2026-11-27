"""Discovery only changes per-conversation visibility, never prompts or instances."""

from types import SimpleNamespace

from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import SystemMessage

from agent_factory import AgentFactory
from agent_factory.tool import load_tool
from middleware.deferred_middleware import DeferredMiddleware, merge_promoted
from tests.test_general_agent import GeneralModel


def test_filter_preserves_prompt_and_input_request():
    tool = load_tool("read_file", "tools.filesystem:read_file")
    middleware = DeferredMiddleware(["read_file"])
    prompt = SystemMessage(content="unchanged")
    request = ModelRequest(model=GeneralModel(), messages=[], tools=[tool], system_message=prompt,
                           state={"messages": []}, runtime=SimpleNamespace())
    filtered = middleware.wrap_model_call(request, lambda value: value)
    assert filtered.tools == []
    assert filtered.system_message is prompt
    assert request.tools == [tool]
    promoted = request.override(state={"messages": [], "promoted_tools": ["read_file"]})
    assert middleware.wrap_model_call(promoted, lambda value: value).tools == [tool]
    assert middleware.deferred_names == frozenset(["read_file"])


def test_same_agent_new_conversation_does_not_inherit_discovery():
    model = GeneralModel(steps=[("tool_research", {"query": "select:write_file"})])
    agent = AgentFactory.create_by_name("general", model_name=model)
    agent.invoke({"messages": [("user", "discover")]})
    assert "write_file" in model.bound_tools[-1]
    agent.invoke({"messages": [("user", "new conversation")]})
    assert "write_file" not in model.bound_tools[-1]


def test_discoveries_accumulate_and_empty_search_does_not_reset():
    assert merge_promoted(["a"], ["a", "b"]) == ["a", "b"]
    assert merge_promoted(["a"], []) == ["a"]


def test_deferred_does_not_implicitly_register_discovery(build_agent):
    model = GeneralModel()
    agent = build_agent(model, "prompt", {"read_file": "tools.filesystem:read_file"},
        [{"ref": "middleware.deferred_middleware:DeferredMiddleware", "deferred_names": ["write_file"]}])
    agent.invoke({"messages": [("user", "hello")]})
    assert model.bound_tools == [["read_file"]]


def test_explicit_discovery_only_sees_selected_tools(build_agent):
    model = GeneralModel(steps=[("tool_research", {"query": "select:read_file,write_file,write_todos,tool_research"})])
    agent = build_agent(
        model,
        tools={"tool_research": "tools.tool_research:tool_research", "read_file": "tools.filesystem:read_file"},
        middleware=[
            {"ref": "middleware.deferred_middleware:DeferredMiddleware", "deferred_names": ["read_file"]},
            {"ref": "langchain.agents.middleware:TodoListMiddleware"},
        ],
    )
    result = agent.invoke({"messages": [("user", "discover")]})
    assert result["promoted_tools"] == ["read_file"]
    search_message = next(m for m in result["messages"] if m.type == "tool")
    assert search_message.content == '["read_file"]'
