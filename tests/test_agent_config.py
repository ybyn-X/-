"""Agent directories are independent and have one explicit config format."""

import pytest

from agent_factory import AgentConfigurationError, AgentFactory
from tests.test_agent_factory import ScriptedModel


def make_agent(root, name, body='[model]\nref = "tests.test_agent_factory:ScriptedModel"', soul="prompt"):
    directory = root / "agents" / name
    directory.mkdir(parents=True)
    (directory / "config.toml").write_text(body, encoding="utf-8")
    (directory / "SOUL.md").write_text(soul, encoding="utf-8")
    return directory


def test_multiple_agent_directories_are_independent(tmp_path, monkeypatch):
    for name in ["first", "second"]:
        make_agent(tmp_path, name, soul=f"{name} prompt")
    monkeypatch.chdir(tmp_path.parent)
    for name in ["first", "second"]:
        model = ScriptedModel()
        agent = AgentFactory.create_by_name(name, model_name=model, config_dir=tmp_path)
        agent.invoke({"messages": [("user", "hello")]})
        assert model.calls[0][0].content == f"{name} prompt"
        assert model.bound_tools == []


@pytest.mark.parametrize("body,match", [
    ('prompt = "missing.md"', "Unknown config fields"),
    ("", "model requires a table"),
    ('model = "default"', "model requires a table"),
    ('[model]', "non-empty ref"),
    ('[model]\nref = "a:b"\n[tools]\nbad = 3', "tools must map names"),
    ('middleware = ["old"]\n[model]\nref = "a:b"', r"middleware\[0\]"),
    ('skills = [42]\n[model]\nref = "a:b"', "list of relative paths"),
])
def test_invalid_agent_config_fails_early(tmp_path, body, match):
    make_agent(tmp_path, "example", body)
    with pytest.raises(ValueError, match=match):
        AgentFactory.create_by_name("example", model_name=ScriptedModel(), config_dir=tmp_path)


def test_missing_soul_is_not_silently_ignored(tmp_path):
    directory = make_agent(tmp_path, "example")
    (directory / "SOUL.md").unlink()
    with pytest.raises(AgentConfigurationError, match=r"agent 'example' prompt configuration failed: .*SOUL\.md"):
        AgentFactory.create_by_name("example", model_name=ScriptedModel(), config_dir=tmp_path)


def test_agent_name_cannot_escape_directory(tmp_path):
    with pytest.raises(ValueError, match="Invalid agent name"):
        AgentFactory.create_by_name("../example", model_name=ScriptedModel(), config_dir=tmp_path)


def test_invalid_model_ref_is_rejected_locally(tmp_path):
    make_agent(tmp_path, "example", '[model]\nref = "tests.test_agent_factory:Missing"')
    with pytest.raises(AgentConfigurationError, match="agent 'example' model configuration failed: .*Missing"):
        AgentFactory.create_by_name("example", config_dir=tmp_path)


def test_explicit_tool_cannot_shadow_middleware_tool(build_agent):
    with pytest.raises(AgentConfigurationError, match="Duplicate tool name 'write_todos'"):
        build_agent(
            tools={"write_todos": "tools.shipping:shipping_fee"},
            middleware=[{"ref": "langchain.agents.middleware:TodoListMiddleware"}],
        )
