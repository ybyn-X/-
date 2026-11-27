"""Build isolated test agents using the production TOML format."""

import json

import pytest

from agent_factory import AgentFactory


@pytest.fixture
def build_agent(tmp_path):
    def build(model=None, prompt="prompt", tools=None, middleware=()):
        directory = tmp_path / "agents" / "example"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "SOUL.md").write_text(prompt, encoding="utf-8")
        lines = ["[model]", 'ref = "tests.test_agent_factory:ScriptedModel"', "[tools]"]
        for name, ref in (tools or {}).items():
            lines.append(f"{name} = {json.dumps(ref)}")
        for spec in middleware:
            lines.append("[[middleware]]")
            lines.extend(f"{key} = {json.dumps(value)}" for key, value in spec.items())
        (directory / "config.toml").write_text("\n".join(lines), encoding="utf-8")
        return AgentFactory.create_by_name("example", model_name=model, config_dir=tmp_path)

    return build
