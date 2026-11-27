"""验证工厂高扩展性契约：自省、异步并发、带参工具、全局默认配置与可观测性。"""

import asyncio
import logging

from langchain_core.tools import tool
import pytest

from agent_factory import AgentCapabilities, AgentConfigurationError, AgentFactory
from tests.test_agent_factory import ScriptedModel


def make_parameterized_fee_tool(unit: str = "元"):
    """按配置参数动态生成工具，验证 [tools] 表参数化构造。"""

    @tool
    def dynamic_fee(subtotal_yuan: int) -> str:
        """Calculate shipping fee with configurable currency unit."""
        fee = 0 if subtotal_yuan >= 99 else 8
        return f"{fee} {unit}"

    return dynamic_fee


OBSERVED_EVENTS: list[dict] = []


def record_event(event: dict) -> None:
    OBSERVED_EVENTS.append(event)


def test_factory_list_and_describe_do_not_instantiate_model():
    factory = AgentFactory()
    agents = factory.list_agents()
    assert set(agents) >= {"general", "shipping"}

    caps = factory.describe("general")
    assert isinstance(caps, AgentCapabilities)
    assert caps.name == "general"
    assert caps.model_ref == "langchain_openai:ChatOpenAI"
    assert "write_file" in caps.tools
    assert set(caps.deferred_tools) == {"write_file", "str_replace"}
    assert "middleware.tool_errors:ToolErrorMiddleware" in caps.middleware

    shipping_caps = factory.describe("shipping")
    assert shipping_caps.skills == ("shipping",)


def test_parameterized_tool_and_defaults_toml(tmp_path):
    (tmp_path / "defaults.toml").write_text(
        '[metadata]\nenvironment = "staging"\nowner = "platform"\n\n'
        '[[middleware]]\nref = "middleware.model_logging:ModelLoggingMiddleware"\n',
        encoding="utf-8",
    )
    agent_dir = tmp_path / "agents" / "custom"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("使用工具计算费用。", encoding="utf-8")
    (agent_dir / "config.toml").write_text(
        '[model]\nref = "tests.test_agent_factory:ScriptedModel"\n\n'
        '[metadata]\nowner = "checkout-team"\ndomain = "orders"\n\n'
        '[tools.shipping_fee]\nref = "tests.test_extensibility_and_observability:make_parameterized_fee_tool"\n'
        'unit = "RMB"\n',
        encoding="utf-8",
    )

    factory = AgentFactory(config_dir=tmp_path)
    caps = factory.describe("custom")
    assert caps.metadata == {"environment": "staging", "owner": "checkout-team", "domain": "orders"}
    assert caps.middleware == ("middleware.model_logging:ModelLoggingMiddleware",)

    agent = factory.create("custom", model_name=ScriptedModel())
    assert agent.name == "custom"
    caller_config = {"configurable": {"trace": "t-1"}}
    result = agent.invoke({"messages": [("user", "商品小计 120")]}, config=caller_config)
    assert result["messages"][-1].content == "运费是 0 RMB。"
    assert "metadata" not in caller_config


def test_invalid_defaults_toml_fails_with_configuration_error(tmp_path):
    (tmp_path / "defaults.toml").write_text('bad_key = "nope"\n', encoding="utf-8")
    agent_dir = tmp_path / "agents" / "demo"
    agent_dir.mkdir(parents=True)
    (agent_dir / "SOUL.md").write_text("hi", encoding="utf-8")
    (agent_dir / "config.toml").write_text('[model]\nref = "tests.test_agent_factory:ScriptedModel"\n', encoding="utf-8")

    with pytest.raises(AgentConfigurationError, match="defaults configuration failed"):
        AgentFactory(config_dir=tmp_path).describe("demo")


def test_async_concurrent_execution_across_agents(tmp_path):
    for name in ("alpha", "beta"):
        agent_dir = tmp_path / "agents" / name
        agent_dir.mkdir(parents=True)
        (agent_dir / "SOUL.md").write_text(f"{name} prompt", encoding="utf-8")
        (agent_dir / "config.toml").write_text(
            '[model]\nref = "tests.test_agent_factory:ScriptedModel"\n'
            '[tools]\nshipping_fee = "tools.shipping:shipping_fee"\n'
            '[[middleware]]\nref = "middleware.model_logging:ModelLoggingMiddleware"\n',
            encoding="utf-8",
        )

    factory = AgentFactory(config_dir=tmp_path)
    alpha = factory.create("alpha", model_name=ScriptedModel())
    beta = factory.create("beta", model_name=ScriptedModel())

    async def run_both():
        return await asyncio.gather(
            alpha.ainvoke({"messages": [("user", "a")]}),
            beta.ainvoke({"messages": [("user", "b")]}),
        )

    res_a, res_b = asyncio.run(run_both())
    assert res_a["messages"][-1].content == "运费是 0 元。"
    assert res_b["messages"][-1].content == "运费是 0 元。"


def test_observability_middleware_sync_and_async_records_ok_and_error(caplog, build_agent):
    OBSERVED_EVENTS.clear()
    with caplog.at_level(logging.INFO, logger="obs.test"):
        agent = build_agent(
            ScriptedModel(),
            "核算运费。",
            {"shipping_fee": "tools.shipping:shipping_fee"},
            [
                {
                    "ref": "middleware.observability:ObservabilityMiddleware",
                    "logger_name": "obs.test",
                    "sink": "tests.test_extensibility_and_observability:record_event",
                }
            ],
        )
        result = agent.invoke({"messages": [("user", "120 元")]})
        assert result["messages"][-1].content == "运费是 0 元。"

        async_result = asyncio.run(agent.ainvoke({"messages": [("user", "120 元")]}))
        assert async_result["messages"][-1].content == "运费是 0 元。"

    events = [item["event"] for item in OBSERVED_EVENTS]
    assert events == [
        "model.end", "tool.end", "model.end",
        "model.end", "tool.end", "model.end",
    ]
    assert all(item["status"] == "ok" and item["duration_ms"] >= 0 for item in OBSERVED_EVENTS)
    assert len(caplog.records) == 6
