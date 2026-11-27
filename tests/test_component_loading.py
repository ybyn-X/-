import logging

import pytest

from agent_factory.tool import load_tool
from tests.test_agent_factory import ScriptedModel


def test_tool_alias_is_bound_to_callable():
    tool = load_tool("quote", "tools.shipping:shipping_fee")
    assert tool.name == "quote"
    assert tool.invoke({"subtotal_yuan": 120}) == "0 元"


def test_configured_logging_hook_runs_for_each_model_call(caplog, build_agent):
    with caplog.at_level(logging.INFO, logger="middleware.model_logging"):
        agent = build_agent(
            ScriptedModel(), "外部系统指令",
            {"shipping_fee": "tools.shipping:shipping_fee"},
            [{"ref": "middleware.model_logging:ModelLoggingMiddleware"}],
        )
        result = agent.invoke({"messages": [("user", "商品小计 120 元，运费多少？")]})
    assert result["messages"][-1].content == "运费是 0 元。"
    assert len(caplog.records) == 2


def test_bad_tool_ref_identifies_agent_and_reference(build_agent):
    with pytest.raises(ValueError, match="agent 'example' tool configuration failed: .*tools.shipping:missing"):
        build_agent(tools={"quote": "tools.shipping:missing"})
