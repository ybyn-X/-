"""验证延迟工具、统一 Agent 配置和中间件注册链。"""

from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import pytest

from agent_factory import AgentFactory
from agent_factory.tool import load_tool
from tests.test_agent_factory import ScriptedModel


class FirstMiddleware(AgentMiddleware):
    """提供执行顺序测试用的第一个中间件。"""

    def before_model(self, state, runtime):
        """记录第一个中间件的模型前置执行。"""
        middleware_events.append("first")


class SecondMiddleware(AgentMiddleware):
    """提供执行顺序测试用的第二个中间件。"""

    def before_model(self, state, runtime):
        """记录第二个中间件的模型前置执行。"""
        middleware_events.append("second")


middleware_events = []


class DiscoveryScriptedModel(BaseChatModel):
    """按发现、正式调用、最终回答三步运行的确定性模型。"""

    calls: list[list] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)

    @property
    def _llm_type(self):
        """返回测试模型的固定类型标识。"""
        return "scripted-discovery"

    def bind_tools(self, tools, **kwargs):
        """记录每轮模型实际看到的工具 schema。"""
        self.bound_tools.append([tool.name for tool in tools])
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """先调用 tool_research，再调用业务工具，最后生成答案。"""
        self.calls.append(list(messages))
        last = messages[-1]
        if isinstance(last, ToolMessage) and last.name == "tool_research":
            reply = AIMessage(
                content="",
                tool_calls=[
                    {"name": "shipping_fee", "args": {"subtotal_yuan": 120}, "id": "fee"}
                ],
            )
        elif isinstance(last, ToolMessage) and last.name == "shipping_fee":
            reply = AIMessage(content=f"运费是 {last.content}。")
        else:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "tool_research",
                        "args": {"query": "select:shipping_fee"},
                        "id": "search",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def test_tool_is_resolved_during_agent_build(monkeypatch):
    """验证工具加载器 在构建阶段就准备完整工具对象。"""
    import agent_factory.tool as tool_module

    calls = []
    original = tool_module.load_symbol
    monkeypatch.setattr(tool_module, "load_symbol", lambda ref: calls.append(ref) or original(ref))
    tool = load_tool("shipping_fee", "tools.shipping:shipping_fee")
    assert calls == ["tools.shipping:shipping_fee"]
    assert tool.invoke({"subtotal_yuan": 120}) == "0 元"


def test_deferred_tools_hide_schema_until_tool_research_promotes_them(build_agent):
    """验证完整工具仍由 ToolNode 持有，但 schema 由 promotion 控制释放。"""
    model = DiscoveryScriptedModel()
    agent = build_agent(
        model,
        "请发现并使用工具。",
        {"shipping_fee": "tools.shipping:shipping_fee", "tool_research": "tools.tool_research:tool_research"},
        [{"ref": "middleware.deferred_middleware:DeferredMiddleware", "deferred_names": ["shipping_fee"]}],
    )
    result = agent.invoke({"messages": [("user", "商品小计 120 元，运费多少？")]})
    assert [set(names) for names in model.bound_tools] == [{"tool_research"}, {"shipping_fee", "tool_research"}, {"shipping_fee", "tool_research"}]
    assert result["promoted_tools"] == ["shipping_fee"]
    assert result["messages"][-1].content == "运费是 0 元。"


def test_standalone_agent_constructs_its_configured_model(build_agent):
    agent = build_agent(tools={"shipping_fee": "tools.shipping:shipping_fee"})
    result = agent.invoke({"messages": [("user", "运费多少？")]})
    assert result["messages"][-1].content == "运费是 0 元。"


def test_named_agent_injects_skill_without_business_file_reads():
    """验证按名称创建时 Skill 内容自动进入系统提示词并保留工具链。"""
    model = ScriptedModel()
    agent = AgentFactory.create_by_name("shipping", model_name=model)
    result = agent.invoke({"messages": [("user", "商品小计 120 元，运费多少？")]})
    assert result["messages"][-1].content == "运费是 0 元。"
    assert "## Enabled skills" in model.calls[0][0].content


@pytest.mark.parametrize("order", [("first", "second"), ("second", "first")])
def test_declared_middleware_order_reaches_every_model_call(build_agent, order):
    middleware_events.clear()
    agent = build_agent(
        tools={"shipping_fee": "tools.shipping:shipping_fee"},
        middleware=[{"ref": f"{__name__}:{name.title()}Middleware"} for name in order],
    )
    agent.invoke({"messages": [("user", "运费多少？")]})
    assert middleware_events == list(order) * 2
