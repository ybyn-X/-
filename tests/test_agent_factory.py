"""验证 目录 Agent 的最小组装和 invoke 链路。"""
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from tools.shipping import shipping_fee
class ScriptedModel(BaseChatModel):
    calls: list[list] = Field(default_factory=list)
    bound_tools: list[list[str]] = Field(default_factory=list)
    @property
    def _llm_type(self):
        """返回测试模型的固定类型标识。"""
        return "scripted-shipping"
    def bind_tools(self, tools, **kwargs):
        """记录绑定的工具名称并返回测试模型实例。"""
        self.bound_tools.append([tool.name for tool in tools]); return self
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        """按消息末尾生成工具调用或最终运费答复。"""
        self.calls.append(list(messages))
        if isinstance(messages[-1], ToolMessage):
            reply = AIMessage(content=f"运费是 {messages[-1].content}。")
        else:
            reply = AIMessage(content="", tool_calls=[{"name":"shipping_fee","args":{"subtotal_yuan":120},"id":"shipping-call"}])
        return ChatResult(generations=[ChatGeneration(message=reply)])
def test_create_assembles_and_invoke_runs_model_tool_model(build_agent):
    """验证工厂组装后完成模型、工具、模型的完整循环。"""
    model = ScriptedModel(); agent = build_agent(model, "用工具核算运费。", {"shipping_fee": "tools.shipping:shipping_fee"})
    result = agent.invoke({"messages": [("user", "商品小计 120 元，运费多少？")]}); messages = result["messages"]
    assert [message.type for message in messages] == ["human", "ai", "tool", "ai"]
    assert messages[2].content == "0 元"; assert messages[-1].content == "运费是 0 元。"
    assert all(call[0].content == "用工具核算运费。" for call in model.calls)
    assert model.bound_tools == [["shipping_fee"], ["shipping_fee"]]
def test_create_accepts_direct_static_prompt_and_no_tools(build_agent):
    """验证没有工具时仍可用静态提示词运行 Agent。"""
    model = ScriptedModel(); agent = build_agent(model, "直接给答复")
    assert agent.invoke({"messages": [("user", "你好")]})["messages"][-1].type == "ai"
def test_unknown_tool_is_rejected_during_create(build_agent):
    """验证未知工具名称在组装阶段报告配置错误。"""
    with pytest.raises(ValueError, match="missing"): build_agent(ScriptedModel(), "prompt", {"missing": "tools.shipping:missing"})
def test_empty_middleware_is_a_valid_default(build_agent):
    """验证空中间件列表是有效的默认输入。"""
    assert build_agent(ScriptedModel(), "prompt") is not None
@pytest.mark.parametrize(("subtotal", "expected"), [(0,"8 元"),(98,"8 元"),(99,"0 元"),(120,"0 元")])
def test_shipping_fee_boundary(subtotal, expected):
    """验证免运费门槛两侧的业务结果。"""
    assert shipping_fee(subtotal) == expected
def test_shipping_fee_rejects_negative_subtotals():
    """验证工具拒绝负商品小计。"""
    with pytest.raises(ValueError): shipping_fee(-1)