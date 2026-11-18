# 自定义 Agent 执行框架

项目入口为 `agent_runtime`。Python 3.12，LangChain 1.3.15，LangGraph 1.2.11，SQLite checkpointer 3.1.3。安装和验证可运行 `uv sync --group dev` 和 `uv run python -m pytest -q`。

## 创建与启动

在异步 Python 入口中注册模型与工具。模型支持 LangChain 的工具绑定接口，工具使用已命名的 LangChain 工具。每个组件显式声明版本，变更实现时同步更新版本。

```python
from agent_runtime import AgentDefinition, Component, Runtime, create_agent

async def run(model, lookup):
    registry = {
        "models": {"main": Component("1", model)},
        "tools": {"lookup": Component("1", lookup)},
        "hooks": {},
    }
    runtime = await Runtime.open("sessions.sqlite", registry)
    try:
        definition = AgentDefinition("assistant", "1", "main", "Use lookup when needed.", ("lookup",))
        agent = create_agent(definition, runtime)
        handle = await agent.start("work-001", "Look up the current value.")
        return await handle.wait()
    finally:
        await runtime.close()
```

## 查询与恢复

同一事件循环中调用 `await runtime.inspect(session_id)` 查询当前尝试、最后确认步骤和最终结果。用 `runtime.history(session_id, after_seq=cursor, limit=100)` 分页，下一页使用返回的 next_cursor。事件只记录组件名、错误类别等诊断元数据，原始输入和模型消息保存在本地数据库中。该数据库应放在可信用户控制的目录。

重启后重新打开同一数据库，使用相同定义及组件版本调用 create_agent。先 inspect，再显式调用 `await agent.resume(session_id)`，返回句柄供 wait。检查点属于图步骤，多个处于同一步骤的工具并不分别获得独立检查点。对于外部效果未知的工具，不应自动重放。工单 04 将专门验收这类中断窗口。

状态扩展通过 state_fields 声明可选字符串字段；hook_refs 引用注册表中的 LangChain middleware。四个框架自定义 hook 的补丁与顺序契约属于后续工单 05。当前部署为单个活跃运行进程。
