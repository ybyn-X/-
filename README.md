# 自定义 Agent 执行框架

项目入口为 `agent_runtime`。Python 3.12，LangChain 1.3.15，LangGraph 1.2.11，SQLite checkpointer 3.1.3。安装和验证可运行 `uv sync --group dev` 和 `uv run python -m pytest -q`。

## 终端对话

在项目目录执行 `uv run --extra chat python main.py`，输入消息并按回车，输入 `/exit` 退出。同一次启动中的多轮消息共用一个 session。入口从本机 Pi 的 `models.json` 读取 `glai2/glm-5.2`，使用本机 SSH 别名 `hz` 的登录配置及已信任的主机密钥，默认连接 HZ 的内网地址 `192.168.17.100`。如果只有 Tailscale 路径可用，启动时加 `--via-hz-alias`。两种方式均由 HZ 连接模型端点。

启动后，把终端打印的 `Get-Content` 命令复制到第二个 PowerShell 窗口。每次运行的 `chat-logs/<session>.jsonl` 都记录完整对话正文，包括输入时的消息、模型调用前的上下文、流式回复片段和最终回复。文件逐条刷新，退出后保留。用 `--data-dir` 可以指定日志及 SQLite 数据库目录。

在终端输入“现在的 UTC 时间是多少？”可以试用内置的 `utc_now` 工具。终端会按实际执行顺序打印 `[middleware]` 行；上下文日志中的 `tool_input` 和 `tool_output` 通过 `call_id` 对应同一次工具调用。

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

重启后重新打开同一数据库，使用相同定义及组件版本调用 create_agent。先 inspect，再显式调用 `await agent.resume(session_id)`，返回句柄供 wait。检查点属于图步骤，多个处于同一步骤的工具并不分别获得独立检查点。对于外部效果未知的工具，不应自动重放。框架会在结果未确认时抛出 OutcomeUnknown，历史中的 call_id 与 attempt_id 可用于核对外部系统。CheckpointError 表示检查点不可读，此时应先处理存储故障。

## 扩展与并发

在注册表 hooks 中登记 Component("1", Hook(stage, callback))，将注册名按顺序填入定义的 hook_refs。callback 可以是同步或异步函数，读取含 session_id 和 state 的副本，返回字典补丁或 None。同阶段按注册顺序应用补丁。before_model 可修改 messages、system_prompt 或缩小 tools，after_model 可修改 content、tool_calls，调用标识受保护。before_tool 可修改 args 或返回 {"deny": True}，after_tool 可修改 content。异常终止当前步骤，history 记录阶段、注册名和错误类别。Python 回调在受信任的宿主进程中运行。

不同 session 可并行等待；同一 session 的并发恢复只接受一次，另一调用返回 SessionBusy。SQLite 数据库由单个活跃进程持有，竞争打开会得到 filelock.Timeout。测试通过真实子进程验证进程退出后的接管、已提交步骤的恢复和外部效果未确认时的拒绝重放。CI 以两个独立 pytest 运行验证可重复性。
