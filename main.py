"""Interactive GLM chat. Run with uv run --extra chat python main.py."""
import argparse
import asyncio
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agent_runtime import AgentDefinition, Component, Hook, Runtime, create_agent


ROOT = Path(__file__).resolve().parent


@tool
def utc_now() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pi_profile():
    directory = Path(os.environ.get("PI_CODING_AGENT_DIR", Path.home() / ".pi" / "agent"))
    providers = json.loads((directory / "models.json").read_text(encoding="utf-8-sig"))["providers"]
    provider = providers["glai2"]
    model = next((m for m in provider["models"] if m["id"] == "glm-5.2"), None)
    if not model or model.get("api", provider.get("api")) != "openai-completions":
        raise ValueError("Pi 中未找到 glai2/glm-5.2 的 OpenAI 兼容配置")
    endpoint = provider["baseUrl"]
    if urlsplit(endpoint).scheme != "https":
        raise ValueError("模型端点必须使用 HTTPS")
    key = provider.get("apiKey", "")
    if not key or key.startswith("!"):
        raise ValueError("需要 Pi 的 API key 或环境变量引用")
    return model["id"], endpoint, os.environ.get(key, key)


@contextmanager
def hz_proxy(via_alias=False):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    # 本机只监听回环地址。HTTP 客户端通过这个 SOCKS 端口把模型请求交给 HZ 转发。
    proc = subprocess.Popen(
        ["ssh", *([] if via_alias else ["-o", "Hostname=192.168.17.100", "-o", "HostKeyAlias=100.90.82.13"]),
         "-N", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
         "-o", "ExitOnForwardFailure=yes", "-o", "ConnectTimeout=8",
         "-o", "ServerAliveInterval=20", "-o", "ServerAliveCountMax=3",
         "-D", f"127.0.0.1:{port}", "hz"],
        stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 12
        while proc.poll() is None and time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.25):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise ConnectionError("无法建立到 hz 的 SSH 转发")
        yield f"socks5h://127.0.0.1:{port}"
    finally:
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


class ContextLog(BaseCallbackHandler):
    """终端对话的观测器。模型回调逐片段写入 JSONL，注册到四个 hook 的方法记录调用上下文并打印执行阶段。"""
    run_inline = True
    raise_error = True

    def __init__(self, stream, sid):
        self.stream = stream
        self.sid = sid

    def record(self, event, **fields):
        row = {"time": datetime.now(timezone.utc).isoformat(), "session": self.sid,
               "event": event, **fields}
        self.stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        # 每个事件都刷新，另一个终端用 Get-Content -Wait 就能在本轮结束前看到它。
        self.stream.flush()

    def before(self, ctx):
        print(f"[middleware] before_model messages={len(ctx['messages'])}", flush=True)
        self.record("model_input", system=ctx["system_prompt"], tools=ctx["tools"],
                    messages=[{"role": m.type, "content": m.content} for m in ctx["messages"]])

    def after(self, ctx):
        selected = ", ".join(call["name"] for call in ctx["tool_calls"]) or "none"
        print(f"[middleware] after_model tool_calls={selected}", flush=True)
        self.record("model_output", content=ctx["content"], tool_calls=ctx["tool_calls"])

    def before_tool(self, ctx):
        print(f"[middleware] before_tool {ctx['name']}", flush=True)
        self.record("tool_input", name=ctx["name"], call_id=ctx["call_id"], args=ctx["args"])

    def after_tool(self, ctx):
        print(f"[middleware] after_tool {ctx['name']}", flush=True)
        self.record("tool_output", name=ctx["name"], call_id=ctx["call_id"], content=ctx["content"])

    def on_llm_new_token(self, token, **kwargs):
        if token:
            self.record("assistant_delta", text=token)


async def chat(data_dir, initial=None, via_alias=False):
    model_name, endpoint, key = pi_profile()
    sid = uuid4().hex
    log_dir = data_dir / "chat-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{sid}.jsonl"
    print(f"上下文文件：{path}", flush=True)
    print("另开 PowerShell 实时查看：Get-Content -Encoding UTF8 -Wait -LiteralPath '"
          + str(path).replace("'", "''") + "'", flush=True)
    with path.open("x", encoding="utf-8") as stream, hz_proxy(via_alias) as proxy:
        trace = ContextLog(stream, sid)
        with httpx.Client(proxy=proxy, trust_env=False) as sync_client:
            async with httpx.AsyncClient(proxy=proxy, trust_env=False) as client:
                model = ChatOpenAI(model=model_name, base_url=endpoint, api_key=key,
                                   http_client=sync_client, http_async_client=client,
                                   streaming=True, callbacks=[trace], max_retries=0, timeout=90)
                # 注册表保存可用组件；下面的 AgentDefinition 用名称选择本次对话启用的工具和 hook。
                registry = {"models": {"glm": Component(model_name, model)}, "tools": {"utc_now": Component("1", utc_now)},
                            "hooks": {"input": Component("1", Hook("before_model", trace.before)),
                                      "output": Component("1", Hook("after_model", trace.after)),
                                      "tool_input": Component("1", Hook("before_tool", trace.before_tool)),
                                      "tool_output": Component("1", Hook("after_tool", trace.after_tool))}}
                runtime = await Runtime.open(data_dir / "sessions.sqlite", registry)
                try:
                    agent = create_agent(AgentDefinition("terminal", "1", "glm",
                                         "你是一位助手，请用中文简短回答。查询当前时间时请调用 utc_now。", tool_refs=("utc_now",),
                                         hook_refs=("input", "output", "tool_input", "tool_output")), runtime)
                    print(f"glai2/{model_name} · /exit 退出", flush=True)
                    first = True
                    while True:
                        try:
                            message = initial if initial is not None else input("你> ")
                        except EOFError:
                            break
                        initial = None
                        if message.strip() == "/exit":
                            break
                        if not message.strip():
                            continue
                        trace.record("user_input", text=message)
                        # 首条消息创建 session，后续消息沿用同一个 sid，从上轮检查点继续对话。
                        handle = await (agent.start(sid, message) if first else agent.send(sid, message))
                        reply = await handle.wait()
                        print(f"GLM> {reply}", flush=True)
                        first = False
                except Exception as exc:
                    trace.record("failed", error=type(exc).__name__)
                    raise
                finally:
                    await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("message", nargs="*", help="可选的首条消息")
    parser.add_argument("--data-dir", type=Path, default=ROOT, help="SQLite 与上下文文件的保存目录")
    parser.add_argument("--via-hz-alias", action="store_true", help="通过本机 SSH 别名 hz 连接跳板")
    args = parser.parse_args()
    try:
        asyncio.run(chat(args.data_dir.resolve(), " ".join(args.message) or None, args.via_hz_alias))
    except ConnectionError as exc:
        parser.exit(1, f"{exc}\n")
    except KeyboardInterrupt:
        pass
