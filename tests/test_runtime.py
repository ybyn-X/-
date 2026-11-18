import asyncio
from dataclasses import replace
import json
import os
import signal
import subprocess
import sys
import time
from uuid import uuid4

import pytest
from agent_runtime import DefinitionMismatch, InvalidDefinition, SessionExists, create_agent
from .scenario import build


def test_factory_and_execution(tmp_path):
    async def scenario():
        rt, agent, definition = await build(tmp_path)
        try:
            assert not (tmp_path / "models").exists()
            for field, value in [("tool_refs", ("bad",)), ("hook_refs", ("bad",)),
                                 ("model_ref", "bad"), ("state_fields", ("messages",))]:
                with pytest.raises(InvalidDefinition, match=field):
                    create_agent(replace(definition, **{field:value}), rt)
            (tmp_path / "value").write_text("tool-result")
            h = await agent.start("one", "original")
            with pytest.raises(SessionExists):
                await agent.start("one", "replacement")
            assert await h.wait() == "tool-result"
            info = await rt.inspect("one")
            assert (info["status"], info["input"], info["result"]) == ("completed", "original", "tool-result")
        finally:
            await rt.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("mode", ["committed", "queued", "waiting"])
def test_real_restart(tmp_path, mode):
    value = uuid4().hex
    (tmp_path / "value").write_text(value)
    proc = subprocess.Popen([sys.executable, "-m", "tests.scenario", str(tmp_path), mode])
    try:
        if mode == "queued":
            assert proc.wait(timeout=30) == 0
        else:
            deadline = time.monotonic() + 30
            while not (tmp_path / "ready").exists():
                assert proc.poll() is None and time.monotonic() < deadline
                time.sleep(.02)
            assert not (tmp_path / "B").exists()
    finally:
        if proc.poll() is None:
            pid_file = tmp_path / 'worker.pid'
            if pid_file.exists():
                os.kill(int(pid_file.read_text()), signal.SIGTERM if os.name == 'nt' else signal.SIGKILL)
            else:
                proc.kill()
        proc.wait(timeout=10)

    async def verify():
        rt, agent, definition = await build(tmp_path)
        try:
            info = await rt.inspect("original")
            assert info["status"] == "interrupted"
            assert not (tmp_path / "B").exists()
            if mode == "waiting":
                assert info["last_committed"]["step"] == "model_tool_request"
                assert info["current_attempt"]["target"] == "lookup"
                assert rt.history("original")["events"][-1]["kind"] == "interrupted"
                return
            if mode == "committed":
                assert info["last_committed"]["step"] == "tool_result"
            for other in [replace(definition, revision="2"), definition]:
                if other is definition:
                    component = rt.registry["models"]["model"]
                    rt.registry["models"]["model"] = replace(component, version="2")
                changed = create_agent(other, rt)
                with pytest.raises(DefinitionMismatch):
                    await changed.resume("original")
            handle = await agent.resume("original")
            assert await handle.wait() == value
            assert (tmp_path / "B").read_text() == value
            assert (tmp_path / "A").read_text().splitlines() == ["called"]
        finally:
            await rt.close()
    asyncio.run(verify())


@pytest.mark.parametrize("mode,error", [("model_error","ValueError"),("tool_error","RuntimeError")])
def test_error_history(tmp_path, mode, error):
    async def verify():
        rt, agent, _ = await build(tmp_path, mode)
        try:
            for sid in ["one","two"]:
                handle = await agent.start(sid, "test-secret")
                with pytest.raises(Exception):
                    await handle.wait()
            full = rt.history("one")["events"]
            first = rt.history("one", limit=2)
            assert first["events"] + rt.history("one", first["next_cursor"])["events"] == full
            assert all(e["sid"] == "one" for e in full)
            assert len({e["seq"] for e in full}) == len(full)
            assert "test-secret" not in json.dumps(full)
            assert any(e["kind"] == "attempt_failed" and e["error"] == error
                       and e["step"] == mode.split("_")[0] for e in full)
            assert (await rt.inspect("one"))["status"] == "failed"
        finally:
            await rt.close()
    asyncio.run(verify())


def test_live_progress(tmp_path):
    async def verify():
        rt, agent, _ = await build(tmp_path,"waiting")
        try:
            await agent.start("live","wait")
            async with asyncio.timeout(10):
                while not (tmp_path / "ready").exists():
                    await asyncio.sleep(.01)
            info = await rt.inspect("live")
            assert info["status"] == "running" and info["current_attempt"]["target"] == "lookup"
            assert info["last_committed"]["step"] == "model_tool_request" and info["last_progress_at"]
        finally:
            await rt.close()
    asyncio.run(verify())
