import asyncio
from dataclasses import replace
import subprocess
import sys

import pytest
from agent_runtime import CheckpointError, Component, Hook, create_agent
from .scenario import build


def test_extension_contract(tmp_path):
    async def run():
        rt, _, definition = await build(tmp_path)
        (tmp_path / "value").write_text("hello")
        stages = ["before_model", "after_model", "before_tool", "before_tool", "after_tool"]
        seen = []
        def make(stage, i):
            def callback(ctx):
                seen.append(stage)
                if i == 2:
                    return {"args":{"suffix":"!"}}
                if i == 3:
                    assert ctx["args"]["suffix"] == "!"
                    return {"args":{"suffix":"!!"}}
                if i == 4:
                    return {"content":ctx["content"] + "?"}
            return Hook(stage, callback)
        try:
            rt.registry["hooks"] = {str(i):Component("1",make(stage,i)) for i,stage in enumerate(stages)}
            agent = create_agent(replace(definition,hook_refs=tuple(rt.registry["hooks"])),rt)
            assert await (await agent.start("test","input")).wait() == "hello!!?"
            assert seen == stages + stages[:2]
            assert (await rt.inspect("test"))["result"] == "hello!!?"
        finally:
            await rt.close()
    asyncio.run(run())


@pytest.mark.parametrize("stage,patch,kind", [
    ("before_model", {"tools":["other"]}, "failed"),
    ("before_tool", {"call_id":"forged"}, "failed"),
    ("before_tool", {"args":{"unknown":1}}, "failed"),
    ("before_tool", {"deny":True}, "completed"),
    ("after_tool", None, "failed"),
])
def test_rejected_and_failed_hooks(tmp_path,stage,patch,kind):
    async def run():
        rt, _, definition = await build(tmp_path)
        (tmp_path / "value").write_text("ok")
        def callback(ctx):
            if patch is None:
                raise RuntimeError("private")
            return patch
        rt.registry["hooks"]["guard"] = Component("1",Hook(stage,callback))
        try:
            agent = create_agent(replace(definition,hook_refs=("guard",)),rt)
            handle = await agent.start("s","input")
            if kind == "failed":
                with pytest.raises((ValueError,RuntimeError)):
                    await handle.wait()
            else:
                assert await handle.wait() == "Tool call denied"
            assert (await rt.inspect("s"))["status"] == kind
            assert (tmp_path / "A").exists() == (stage == "after_tool")
            events = rt.history("s")["events"]
            if stage == "after_tool":
                returned = next(e for e in events if e["step"]=="tool" and e["kind"]=="attempt_returned")
                failed = next(e for e in events if e["step"]==stage and e["kind"]=="attempt_failed")
                assert failed["target"]=="guard" and failed["seq"] > returned["seq"]
        finally:
            await rt.close()
    asyncio.run(run())


def test_locked_database_and_concurrency(tmp_path):
    async def run():
        rt, agent, definition = await build(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        async def pause(ctx):
            if ctx["session_id"] == "a":
                entered.set()
                await release.wait()
        rt.registry["hooks"]["pause"] = Component("1",Hook("before_tool",pause))
        agent = create_agent(replace(definition,hook_refs=("pause",)),rt)
        (tmp_path / "value").write_text("result")
        try:
            a = await agent.start("a","one")
            await asyncio.wait_for(entered.wait(),5)
            b = await agent.start("b","two")
            assert await asyncio.wait_for(b.wait(),5) == "result"
            assert (await rt.inspect("a"))["status"] == "running"
            for sid, initial in [("a","one"),("b","two")]:
                assert (await rt.inspect(sid))["input"] == initial
                assert all(e["sid"]==sid for e in rt.history(sid)["events"])
            code = """import asyncio,sys
from agent_runtime import Runtime
from filelock import Timeout
async def check():
    try:
        rt = await Runtime.open(sys.argv[1], {})
    except Timeout:
        return
    await rt.close()
    raise AssertionError("second owner")
asyncio.run(check())
"""
            subprocess.run([sys.executable,"-c",code,str(tmp_path/"sessions.db")],check=True,timeout=20)
            release.set()
            assert await a.wait() == "result"
        finally:
            await rt.close()
    asyncio.run(run())


def test_corrupt_checkpoint(tmp_path):
    async def run():
        rt, agent, _ = await build(tmp_path)
        try:
            (tmp_path / "value").write_text("done")
            await (await agent.start("s","run")).wait()
            with rt.db:
                rt.db.execute("UPDATE checkpoints SET checkpoint=x'ff' WHERE thread_id='s'")
            with pytest.raises(CheckpointError):
                await rt.inspect("s")
        finally:
            await rt.close()
    asyncio.run(run())
