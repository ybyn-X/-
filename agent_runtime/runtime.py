import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

import aiosqlite
from filelock import FileLock
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class SessionExists(ValueError):
    """start 使用了数据库里已有的 session ID。已完成的会话可通过 send 追加消息。"""


class DefinitionMismatch(ValueError):
    """续聊或恢复时，当前 Agent 指纹与创建 session 时保存的指纹不同。"""


class SessionBusy(RuntimeError):
    """session 当前无法接受操作，例如执行尚未结束，或当前状态不允许 send 或 resume。"""


class OutcomeUnknown(RuntimeError):
    """恢复时发现工具调用已开始，但检查点无法确认结果。异常携带 session、调用和尝试标识，供核对外部效果。"""


class CheckpointError(RuntimeError):
    """执行检查点缺失、损坏或无法读取，运行时无法据此判断安全的恢复位置。"""


def config(sid):
    # LangGraph 以 thread_id 查找检查点，因此同一 session 的每一轮都使用同一个 sid。
    return {"configurable": {"thread_id": sid, "checkpoint_ns": ""}}


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Handle:
    """代表一次已调度的执行。wait 等待本次结果；调用方取消等待时，底层执行任务仍可继续。"""
    session_id: str
    _task: asyncio.Task

    async def wait(self):
        return await asyncio.shield(self._task)


class Runtime:
    """管理同一数据库上的 session、后台任务和检查点。负责调度执行、记录状态，以及在进程重启后恢复中断的工作。"""
    @classmethod
    async def open(cls, path, registry):
        self = cls()
        self.registry = registry
        self.active = {}
        self.closed = False
        self.gate = asyncio.Lock()
        # 持锁期间只有本进程可以接管数据库中的 session，避免两个运行时同时恢复同一任务。
        self.lock = FileLock(str(Path(path).resolve()) + ".lock", timeout=0)
        self.lock.acquire()
        try:
            self.db = sqlite3.connect(path)
            self.db.row_factory = sqlite3.Row
            self.db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS sessions (
                  sid TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, input TEXT NOT NULL,
                  status TEXT NOT NULL, result TEXT, error TEXT);
                CREATE TABLE IF NOT EXISTS followups (
                  sid TEXT PRIMARY KEY, base_checkpoint TEXT NOT NULL, message TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  seq INTEGER PRIMARY KEY, sid TEXT NOT NULL, time TEXT NOT NULL,
                  kind TEXT NOT NULL, attempt_id TEXT, step TEXT, target TEXT, error TEXT);
                CREATE INDEX IF NOT EXISTS events_by_session ON events(sid,seq);
            """)
            if "call_id" not in {r["name"] for r in self.db.execute("PRAGMA table_info(events)")}:
                self.db.execute("ALTER TABLE events ADD COLUMN call_id TEXT")
            self.conn = await aiosqlite.connect(path)
            self.saver = AsyncSqliteSaver(self.conn)
            await self.saver.setup()
            with self.db:
                # 上一个运行时遗留的排队或运行中任务，在新进程里标为中断，等待显式恢复。
                for row in self.db.execute("SELECT sid FROM sessions WHERE status IN ('running','queued')").fetchall():
                    self.status(row["sid"], "interrupted")
            return self
        except BaseException:
            if hasattr(self, "conn"):
                await self.conn.close()
            if hasattr(self, "db"):
                self.db.close()
            self.lock.release()
            raise

    def event(self, sid, kind, attempt=None, step=None, target=None, error=None, call_id=None):
        self.db.execute("INSERT INTO events(sid,time,kind,attempt_id,step,target,error,call_id) VALUES(?,?,?,?,?,?,?,?)",
                        (sid, now(), kind, attempt, step, target, error, call_id))

    def status(self, sid, value, error=None, result=None):
        self.db.execute("UPDATE sessions SET status=?,error=?,result=? WHERE sid=?",
                        (value, error, json.dumps(result) if result is not None else None, sid))
        self.event(sid, value, error=error)

    def begin(self, sid, step, target, call_id=None):
        attempt = uuid4().hex
        with self.db:
            self.event(sid, "attempt_started", attempt, step, target, call_id=call_id)
        return attempt

    def end(self, sid, attempt, error=None):
        source = self.db.execute("SELECT step,target FROM events WHERE sid=? AND attempt_id=?",
                                 (sid, attempt)).fetchone()
        with self.db:
            self.event(sid, "attempt_failed" if error else "attempt_returned", attempt,
                       source["step"], source["target"], error)

    def session(self, sid):
        row = self.db.execute("SELECT * FROM sessions WHERE sid=?", (sid,)).fetchone()
        if row is None:
            raise KeyError(sid)
        return row

    def start(self, agent, sid, input):
        if self.closed:
            raise RuntimeError("runtime closed")
        if not isinstance(input, str) or not isinstance(sid, str) or not sid.strip():
            raise ValueError("expected session ID and text input")
        try:
            with self.db:
                self.db.execute("INSERT INTO sessions(sid,fingerprint,input,status) VALUES(?,?,?,'queued')",
                                (sid, agent.fingerprint, input))
                self.event(sid, "queued")
        except sqlite3.IntegrityError:
            raise SessionExists(sid) from None
        return self.schedule(agent, sid, {"messages": [("user", input)]})

    def schedule(self, agent, sid, payload):
        task = asyncio.create_task(self.run(agent, sid, payload))
        self.active[sid] = task
        return Handle(sid, task)

    async def run(self, agent, sid, payload):
        with self.db:
            self.status(sid, "running")
        try:
            output = await agent.graph.ainvoke(payload, config(sid), durability="sync")
            result = output["messages"][-1].content
            with self.db:
                self.status(sid, "completed", result=result)
            return result
        except asyncio.CancelledError:
            with self.db:
                self.status(sid, "interrupted")
            raise
        except Exception as exc:
            with self.db:
                self.status(sid, "failed", type(exc).__name__)
            raise
        finally:
            self.active.pop(sid, None)

    async def checkpoint(self, sid):
        try:
            cp = await self.saver.aget_tuple(config(sid))
            if cp is not None:
                state = cp.checkpoint
                if not isinstance(state, dict) or not isinstance(state.get("id"), str) or not isinstance(state.get("channel_values"), dict):
                    raise ValueError("invalid persisted checkpoint structure")
        except Exception as exc:
            raise CheckpointError(sid) from exc
        if cp is None and self.db.execute(
            "SELECT 1 FROM events WHERE sid=? AND kind='attempt_started' LIMIT 1", (sid,)
        ).fetchone():
            raise CheckpointError(f"{sid}: missing checkpoint after execution began")
        return cp

    async def send(self, agent, sid, message):
        if not isinstance(message, str) or not message.strip():
            raise ValueError("expected a non-empty message")
        async with self.gate:
            row = self.session(sid)
            if row["fingerprint"] != agent.fingerprint:
                raise DefinitionMismatch(sid)
            if self.closed or sid in self.active or row["status"] != "completed":
                raise SessionBusy(sid)
            cp = await self.checkpoint(sid)
            with self.db:
                # 先在同一事务保存新消息和上轮检查点，再调度执行。恢复时据此判断是否需要重新提交输入。
                self.db.execute("INSERT OR REPLACE INTO followups VALUES(?,?,?)",
                                (sid, cp.checkpoint["id"], message))
                self.status(sid, "queued")
            return self.schedule(agent, sid, {"messages": [("user", message)]})

    async def resume(self, agent, sid):
        async with self.gate:
            if self.closed:
                raise RuntimeError("runtime closed")
            row = self.session(sid)
            if row["fingerprint"] != agent.fingerprint:
                raise DefinitionMismatch(sid)
            if sid in self.active or row["status"] != "interrupted":
                raise SessionBusy(sid)
            cp = await self.checkpoint(sid)
            payload = None if cp else {"messages": [("user", row["input"])]}
            followup = self.db.execute("SELECT * FROM followups WHERE sid=?", (sid,)).fetchone()
            # 最新检查点仍是上轮终点，说明新消息尚未写入图检查点，需要再次提交。
            if followup and cp and followup["base_checkpoint"] == cp.checkpoint["id"]:
                payload = {"messages": [("user", followup["message"])]}
            elif cp:
                try:
                    snapshot = await agent.graph.aget_state(config(sid))
                except Exception as exc:
                    raise CheckpointError(sid) from exc
                messages = snapshot.values.get("messages", [])
                # 图可能已经写入最终回答，进程却来不及更新 session 状态。直接采用已保存的结果。
                if not snapshot.next and messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls:
                    result = messages[-1].content
                    with self.db:
                        self.status(sid, "completed", result=result)
                    return Handle(sid, asyncio.create_task(asyncio.sleep(0, result=result)))
                if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                    last_model = self.db.execute(
                        "SELECT max(seq) FROM events WHERE sid=? AND step='model' AND kind='attempt_returned'",
                        (sid,)).fetchone()[0] or 0
                    intent = self.db.execute("SELECT attempt_id, call_id FROM events WHERE sid=? AND seq>? AND step='tool' AND kind='attempt_started' LIMIT 1",
                                       (sid, last_model)).fetchone()
                    # 工具可能已产生外部副作用，但检查点还没有结果；自动重放可能重复执行。
                    if intent:
                        raise OutcomeUnknown(sid, intent["call_id"], intent["attempt_id"])
            with self.db:
                self.event(sid, "resume_requested")
            return self.schedule(agent, sid, payload)

    async def inspect(self, sid):
        self.session(sid)
        cp = await self.checkpoint(sid)
        committed = None
        if cp:
            messages = cp.checkpoint["channel_values"].get("messages", [])
            message = messages[-1] if messages else None
            if isinstance(message, AIMessage) and message.tool_calls:
                step = "model_tool_request"
            elif isinstance(message, ToolMessage):
                step = "tool_result"
            else:
                step = getattr(message, "type", "input")
            committed = {"checkpoint_id": cp.checkpoint["id"], "step": step}
        row = self.session(sid)
        attempts = self.db.execute("""
            SELECT a.* FROM events a WHERE a.sid=? AND a.kind='attempt_started'
            AND NOT EXISTS (SELECT 1 FROM events e WHERE e.sid=a.sid AND e.attempt_id=a.attempt_id
                AND e.kind IN ('attempt_returned','attempt_failed'))
            AND a.seq > COALESCE((SELECT MAX(seq) FROM events WHERE sid=a.sid AND kind='resume_requested'),0)
            ORDER BY a.seq DESC
        """, (sid,)).fetchall()
        latest = self.db.execute("SELECT time FROM events WHERE sid=? ORDER BY seq DESC LIMIT 1",(sid,)).fetchone()
        return {"session_id": sid, "status": row["status"], "input": row["input"],
                "result": json.loads(row["result"]) if row["result"] else None, "error": row["error"],
                "current_attempt": dict(attempts[0]) if attempts else None,
                "last_committed": committed, "last_progress_at": latest[0]}

    def history(self, sid, after_seq=0, limit=100):
        self.session(sid)
        if not 1 <= limit <= 1000 or after_seq < 0:
            raise ValueError("invalid event page")
        rows = self.db.execute("SELECT * FROM events WHERE sid=? AND seq>? ORDER BY seq LIMIT ?",
                               (sid, after_seq, limit))
        events = [dict(row) for row in rows]
        return {"events": events, "next_cursor": events[-1]["seq"] if events else after_seq}

    async def close(self):
        self.closed = True
        tasks = list(self.active.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.conn.close()
        self.db.close()
        self.lock.release()
