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
    pass


class DefinitionMismatch(ValueError):
    pass


class SessionBusy(RuntimeError):
    pass


class OutcomeUnknown(RuntimeError):
    pass


def config(sid):
    return {"configurable": {"thread_id": sid, "checkpoint_ns": ""}}


def now():
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Handle:
    session_id: str
    _task: asyncio.Task

    async def wait(self):
        return await asyncio.shield(self._task)


class Runtime:
    @classmethod
    async def open(cls, path, registry):
        self = cls()
        self.registry = registry
        self.active = {}
        self.closed = False
        self.gate = asyncio.Lock()
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
                CREATE TABLE IF NOT EXISTS events (
                  seq INTEGER PRIMARY KEY, sid TEXT NOT NULL, time TEXT NOT NULL,
                  kind TEXT NOT NULL, attempt_id TEXT, step TEXT, target TEXT, error TEXT);
                CREATE INDEX IF NOT EXISTS events_by_session ON events(sid,seq);
            """)
            self.conn = await aiosqlite.connect(path)
            self.saver = AsyncSqliteSaver(self.conn)
            await self.saver.setup()
            with self.db:
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

    def event(self, sid, kind, attempt=None, step=None, target=None, error=None):
        self.db.execute("INSERT INTO events(sid,time,kind,attempt_id,step,target,error) VALUES(?,?,?,?,?,?,?)",
                        (sid, now(), kind, attempt, step, target, error))

    def status(self, sid, value, error=None, result=None):
        self.db.execute("UPDATE sessions SET status=?,error=?,result=? WHERE sid=?",
                        (value, error, json.dumps(result) if result is not None else None, sid))
        self.event(sid, value, error=error)

    def begin(self, sid, step, target):
        attempt = uuid4().hex
        with self.db:
            self.event(sid, "attempt_started", attempt, step, target)
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

    async def resume(self, agent, sid):
        async with self.gate:
            if self.closed:
                raise RuntimeError("runtime closed")
            row = self.session(sid)
            if row["fingerprint"] != agent.fingerprint:
                raise DefinitionMismatch(sid)
            if sid in self.active or row["status"] != "interrupted":
                raise SessionBusy(sid)
            cp = await self.saver.aget_tuple(config(sid))
            payload = None if cp else {"messages": [("user", row["input"])]}
            if cp:
                snapshot = await agent.graph.aget_state(config(sid))
                messages = snapshot.values.get("messages", [])
                if not snapshot.next and messages and isinstance(messages[-1], AIMessage):
                    result = messages[-1].content
                    with self.db:
                        self.status(sid, "completed", result=result)
                    return Handle(sid, asyncio.create_task(asyncio.sleep(0, result=result)))
                if messages and isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
                    last_model = self.db.execute(
                        "SELECT max(seq) FROM events WHERE sid=? AND step='model' AND kind='attempt_returned'",
                        (sid,)).fetchone()[0] or 0
                    if self.db.execute("SELECT 1 FROM events WHERE sid=? AND seq>? AND step='tool' LIMIT 1",
                                       (sid, last_model)).fetchone():
                        raise OutcomeUnknown(sid)
            with self.db:
                self.event(sid, "resume_requested")
            return self.schedule(agent, sid, payload)

    async def inspect(self, sid):
        self.session(sid)
        cp = await self.saver.aget_tuple(config(sid))
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
