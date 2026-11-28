"""Ledger access off the event loop, and exact caps under concurrency (#41 AC1)."""

import asyncio
import json
import os
import socket
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from app.main import app, lifespan
from core import budget_store
from core.config import PROJECT_ROOT, get_settings, token_cost_gbp
from core.schemas import LedgerEntry
from storage_helpers import create_ledger, reserve, use_database, write_lock

SERVER = Path(__file__).parent / "storage_server.py"
LAG_BOUND_MS = 100.0
API = {"X-Requested-With": "AgentGuard"}


@pytest.fixture(autouse=True)
def database(monkeypatch, tmp_path):
    for name in ("SESSION_BUDGET_CAP_GBP", "GLOBAL_BUDGET_CAP_GBP", "CLIENT_BUDGET_CAP_GBP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    path = tmp_path / "ledger.db"
    use_database(monkeypatch, path)
    return path


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@contextmanager
def server(tmp_path: Path):
    port = free_port()
    log = tmp_path / "server.log"
    with log.open("w") as stream:
        process = subprocess.Popen(
            [sys.executable, str(SERVER), str(port)],
            cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 30
            while True:
                try:
                    httpx.get(f"{url}/__test__/state", trust_env=False, timeout=2).raise_for_status()
                    break
                except httpx.TransportError:
                    assert process.poll() is None and time.monotonic() < deadline, log.read_text()
                    time.sleep(0.05)
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def estimate(tool_name: str, args: dict) -> float:
    """What the classifier reserves for this call."""
    settings = get_settings()
    payload = json.dumps({"tool_name": tool_name, "args": args}, ensure_ascii=False)
    return token_cost_gbp(
        settings.rates_for(settings.classifier),
        settings.budget_reserve_input_tokens + len(payload.encode()),
        settings.classifier.max_output_tokens,
    )


async def run(client: httpx.AsyncClient, **body) -> list[dict]:
    response = await client.post("/api/run", json=body)
    assert response.status_code == 200, response.text
    return [json.loads(line) for line in response.text.splitlines() if line]


async def test_real_server_under_a_write_lock_keeps_the_loop_free_and_the_caps_exact(
    database, tmp_path
):
    admitted, calls = 5, 15
    web_args = {"prompt": "add the numbers"}
    mcp_args = {"code": "print(2)"}
    web_cost = estimate("agent_prompt", web_args)
    mcp_cost = estimate("execute_python", mcp_args)
    tracker = budget_store.get_tracker()
    # Room for exactly `admitted` classifier calls in each session.
    tracker.set_session_cap("race-web", (admitted + 0.5) * web_cost)
    tracker.set_session_cap("race-mcp", (admitted + 0.5) * mcp_cost)

    with server(tmp_path) as url:
        async with asyncio.timeout(60):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=30) as client:
                async with streamable_http_client(f"{url}/mcp", http_client=client) as (r, w, _):
                    async with ClientSession(r, w) as mcp:
                        await mcp.initialize()

                        def check():
                            return mcp.call_tool("guard_check_action", {
                                "tool_name": "execute_python",
                                "args_json": json.dumps(mcp_args),
                                "session_id": "race-mcp",
                            })

                        # Warm up lazy imports, then measure only the load.
                        await run(client, prompt="warm up", session_id="warm")
                        await mcp.call_tool("guard_get_budget_status", {"session_id": "warm"})
                        await client.get("/__test__/state", params={"reset": True})

                        finished: list[float] = []

                        async def timed(call):
                            result = await call
                            finished.append(time.monotonic())
                            return result

                        with write_lock(database):
                            load = [
                                asyncio.create_task(timed(run(client, session_id="race-web", **web_args)))
                                for _ in range(calls)
                            ] + [asyncio.create_task(timed(check())) for _ in range(calls)]
                            await asyncio.sleep(1.0)
                            released = time.monotonic()
                        results = await asyncio.gather(*load)
                        loaded = (await client.get("/__test__/state")).json()

                        # Every other product path, for the thread check.
                        await run(client, prompt="work", session_id="paths", budget_cap_gbp=0.04)
                        await run(client, prompt="fail-agent", session_id="paths")
                        await run(client, prompt="fail-classify", session_id="paths")
                        await run(client, prompt="loop", preset="loop")
                        await mcp.call_tool("guard_execute_sandbox", {"code": "print(3)", "session_id": "paths"})
                        await mcp.call_tool("guard_check_action", {"tool_name": "x", "args_json": "no"})
                        await mcp.call_tool("guard_execute_sandbox", {"code": "1", "session_id": "bad id"})
                        await mcp.call_tool("guard_get_budget_status", {"session_id": "bad id"})
                        await mcp.call_tool("guard_get_budget_status", {"session_id": "paths"})
                        assert (await client.get("/api/budget/paths")).status_code == 200
                        assert (await client.post("/api/budget/paths/reset", headers=API)).status_code == 200
                        assert (await client.post("/api/budget/bad%20id/reset", headers=API)).status_code == 422
                        assert (await client.get("/api/ledger/recent")).status_code == 200
                        final = (await client.get("/__test__/state")).json()

    assert min(finished) >= released, "a call finished without waiting for the write lock"
    print(f"max loop lag under load: {loaded['max_lag_ms']:.1f} ms")
    assert loaded["max_lag_ms"] < LAG_BOUND_MS
    assert final["violations"] == []
    for method in (
        "_create_schema", "check_budget", "reserve", "commit", "release",
        "set_session_cap", "reset", "record", "recent_entries",
    ):
        assert final["calls"].get(method), f"{method} was not exercised"

    web, checks = results[:calls], results[calls:]
    screens = [
        next(e for e in events if e["type"] == "step_finished")["result"]["guard"] for events in web
    ]
    assert sorted(s["decision"] for s in screens) == ["ALLOW"] * admitted + ["BLOCK"] * (calls - admitted)
    assert all(s["stage"] == "BUDGET" for s in screens if s["decision"] == "BLOCK")
    verdicts = [c.structuredContent for c in checks]
    assert sorted(v["decision"] for v in verdicts) == ["ALLOW"] * admitted + ["BLOCK"] * (calls - admitted)
    assert all(v["stage"] == "BUDGET" for v in verdicts if v["decision"] == "BLOCK")
    for session, cost in (("race-web", web_cost), ("race-mcp", mcp_cost)):
        status = tracker.check_budget(session)
        assert status.spent_gbp == pytest.approx(admitted * cost, abs=1e-12)
        assert status.spent_gbp <= status.cap_gbp and status.reserved_gbp == 0


async def test_disconnect_during_the_agent_turn_is_recorded(database, tmp_path):
    with server(tmp_path) as url:
        async with asyncio.timeout(30):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=20) as client:
                body = {"prompt": "slow-agent", "session_id": "gone"}
                async with client.stream("POST", "/api/run", json=body) as response:
                    async for _ in response.aiter_lines():
                        while not (await client.get("/__test__/state")).json()["agent_calls"]:
                            await asyncio.sleep(0.02)
                        break
                # Leaving the block closed the connection in the middle of the turn.
                tracker = budget_store.get_tracker()
                for _ in range(200):
                    entries = [e for e in tracker.recent_entries() if e.session_id == "gone"]
                    if len(entries) == 2 and tracker.check_budget("gone").reserved_gbp == 0:
                        break
                    await asyncio.sleep(0.02)
    assert [(e.label, e.outcome) for e in entries] == [("screen", "ALLOW"), ("agent turn", "ERROR")]
    assert entries[1].cost_gbp == 0
    assert tracker.check_budget("gone").reserved_gbp == 0


def test_server_does_not_start_on_a_newer_schema(database, tmp_path):
    create_ledger(database, "s")
    conn = sqlite3.connect(database)
    conn.execute("PRAGMA user_version = 9")
    conn.close()
    process = subprocess.run(
        [sys.executable, str(SERVER), str(free_port())],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert process.returncode != 0
    assert "newer than supported" in process.stderr


async def test_startup_opens_the_ledger_once_and_off_the_loop(database, monkeypatch):
    create_ledger(database, "s")
    threads = []
    real = budget_store.SessionBudgetTracker._create_schema

    def spy(self):
        threads.append(threading.current_thread())
        real(self)

    monkeypatch.setattr(budget_store.SessionBudgetTracker, "_create_schema", spy)
    lag = 0.0

    async def watch():
        nonlocal lag
        while True:
            start = time.perf_counter()
            await asyncio.sleep(0.005)
            lag = max(lag, time.perf_counter() - start - 0.005)

    async with asyncio.timeout(20):
        watcher = asyncio.create_task(watch())
        await asyncio.sleep(0.02)
        held = threading.Event()

        def hold():
            with write_lock(database):
                held.set()
                time.sleep(0.8)

        holder = threading.Thread(target=hold)
        holder.start()
        held.wait(5)
        async with lifespan(app):
            watcher.cancel()
        holder.join()
    assert len(threads) == 1 and threads[0] is not threading.main_thread()
    assert lag < LAG_BOUND_MS / 1000


async def test_startup_fails_when_the_ledger_cannot_open(database, monkeypatch):
    def broken(self):
        raise ValueError("cannot open")

    monkeypatch.setattr(budget_store.SessionBudgetTracker, "_create_schema", broken)
    with pytest.raises(ValueError, match="cannot open"):
        async with lifespan(app):
            pass


def test_threads_share_one_tracker(monkeypatch):
    real = budget_store.SessionBudgetTracker._create_schema
    built = []

    def slow(self):
        built.append(self)
        time.sleep(0.1)
        real(self)

    monkeypatch.setattr(budget_store.SessionBudgetTracker, "_create_schema", slow)
    seen = []
    threads = [threading.Thread(target=lambda: seen.append(budget_store.get_tracker())) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert len(built) == 1 and len(set(map(id, seen))) == 1


async def test_reads_complete_while_writes_wait_for_the_lock(database):
    tracker = budget_store.get_tracker()
    entry = LedgerEntry(
        time="2026-11-28T00:00:00Z", source="web", session_id="r",
        label="screen", outcome="ALLOW", cost_gbp=0,
    )
    async with asyncio.timeout(20):
        with write_lock(database):
            # More writes than the writer pool has threads.
            writes = [asyncio.create_task(tracker.arecord(entry)) for _ in range(12)]
            await asyncio.sleep(0.2)
            async with asyncio.timeout(1.0):
                await tracker.acheck_budget("r")
                await tracker.arecent_entries()
                await tracker.arecent_entries(client_id="someone")
            assert not any(w.done() for w in writes)
        await asyncio.gather(*writes)


def test_two_processes_share_one_client_cap(monkeypatch, database):
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.01")
    tracker = budget_store.get_tracker()
    code = """
import sys, uuid
from core import budget_store
from core.schemas import ModelRates, TokenUsage
rates = ModelRates(provider="g", model="m", input_cost_per_1m_usd=1, output_cost_per_1m_usd=1, usd_to_gbp=1)
tracker = budget_store.SessionBudgetTracker()
admitted = 0
for _ in range(20):
    key = budget_store.session_key("acme", uuid.uuid4().hex)
    held = tracker.reserve(key, 0.001, rates=rates, ttl_sec=60)
    if held:
        tracker.commit(held, TokenUsage(prompt_tokens=1000, candidates_tokens=0))
        admitted += 1
print(admitted)
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT, stdout=subprocess.PIPE, text=True,
            env=os.environ,
        )
        for _ in range(2)
    ]
    totals = [int(p.communicate(timeout=60)[0]) for p in processes]
    assert sum(totals) == 10
    status = tracker.check_budget("acme:fresh")
    assert status.client_spent_gbp == pytest.approx(0.01) and status.is_exceeded
    assert reserve(tracker, "acme:fresh", 0.001) is None
    assert reserve(tracker, "fresh", 0.001) is not None  # a local session has no client cap


def test_product_modules_keep_no_unused_ledger_wrappers():
    from app import mcp_server
    from core import guard_core, pipeline

    assert not hasattr(pipeline, "_record") and not hasattr(guard_core, "_record")
    assert not hasattr(mcp_server, "_arecord")
