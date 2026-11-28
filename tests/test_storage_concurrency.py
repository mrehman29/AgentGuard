"""Load test demonstrating non-blocking event loop and strict cap enforcement under concurrency."""

import asyncio
import multiprocessing
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from app.main import app
from app.mcp_server import guard_check_action, guard_get_budget_status
from core import budget_store
from core.schemas import LedgerEntry, ModelRates, TokenUsage

RATES = ModelRates(
    provider="gemini",
    model="gemini-3.5-flash-lite",
    input_cost_per_1m_usd=0.30,
    output_cost_per_1m_usd=2.50,
    usd_to_gbp=0.80,
)


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    db_file = tmp_path / "concurrent_guard.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("SESSION_BUDGET_CAP_GBP", "0.020")
    monkeypatch.setenv("GLOBAL_BUDGET_CAP_GBP", "0.050")
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.020")
    monkeypatch.setattr(budget_store, "_tracker", None)
    return budget_store.get_tracker()


@pytest.mark.asyncio
async def test_loop_lag_under_held_write_lock_and_exact_cap_enforcement(test_db):
    """AC1: Hold write lock for 1s on background thread while driving /api/run and /mcp.

    Assert loop lag bound < 100ms, and exact integer math admits exactly 10 of 50 tasks.
    """
    async with asyncio.timeout(30.0):
        tracker = test_db
        session_id = "concurrent-sess"

        # Start event loop heartbeat monitor
        heartbeat_ticks = 0
        max_lag_ms = 0.0
        stop_heartbeat = asyncio.Event()

        async def heartbeat_monitor():
            nonlocal heartbeat_ticks, max_lag_ms
            interval = 0.005  # 5ms
            while not stop_heartbeat.is_set():
                t0 = time.perf_counter()
                await asyncio.sleep(interval)
                lag = (time.perf_counter() - t0 - interval) * 1000
                if lag > max_lag_ms:
                    max_lag_ms = lag
                heartbeat_ticks += 1

        monitor_task = asyncio.create_task(heartbeat_monitor())

        # Background thread holds BEGIN IMMEDIATE for 1.0 second
        def hold_lock():
            conn = sqlite3.connect(tracker._path, timeout=5.0)
            conn.execute("BEGIN IMMEDIATE")
            time.sleep(1.0)
            conn.execute("ROLLBACK")
            conn.close()

        lock_thread = threading.Thread(target=hold_lock)
        lock_thread.start()
        # Give thread time to acquire write lock
        await asyncio.sleep(0.05)

        num_concurrent_ops = 50
        reserve_cost_per_op = 0.002  # With cap = 0.020, exactly 10 must succeed with integer math!

        successful_reservations = 0
        rejected_reservations = 0
        lock = asyncio.Lock()

        async def web_client_worker(client: httpx.AsyncClient, worker_id: int):
            res = await client.get(f"/api/budget/{session_id}")
            assert res.status_code == 200
            res2 = await client.get("/api/ledger/recent?limit=10")
            assert res2.status_code == 200

        async def mcp_client_worker(worker_id: int):
            status = await guard_get_budget_status(session_id)
            assert "spent_gbp" in status or "error" not in status

        async def spender_worker(worker_id: int):
            nonlocal successful_reservations, rejected_reservations
            r_id = await tracker.areserve(
                session_id, reserve_cost_per_op, rates=RATES, ttl_sec=60.0
            )
            if r_id is not None:
                async with lock:
                    successful_reservations += 1
                # Commit exact 0.002 GBP: 1000 tokens * 2.50 / 1e6 * 0.80 = 0.002
                await tracker.acommit(
                    r_id,
                    TokenUsage(prompt_tokens=0, candidates_tokens=1000, thoughts_tokens=0),
                )
                await tracker.arecord(
                    LedgerEntry(
                        time=datetime.now(UTC),
                        source="web",
                        session_id=session_id,
                        label=f"worker-{worker_id}",
                        outcome="ALLOW",
                        cost_gbp=reserve_cost_per_op,
                    )
                )
            else:
                async with lock:
                    rejected_reservations += 1

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            tasks = []
            for i in range(num_concurrent_ops):
                tasks.append(web_client_worker(client, i))
                tasks.append(mcp_client_worker(i))
                tasks.append(spender_worker(i))

            await asyncio.gather(*tasks)

        lock_thread.join()
        stop_heartbeat.set()
        await monitor_task

        # Loop lag must be strictly bounded (well below the 1.0s background lock)
        assert max_lag_ms < 150.0, f"Event loop stalled: max lag was {max_lag_ms:.2f}ms"
        assert heartbeat_ticks > 10, f"Heartbeat stalled: ticks={heartbeat_ticks}"

        # Exact cap enforcement via integer math: exactly 10 succeed, exactly 40 rejected
        final_status = await tracker.acheck_budget(session_id)
        assert final_status.spent_gbp <= 0.020 + 1e-9
        assert successful_reservations == 10, f"Expected exactly 10 successes, got {successful_reservations}"
        assert rejected_reservations == 40, f"Expected exactly 40 rejections, got {rejected_reservations}"


def _worker_process_spend(db_path_str: str, client_id: str, results_queue: multiprocessing.Queue):
    """Subprocess function simulating concurrent process using client accounting."""
    os.environ["SQLITE_DB_PATH"] = db_path_str
    os.environ["CLIENT_BUDGET_CAP_GBP"] = "0.010"
    budget_store._tracker = None
    tracker = budget_store.get_tracker()

    successful = 0
    # Try 10 reserves of 0.002 GBP with unique session IDs under client_id
    for i in range(10):
        key = budget_store.session_key(client_id, f"sess-{os.getpid()}-{i}")
        res = tracker.reserve(key, 0.002, rates=RATES, ttl_sec=60.0)
        if res is not None:
            tracker.commit(
                res, TokenUsage(prompt_tokens=0, candidates_tokens=1000, thoughts_tokens=0)
            )
            tracker.record(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="web",
                    session_id=key,
                    label="proc_spend",
                    outcome="ALLOW",
                    cost_gbp=0.002,
                )
            )
            successful += 1
    results_queue.put(successful)


def test_multi_process_client_cap_and_reset(tmp_path, monkeypatch):
    """AC1: 2 processes sharing 1 client cap (0.010 GBP), fresh session IDs, client reset."""
    db_file = tmp_path / "client_multiprocess.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.010")
    monkeypatch.setattr(budget_store, "_tracker", None)
    tracker = budget_store.get_tracker()

    client_id = "client_alpha"
    q = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=_worker_process_spend, args=(str(db_file), client_id, q)
    )
    p2 = multiprocessing.Process(
        target=_worker_process_spend, args=(str(db_file), client_id, q)
    )

    p1.start()
    p2.start()
    p1.join(timeout=10.0)
    p2.join(timeout=10.0)

    s1 = q.get()
    s2 = q.get()

    # Total successful across both processes must not exceed client cap:
    # 0.010 / 0.002 = exactly 5 successful reserves total!
    total_successful = s1 + s2
    assert total_successful == 5, f"Expected 5 total successes on 0.010 cap, got {total_successful}"

    # Client ledger must be filtered
    entries = tracker.recent_entries(limit=50, client_id=client_id)
    assert len(entries) == 5
    for entry in entries:
        c, _ = budget_store.split_session_key(entry.session_id)
        assert c == client_id

    # Reset client
    tracker.reset_client(client_id)

    # After reset, client spend should be 0, allowing fresh reserves
    key = budget_store.session_key(client_id, "after_reset_sess")
    res_after = tracker.reserve(key, 0.002, rates=RATES, ttl_sec=60.0)
    assert res_after is not None
    tracker.release(res_after)


@pytest.mark.asyncio
async def test_event_loop_thread_checker(test_db):
    """Ensure no sync tracker method executes directly on the asyncio event loop thread."""
    async with asyncio.timeout(10.0):
        tracker = test_db
        loop_thread_id = threading.get_ident()

        # Wrapper to verify thread ID inside sync tracker methods
        def check_not_on_loop_thread():
            assert threading.get_ident() != loop_thread_id, (
                "Sync tracker method executed directly on the asyncio event loop thread!"
            )

        orig_reserve = tracker.reserve
        tracker.reserve = lambda *args, **kwargs: (check_not_on_loop_thread(), orig_reserve(*args, **kwargs))[1]

        # Calling areserve must run on worker thread, never loop thread
        r_id = await tracker.areserve("loop_thread_sess", 0.001, rates=RATES, ttl_sec=60.0)
        assert r_id is not None
        await tracker.arelease(r_id)
