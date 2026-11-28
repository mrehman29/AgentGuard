"""Tests for cancellation safety, CommitCancelled propagation, and fail-closed behaviors (AC5)."""

import asyncio
import sqlite3
import time
from datetime import UTC, datetime

import pytest

from core import budget_store, guard_core
from core.pipeline import execute_guarded
from core.schemas import CommitCancelled, Decision, LedgerEntry, ModelRates, TokenUsage

RATES = ModelRates(
    provider="gemini",
    model="gemini-3.5-flash-lite",
    input_cost_per_1m_usd=0.30,
    output_cost_per_1m_usd=2.50,
    usd_to_gbp=0.80,
)


@pytest.fixture
def clean_tracker(tmp_path, monkeypatch):
    db_file = tmp_path / "cancellation.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(budget_store, "_tracker", None)
    return budget_store.get_tracker()


@pytest.mark.asyncio
async def test_cancelled_reserve_releases_hold(clean_tracker):
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-cancel-reserve"

        # Inject delay so task cancellation occurs while reserve is in-flight on worker thread
        orig_reserve = tracker.reserve
        tracker.reserve = lambda *a, **k: (time.sleep(0.05), orig_reserve(*a, **k))[1]

        task = asyncio.create_task(tracker.areserve(session_id, 0.010, rates=RATES, ttl_sec=60.0))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Give threadpool executor a moment to complete release
        await asyncio.sleep(0.05)

        status = await tracker.acheck_budget(session_id)
        assert status.reserved_gbp == 0.0, "Cancelled reserve leaked a reservation hold!"


@pytest.mark.asyncio
async def test_cancelled_commit_raises_commit_cancelled_with_result(clean_tracker):
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-cancel-commit"

        r_id = await tracker.areserve(session_id, 0.005, rates=RATES, ttl_sec=60.0)
        assert r_id is not None

        # Inject delay so task cancellation occurs while commit is in-flight on worker thread
        orig_commit = tracker.commit
        tracker.commit = lambda *a, **k: (time.sleep(0.05), orig_commit(*a, **k))[1]

        usage = TokenUsage(prompt_tokens=1000, candidates_tokens=500, thoughts_tokens=0)

        task = asyncio.create_task(tracker.acommit(r_id, usage))
        await asyncio.sleep(0.01)
        task.cancel()

        with pytest.raises(CommitCancelled) as exc_info:
            await task

        assert isinstance(exc_info.value, asyncio.CancelledError)
        assert exc_info.value.result is not None
        assert exc_info.value.result.cost_gbp > 0
        assert exc_info.value.result.budget.spent_gbp > 0


@pytest.mark.asyncio
async def test_double_cancel_does_not_drop_ledger_entry(clean_tracker):
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-double-cancel"

        entry = LedgerEntry(
            time=datetime.now(UTC),
            source="web",
            session_id=session_id,
            label="test-cancel",
            outcome="BLOCK error",
            cost_gbp=0.002,
        )

        orig_record = tracker.record
        tracker.record = lambda *a, **k: (time.sleep(0.05), orig_record(*a, **k))[1]

        # arecord is shielded against cancellation: the DB write completes even if cancelled
        task = asyncio.create_task(tracker.arecord(entry))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        entries = await tracker.arecent_entries(limit=10)
        assert len(entries) == 1
        assert entries[0].session_id == session_id
        assert entries[0].cost_gbp == 0.002


@pytest.mark.asyncio
async def test_cancelled_commit_records_cost_in_pipeline(clean_tracker, monkeypatch):
    """When a commit is cancelled during execute_guarded, cost must be recorded to the ledger."""
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-pipeline-cancel"

        from core.schemas import BudgetStatus, CommitResult

        fake_budget = BudgetStatus(
            session_id=session_id,
            prompt_tokens=100,
            candidates_tokens=50,
            thoughts_tokens=0,
            total_tokens=150,
            spent_gbp=0.003,
            reserved_gbp=0.0,
            cap_gbp=0.05,
            remaining_gbp=0.047,
            global_spent_gbp=0.003,
            global_cap_gbp=0.50,
            is_exceeded=False,
        )
        fake_commit_result = CommitResult(cost_gbp=0.003, budget=fake_budget)

        async def fake_classify(*args, **kwargs):
            raise CommitCancelled(fake_commit_result)

        monkeypatch.setattr(guard_core, "_classify", fake_classify)

        with pytest.raises(CommitCancelled):
            await execute_guarded(
                intent="Test cancellation",
                tool_name="execute_python",
                args={"code": "1 + 1"},
                session_id=session_id,
            )

        # Verify ledger recorded the cost despite the cancellation
        entries = await tracker.arecent_entries(limit=10)
        assert len(entries) == 1
        assert entries[0].cost_gbp == 0.003
        assert entries[0].outcome == "BLOCK error"


@pytest.mark.asyncio
async def test_fail_closed_on_lock_timeout(clean_tracker):
    """Fail-closed on SQLite busy timeout."""
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-timeout-sess"

        # Hold write lock on separate connection without committing
        conn = tracker._connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            tracker._connect = lambda: sqlite3.connect(tracker._path, timeout=0.1)
            with pytest.raises(sqlite3.OperationalError):
                tracker.reserve(session_id, 0.001, rates=RATES, ttl_sec=60.0)
        finally:
            conn.rollback()
            conn.close()
