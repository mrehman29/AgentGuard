"""Tests for retention policy: verifying caps and spend never change after pruning."""

import asyncio
import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

import pytest

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
def clean_tracker(tmp_path, monkeypatch):
    db_file = tmp_path / "retention_test.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setenv("LEDGER_RETENTION_DAYS", "0")
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.10")
    monkeypatch.setattr(budget_store, "_tracker", None)
    return budget_store.get_tracker()


def test_retention_preserves_caps_and_status(clean_tracker):
    tracker = clean_tracker
    session_id = "test-retention"

    # 1. Spend in session
    res_1 = tracker.reserve(session_id, 0.010, rates=RATES, ttl_sec=60.0)
    commit_1 = tracker.commit(
        res_1,
        TokenUsage(prompt_tokens=400, candidates_tokens=100, thoughts_tokens=20),
    )
    spent_1 = commit_1.cost_gbp

    res_2 = tracker.reserve(session_id, 0.010, rates=RATES, ttl_sec=60.0)
    commit_2 = tracker.commit(
        res_2,
        TokenUsage(prompt_tokens=300, candidates_tokens=80, thoughts_tokens=10),
    )
    spent_2 = commit_2.cost_gbp

    # Live committable reservation that should NOT be pruned
    live_res = tracker.reserve(session_id, 0.005, rates=RATES, ttl_sec=3600.0)
    assert live_res is not None

    # Record entries: one legacy (created_at = 0), one old, one recent
    tracker.record(
        LedgerEntry(
            time=datetime.now(UTC) - timedelta(days=45),
            source="web",
            session_id=session_id,
            label="old-run",
            outcome="ALLOW",
            cost_gbp=spent_1,
        )
    )
    tracker.record(
        LedgerEntry(
            time=datetime.now(UTC),
            source="web",
            session_id=session_id,
            label="recent-run",
            outcome="ALLOW",
            cost_gbp=spent_2,
        )
    )

    # Insert a legacy entry with created_at = 0.0
    conn = tracker._connect()
    try:
        conn.execute(
            "INSERT INTO entries (entry, created_at, client_id) VALUES (?, 0.0, NULL)",
            ('{"label": "legacy"}',),
        )
        # Age the first usage row to 45 days ago
        old_timestamp = time.time() - (45 * 86400)
        cursor = conn.execute("SELECT id FROM usage ORDER BY id ASC LIMIT 1")
        (first_id,) = cursor.fetchone()
        conn.execute("UPDATE usage SET created_at = ? WHERE id = ?", (old_timestamp, first_id))
        conn.commit()
    finally:
        conn.close()

    # Capture status before retention
    status_before = tracker.check_budget(session_id)

    # Run retention pruning (30 days window)
    stats = tracker.apply_retention(retention_days=30)
    assert stats.usage_pruned == 1
    assert stats.entries_pruned == 1  # only the 45-day-old entry pruned; legacy and recent kept!
    assert stats.reservations_pruned == 0  # live reservation kept!

    # Check status after retention
    status_after = tracker.check_budget(session_id)
    assert status_after.spent_gbp == status_before.spent_gbp
    assert status_after.total_tokens == status_before.total_tokens
    assert status_after.prompt_tokens == status_before.prompt_tokens
    assert status_after.candidates_tokens == status_before.candidates_tokens
    assert status_after.thoughts_tokens == status_before.thoughts_tokens
    assert status_after.reserved_gbp == status_before.reserved_gbp
    assert status_after.remaining_gbp == status_before.remaining_gbp
    assert status_after.global_spent_gbp == status_before.global_spent_gbp
    assert status_after.is_exceeded == status_before.is_exceeded

    # Verify summary row is inside usage with created_at = 0.0
    conn = tracker._connect()
    try:
        row = conn.execute("SELECT created_at, cost_gbp FROM usage WHERE created_at = 0.0").fetchone()
        assert row is not None
        assert abs(row[1] - spent_1) < 1e-9

        # Verify legacy entry is still present
        (legacy_count,) = conn.execute("SELECT count(*) FROM entries WHERE created_at = 0.0").fetchone()
        assert legacy_count == 1

        # Verify live reservation is still present and committable
        res_row = conn.execute("SELECT id FROM reservations WHERE id = ?", (live_res,)).fetchone()
        assert res_row is not None
    finally:
        conn.close()

    # Commit live reservation after retention succeeds!
    commit_live = tracker.commit(live_res, TokenUsage(prompt_tokens=50, candidates_tokens=10, thoughts_tokens=0))
    assert commit_live.cost_gbp > 0


def test_retention_cap_invariance_3000_sessions(clean_tracker):
    """AC3: Cap invariance across 3,000 sessions (identical is_exceeded results)."""
    tracker = clean_tracker
    conn = tracker._connect()

    old_ts = time.time() - (40 * 86400)
    now_ts = time.time()

    # Insert 3,000 sessions with varying spend and caps
    # Some exceeded, some close to boundary
    cap_val = 0.05
    sessions_to_test = []

    usage_inserts = []
    caps_inserts = []

    for i in range(3000):
        s_id = f"bulk-sess-{i}"
        client_id = f"client-{i % 10}"
        key = budget_store.session_key(client_id, s_id)

        # Diverse costs to test boundary conditions
        cost1 = 0.01 + (i % 50) * 0.0008
        cost2 = 0.005 + (i % 25) * 0.0004
        # Session cap
        caps_inserts.append((key, cap_val))

        # Two usage rows: one old (will be pruned), one recent
        usage_inserts.append((key, client_id, 100, 50, 10, cost1, 0, old_ts))
        usage_inserts.append((key, client_id, 50, 20, 5, cost2, 0, now_ts))

        if i < 100 or i % 30 == 0:
            sessions_to_test.append(key)

    conn.executemany("INSERT INTO session_caps (session_id, cap_gbp) VALUES (?, ?)", caps_inserts)
    conn.executemany(
        "INSERT INTO usage (session_id, client_id, prompt_tokens, candidates_tokens, thoughts_tokens, cost_gbp, cleared, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        usage_inserts,
    )
    conn.commit()
    conn.close()

    # Capture status before
    statuses_before = {s: tracker.check_budget(s) for s in sessions_to_test}

    # Apply retention
    stats = tracker.apply_retention(retention_days=30)
    assert stats.usage_pruned == 3000

    # Capture status after
    statuses_after = {s: tracker.check_budget(s) for s in sessions_to_test}

    for s in sessions_to_test:
        sb = statuses_before[s]
        sa = statuses_after[s]
        assert abs(sb.spent_gbp - sa.spent_gbp) < 1e-9
        assert sb.is_exceeded == sa.is_exceeded, f"is_exceeded changed for {s}: {sb.is_exceeded} vs {sa.is_exceeded}"
        assert abs(sb.remaining_gbp - sa.remaining_gbp) < 1e-9


def test_retention_cli_days_flag(tmp_path, monkeypatch):
    """Test CLI prune with --days flag."""
    db_file = tmp_path / "cli_retention.db"
    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(budget_store, "_tracker", None)
    tracker = budget_store.get_tracker()

    # Seed data
    r = tracker.reserve("sess-cli", 0.005, rates=RATES, ttl_sec=60.0)
    tracker.commit(r, TokenUsage(prompt_tokens=100, candidates_tokens=50, thoughts_tokens=0))

    # Age usage row
    conn = tracker._connect()
    try:
        conn.execute("UPDATE usage SET created_at = ?", (time.time() - (45 * 86400),))
        conn.commit()
    finally:
        conn.close()

    res = subprocess.run(
        [sys.executable, "-m", "core.storage", "prune", "--db", str(db_file), "--days", "30"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "Pruned" in res.stdout
    assert "1 usage rows" in res.stdout


@pytest.mark.asyncio
async def test_async_retention_and_reset(clean_tracker):
    async with asyncio.timeout(10.0):
        tracker = clean_tracker
        session_id = "test-async-retention"

        res = await tracker.areserve(session_id, 0.010, rates=RATES, ttl_sec=60.0)
        await tracker.acommit(
            res,
            TokenUsage(prompt_tokens=200, candidates_tokens=50, thoughts_tokens=10),
        )

        # Age usage row
        conn = tracker._connect()
        try:
            old_timestamp = time.time() - (40 * 86400)
            conn.execute("UPDATE usage SET created_at = ?", (old_timestamp,))
            conn.commit()
        finally:
            conn.close()

        status_before = await tracker.acheck_budget(session_id)
        await tracker.aapply_retention(retention_days=30)
        status_after = await tracker.acheck_budget(session_id)

        assert status_after.spent_gbp == status_before.spent_gbp
        assert status_after.global_spent_gbp == status_before.global_spent_gbp

        # Reset session: session spend should be cleared, but global spend must persist!
        reset_status = await tracker.areset(session_id)
        assert reset_status.spent_gbp == 0.0
        assert reset_status.global_spent_gbp == status_before.global_spent_gbp
