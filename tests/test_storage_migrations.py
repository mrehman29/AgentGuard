"""Tests for storage schema versioning and database upgrades."""

import json
import sqlite3
import threading
import time
from datetime import UTC, datetime

import pytest

from core import budget_store, storage
from core.schemas import LedgerEntry, ModelRates, TokenUsage

RATES = ModelRates(
    provider="gemini",
    model="gemini-3.5-flash-lite",
    input_cost_per_1m_usd=0.30,
    output_cost_per_1m_usd=2.50,
    usd_to_gbp=0.80,
)

V1_SCHEMA_STATEMENTS = (
    """CREATE TABLE usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        candidates_tokens INTEGER NOT NULL,
        thoughts_tokens INTEGER NOT NULL,
        cost_gbp REAL NOT NULL,
        cleared INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX usage_session ON usage (session_id, cleared)",
    """CREATE TABLE reservations (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        amount_gbp REAL NOT NULL,
        expires_at REAL NOT NULL
    )""",
    """CREATE TABLE session_caps (
        session_id TEXT PRIMARY KEY,
        cap_gbp REAL NOT NULL
    )""",
    """CREATE TABLE entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry TEXT NOT NULL
    )""",
)


def create_v1_database(db_path):
    """Build an exact v1.0 database matching pre-#41 code (user_version = 0)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    for stmt in V1_SCHEMA_STATEMENTS:
        conn.execute(stmt)

    now_iso = datetime.now(UTC).isoformat()
    # Insert legacy sample records
    conn.execute(
        "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens, thoughts_tokens, cost_gbp, cleared)"
        " VALUES ('legacy-sess', 100, 50, 10, 0.005, 0)"
    )
    conn.execute(
        "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens, thoughts_tokens, cost_gbp, cleared)"
        " VALUES ('legacy-sess', 80, 20, 5, 0.002, 1)"
    )
    conn.execute(
        "INSERT INTO reservations (id, session_id, amount_gbp, expires_at)"
        " VALUES ('res-legacy', 'legacy-sess', 0.0015, ?)",
        (time.time() + 60,),
    )
    conn.execute(
        "INSERT INTO session_caps (session_id, cap_gbp) VALUES ('legacy-sess', 0.03)"
    )
    legacy_entry = json.dumps(
        {
            "time": now_iso,
            "source": "web",
            "session_id": "legacy-sess",
            "label": "screen",
            "outcome": "ALLOW",
            "cost_gbp": 0.005,
        }
    )
    conn.execute("INSERT INTO entries (entry) VALUES (?)", (legacy_entry,))
    conn.commit()
    conn.close()


def test_migration_from_v1_database(tmp_path, monkeypatch):
    db_file = tmp_path / "legacy_v1.db"
    create_v1_database(db_file)

    # Verify initial v1 state
    conn = sqlite3.connect(db_file)
    assert storage.get_schema_version(conn) == 0
    conn.close()

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(budget_store, "_tracker", None)

    # Initialize tracker, which runs migrate_database
    tracker = budget_store.get_tracker()

    # Verify backup file <db>.v1.bak was created with mode 0600
    bak_file = db_file.with_name(f"{db_file.name}.v1.bak")
    assert bak_file.exists()
    assert (bak_file.stat().st_mode & 0o777) == 0o600

    conn = sqlite3.connect(db_file)
    try:
        assert storage.get_schema_version(conn) == storage.CURRENT_SCHEMA_VERSION
        # Check migrations table
        cursor = conn.execute("SELECT version, description FROM schema_migrations ORDER BY version")
        migrations = cursor.fetchall()
        assert len(migrations) == 2
        assert migrations[0][0] == 1
        assert migrations[1][0] == 2

        # Check columns
        cursor = conn.execute("PRAGMA table_info(usage)")
        col_names = [r[1] for r in cursor.fetchall()]
        assert "created_at" in col_names
        assert "client_id" in col_names

        cursor = conn.execute("PRAGMA table_info(entries)")
        col_names = [r[1] for r in cursor.fetchall()]
        assert "created_at" in col_names
        assert "client_id" in col_names

        # Check backfilled timestamp on legacy entry
        cursor = conn.execute("SELECT created_at FROM entries LIMIT 1")
        (ts,) = cursor.fetchone()
        assert ts > 0
    finally:
        conn.close()

    # Pre-v2 reservation commits at amount_gbp (0.0015)
    commit_res = tracker.commit(
        "res-legacy", TokenUsage(prompt_tokens=100, candidates_tokens=100, thoughts_tokens=0)
    )
    assert commit_res.cost_gbp == 0.0015

    # Verify budget status calculation on migrated database
    status = tracker.check_budget("legacy-sess")
    assert status.session_id == "legacy-sess"
    assert status.prompt_tokens == 200
    assert abs(status.spent_gbp - (0.005 + 0.0015)) < 1e-6
    assert status.reserved_gbp == 0.0
    assert status.cap_gbp == 0.03


def test_reject_unsupported_future_version(tmp_path):
    future_db = tmp_path / "future.db"
    conn = sqlite3.connect(future_db)
    conn.execute("PRAGMA user_version = 9")
    conn.close()

    conn2 = sqlite3.connect(future_db)
    try:
        with pytest.raises(ValueError, match="is newer than supported version"):
            storage.migrate_database(conn2)
    finally:
        conn2.close()


def test_concurrent_starters_migration(tmp_path, monkeypatch):
    """AC2: Concurrent starters both migrate cleanly without corrupting schema or backup."""
    db_file = tmp_path / "concurrent_mig.db"
    create_v1_database(db_file)

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))

    errors = []

    def starter():
        try:
            tracker = budget_store.SessionBudgetTracker()
            st = tracker.check_budget("legacy-sess")
            assert st.spent_gbp > 0
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=starter) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent starters failed with errors: {errors}"
    conn = sqlite3.connect(db_file)
    try:
        assert storage.get_schema_version(conn) == storage.CURRENT_SCHEMA_VERSION
    finally:
        conn.close()
