"""Tests for online backup and restore functionality."""

import asyncio
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

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


def test_backup_and_restore(tmp_path, monkeypatch):
    db_file = tmp_path / "origin.db"
    backup_file = tmp_path / "origin_backup.bak"
    restored_db_file = tmp_path / "restored.db"

    monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
    monkeypatch.setattr(budget_store, "_tracker", None)

    tracker = budget_store.get_tracker()
    session_id = "backup-sess"

    # Seed data
    tracker.set_session_cap(session_id, 0.04)
    res_id = tracker.reserve(session_id, 0.005, rates=RATES, ttl_sec=60.0)
    tracker.commit(
        res_id,
        TokenUsage(prompt_tokens=350, candidates_tokens=70, thoughts_tokens=15),
    )
    status_before = tracker.check_budget(session_id)

    # 1. Hot online backup
    backed_path = tracker.backup(backup_file)
    assert backed_path.exists()
    assert (backed_path.stat().st_mode & 0o777) == 0o600

    # 2. Refuse backup when dest == src
    with pytest.raises(ValueError, match="cannot be the same"):
        tracker.backup(db_file)

    # 3. Restore to new location
    storage.restore_database(backup_file, restored_db_file)
    assert restored_db_file.exists()
    assert (restored_db_file.stat().st_mode & 0o777) == 0o600

    # 4. Connect to restored DB and verify state
    monkeypatch.setenv("SQLITE_DB_PATH", str(restored_db_file))
    monkeypatch.setattr(budget_store, "_tracker", None)
    restored_tracker = budget_store.get_tracker()

    status_restored = restored_tracker.check_budget(session_id)
    assert status_restored.session_id == status_before.session_id
    assert status_restored.spent_gbp == status_before.spent_gbp
    assert status_restored.cap_gbp == status_before.cap_gbp
    assert status_restored.total_tokens == status_before.total_tokens
    assert status_restored.global_spent_gbp == status_before.global_spent_gbp


def test_restore_accepts_versions_0_1_2(tmp_path):
    """Restore accepts version 0, 1, or 2 if required tables exist."""
    for ver in (0, 1, 2):
        bak_file = tmp_path / f"ver_{ver}.bak"
        target_file = tmp_path / f"restored_{ver}.db"

        conn = sqlite3.connect(bak_file)
        conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE reservations (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE session_caps (session_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
        conn.execute(f"PRAGMA user_version = {ver}")
        conn.commit()
        conn.close()

        storage.restore_database(bak_file, target_file)
        assert target_file.exists()


def test_restore_rejects_missing_tables_or_version_9(tmp_path):
    target = tmp_path / "target.db"

    # Missing table
    bak_missing = tmp_path / "missing_table.bak"
    conn = sqlite3.connect(bak_missing)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY)")
    conn.close()
    with pytest.raises(ValueError, match="missing required table"):
        storage.restore_database(bak_missing, target)

    # Version 9
    bak_ver9 = tmp_path / "ver9.bak"
    conn = sqlite3.connect(bak_ver9)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE reservations (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE session_caps (session_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
    conn.execute("PRAGMA user_version = 9")
    conn.commit()
    conn.close()
    with pytest.raises(ValueError, match="Unsupported backup schema version"):
        storage.restore_database(bak_ver9, target)


def test_restore_requires_force_and_creates_pre_restore_bak(tmp_path):
    src_bak = tmp_path / "source.bak"
    target = tmp_path / "target_existing.db"

    conn = sqlite3.connect(src_bak)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE reservations (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE session_caps (session_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    target.write_text("existing content")

    # Refuse overwrite without --force
    with pytest.raises(FileExistsError, match="Use --force to overwrite"):
        storage.restore_database(src_bak, target, force=False)

    # Restore with force=True creates <target>.pre-restore.bak
    storage.restore_database(src_bak, target, force=True)
    pre_bak = target.with_name(f"{target.name}.pre-restore.bak")
    assert pre_bak.exists()
    assert (pre_bak.stat().st_mode & 0o777) == 0o600

    # If <target>.pre-restore.bak already exists, refuse to overwrite it!
    with pytest.raises(FileExistsError, match="Pre-restore backup already exists"):
        storage.restore_database(src_bak, target, force=True)


def test_restore_detects_running_server_via_exclusive_lock(tmp_path):
    src_bak = tmp_path / "source.bak"
    target = tmp_path / "running_server.db"

    conn = sqlite3.connect(src_bak)
    conn.execute("CREATE TABLE usage (id INTEGER PRIMARY KEY)")
    conn.execute("CREATE TABLE reservations (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE session_caps (session_id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE entries (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    target_conn = sqlite3.connect(target)
    target_conn.execute("CREATE TABLE dummy (id INTEGER)")
    target_conn.execute("BEGIN EXCLUSIVE")  # active transaction

    try:
        with pytest.raises(RuntimeError, match="currently locked by a running process"):
            storage.restore_database(src_bak, target, force=True)
    finally:
        target_conn.rollback()
        target_conn.close()


def test_storage_cli_backup_restore(tmp_path):
    db_file = tmp_path / "cli_src.db"
    backup_file = tmp_path / "cli_dest.bak"
    restored_file = tmp_path / "cli_restored.db"

    conn = sqlite3.connect(db_file)
    storage.migrate_database(conn)
    conn.close()

    res_bak = subprocess.run(
        [sys.executable, "-m", "core.storage", "backup", str(db_file), str(backup_file)],
        capture_output=True,
        text=True,
    )
    assert res_bak.returncode == 0
    assert backup_file.exists()

    res_res = subprocess.run(
        [sys.executable, "-m", "core.storage", "restore", str(backup_file), str(restored_file)],
        capture_output=True,
        text=True,
    )
    assert res_res.returncode == 0
    assert restored_file.exists()


@pytest.mark.asyncio
async def test_async_backup(tmp_path, monkeypatch):
    async with asyncio.timeout(10.0):
        db_file = tmp_path / "async_src.db"
        backup_file = tmp_path / "async_dest.bak"

        monkeypatch.setenv("SQLITE_DB_PATH", str(db_file))
        monkeypatch.setattr(budget_store, "_tracker", None)

        tracker = budget_store.get_tracker()
        res_id = await tracker.areserve("sess_async", 0.002, rates=RATES, ttl_sec=60.0)
        await tracker.acommit(res_id, TokenUsage(prompt_tokens=50, candidates_tokens=20, thoughts_tokens=0))

        dest = await tracker.abackup(backup_file)
        assert dest.exists()
        assert (dest.stat().st_mode & 0o777) == 0o600
