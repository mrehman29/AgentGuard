"""Schema upgrade from a v1.0 ledger (#41 AC2)."""

import os
import sqlite3
import subprocess
import sys

import pytest
from pydantic import ValidationError

from core import budget_store, storage
from core.config import PROJECT_ROOT
from storage_helpers import (
    USAGE, create_ledger, create_v1_database, dump, query, reserve, use_database, write_lock,
)

RATE_COLUMNS = ("input_cost_per_1m_usd", "output_cost_per_1m_usd", "usd_to_gbp")


@pytest.fixture
def v1(tmp_path, monkeypatch):
    path = create_v1_database(tmp_path / "ledger.db")
    use_database(monkeypatch, path)
    return path


def migrate(path):
    conn = sqlite3.connect(path)
    try:
        return storage.migrate_database(conn)
    finally:
        conn.close()


def backups(path):
    return sorted(path.parent.glob(f"{path.name}.v1.bak*"))


def test_upgrade_from_the_committed_v1_dump(v1):
    before = dump(v1)
    tracker = budget_store.get_tracker()
    assert query(v1, "PRAGMA user_version") == [(2,)]
    [backup] = backups(v1)
    assert backup.name == "ledger.db.v1.bak" and backup.stat().st_mode & 0o777 == 0o600
    assert query(backup, "PRAGMA user_version") == [(0,)] and dump(backup) == before

    status = tracker.check_budget("legacy-sess")
    assert (status.spent_gbp, status.reserved_gbp, status.cap_gbp) == (0.005, 0.0015, 0.03)
    assert status.global_spent_gbp == pytest.approx(0.008)
    assert query(v1, "SELECT COUNT(*) FROM entries WHERE created_at > 1767000000") == [(2,)]


def test_legacy_rows_get_no_client_even_with_a_colon_in_the_key(v1):
    tracker = budget_store.get_tracker()
    for table in ("usage", "reservations", "entries"):
        assert query(v1, f"SELECT COUNT(*) FROM {table} WHERE client_id IS NOT NULL") == [(0,)]
    assert tracker.recent_entries(client_id="demo") == []
    assert len(tracker.recent_entries()) == 2


def test_rate_columns_are_nullable_with_no_default(v1):
    budget_store.get_tracker()
    columns = {row[1]: row for row in query(v1, "PRAGMA table_info(reservations)")}
    for name in RATE_COLUMNS:
        _, _, _, not_null, default, _ = columns[name]
        assert (not_null, default) == (0, None)


def test_a_reservation_with_null_rates_is_charged_its_amount(v1):
    tracker = budget_store.get_tracker()
    assert tracker.commit("res-legacy", USAGE).cost_gbp == 0.0015
    # A process still running the old code inserts no rates either.
    conn = sqlite3.connect(v1)
    conn.execute(
        "INSERT INTO reservations (id, session_id, amount_gbp, expires_at)"
        " VALUES ('old-code', 's', 0.002, 4102444800.0)"
    )
    conn.commit()
    conn.close()
    assert tracker.commit("old-code", USAGE).cost_gbp == 0.002


def test_a_zero_rate_is_not_taken_for_a_pre_v2_reservation(v1):
    """Pricing free rates is #30's; here they must never be charged as the amount."""
    tracker = budget_store.get_tracker()
    held = reserve(tracker, "free", 0.002)
    conn = sqlite3.connect(v1)
    conn.execute(
        "UPDATE reservations SET input_cost_per_1m_usd = 0, output_cost_per_1m_usd = 0,"
        " usd_to_gbp = 0 WHERE id = ?", (held,)
    )
    conn.commit()
    conn.close()
    try:
        charged = tracker.commit(held, USAGE).cost_gbp
    except ValidationError:
        charged = 0
    assert charged == tracker.check_budget("free").spent_gbp == 0


def test_a_failed_upgrade_changes_nothing_and_can_be_retried(v1, monkeypatch):
    before = dump(v1)
    with monkeypatch.context() as broken:
        broken.setattr(storage, "_V2_INDEXES", (*storage._V2_INDEXES, "CREATE INDEX x ON no_table (y)"))
        with pytest.raises(sqlite3.OperationalError):
            migrate(v1)
    assert dump(v1) == before and query(v1, "PRAGMA user_version") == [(0,)]
    [first] = backups(v1)
    stamp = first.stat().st_mtime_ns
    assert migrate(v1) == 2
    # The name was taken: the retry made a new backup and left the first alone.
    assert len(backups(v1)) == 2 and first.stat().st_mtime_ns == stamp
    assert all(dump(b) == before for b in backups(v1))


def test_a_new_ledger_leaves_no_backup(tmp_path, monkeypatch):
    path = tmp_path / "new.db"
    use_database(monkeypatch, path)
    budget_store.get_tracker()
    assert backups(path) == []


def test_an_existing_backup_is_never_overwritten_or_relied_on(v1):
    old = v1.with_name("ledger.db.v1.bak")
    old.write_text("an older backup")
    before = dump(v1)
    migrate(v1)
    assert old.read_text() == "an older backup"
    [new] = [b for b in backups(v1) if b != old]
    assert dump(new) == before and new.stat().st_mode & 0o777 == 0o600


def test_newer_schema_is_refused(tmp_path):
    path = create_ledger(tmp_path / "future.db", "s")
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version = 9")
    with pytest.raises(ValueError, match="newer than supported"):
        storage.migrate_database(conn)
    conn.close()


def test_rate_columns_from_an_unreleased_build_are_refused(tmp_path):
    path = create_ledger(tmp_path / "odd.db", "s")
    conn = sqlite3.connect(path)
    conn.execute("ALTER TABLE reservations DROP COLUMN usd_to_gbp")
    conn.execute("ALTER TABLE reservations ADD COLUMN usd_to_gbp REAL NOT NULL DEFAULT 0.0")
    with pytest.raises(ValueError, match="nullable with no default"):
        storage.migrate_database(conn)
    conn.close()


def test_concurrent_process_starts_upgrade_once(v1):
    conn = sqlite3.connect(v1)
    conn.executemany(
        "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens, thoughts_tokens,"
        " cost_gbp) VALUES ('bulk', 1, 1, 0, 0.0000001)", [()] * 20_000
    )
    conn.commit()
    conn.close()
    code = (
        "from core import budget_store\n"
        "print(budget_store.SessionBudgetTracker().check_budget('bulk').prompt_tokens)\n"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code], cwd=PROJECT_ROOT, env=os.environ,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(6)
    ]
    for process in processes:
        out, err = process.communicate(timeout=60)
        assert (process.returncode, out.strip()) == (0, "20000"), err
    [backup] = backups(v1)
    assert query(backup, "PRAGMA user_version") == [(0,)]
    assert query(backup, "SELECT COUNT(*) FROM usage") == [(20_003,)]
    assert query(v1, "PRAGMA user_version") == [(2,)]


def test_a_current_ledger_opens_while_another_process_is_writing(tmp_path, monkeypatch):
    path = create_ledger(tmp_path / "busy.db", "s")
    monkeypatch.setattr(budget_store, "BUSY_TIMEOUT_MS", 100)
    use_database(monkeypatch, path)
    with write_lock(path):
        assert budget_store.get_tracker().check_budget("s").prompt_tokens == 1
