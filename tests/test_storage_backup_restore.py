"""Online backup and restore (#41 AC3)."""

import asyncio
import os
import sqlite3
import subprocess
import sys
from contextlib import contextmanager

import pytest

from core import budget_store, storage
from core.config import PROJECT_ROOT
from storage_helpers import (
    USAGE, create_ledger, create_v1_database, dump, query, reserve, use_database,
)


@pytest.fixture
def ledgers(tmp_path):
    """A one-row backup and a different, larger target."""
    return (
        create_ledger(tmp_path / "source.bak", "from-backup", 1),
        create_ledger(tmp_path / "target.db", "live", 101),
    )


def mode(path):
    return path.stat().st_mode & 0o777


@contextmanager
def other_process(code: str, *args):
    """A child that runs code, says ready, then keeps everything open."""
    process = subprocess.Popen(
        [sys.executable, "-c", code + "\nprint('ready', flush=True)\nimport sys; sys.stdin.read()",
         *map(str, args)],
        cwd=PROJECT_ROOT, env=os.environ, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        yield process
    finally:
        process.communicate(timeout=30)


IDLE_CONNECTION = (
    "import sqlite3, sys\n"
    "conn = sqlite3.connect(sys.argv[1])\n"
    "conn.execute('SELECT COUNT(*) FROM usage').fetchall()"
)
IDLE_TRACKER = (
    "import os, sys\n"
    "os.environ['SQLITE_DB_PATH'] = sys.argv[1]\n"
    "from core import budget_store\n"
    "tracker = budget_store.get_tracker()"
)


def test_backup_and_restore_round_trip(tmp_path, monkeypatch):
    origin = tmp_path / "origin.db"
    use_database(monkeypatch, origin)
    tracker = budget_store.get_tracker()
    tracker.set_session_cap("s", 0.04)
    tracker.commit(reserve(tracker, "s", 0.005), USAGE)
    before = tracker.check_budget("s")
    backup = tracker.backup(tmp_path / "origin.bak")
    assert mode(backup) == 0o600
    restored = tmp_path / "restored.db"
    storage.restore_database(backup, restored)
    assert mode(restored) == 0o600 and dump(restored) == dump(origin)
    use_database(monkeypatch, restored)
    assert budget_store.get_tracker().check_budget("s") == before


@pytest.mark.parametrize("opener", [IDLE_CONNECTION, IDLE_TRACKER], ids=["connection", "tracker"])
def test_restore_refuses_while_an_idle_process_has_the_ledger_open(ledgers, opener):
    backup, target = ledgers
    before = dump(target)
    with other_process(opener, target):
        result = subprocess.run(
            [sys.executable, "-m", "core.storage", "restore", str(backup), str(target), "--force"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 1, result.stdout
        assert "locked by a running process" in result.stderr
        assert dump(target) == before
    assert not target.with_name("target.db.pre-restore.bak").exists()
    storage.restore_database(backup, target, force=True)
    assert dump(target) == dump(backup)


def test_restore_refuses_while_a_tracker_in_this_process_is_alive(ledgers, monkeypatch):
    backup, target = ledgers
    use_database(monkeypatch, target)
    tracker = budget_store.get_tracker()
    with pytest.raises(storage.LedgerInUse):
        storage.restore_database(backup, target, force=True)
    assert tracker.check_budget("live").prompt_tokens == 101


def test_restore_refuses_during_a_transaction(ledgers):
    backup, target = ledgers
    conn = sqlite3.connect(target)
    conn.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(RuntimeError, match="locked by a running process"):
            storage.restore_database(backup, target, force=True)
    finally:
        conn.close()


def test_restore_replaces_the_whole_ledger_after_a_killed_writer(ledgers):
    """A process killed with the ledger open leaves a WAL full of its rows."""
    backup, target = ledgers
    code = IDLE_CONNECTION + (
        "\nconn.execute('PRAGMA wal_autocheckpoint=0')\n"
        "conn.executemany(\"INSERT INTO entries (entry) VALUES (?)\", [('x' * 200,)] * 3000)\n"
        "conn.commit()"
    )
    with other_process(code, target) as process:
        process.kill()
        process.wait(timeout=10)
    assert target.with_name("target.db-wal").stat().st_size > 0
    storage.restore_database(backup, target, force=True)
    assert dump(target) == dump(backup)
    assert query(target, "PRAGMA integrity_check") == [("ok",)]
    pre = target.with_name("target.db.pre-restore.bak")
    assert query(pre, "SELECT COUNT(*) FROM entries") == [(3000,)] and mode(pre) == 0o600


def test_restore_to_a_missing_target_ignores_a_stale_journal(tmp_path):
    backup = create_ledger(tmp_path / "source.bak", "from-backup")
    donor = create_ledger(tmp_path / "donor.db", "stale")
    conn = sqlite3.connect(donor)
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("INSERT INTO entries (entry) VALUES ('stale')")
    conn.commit()
    stale = (tmp_path / "donor.db-wal").read_bytes()
    conn.close()
    target = tmp_path / "gone.db"
    (tmp_path / "gone.db-wal").write_bytes(stale)
    storage.restore_database(backup, target)
    assert dump(target) == dump(backup)


def test_the_ledger_stays_locked_for_the_whole_restore(ledgers, monkeypatch):
    backup, target = ledgers
    real = storage._copy_database
    seen = []

    def probe(source, destination):
        conn = sqlite3.connect(target, timeout=0.05)
        try:
            conn.execute("SELECT COUNT(*) FROM usage").fetchall()
            seen.append("readable")
        except sqlite3.OperationalError:
            seen.append("locked")
        finally:
            conn.close()
        with pytest.raises(storage.LedgerInUse):
            storage.acquire_ledger_lock(target, timeout=0.05)
        seen.append("refused")
        real(source, destination)

    monkeypatch.setattr(storage, "_copy_database", probe)
    storage.restore_database(backup, target, force=True)
    # Once for the pre-restore copy and once for the restore itself.
    assert seen == ["locked", "refused"] * 2


def test_a_restore_that_fails_leaves_the_old_ledger(ledgers, monkeypatch):
    backup, target = ledgers
    before = dump(target)
    real = storage._copy_database
    copies = []

    def fail_on_restore(source, destination):
        copies.append(1)
        if len(copies) == 2:
            raise OSError("disk full")
        real(source, destination)

    monkeypatch.setattr(storage, "_copy_database", fail_on_restore)
    with pytest.raises(OSError, match="disk full"):
        storage.restore_database(backup, target, force=True)
    assert dump(target) == before


def test_restore_needs_force_and_never_overwrites_the_pre_restore_copy(ledgers):
    backup, target = ledgers
    before = dump(target)
    with pytest.raises(FileExistsError, match="Use --force"):
        storage.restore_database(backup, target)
    storage.restore_database(backup, target, force=True)
    pre = target.with_name("target.db.pre-restore.bak")
    assert dump(pre) == before and mode(pre) == 0o600 and mode(target) == 0o600
    with pytest.raises(FileExistsError, match="Pre-restore backup already exists"):
        storage.restore_database(backup, target, force=True)
    assert dump(pre) == before


def test_restore_over_a_file_that_is_not_a_database(tmp_path):
    backup = create_ledger(tmp_path / "source.bak", "from-backup")
    target = tmp_path / "broken.db"
    target.write_text("not a database " * 100)
    storage.restore_database(backup, target, force=True)
    assert dump(target) == dump(backup)
    pre = tmp_path / "broken.db.pre-restore.bak"
    assert pre.read_text().startswith("not a database") and mode(pre) == 0o600


@pytest.mark.parametrize("version", [0, 1, 2])
def test_restore_accepts_versions_0_1_2(tmp_path, version):
    if version == 2:
        backup = create_ledger(tmp_path / "v.bak", "s")
    else:
        backup = create_v1_database(tmp_path / "v.bak")
        conn = sqlite3.connect(backup)
        conn.execute(f"PRAGMA user_version = {version}")
        conn.close()
    storage.restore_database(backup, tmp_path / "target.db")
    assert dump(tmp_path / "target.db") == dump(backup)


@pytest.mark.parametrize(
    "sql,message",
    [
        ("DROP TABLE session_caps", "missing required table"),
        ("ALTER TABLE usage DROP COLUMN thoughts_tokens", "missing columns: thoughts_tokens"),
        ("ALTER TABLE reservations DROP COLUMN usd_to_gbp", "missing columns: usd_to_gbp"),
        ("PRAGMA user_version = 9", "Unsupported backup schema version"),
    ],
)
def test_restore_rejects_an_unusable_backup(ledgers, sql, message):
    backup, target = ledgers
    conn = sqlite3.connect(backup)
    conn.execute(sql)
    conn.commit()
    conn.close()
    before = dump(target)
    with pytest.raises(ValueError, match=message):
        storage.restore_database(backup, target, force=True)
    assert dump(target) == before


@pytest.mark.parametrize("operation", ["upgrade", "backup", "restore-new", "restore-over", "restore-broken"])
def test_copies_are_private_while_being_written(tmp_path, monkeypatch, operation):
    real_copy, real_copyfile = storage._copy_database, storage.shutil.copyfile
    modes = []

    def observe(source, destination):
        modes.extend(mode(p) for p in tmp_path.glob("*.tmp.*") if not p.name.endswith("-journal"))
        real_copy(source, destination)

    def observe_file(source, destination):
        modes.append(mode(destination))
        return real_copyfile(source, destination)

    monkeypatch.setattr(storage, "_copy_database", observe)
    monkeypatch.setattr(storage.shutil, "copyfile", observe_file)
    old_umask = os.umask(0o022)
    try:
        backup = create_ledger(tmp_path / "source.bak", "s")
        target = tmp_path / "target.db"
        if operation == "upgrade":
            create_v1_database(target)
            conn = sqlite3.connect(target)
            storage.migrate_database(conn)
            conn.close()
        elif operation == "backup":
            storage.backup_database(backup, target)
        else:
            if operation == "restore-over":
                create_ledger(target, "live")
            elif operation == "restore-broken":
                target.write_text("not a database " * 100)
            storage.restore_database(backup, target, force=True)
    finally:
        os.umask(old_umask)
    assert modes and set(modes) == {0o600}
    assert list(tmp_path.glob("*.tmp.*")) == []


def test_backup_refuses_to_damage_a_ledger(ledgers, monkeypatch):
    backup, target = ledgers
    for dest in (target, target.with_name("target.db-wal")):
        with pytest.raises(ValueError):
            storage.backup_database(target, dest)
    use_database(monkeypatch, target)
    tracker = budget_store.get_tracker()
    with pytest.raises(storage.LedgerInUse):
        storage.backup_database(backup, target)
    assert tracker.check_budget("live").prompt_tokens == 101


def test_backup_overwrites_an_old_backup_completely(ledgers):
    backup, target = ledgers
    storage.backup_database(target, backup)
    assert dump(backup) == dump(target)


def test_command_line_backup_and_restore(tmp_path):
    source = create_ledger(tmp_path / "src.db", "s")
    for command in (
        ["backup", str(source), str(tmp_path / "copy.bak")],
        ["restore", str(tmp_path / "copy.bak"), str(tmp_path / "restored.db")],
    ):
        result = subprocess.run(
            [sys.executable, "-m", "core.storage", *command],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
    assert dump(tmp_path / "restored.db") == dump(source)


async def test_async_backup(tmp_path, monkeypatch):
    use_database(monkeypatch, tmp_path / "async.db")
    async with asyncio.timeout(10):
        copy = await budget_store.get_tracker().abackup(tmp_path / "async.bak")
    assert mode(copy) == 0o600
