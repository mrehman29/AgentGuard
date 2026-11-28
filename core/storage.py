"""Storage management for AgentGuard: schema migrations, retention, and backups."""

import argparse
import fcntl
import json
import logging
import math
import os
import shutil
import sqlite3
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2

# One retention transaction removes at most this many entries and this many usage
# rows, so how long it holds the write lock does not grow with the ledger.
RETENTION_BATCH_ROWS = 500
# Between two of them. A writer that is waiting sleeps up to 100 ms between tries
# (SQLite's busy handler), so a longer gap lets every waiter try while the lock is free.
RETENTION_PAUSE_SEC = 0.125
# Writers that keep a ledger busy leave only brief gaps, which a prune using
# SQLite's own back-off can miss for good. It asks this often instead, and gives
# up only after this long without once getting the lock.
RETENTION_POLL_SEC = 0.001
RETENTION_LOCK_WAIT_SEC = 60.0

_V1_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        candidates_tokens INTEGER NOT NULL,
        thoughts_tokens INTEGER NOT NULL,
        cost_gbp REAL NOT NULL,
        cleared INTEGER NOT NULL DEFAULT 0
    )""",
    "CREATE INDEX IF NOT EXISTS usage_session ON usage (session_id, cleared)",
    """CREATE TABLE IF NOT EXISTS reservations (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        amount_gbp REAL NOT NULL,
        expires_at REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS session_caps (
        session_id TEXT PRIMARY KEY,
        cap_gbp REAL NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        entry TEXT NOT NULL
    )""",
)

# Columns added by schema v2. The rate columns are nullable with no default: NULL
# marks a reservation made before v2, and zero is a rate like any other.
_V2_COLUMNS = (
    ("usage", "created_at", "REAL NOT NULL DEFAULT 0.0"),
    ("usage", "client_id", "TEXT DEFAULT NULL"),
    ("usage", "summary", "INTEGER NOT NULL DEFAULT 0"),
    ("entries", "created_at", "REAL NOT NULL DEFAULT 0.0"),
    ("entries", "client_id", "TEXT DEFAULT NULL"),
    ("reservations", "client_id", "TEXT DEFAULT NULL"),
    ("reservations", "input_cost_per_1m_usd", "REAL"),
    ("reservations", "output_cost_per_1m_usd", "REAL"),
    ("reservations", "usd_to_gbp", "REAL"),
)

_RATE_COLUMNS = ("input_cost_per_1m_usd", "output_cost_per_1m_usd", "usd_to_gbp")

_V2_INDEXES = (
    "CREATE INDEX IF NOT EXISTS usage_session ON usage (session_id, cleared)",
    "CREATE INDEX IF NOT EXISTS usage_client ON usage (client_id, cleared)",
    "CREATE INDEX IF NOT EXISTS usage_created_at ON usage (created_at)",
    "CREATE INDEX IF NOT EXISTS entries_client_id ON entries (client_id, id DESC)",
    "CREATE INDEX IF NOT EXISTS entries_created_at ON entries (created_at)",
    "CREATE INDEX IF NOT EXISTS reservations_client_id ON reservations (client_id)",
    "CREATE INDEX IF NOT EXISTS reservations_expires_at ON reservations (expires_at)",
    # Finds a group's summary row without reading the group's other rows.
    "CREATE INDEX IF NOT EXISTS usage_summary ON usage (session_id, cleared, client_id)"
    " WHERE summary = 1",
)

# What a restorable backup must hold. Version 0 and 1 are the v1.0 layout.
_BASE_COLUMNS = {
    "usage": {
        "id", "session_id", "prompt_tokens", "candidates_tokens",
        "thoughts_tokens", "cost_gbp", "cleared",
    },
    "reservations": {"id", "session_id", "amount_gbp", "expires_at"},
    "session_caps": {"session_id", "cap_gbp"},
    "entries": {"id", "entry"},
}

_SIDECARS = ("-wal", "-shm", "-journal")


class LedgerInUse(RuntimeError):
    """Another process holds the ledger, or it is being restored."""


@dataclass(frozen=True)
class RetentionStats:
    entries_pruned: int
    usage_pruned: int
    cost_archived_gbp: float


def lock_path(db_path: Path | str) -> Path:
    path = Path(db_path).resolve()
    return path.with_name(f"{path.name}.lock")


def acquire_ledger_lock(db_path: Path | str, *, exclusive: bool = False, timeout: float = 0.0) -> int:
    """Lock the ledger's lock file and return its descriptor. Closing it unlocks.

    Every user of a ledger holds it shared for as long as it has the ledger open;
    restore holds it exclusive. The file is never deleted: that would let two
    processes lock different files.
    """
    fd = os.open(lock_path(db_path), os.O_RDWR | os.O_CREAT, 0o600)
    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    deadline = time.monotonic() + timeout
    while True:
        try:
            fcntl.flock(fd, mode | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(fd)
                holder = "another process" if exclusive else "a restore"
                raise LedgerInUse(
                    f"Ledger is currently locked by a running process ({holder}): {db_path}"
                ) from None
            time.sleep(0.02)
        except BaseException:
            os.close(fd)
            raise


@contextmanager
def ledger_lock(db_path: Path | str, *, exclusive: bool = False, timeout: float = 0.0) -> Iterator[None]:
    fd = acquire_ledger_lock(db_path, exclusive=exclusive, timeout=timeout)
    try:
        yield
    finally:
        os.close(fd)


@contextmanager
def _write_transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """BEGIN IMMEDIATE, then COMMIT, or roll back and raise the first error."""
    conn.execute("BEGIN IMMEDIATE")
    with _commit_or_roll_back(conn):
        yield


@contextmanager
def _commit_or_roll_back(conn: sqlite3.Connection) -> Iterator[None]:
    """End the transaction already begun on conn."""
    try:
        yield
        conn.execute("COMMIT")
    except BaseException:
        _roll_back(conn)
        raise


def _roll_back(conn: sqlite3.Connection) -> None:
    """Roll back while another error is being raised, without hiding it."""
    # SQLite has already rolled back after some errors (full disk, I/O, interrupt).
    if conn.in_transaction:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            logger.exception("rollback failed")


def get_schema_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("PRAGMA user_version")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    )
    return cursor.fetchone() is not None


def _columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}


def _get_db_file_path(conn: sqlite3.Connection) -> Path | None:
    for row in conn.execute("PRAGMA database_list").fetchall():
        if row[1] == "main" and row[2]:
            return Path(row[2])
    return None


def _private_temp(dest: Path) -> Path:
    """A new empty file beside dest, readable by its owner only from the start."""
    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}_{time.time_ns()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.fchmod(fd, 0o600)  # whatever the umask
    finally:
        os.close(fd)
    return tmp


def _remove_sidecars(db_path: Path) -> None:
    """A journal left beside a replaced database would be applied to the new file."""
    for suffix in _SIDECARS:
        db_path.with_name(db_path.name + suffix).unlink(missing_ok=True)


def _copy_database(source: sqlite3.Connection, target: sqlite3.Connection) -> None:
    source.backup(target)


def _snapshot(source: sqlite3.Connection, dest: Path, *, overwrite: bool) -> None:
    """Write a complete copy of source to dest, which appears all at once, mode 0600.

    Without overwrite, an existing dest raises FileExistsError and is untouched.
    """
    if not overwrite and dest.exists():
        raise FileExistsError(f"File already exists and will not be overwritten: {dest}")
    tmp = _private_temp(dest)
    try:
        target = sqlite3.connect(tmp, timeout=10.0)
        try:
            _copy_database(source, target)
        finally:
            target.close()
        if overwrite:
            _remove_sidecars(dest)
            os.replace(tmp, dest)
        else:
            if not dest.exists():
                _remove_sidecars(dest)
            os.link(tmp, dest)  # atomic, and fails if dest exists
    finally:
        tmp.unlink(missing_ok=True)
        _remove_sidecars(tmp)


def _backup_before_upgrade(db_file: Path) -> Path:
    """Copy the database to <db>.v1.bak, or to a new name if that one is taken."""
    bak_path = db_file.with_name(f"{db_file.name}.v1.bak")
    # A second connection: backing up from the one holding BEGIN IMMEDIATE deadlocks.
    source = sqlite3.connect(db_file, timeout=10.0)
    try:
        try:
            _snapshot(source, bak_path, overwrite=False)
        except FileExistsError:
            bak_path = db_file.with_name(f"{db_file.name}.v1.bak.{time.time_ns()}")
            _snapshot(source, bak_path, overwrite=False)
    finally:
        source.close()
    logger.warning("backed up the ledger to %s before upgrading it", bak_path)
    return bak_path


def _entry_timestamp(entry_json: str) -> float:
    """0.0 (kept for ever) if the entry holds no readable time."""
    try:
        return datetime.fromisoformat(json.loads(entry_json)["time"]).timestamp()
    except (ValueError, TypeError, KeyError):
        return 0.0


def _add_v2_schema(conn: sqlite3.Connection) -> None:
    for table, column, definition in _V2_COLUMNS:
        if column not in _columns(conn, table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    for statement in _V2_INDEXES:
        conn.execute(statement)
    _check_rate_columns(conn)


def _check_rate_columns(conn: sqlite3.Connection) -> None:
    for _, name, _, not_null, default, _ in conn.execute("PRAGMA table_info(reservations)"):
        if name in _RATE_COLUMNS and (not_null or default is not None):
            raise ValueError(
                f"Column reservations.{name} must be nullable with no default."
                " This database was migrated by an unreleased build; restore its .v1.bak copy."
            )


def _is_current(conn: sqlite3.Connection) -> bool:
    """Nothing to change. Reads only, so a start under load needs no write lock."""
    if get_schema_version(conn) != CURRENT_SCHEMA_VERSION:
        return False
    names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master")}
    wanted = {statement.split()[5] for statement in _V2_INDEXES}
    if not wanted <= names or any(c not in _columns(conn, t) for t, c, _ in _V2_COLUMNS):
        return False
    _check_rate_columns(conn)
    return True


def migrate_database(conn: sqlite3.Connection) -> int:
    """Migrate the database to schema v2 in a single atomic transaction.

    Before altering an existing ledger, copies it to <db>.v1.bak (mode 0600),
    or to a new name if that exists. Refuses user_version > CURRENT_SCHEMA_VERSION.
    Existing rows get no client: client names are never read out of old keys.
    """
    if _is_current(conn):
        return CURRENT_SCHEMA_VERSION
    with _write_transaction(conn):
        version = get_schema_version(conn)
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported version {CURRENT_SCHEMA_VERSION}."
            )

        if version == CURRENT_SCHEMA_VERSION:
            _add_v2_schema(conn)  # no change unless something is missing
            return version

        db_file = _get_db_file_path(conn)
        if db_file and any(_table_exists(conn, t) for t in ("usage", "reservations", "entries")):
            _backup_before_upgrade(db_file)

        for statement in _V1_SCHEMA:
            conn.execute(statement)
        _add_v2_schema(conn)

        rows = conn.execute("SELECT id, entry FROM entries WHERE created_at = 0.0").fetchall()
        conn.executemany(
            "UPDATE entries SET created_at = ? WHERE id = ?",
            [(_entry_timestamp(entry), row_id) for row_id, entry in rows],
        )

        conn.execute(
            """CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at REAL NOT NULL,
                description TEXT NOT NULL
            )"""
        )
        now = time.time()
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at, description)"
            " VALUES (1, ?, 'Baseline v1.0 schema')",
            (now,),
        )
        conn.execute(
            "INSERT OR REPLACE INTO schema_migrations (version, applied_at, description)"
            " VALUES (2, ?, 'Schema v2: client accounting, timestamps, rate storage')",
            (now,),
        )
        conn.execute("PRAGMA user_version = 2")
    return CURRENT_SCHEMA_VERSION


# A summary group is (session_id, client_id, cleared). Rows below are
# (id, prompt_tokens, candidates_tokens, thoughts_tokens, cost_gbp).


def _summary_rows(conn: sqlite3.Connection, group: tuple, limit: int) -> list[tuple]:
    session_id, client_id, cleared = group
    return conn.execute(
        "SELECT id, prompt_tokens, candidates_tokens, thoughts_tokens, cost_gbp"
        " FROM usage INDEXED BY usage_summary"
        " WHERE summary = 1 AND session_id = ? AND client_id IS ? AND cleared = ? LIMIT ?",
        (session_id, client_id, cleared, limit),
    ).fetchall()


def _delete_usage(conn: sqlite3.Connection, ids: list[int]) -> None:
    if ids:
        marks = ",".join("?" * len(ids))
        conn.execute(f"DELETE FROM usage WHERE id IN ({marks})", ids)


def _fold(
    conn: sqlite3.Connection, group: tuple, summaries: list[tuple], rows: list[tuple]
) -> list[int]:
    """Add rows, and any further summaries, into the group's first summary row.

    Returns the ids now counted there, which the caller must delete in the same
    transaction.
    """
    parts = summaries + rows
    _, prompt, candidates, thoughts, cost = zip(*parts)
    totals = (sum(prompt), sum(candidates), sum(thoughts), math.fsum(cost))
    if summaries:
        conn.execute(
            "UPDATE usage SET prompt_tokens = ?, candidates_tokens = ?, thoughts_tokens = ?,"
            " cost_gbp = ? WHERE id = ?",
            (*totals, summaries[0][0]),
        )
        return [part[0] for part in parts[1:]]
    session_id, client_id, cleared = group
    conn.execute(
        "INSERT INTO usage (session_id, client_id, cleared, prompt_tokens, candidates_tokens,"
        " thoughts_tokens, cost_gbp, created_at, summary) VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 1)",
        (session_id, client_id, cleared, *totals),
    )
    return [part[0] for part in parts]


def _prune_usage_batch(conn: sqlite3.Connection, cutoff: float, limit: int) -> list[float]:
    """Move up to limit old usage rows into their summaries. Returns their costs."""
    found = conn.execute(
        "SELECT session_id, client_id, cleared, id, prompt_tokens, candidates_tokens,"
        " thoughts_tokens, cost_gbp FROM usage INDEXED BY usage_created_at"
        " WHERE created_at > 0 AND created_at < ? AND summary = 0"
        " ORDER BY created_at LIMIT ?",
        (cutoff, limit),
    ).fetchall()
    groups: dict[tuple, list[tuple]] = {}
    for row in found:
        groups.setdefault(row[:3], []).append(row[3:])
    counted: list[int] = []
    for group, rows in groups.items():
        counted += _fold(conn, group, _summary_rows(conn, group, 1), rows)
    _delete_usage(conn, counted)
    return [row[-1] for row in found]


def _merge_duplicate_summaries(conn: sqlite3.Connection, stop: threading.Event) -> None:
    """Leave one summary per group. A reset can bring two summaries into one group."""
    # Read without the write lock: this looks at every summary.
    groups = conn.execute(
        "SELECT session_id, client_id, cleared FROM usage INDEXED BY usage_summary"
        " WHERE summary = 1 GROUP BY session_id, cleared, client_id HAVING COUNT(*) > 1"
    ).fetchall()
    while groups:
        _let_writers_in(stop)
        if not _begin_retention(conn, stop):
            return
        with _commit_or_roll_back(conn):
            room = RETENTION_BATCH_ROWS
            while groups and room >= 2:
                # Read again under the lock: another prune may have merged them.
                summaries = _summary_rows(conn, groups[-1], room)
                if len(summaries) >= 2:
                    _delete_usage(conn, _fold(conn, groups[-1], summaries, []))
                if len(summaries) < room:
                    groups.pop()  # all of this group's summaries were read
                room -= max(len(summaries), 1)


def _begin_retention(conn: sqlite3.Connection, stop: threading.Event) -> bool:
    """Take the write lock, polling for it. False if stop was set first."""
    (busy_timeout,) = conn.execute("PRAGMA busy_timeout").fetchone()
    deadline = time.monotonic() + RETENTION_LOCK_WAIT_SEC
    try:
        conn.execute("PRAGMA busy_timeout = 0")  # no back-off: a miss returns at once
        try:
            while not stop.is_set():
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    return True
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc) or time.monotonic() >= deadline:
                        raise
                stop.wait(RETENTION_POLL_SEC)
            return False
        finally:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout)}")
    except BaseException:
        _roll_back(conn)  # the lock may have been taken before the error
        raise


def _let_writers_in(stop: threading.Event) -> None:
    """Called between retention transactions, with no lock held."""
    stop.wait(RETENTION_PAUSE_SEC)


def apply_retention(
    conn: sqlite3.Connection, retention_days: int, *, stop: threading.Event | None = None
) -> RetentionStats:
    """Prune entries and usage older than retention_days.

    Pruned usage is added to one summary row per (session_id, client_id, cleared),
    with created_at = 0. Other rows with created_at = 0 (legacy v1) are never
    deleted, and neither is any reservation: an expired one can still be committed
    by a late call.

    The work is a series of short transactions, each of which leaves every total as
    it was. After an error or a kill, what was committed stays and the next run does
    the rest. Setting stop ends the run before its next transaction.
    """
    if retention_days <= 0:
        return RetentionStats(0, 0, 0.0)

    cutoff = time.time() - (retention_days * 86400)
    stop = stop or threading.Event()
    entries_pruned = usage_pruned = 0
    cost_archived_gbp = 0.0

    while _begin_retention(conn, stop):
        limit = RETENTION_BATCH_ROWS
        with _commit_or_roll_back(conn):
            entries = conn.execute(
                "DELETE FROM entries WHERE id IN (SELECT id FROM entries"
                " WHERE created_at > 0 AND created_at < ? ORDER BY created_at LIMIT ?)",
                (cutoff, limit),
            ).rowcount
            costs = _prune_usage_batch(conn, cutoff, limit)
        # Counted only once committed.
        entries_pruned += entries
        usage_pruned += len(costs)
        cost_archived_gbp += math.fsum(costs)
        if entries < limit and len(costs) < limit:
            break
        _let_writers_in(stop)

    if not stop.is_set():
        _merge_duplicate_summaries(conn, stop)

    return RetentionStats(
        entries_pruned=entries_pruned,
        usage_pruned=usage_pruned,
        cost_archived_gbp=cost_archived_gbp,
    )


def backup_database(src_db_path: Path | str, dest_backup_path: Path | str) -> Path:
    """Perform an online hot backup using SQLite's backup API.

    Refuses dest == src and a dest that is a ledger in use. The copy is mode 0600
    from its creation and replaces dest only when complete.
    """
    src = Path(src_db_path).resolve()
    dest = Path(dest_backup_path).resolve()
    if src == dest or (src.exists() and dest.exists() and src.samefile(dest)):
        raise ValueError(f"Destination backup path cannot be the same as source: {src}")
    if dest in [src.with_name(src.name + s) for s in (*_SIDECARS, ".lock")]:
        raise ValueError(f"Destination backup path belongs to the source database: {dest}")
    if not src.exists():
        raise FileNotFoundError(f"Source database does not exist: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest_fd = None
    if lock_path(dest).exists():
        dest_fd = acquire_ledger_lock(dest, exclusive=True)
    try:
        with ledger_lock(src, timeout=10.0):
            source = sqlite3.connect(src, timeout=10.0)
            try:
                _snapshot(source, dest, overwrite=True)
            finally:
                source.close()
    finally:
        if dest_fd is not None:
            os.close(dest_fd)
    return dest


def _verify_backup(backup: Path) -> None:
    conn = sqlite3.connect(backup, timeout=10.0)
    try:
        row = conn.execute("PRAGMA integrity_check").fetchone()
        if not row or row[0] != "ok":
            raise ValueError(f"Backup file failed integrity check: {row}")

        version = get_schema_version(conn)
        if version not in (0, 1, 2):
            raise ValueError(f"Unsupported backup schema version: {version}")

        for table, base in _BASE_COLUMNS.items():
            if not _table_exists(conn, table):
                raise ValueError(f"Backup file missing required table: {table}")
            required = set(base)
            if version == 2:
                # summary is added on the next start if it is absent.
                required |= {c for t, c, _ in _V2_COLUMNS if t == table and c != "summary"}
            missing = required - _columns(conn, table)
            if missing:
                raise ValueError(
                    f"Backup table {table} is missing columns: {', '.join(sorted(missing))}"
                )
    finally:
        conn.close()


def _open_exclusive(target: Path) -> sqlite3.Connection:
    """Open target so that no other SQLite connection can read or write it.

    Leaves it in rollback-journal mode, which folds in and removes any WAL.
    """
    conn = sqlite3.connect(target, timeout=0.5, isolation_level=None)
    try:
        conn.execute("PRAGMA locking_mode=EXCLUSIVE")
        conn.execute("BEGIN EXCLUSIVE")
        conn.execute("COMMIT")
        (mode,) = conn.execute("PRAGMA journal_mode=DELETE").fetchone()
        if mode != "delete":
            raise sqlite3.OperationalError("database is locked")
    except BaseException:
        conn.close()
        raise
    return conn


def _replace_unreadable(backup: Path, target: Path, pre_bak: Path) -> None:
    """The target is not a readable database: keep its bytes and replace the file."""
    for suffix in ("", "-wal"):
        original = target.with_name(target.name + suffix)
        if original.exists():
            copy = _private_temp(pre_bak)
            try:
                shutil.copyfile(original, copy)
                os.link(copy, pre_bak.with_name(pre_bak.name + suffix))
            finally:
                copy.unlink(missing_ok=True)
    source = sqlite3.connect(backup, timeout=10.0)
    try:
        _snapshot(source, target, overwrite=True)
    finally:
        source.close()


def restore_database(
    src_backup_path: Path | str, target_db_path: Path | str, force: bool = False
) -> None:
    """Restore target database from a verified backup.

    Checks:
    - Backup exists, passes PRAGMA integrity_check and has the required tables and
      columns, with user_version in (0, 1, 2).
    - If target exists, requires force=True.
    - No process holds the ledger's lock file and no SQLite connection holds the
      database; both stay locked until the restore ends.
    - Creates <db>.pre-restore.bak (mode 0600, never overwriting an existing one).

    An existing database is overwritten through SQLite's backup API, in one
    transaction: it ends up as the old ledger or the backup, never a mixture.
    """
    backup = Path(src_backup_path).resolve()
    target = Path(target_db_path).resolve()
    if not backup.exists():
        raise FileNotFoundError(f"Backup file does not exist: {backup}")
    if backup == target or (target.exists() and backup.samefile(target)):
        raise ValueError(f"Backup and target are the same file: {target}")
    _verify_backup(backup)

    target.parent.mkdir(parents=True, exist_ok=True)
    with ledger_lock(target, exclusive=True):
        if not target.exists():
            source = sqlite3.connect(backup, timeout=10.0)
            try:
                _snapshot(source, target, overwrite=False)
            finally:
                source.close()
            return

        if not force:
            raise FileExistsError(f"Target database already exists: {target}. Use --force to overwrite.")
        pre_bak = target.with_name(f"{target.name}.pre-restore.bak")
        if pre_bak.exists():
            raise FileExistsError(f"Pre-restore backup already exists and will not be overwritten: {pre_bak}")

        try:
            target_conn = _open_exclusive(target)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc) and "busy" not in str(exc):
                raise
            raise LedgerInUse(f"Target database is currently locked by a running process: {exc}") from exc
        except sqlite3.DatabaseError:
            _replace_unreadable(backup, target, pre_bak)
            return

        try:
            _snapshot(target_conn, pre_bak, overwrite=False)
            source = sqlite3.connect(backup, timeout=10.0)
            try:
                _copy_database(source, target_conn)
            finally:
                source.close()
            os.chmod(target, 0o600)
            target_conn.execute("PRAGMA journal_mode=WAL")
        finally:
            target_conn.close()


def _open_ledger(db_path: Path) -> sqlite3.Connection:
    """Open an existing ledger. A mistyped path must not create one."""
    if not db_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {db_path}")
    conn = sqlite3.connect(
        f"{db_path.resolve().as_uri()}?mode=rw", uri=True, timeout=10.0, isolation_level=None
    )
    try:
        if not all(_table_exists(conn, table) for table in _BASE_COLUMNS):
            raise ValueError(f"Not an AgentGuard ledger: {db_path}")
    except BaseException:
        conn.close()
        raise
    return conn


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentGuard Storage CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # migrate
    migrate_parser = subparsers.add_parser("migrate", help="Migrate database to latest schema")
    migrate_parser.add_argument("--db", default=None, help="SQLite DB path (default from settings)")

    # prune
    prune_parser = subparsers.add_parser("prune", help="Prune records per retention policy")
    prune_parser.add_argument("--db", default=None, help="SQLite DB path (default from settings)")
    prune_parser.add_argument("--days", type=int, default=None, help="Retention window in days (default from settings)")

    # backup
    backup_parser = subparsers.add_parser("backup", help="Back up database")
    backup_parser.add_argument("src", help="Source SQLite DB path")
    backup_parser.add_argument("dest", help="Destination backup file path")

    # restore
    restore_parser = subparsers.add_parser("restore", help="Restore database from backup")
    restore_parser.add_argument("src", help="Source backup file path")
    restore_parser.add_argument("dest", help="Target SQLite DB path")
    restore_parser.add_argument("--force", action="store_true", help="Force overwrite existing target")

    args = parser.parse_args()

    try:
        if args.command in ("migrate", "prune"):
            from core.config import get_settings

            settings = None
            if args.db is None or (args.command == "prune" and args.days is None):
                settings = get_settings()
            db_path = Path(args.db) if args.db else settings.sqlite_db_path
            if not db_path.is_file():
                raise FileNotFoundError(f"Database does not exist: {db_path}")
            with ledger_lock(db_path, timeout=10.0):
                conn = _open_ledger(db_path)
                try:
                    version = migrate_database(conn)
                    if args.command == "migrate":
                        print(f"Database migrated to version {version}: {db_path}")
                    else:
                        days = args.days if args.days is not None else settings.ledger_retention_days
                        stats = apply_retention(conn, days)
                        if days <= 0:
                            print("Retention is off (0 days): nothing was pruned")
                        else:
                            print(
                                f"Pruned {stats.entries_pruned} entries, {stats.usage_pruned} usage rows "
                                f"({stats.cost_archived_gbp:.6f} GBP archived)"
                            )
                finally:
                    conn.close()

        elif args.command == "backup":
            dest = backup_database(args.src, args.dest)
            print(f"Database backed up successfully to {dest}")

        elif args.command == "restore":
            restore_database(args.src, args.dest, force=args.force)
            print(f"Database restored successfully from {args.src} to {args.dest}")

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
