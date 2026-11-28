"""Storage management for AgentGuard: schema migrations, retention, and backups."""

import argparse
import fcntl
import json
import logging
import os
import shutil
import sqlite3
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

CURRENT_SCHEMA_VERSION = 2

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
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = get_schema_version(conn)
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported version {CURRENT_SCHEMA_VERSION}."
            )

        if version == CURRENT_SCHEMA_VERSION:
            _add_v2_schema(conn)  # no change unless something is missing
            conn.execute("COMMIT")
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
        conn.execute("COMMIT")
        return 2
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def apply_retention(
    conn: sqlite3.Connection, retention_days: int
) -> RetentionStats:
    """Prune entries and usage older than retention_days.

    Pruned usage is added to one summary row per (session_id, client_id, cleared)
    holding the exact sum, with created_at = 0; older summaries are merged into it.
    Other rows with created_at = 0 (legacy v1) are never deleted, and neither is
    any reservation: an expired one can still be committed by a late call.
    """
    if retention_days <= 0:
        return RetentionStats(0, 0, 0.0)

    cutoff = time.time() - (retention_days * 86400)
    old = "(summary = 0 AND created_at > 0 AND created_at < ?)"

    conn.execute("BEGIN IMMEDIATE")
    try:
        entries_pruned = conn.execute(
            "DELETE FROM entries WHERE created_at > 0 AND created_at < ?", (cutoff,)
        ).rowcount

        # The one full pass reads the table in order. The planner would walk
        # usage_session and fetch each row from its own place on disk.
        groups = conn.execute(
            "SELECT session_id, client_id, cleared, SUM(prompt_tokens), SUM(candidates_tokens),"
            " SUM(thoughts_tokens), TOTAL(cost_gbp), SUM(summary = 0),"
            " TOTAL(CASE WHEN summary = 0 THEN cost_gbp END)"
            f" FROM usage NOT INDEXED WHERE summary = 1 OR {old}"
            " GROUP BY session_id, client_id, cleared"
            " HAVING SUM(summary = 0) > 0 OR COUNT(*) > 1",
            (cutoff,),
        ).fetchall()

        usage_pruned = 0
        cost_archived_gbp = 0.0
        for session_id, client_id, cleared, p, c, t, total, count, archived in groups:
            # One of these runs per group, so it must find the group by session.
            # The planner picks usage_client, which makes every local row
            # (client_id NULL) a candidate each time: quadratic, under the write lock.
            conn.execute(
                "DELETE FROM usage INDEXED BY usage_session"
                " WHERE session_id = ? AND client_id IS ? AND cleared = ?"
                f" AND (summary = 1 OR {old})",
                (session_id, client_id, cleared, cutoff),
            )
            conn.execute(
                "INSERT INTO usage (session_id, client_id, prompt_tokens, candidates_tokens, "
                "thoughts_tokens, cost_gbp, cleared, created_at, summary) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 0.0, 1)",
                (session_id, client_id, p, c, t, total, cleared),
            )
            usage_pruned += count
            cost_archived_gbp += archived

        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise

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
