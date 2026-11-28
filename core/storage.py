"""Storage management for AgentGuard: schema migrations, retention, and backups."""

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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

_V2_INDEXES = (
    "CREATE INDEX IF NOT EXISTS usage_session ON usage (session_id, cleared)",
    "CREATE INDEX IF NOT EXISTS usage_client ON usage (client_id, cleared)",
    "CREATE INDEX IF NOT EXISTS usage_created_at ON usage (created_at)",
    "CREATE INDEX IF NOT EXISTS entries_client_id ON entries (client_id, id DESC)",
    "CREATE INDEX IF NOT EXISTS entries_created_at ON entries (created_at)",
    "CREATE INDEX IF NOT EXISTS reservations_client_id ON reservations (client_id)",
    "CREATE INDEX IF NOT EXISTS reservations_expires_at ON reservations (expires_at)",
)


@dataclass(frozen=True)
class RetentionStats:
    entries_pruned: int
    usage_pruned: int
    cost_archived_gbp: float
    reservations_pruned: int


def get_schema_version(conn: sqlite3.Connection) -> int:
    cursor = conn.execute("PRAGMA user_version")
    row = cursor.fetchone()
    return int(row[0]) if row else 0


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cursor = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)
    )
    return cursor.fetchone() is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return any(row[1] == column_name for row in cursor.fetchall())


def _get_db_file_path(conn: sqlite3.Connection) -> Path | None:
    try:
        cursor = conn.execute("PRAGMA database_list")
        for row in cursor.fetchall():
            if row[1] == "main" and row[2]:
                path = Path(row[2])
                if path.is_file():
                    return path
    except Exception:
        pass
    return None


def migrate_database(conn: sqlite3.Connection) -> int:
    """Migrate the database to schema v2 in a single atomic transaction.

    Before altering, if old version is seen, takes <db>.v1.bak via a temporary
    file with mode 0600 without overwriting an existing backup.
    Refuses databases with user_version > CURRENT_SCHEMA_VERSION.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        version = get_schema_version(conn)
        if version > CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Database schema version {version} is newer than supported version {CURRENT_SCHEMA_VERSION}."
            )

        if version == CURRENT_SCHEMA_VERSION:
            # Ensure indexes exist
            for idx_stmt in _V2_INDEXES:
                conn.execute(idx_stmt)
            conn.execute("COMMIT")
            return version

        # Version < 2: backup existing database if it's an existing file on disk
        db_file = _get_db_file_path(conn)
        if db_file and db_file.exists() and db_file.stat().st_size > 0:
            bak_path = db_file.with_name(f"{db_file.name}.v1.bak")
            if not bak_path.exists():
                tmp_bak = db_file.with_name(f"{db_file.name}.v1.bak.tmp.{os.getpid()}_{time.time_ns()}")
                src2_conn = sqlite3.connect(db_file, timeout=10.0)
                bak_conn = sqlite3.connect(tmp_bak, timeout=10.0)
                try:
                    src2_conn.backup(bak_conn)
                finally:
                    bak_conn.close()
                    src2_conn.close()
                os.chmod(tmp_bak, 0o600)
                if not bak_path.exists():
                    tmp_bak.replace(bak_path)
                else:
                    if tmp_bak.exists():
                        tmp_bak.unlink()

        # Version 0 -> Baseline tables
        if version == 0:
            for statement in _V1_SCHEMA:
                conn.execute(statement)

        # Version 1 -> 2: Add columns, migrations table, backfills
        if not _column_exists(conn, "usage", "created_at"):
            conn.execute("ALTER TABLE usage ADD COLUMN created_at REAL NOT NULL DEFAULT 0.0")
        if not _column_exists(conn, "usage", "client_id"):
            conn.execute("ALTER TABLE usage ADD COLUMN client_id TEXT DEFAULT NULL")

        if not _column_exists(conn, "entries", "created_at"):
            conn.execute("ALTER TABLE entries ADD COLUMN created_at REAL NOT NULL DEFAULT 0.0")
        if not _column_exists(conn, "entries", "client_id"):
            conn.execute("ALTER TABLE entries ADD COLUMN client_id TEXT DEFAULT NULL")

        if not _column_exists(conn, "reservations", "client_id"):
            conn.execute("ALTER TABLE reservations ADD COLUMN client_id TEXT DEFAULT NULL")
        if not _column_exists(conn, "reservations", "input_cost_per_1m_usd"):
            conn.execute("ALTER TABLE reservations ADD COLUMN input_cost_per_1m_usd REAL NOT NULL DEFAULT 0.0")
        if not _column_exists(conn, "reservations", "output_cost_per_1m_usd"):
            conn.execute("ALTER TABLE reservations ADD COLUMN output_cost_per_1m_usd REAL NOT NULL DEFAULT 0.0")
        if not _column_exists(conn, "reservations", "usd_to_gbp"):
            conn.execute("ALTER TABLE reservations ADD COLUMN usd_to_gbp REAL NOT NULL DEFAULT 0.0")

        # Backfill entries timestamps and client_id
        cursor = conn.execute("SELECT id, entry FROM entries WHERE created_at = 0.0 OR client_id IS NULL")
        updates = []
        for row_id, entry_str in cursor.fetchall():
            ts = 0.0
            client = None
            try:
                data = json.loads(entry_str)
                time_val = data.get("time")
                if time_val:
                    ts = datetime.fromisoformat(time_val).timestamp()
                session_val = data.get("session_id", "")
                if ":" in session_val:
                    client = session_val.split(":", 1)[0]
            except Exception:
                pass
            updates.append((ts, client, row_id))
        if updates:
            conn.executemany("UPDATE entries SET created_at = ?, client_id = ? WHERE id = ?", updates)

        # Legacy rows in usage get NULL client_id (no prefix parsing)
        # Apply indexes
        for idx_stmt in _V2_INDEXES:
            conn.execute(idx_stmt)

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
    except Exception:
        conn.execute("ROLLBACK")
        raise


def apply_retention(
    conn: sqlite3.Connection, retention_days: int
) -> RetentionStats:
    """Prune entries and usage older than retention_days.

    Pruned rows are replaced by summary rows inside usage grouped by
    (session_id, client_id, cleared) holding the exact sum with created_at = 0.
    Rows with created_at = 0 (legacy v1 and summaries) and live reservations
    are NEVER deleted.
    """
    if retention_days <= 0:
        return RetentionStats(0, 0, 0.0, 0)

    cutoff = time.time() - (retention_days * 86400)
    now = time.time()

    conn.execute("BEGIN IMMEDIATE")
    try:
        # 1. Clean expired reservations
        res_cursor = conn.execute("DELETE FROM reservations WHERE expires_at < ?", (now,))
        reservations_pruned = res_cursor.rowcount

        # 2. Prune old audit log entries (legacy created_at = 0 kept)
        ent_cursor = conn.execute(
            "DELETE FROM entries WHERE created_at > 0 AND created_at < ?", (cutoff,)
        )
        entries_pruned = ent_cursor.rowcount

        # 3. Find usage rows older than cutoff grouped by (session_id, client_id, cleared)
        groups = conn.execute(
            "SELECT session_id, client_id, cleared, "
            "COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidates_tokens), 0), "
            "COALESCE(SUM(thoughts_tokens), 0), TOTAL(cost_gbp), COUNT(*) "
            "FROM usage WHERE created_at > 0 AND created_at < ? "
            "GROUP BY session_id, client_id, cleared",
            (cutoff,),
        ).fetchall()

        usage_pruned = 0
        cost_archived_gbp = 0.0

        if groups:
            # Delete the pruned usage rows
            del_cursor = conn.execute("DELETE FROM usage WHERE created_at > 0 AND created_at < ?", (cutoff,))
            usage_pruned = del_cursor.rowcount

            # Insert summary rows holding exact sums with created_at = 0
            for session_id, client_id, cleared, p, c, t, spent, count in groups:
                cost_archived_gbp += spent
                conn.execute(
                    "INSERT INTO usage (session_id, client_id, prompt_tokens, candidates_tokens, "
                    "thoughts_tokens, cost_gbp, cleared, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0.0)",
                    (session_id, client_id, int(p), int(c), int(t), spent, cleared),
                )

        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    return RetentionStats(
        entries_pruned=entries_pruned,
        usage_pruned=usage_pruned,
        cost_archived_gbp=cost_archived_gbp,
        reservations_pruned=reservations_pruned,
    )


def backup_database(src_db_path: Path | str, dest_backup_path: Path | str) -> Path:
    """Perform an online hot backup using SQLite's backup API.

    Refuses dest == src. Writes to a temporary file with mode 0600 before atomic rename.
    """
    src = Path(src_db_path).resolve()
    dest = Path(dest_backup_path).resolve()
    if src == dest:
        raise ValueError(f"Destination backup path cannot be the same as source: {src}")
    if not src.exists():
        raise FileNotFoundError(f"Source database does not exist: {src}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_dest = dest.with_name(f"{dest.name}.tmp.{os.getpid()}_{time.time_ns()}")
    src_conn = sqlite3.connect(src, timeout=10.0)
    try:
        dest_conn = sqlite3.connect(tmp_dest, timeout=10.0)
        try:
            src_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        src_conn.close()

    os.chmod(tmp_dest, 0o600)
    tmp_dest.replace(dest)
    os.chmod(dest, 0o600)
    return dest


def restore_database(
    src_backup_path: Path | str, target_db_path: Path | str, force: bool = False
) -> None:
    """Restore target database from a verified backup.

    Checks:
    - Backup exists and passes PRAGMA integrity_check.
    - Required tables exist: usage, reservations, session_caps, entries.
    - user_version in (0, 1, 2).
    - If target exists, requires force=True.
    - Tests exclusive lock on target to ensure no running server.
    - Creates <db>.pre-restore.bak via temporary file (mode 0600, never overwriting existing).
    """
    backup = Path(src_backup_path).resolve()
    target = Path(target_db_path).resolve()
    if not backup.exists():
        raise FileNotFoundError(f"Backup file does not exist: {backup}")

    # 1. Verify backup integrity
    conn = sqlite3.connect(backup, timeout=10.0)
    try:
        cursor = conn.execute("PRAGMA integrity_check")
        row = cursor.fetchone()
        if not row or row[0] != "ok":
            raise ValueError(f"Backup file failed integrity check: {row}")

        # Check required tables
        for tbl in ("usage", "reservations", "session_caps", "entries"):
            if not _table_exists(conn, tbl):
                raise ValueError(f"Backup file missing required table: {tbl}")

        # Check user_version in (0, 1, 2)
        version = get_schema_version(conn)
        if version not in (0, 1, 2):
            raise ValueError(f"Unsupported backup schema version: {version}")
    finally:
        conn.close()

    # 2. Check target database
    if target.exists():
        if not force:
            raise FileExistsError(f"Target database already exists: {target}. Use --force to overwrite.")

        # Test exclusive lock to detect running server
        try:
            target_conn = sqlite3.connect(target, timeout=0.5)
            try:
                target_conn.execute("BEGIN EXCLUSIVE")
                target_conn.rollback()
            finally:
                target_conn.close()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(f"Target database is currently locked by a running process: {exc}")
        except sqlite3.DatabaseError:
            pass

        # Pre-restore backup
        pre_bak = target.with_name(f"{target.name}.pre-restore.bak")
        if pre_bak.exists():
            raise FileExistsError(f"Pre-restore backup already exists and will not be overwritten: {pre_bak}")
        tmp_pre_bak = target.with_name(f"{target.name}.pre-restore.bak.tmp.{os.getpid()}_{time.time_ns()}")
        try:
            src_conn = sqlite3.connect(target, timeout=10.0)
            try:
                dest_conn = sqlite3.connect(tmp_pre_bak, timeout=10.0)
                try:
                    src_conn.backup(dest_conn)
                finally:
                    dest_conn.close()
            finally:
                src_conn.close()
        except sqlite3.DatabaseError:
            import shutil
            shutil.copy2(target, tmp_pre_bak)

        os.chmod(tmp_pre_bak, 0o600)
        tmp_pre_bak.replace(pre_bak)
        os.chmod(pre_bak, 0o600)

    # 3. Restore via atomic temp file
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.tmp.{os.getpid()}_{time.time_ns()}")
    backup_conn = sqlite3.connect(backup, timeout=10.0)
    try:
        dest_conn = sqlite3.connect(tmp_target, timeout=10.0)
        try:
            backup_conn.backup(dest_conn)
        finally:
            dest_conn.close()
    finally:
        backup_conn.close()

    os.chmod(tmp_target, 0o600)
    tmp_target.replace(target)
    os.chmod(target, 0o600)


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
        from core.config import get_settings
        settings = get_settings()
    except Exception:
        settings = None

    try:
        if args.command == "migrate":
            db_path = Path(args.db) if args.db else (settings.sqlite_db_path if settings else Path("data/guard_state.db"))
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                version = migrate_database(conn)
                print(f"Database migrated to version {version}: {db_path}")
            finally:
                conn.close()

        elif args.command == "prune":
            db_path = Path(args.db) if args.db else (settings.sqlite_db_path if settings else Path("data/guard_state.db"))
            retention_days = args.days if args.days is not None else (settings.ledger_retention_days if settings else 0)
            conn = sqlite3.connect(db_path, timeout=10.0)
            try:
                # Migrate first
                migrate_database(conn)
                stats = apply_retention(conn, retention_days)
                print(
                    f"Pruned {stats.entries_pruned} entries, {stats.usage_pruned} usage rows "
                    f"({stats.cost_archived_gbp:.6f} GBP archived), {stats.reservations_pruned} expired reservations"
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
