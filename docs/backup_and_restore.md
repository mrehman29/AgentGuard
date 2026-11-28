# AgentGuard Storage: Retention, Backup, and Restore Guide

This document describes the retention management, schema versioning, and backup/restore procedures for AgentGuard (Story #41).

## 1. Retention Settings

AgentGuard stores token usage metrics and audit log entries in its SQLite database (`SQLITE_DB_PATH`, default: `data/guard_state.db`). To prevent unbounded disk growth while preserving strict budget cap guarantees, AgentGuard provides configurable retention policies.

### Configuration (`.env` or Environment Variables)

- `LEDGER_RETENTION_DAYS`: Number of days to retain detailed audit entries and granular usage rows (default: `0`, meaning retention pruning is disabled by default and records are kept indefinitely). Set to an integer > 0 (e.g. `30`) to prune records older than that window when running retention.
- `CLIENT_BUDGET_CAP_GBP`: Default budget cap for clients (default: `0.10`).

### Cap Invariance Guarantee

> [!IMPORTANT]
> **Retention never alters what counts against a session, client, or global budget cap.**
> When granular `usage` rows older than the retention window (`created_at > 0 AND created_at < cutoff`) are pruned, their token counts and costs are rolled up into exact summary rows **inside the `usage` table** grouped by `(session_id, client_id, cleared)` holding the exact sum with `created_at = 0.0`.
> Legacy rows with `created_at = 0.0` (including v1 rows and earlier summaries) and live reservations are **never deleted**.
> As a result, `spent_gbp`, `remaining_gbp`, and `is_exceeded` calculations yield identical values before and after pruning across all sessions.
>
> Note: Summary rows roll up older events and therefore lose fine-grained per-call timestamps (`created_at = 0.0`). Future calendar/period budget resets (e.g., monthly spending limits in Story #39) must treat `created_at = 0.0` as historical spend according to story rules.

### Running Retention Pruning

Retention pruning runs via CLI or cron only (never automatically on request paths):

#### Storage CLI
```bash
# Reads SQLITE_DB_PATH and LEDGER_RETENTION_DAYS from settings by default,
# automatically migrates schema if needed, and applies retention pruning:
.venv/bin/python -m core.storage prune

# Or specify database and retention days explicitly:
.venv/bin/python -m core.storage prune --db data/guard_state.db --days 30
```

#### Python API
```python
from core.budget_store import get_tracker

tracker = get_tracker()
# Synchronous
stats = tracker.apply_retention(retention_days=30)
# Asynchronous (non-blocking)
stats = await tracker.aapply_retention(retention_days=30)
```

---

## 2. Backup Procedure

AgentGuard databases operate in SQLite WAL (Write-Ahead Logging) mode. Copying the active `.db` file using standard OS commands (`cp`, `rsync`) while transactions are running can yield a corrupted or inconsistent snapshot.

AgentGuard utilizes the **SQLite Online Backup API** (`sqlite3.Connection.backup`), which locks pages safely and writes out a complete, clean, self-contained database file. Backups are written to a temporary file first before atomic rename, and created with file permissions `0600`. The backup command refuses destination paths that match the source path.

### Performing a Hot Online Backup

#### Storage CLI
```bash
.venv/bin/python -m core.storage backup data/guard_state.db backups/guard_state_$(date +%F).bak
```

#### Python API
```python
from core.budget_store import get_tracker

tracker = get_tracker()
# Synchronous
backup_path = tracker.backup("backups/guard_state_2026-11-27.bak")
# Asynchronous (non-blocking)
backup_path = await tracker.abackup("backups/guard_state_2026-11-27.bak")
```

---

## 3. Restore Procedure

Restoring replaces the target database with a verified backup.

### Safety Requirements & Checks
1. **Server Stopped**: The restore procedure acquires an exclusive lock on the target database (`BEGIN EXCLUSIVE`). If an active server is running, restore aborts immediately with an error detecting the active process.
2. **Integrity & Schema Verification**: The backup file must pass `PRAGMA integrity_check`, must contain all required tables (`usage`, `reservations`, `session_caps`, `entries`), and must have `user_version in (0, 1, 2)`.
3. **Safety Snapshot**: If the target database already exists, restoring requires `--force`. Before overwriting the target, a safety backup `<target>.pre-restore.bak` (mode `0600`) is created via a temporary file. If `<target>.pre-restore.bak` already exists, restore refuses to proceed to avoid overwriting existing rollback snapshots.
4. **Forgotten Spend Warning**: Restoring a backup reverts the budget state to the moment the backup was captured. Any calls committed between the backup time and restore time are lost from the ledger. In multi-client authentication setups (Story #33), API tokens revoked after the backup will be re-activated unless re-revoked in `auth.db`.

### Restoring from Backup

#### Storage CLI
```bash
# Requires server to be stopped:
.venv/bin/python -m core.storage restore backups/guard_state_backup.bak data/guard_state.db --force
```

#### Python API
```python
from core.storage import restore_database

restore_database("backups/guard_state_backup.bak", "data/guard_state.db", force=True)
```

---

## 4. Automated Maintenance (Cron Example)

To schedule daily backups and weekly retention pruning, standard cron entries can be configured:

```cron
# Daily backup at 02:00 (mode 0600)
0 2 * * * /path/to/project/.venv/bin/python -m core.storage backup /path/to/project/data/guard_state.db /path/to/backups/guard_$(date +\%Y\%m\%d).bak

# Weekly retention prune (prunes records older than 30 days) on Sunday at 03:00
0 3 * * 0 /path/to/project/.venv/bin/python -m core.storage prune --days 30
```
