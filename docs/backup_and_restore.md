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
> `usage` rows older than the window (`created_at > 0 AND created_at < cutoff`) are added to one summary row per `(session_id, client_id, cleared)`, holding their sum with `created_at = 0.0` (a floating-point sum, rounded once each time rows are added to it). Later prunes merge into that summary, and merge any second summary a reset has brought into the same group.
> Legacy rows (made before schema v2, `created_at = 0.0`) are **never deleted**.
> **No reservation is ever deleted by retention.** An expired reservation stops counting against the caps but can still be committed if its call finishes late. Reservations left by processes that died therefore stay in the table.
>
> Summary rows lose per-call timestamps. Future period budgets (Story #39) must not count `created_at = 0.0` as current spend without a rule.

Old audit `entries` are deleted; entries with no readable time are kept.

### Who may reset

Resetting a session also lowers its client's spend, so it reopens the client cap. Only an administrator may reset a session or a client; Story #33's routes enforce this. Global spend is never lowered by a reset.

### How a prune runs

A prune is **not one transaction**. It is a series of short ones, so that calls can keep using the ledger:

- Each transaction removes at most 500 entries and 500 usage rows and adds those rows to their summaries. How long it holds the write lock does not grow with the ledger, with the amount to prune, or with the size of one session or client.
- After each one the prune waits 0.125 s. That is longer than the longest sleep of a waiting SQLite writer (0.1 s), so waiting calls get the lock in the gap.
- **Every committed transaction leaves session, client and global totals as they were.** While a prune runs, and after one that stopped early, some old rows are summarised and others not yet.
- **Stopping is safe.** After an error, a cancelled `aapply_retention` (it stops after the transaction in progress) or a killed process, committed work stays. Run the prune again to finish it. The command prints counts only when it completes.
- Two prunes at once are safe: each row is summarised once. So is a backup during a prune: it copies the ledger as of one committed transaction. Restore refuses while a prune runs.
- While calls keep the ledger busy, the prune asks for the write lock every millisecond. It stops with `database is locked` only after 60 s without getting it once. Run it again.

It takes time: the pauses alone are 0.125 s for every 500 rows (about 2 minutes for 500,000). Measured on a local ext4 disk with one audit entry per usage row: 100,000 usage rows in 20,000 sessions took 32 s, with no single lock over 0.08 s; 500,000 rows took 2 to 5 minutes depending on the number of sessions and the load, with no single lock over 0.41 s. Another disk will differ.

On a large ledger, calls can fail with `database is locked` when several processes write without pause, **with or without a prune**: every reservation adds up the whole `usage` table under the write lock (see the [server database decision](server_database_decision.md)). A completed prune shrinks that table.

The first start of this version on an existing ledger builds the index that finds summaries, in one transaction that grows with the ledger (0.25 s with 500,000 usage rows).

### Running Retention Pruning

Retention pruning runs via CLI or cron only (never automatically on request paths):

#### Storage CLI
```bash
# Reads SQLITE_DB_PATH and LEDGER_RETENTION_DAYS from settings by default,
# automatically migrates schema if needed, and applies retention pruning:
.venv/bin/python -m core.storage prune

# Or specify database and retention days explicitly (the database must exist):
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

AgentGuard utilizes the **SQLite Online Backup API** (`sqlite3.Connection.backup`), which locks pages safely and writes out a complete, clean, self-contained database file. A backup is written to a temporary file that has permissions `0600` from its creation, and replaces the destination only when complete. The command refuses a destination that is the source, one of the source's journal files, or a ledger that is in use.

Before upgrading an older ledger, startup copies it to `<db>.v1.bak`. If that name is taken, the copy gets a new name, `<db>.v1.bak.<number>`; existing copies are never overwritten.

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
1. **Stop every process that uses the ledger** (web server, MCP stdio servers, CLI jobs). Each AgentGuard process holds a shared lock on `<db>.lock` while it has the ledger open, and restore needs it exclusively, so restore refuses while any of them is running, even if idle. Restore also takes SQLite's exclusive lock, which refuses while another program holds the database open. Both locks are held until the restore ends. Never delete the `.lock` file.
   - Not detected: a server running code from before this lock (older than Story #41), or another tool, that has no connection open at that moment. Stop those yourself.
   - The lock needs a local POSIX filesystem.
2. **Integrity & Schema Verification**: The backup must pass `PRAGMA integrity_check`, contain the tables `usage`, `reservations`, `session_caps` and `entries` with the columns its version needs, and have `user_version` 0, 1 or 2.
3. **Safety Snapshot**: Overwriting an existing database requires `--force`. First `<db>.pre-restore.bak` (mode `0600`) is made. If that file exists, restore refuses; move it elsewhere first.
4. **One piece**: An existing database is overwritten through SQLite's backup API in a single transaction, so it ends as either the old ledger or the backup. If the target is not a readable database, its bytes are copied to the pre-restore file and the file is replaced.
5. **Forgotten Spend Warning**: Restoring reverts the budget state to the moment of the backup. Calls committed since then are lost from the ledger. With client authentication (Story #33), keys revoked after the backup must be revoked again in `auth.db`.

### Restoring from Backup

#### Storage CLI
```bash
# Requires every process using the ledger to be stopped:
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
