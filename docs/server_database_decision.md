# Architecture Decision: When AgentGuard Needs a Server Database

## Status
Approved (Story #41 / v1.1 Production Storage)

## Decision
AgentGuard stays on SQLite in WAL mode for a single host. Move to a server
database (PostgreSQL) when any trigger below applies.

Ledger calls made while serving requests run on thread pools (8 writer threads,
4 reader threads), not on the event loop. Writes are serialised with
`BEGIN IMMEDIATE`; reads take no write lock.

## Measured throughput

### Method
`scripts/measure_storage.py NEW_DIRECTORY` runs the product's own `reserve` and
`commit` against temporary databases for 5 seconds per row. A **pair** is one
reserve plus one commit: two write transactions. Each pair uses a new session id.
The old usage rows are spread over 1,000 other sessions. One model call in the
product also records a ledger entry (a third write), and the guard makes extra
status reads, so a request costs more than a pair.

Measured on 2026-11-28: Intel i7-6700 (8 threads), Linux 7.0, Python 3.11.16,
SQLite 3.53.1. One run of each row; expect variation between runs.

### Disk (ext4)

| Usage rows | Processes | Pairs/s | Write tx/s | Pair median ms | Pair p99 ms | Status read median ms | Failed pairs |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 73 | 146 | 13.0 | 21.7 | 1.9 | 0 |
| 0 | 4 | 78 | 156 | 12.4 | 1045.5 | 0.9 | 0 |
| 10,000 | 1 | 66 | 131 | 14.2 | 26.1 | 1.4 | 0 |
| 10,000 | 4 | 141 | 281 | 7.0 | 937.3 | 1.6 | 0 |
| 100,000 | 1 | 38 | 76 | 26.2 | 42.9 | 7.7 | 0 |
| 100,000 | 4 | 53 | 106 | 18.6 | 2450.9 | 8.1 | 0 |
| 500,000 | 1 | 11 | 22 | 89.4 | 171.6 | 29.5 | 0 |
| 500,000 | 4 | 6 | 12 | 180.7 | 3707.6 | 33.2 | 2 |

### RAM disk (tmpfs): the cost without disk syncs

| Usage rows | Processes | Pairs/s | Write tx/s | Pair median ms | Pair p99 ms | Status read median ms | Failed pairs |
|---|---|---|---|---|---|---|---|
| 0 | 1 | 584 | 1,168 | 1.7 | 2.9 | 0.8 | 0 |
| 0 | 4 | 652 | 1,305 | 1.5 | 105.3 | 0.8 | 0 |
| 10,000 | 1 | 355 | 711 | 2.6 | 4.9 | 1.2 | 0 |
| 10,000 | 4 | 393 | 785 | 2.4 | 106.3 | 1.4 | 0 |
| 100,000 | 1 | 67 | 134 | 14.5 | 21.2 | 7.3 | 0 |
| 100,000 | 4 | 67 | 135 | 14.4 | 1752.3 | 7.7 | 0 |
| 500,000 | 1 | 16 | 33 | 58.2 | 101.9 | 29.0 | 0 |
| 500,000 | 4 | 17 | 35 | 58.0 | 5098.3 | 32.1 | 0 |

### What the figures show
- On this disk an empty ledger sustained about **150 write transactions per
  second** (about 75 pairs). Each commit is synced to disk, so faster or slower
  storage changes this figure.
- **Throughput falls as `usage` grows.** Every status check, including the one
  inside `reserve`'s write transaction, adds up the whole `usage` table for the
  global cap. At 500,000 rows throughput was 12 to 22 write transactions per
  second on disk.
- With 4 processes, waiting is uneven: the 99th percentile on disk was about 1
  second even on a small ledger and 2.5 to 3.7 seconds at 100,000 rows and above.
  At 500,000 rows on disk 2 pairs failed on the 5 second `busy_timeout` (a failed
  call is blocked: fail closed).
- Single runs vary: on disk, 4 processes at 10,000 rows measured faster than at 0.
- Retention is **off by default**. Pruning (`LEDGER_RETENTION_DAYS`, see the
  backup guide) replaces old rows with one summary row per session and so bounds
  the table. It runs as many short transactions with a pause after each, so calls
  continue while it does.

Not measured: network or cloud disks, other machines, more than 4 processes, and
a full request through the web server at saturation.

## Triggers for a server database

1. **More than one host or container** must share the caps and ledger. SQLite
   locking needs one local filesystem; network filesystems (NFS, SMB and similar)
   are not supported.
2. **Write load near what was measured.** Plan to move, or to replace the global
   total with a maintained counter, if the sustained need approaches half of the
   measured rate for your disk and row count (on the machine above, about 70 write
   transactions per second on an empty ledger), or if `database is locked` errors
   appear. This margin is a judgement, not a measurement.
3. **High availability** across zones or regions with automatic failover.
4. **Continuous audit export** (change data capture) to external systems.

## Migration boundary
Callers use `SessionBudgetTracker` (`check_budget`, `reserve`, `commit`,
`release`, `record`, `recent_entries`, resets and their `async` forms). A server
database tracker would implement the same methods. Backup, restore, retention and
migration in `core/storage.py` are SQLite specific and would be replaced.
