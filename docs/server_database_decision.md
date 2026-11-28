# Architecture Decision: When AgentGuard Needs a Server Database

## Status
Approved (Story #41 / v1.1 Production Storage)

## Context
AgentGuard currently uses an embedded SQLite database in WAL (Write-Ahead Logging) mode, with strict immediate transactions (`BEGIN IMMEDIATE`) for reserve/commit concurrency control and offloaded asynchronous operations via `loop.run_in_executor` on a dedicated `ThreadPoolExecutor(max_workers=8)`. This guarantees that SQLite file locking and disk I/O never block the Python event loop, and ensures storage tasks cannot be starved by CPU-intensive operations (such as scrypt password hashing in Story #33).

This architecture offers substantial benefits:
- Zero operational dependencies or external network round-trips.
- ACID compliance with strict single-writer lock serialization preventing double-spending or race conditions.
- Sub-millisecond reads (concurrent non-blocking reads in WAL mode without acquiring the write lock).
- Simple local backup and migration procedures.

However, as production workloads grow and deployment architectures evolve, architectural boundaries must be clearly defined to know when a server-grade database (such as PostgreSQL) becomes necessary.

## Evaluation & Decision Matrix

A server database is **NOT** needed when:
1. **Single-Host Deployments**: AgentGuard runs on a single container instance, VM, or local workstation where multiple processes/threads communicate with a local SQLite file in WAL mode.
2. **Moderate Throughput**: Aggregate write transaction rates remain below ~1,000 commits per second (easily handled by SQLite on modern SSDs).
3. **Local Storage**: The database resides on local POSIX filesystem storage (not NFS/SMB or distributed virtual network mounts where SQLite file locking is unreliable).

A server database (e.g., PostgreSQL) **IS REQUIRED** when any of the following operational triggers are met:

### 1. Horizontal Multi-Instance Scaling
- **Trigger**: Running AgentGuard across multiple independent nodes or containers (e.g., multi-replica Kubernetes Deployments, multiple Google Cloud Run instances) that must share a unified global spend cap, client caps, and ledger.
- **Why SQLite Fails**: SQLite locks are process-level POSIX advisory locks on a single shared filesystem. Sharing a SQLite database over network filesystems (NFS, CIFS, GlusterFS, EFS) risks database corruption, lock deadlocks, or broken WAL synchronization across independent nodes.
- **Target**: PostgreSQL with connection pooling (e.g., PgBouncer).

### 2. Extreme Concurrent Write Saturation
- **Trigger**: Workloads exceeding sustained 1,500–2,000 write transactions per second.
- **Why SQLite Fails**: In SQLite, while multiple readers can run concurrently in WAL mode without locking writers, there can only ever be **one writer** at a time. High concurrent write demand leads to queueing behind `busy_timeout` (5000ms), ultimately returning `OperationalError: database is locked`.
- **Target**: PostgreSQL row-level locking (`SELECT ... FOR UPDATE`), distributed sequence generators, or partitioned tables.

### 3. Cross-Region High Availability & Instant Failover
- **Trigger**: Requirements for sub-second disaster recovery failover across multiple availability zones or regions with active-active or active-passive replication.
- **Why SQLite Fails**: Embedded SQLite files require disk snapshots or Litestream/replicated streaming. While tools like Litestream provide point-in-time recovery, they do not provide zero-RPO active multi-master consensus.
- **Target**: Managed PostgreSQL (e.g., Google Cloud SQL, AWS RDS Aurora) with multi-AZ replication.

### 4. Advanced Federated Auditing and Analytics
- **Trigger**: Enterprise compliance demands where ledger audit trails must be continuously ingested into central security SIEMs, data warehouses (BigQuery/Snowflake), or partitioned by enterprise organization hierarchies.
- **Target**: Database Change Data Capture (CDC via Debezium/Postgres WAL logical replication).

## Migration Path & Abstraction Boundary
The AgentGuard codebase isolates storage behind the `SessionBudgetTracker` interface:
- Callers interact exclusively with `check_budget`, `reserve`, `commit`, `release`, `record`, and their non-blocking `async` counterparts (`acheck_budget`, `areserve`, etc.).
- The underlying relational model (sessions, reservations, usage aggregates, audit entries) maps directly to PostgreSQL tables.
- When migrating to PostgreSQL, a `PostgresBudgetTracker` implementing the identical tracker contract can be introduced with zero changes to caller code in `app/` and `core/guard_core.py`.
