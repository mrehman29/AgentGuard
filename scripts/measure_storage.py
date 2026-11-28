"""Measure ledger throughput for docs/server_database_decision.md.

Runs the product's own reserve and commit against temporary databases. One
"pair" is one reserve plus one commit: two write transactions.

    .venv/bin/python scripts/measure_storage.py NEW_DIRECTORY

The directory must not exist. Put it on the disk whose speed matters (not a
tmpfs). The databases are left there.
"""

import multiprocessing
import os
import platform
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import budget_store
from core.schemas import ModelRates, TokenUsage

RATES = ModelRates(
    provider="gemini", model="measure", input_cost_per_1m_usd=0.3,
    output_cost_per_1m_usd=2.5, usd_to_gbp=0.8,
)
SECONDS = 5.0


def fill(path: Path, rows: int) -> None:
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens, thoughts_tokens,"
        " cost_gbp, created_at) VALUES (?, 1, 1, 0, 1e-9, 1)",
        ((f"old-{i % 1000}",) for i in range(rows)),
    )
    conn.commit()
    conn.close()


def work(session: str, results) -> None:
    tracker = budget_store.SessionBudgetTracker()
    times, failures = [], 0
    end = time.perf_counter() + SECONDS
    while time.perf_counter() < end:
        start = time.perf_counter()
        try:
            held = tracker.reserve(f"{session}-{len(times)}", 1e-6, rates=RATES, ttl_sec=60)
            tracker.commit(held, TokenUsage(prompt_tokens=1, candidates_tokens=1))
        except sqlite3.OperationalError:
            failures += 1
            continue
        times.append(time.perf_counter() - start)
    reads = []
    for _ in range(20):
        start = time.perf_counter()
        tracker.check_budget(session)
        reads.append(time.perf_counter() - start)
    results.put((times, failures, reads))


def measure(rows: int, processes: int, directory: Path) -> str:
    path = directory / f"{rows}-{processes}.db"
    os.environ["SQLITE_DB_PATH"] = str(path)
    os.environ["GLOBAL_BUDGET_CAP_GBP"] = "1000000"
    budget_store.SessionBudgetTracker()
    fill(path, rows)
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=work, args=(f"w{n}", results)) for n in range(processes)]
    for worker in workers:
        worker.start()
    done = [results.get(timeout=120) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)
    times = sorted(t for d in done for t in d[0])
    reads = sorted(t for d in done for t in d[2])
    pairs = len(times) / SECONDS
    return (
        f"| {rows:,} | {processes} | {pairs:,.0f} | {2 * pairs:,.0f} |"
        f" {times[len(times) // 2] * 1000:.1f} | {times[int(len(times) * 0.99)] * 1000:.1f} |"
        f" {reads[len(reads) // 2] * 1000:.1f} | {sum(d[1] for d in done)} |"
    )


if __name__ == "__main__":
    directory = Path(sys.argv[1]).resolve()
    directory.mkdir()
    print(f"{platform.platform()}, Python {platform.python_version()},"
          f" SQLite {sqlite3.sqlite_version}, {os.cpu_count()} CPUs, directory {directory}")
    print("| Usage rows | Processes | Pairs/s | Write tx/s | Pair median ms |"
          " Pair p99 ms | Status read median ms | Failed pairs |")
    print("|---|---|---|---|---|---|---|---|")
    for rows in (0, 10_000, 100_000, 500_000):
        for processes in (1, 4):
            print(measure(rows, processes, directory), flush=True)
