"""Shared by the storage tests (#41)."""

import asyncio
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from core import budget_store, storage
from core.schemas import ModelRates, TokenUsage

RATES = ModelRates(
    provider="gemini",
    model="gemini-3.5-flash-lite",
    input_cost_per_1m_usd=0.30,
    output_cost_per_1m_usd=2.50,
    usd_to_gbp=0.80,
)
USAGE = TokenUsage(prompt_tokens=400, candidates_tokens=60, thoughts_tokens=20)
V1_DUMP = Path(__file__).parent / "data" / "ledger_v1.sql"


def use_database(monkeypatch, path: Path) -> None:
    monkeypatch.setenv("SQLITE_DB_PATH", str(path))
    monkeypatch.setattr(budget_store, "_tracker", None)


def reserve(tracker, session_id: str, amount: float, ttl_sec: float = 60.0) -> str | None:
    return tracker.reserve(session_id, amount, rates=RATES, ttl_sec=ttl_sec)


def create_v1_database(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(V1_DUMP.read_text())
    conn.close()
    return path


def create_ledger(path: Path, session_id: str, rows: int = 1) -> Path:
    """A current ledger holding `rows` usage rows of one session."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    storage.migrate_database(conn)
    conn.execute("BEGIN")
    conn.executemany(
        "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens, thoughts_tokens,"
        " cost_gbp) VALUES (?, 1, 1, 0, 0.000001)",
        [(session_id,)] * rows,
    )
    conn.execute("COMMIT")
    conn.close()
    return path


def dump(path: Path) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return list(conn.iterdump())
    finally:
        conn.close()


def query(path: Path, sql: str, args: tuple = ()) -> list[tuple]:
    conn = sqlite3.connect(path)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


@contextmanager
def write_lock(path: Path):
    """Hold the database's write lock, as another busy process would."""
    conn = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    finally:
        conn.execute("ROLLBACK")
        conn.close()


class Gate:
    """Makes one tracker method wait in its thread until opened."""

    def __init__(self, monkeypatch, tracker, name: str, *, error: Exception | None = None) -> None:
        self.entered = threading.Event()
        self.open = threading.Event()
        real = getattr(tracker, name)

        def gated(*args, **kwargs):
            self.entered.set()
            assert self.open.wait(10), f"{name} was never released"
            if error is not None:
                raise error
            return real(*args, **kwargs)

        monkeypatch.setattr(tracker, name, gated)

    async def wait_entered(self) -> None:
        async with asyncio.timeout(10):
            while not self.entered.is_set():
                await asyncio.sleep(0.005)
