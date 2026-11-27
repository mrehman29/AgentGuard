"""Budget ledger (Track A): SQLite, shared by every process on the same database.

Spend is reserved before a model call, then committed with the actual usage or
released. Signatures, including get_tracker(), are frozen.
"""

import logging
import math
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

from core.config import get_settings, token_cost_gbp
from core.schemas import BudgetStatus, CommitResult, LedgerEntry, ModelRates, TokenUsage

logger = logging.getLogger(__name__)

BUSY_TIMEOUT_MS = 5000
# A reservation left by a process that died stops counting after this many
# of the longer model timeouts. It can still be committed if its call finishes late.
RESERVATION_TTL_TIMEOUTS = 12

_SCHEMA = (
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


def _check_amount(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("amount must be a finite number, zero or more")


class SessionBudgetTracker:
    def __init__(self) -> None:
        self._path = get_settings().sqlite_db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None
        )
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        return connection

    def _create_schema(self) -> None:
        # Switching to WAL can report "locked" at once, ignoring busy_timeout,
        # when two processes open a new database together.
        deadline = time.monotonic() + BUSY_TIMEOUT_MS / 1000
        while True:
            connection = self._connect()
            try:
                connection.execute("PRAGMA journal_mode=WAL")
                for statement in _SCHEMA:
                    connection.execute(statement)
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
            finally:
                connection.close()

    @contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Connection]:
        """IMMEDIATE takes the write lock at once; DEFERRED gives a read snapshot."""
        connection = self._connect()
        try:
            connection.execute(f"BEGIN {mode}")
            yield connection
            connection.execute("COMMIT")
        finally:
            connection.close()  # rolls back anything not committed

    def _status(self, db: sqlite3.Connection, session_id: str) -> BudgetStatus:
        settings = get_settings()
        prompt, candidates, thoughts, spent = db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidates_tokens), 0),"
            " COALESCE(SUM(thoughts_tokens), 0), TOTAL(cost_gbp)"
            " FROM usage WHERE session_id = ? AND cleared = 0",
            (session_id,),
        ).fetchone()
        (reserved,) = db.execute(
            "SELECT TOTAL(amount_gbp) FROM reservations WHERE session_id = ? AND expires_at > ?",
            (session_id, time.time()),
        ).fetchone()
        (global_spent,) = db.execute("SELECT TOTAL(cost_gbp) FROM usage").fetchone()
        row = db.execute(
            "SELECT cap_gbp FROM session_caps WHERE session_id = ?", (session_id,)
        ).fetchone()
        cap = settings.session_budget_cap_gbp
        if row is not None:
            cap = min(cap, row[0])
        return BudgetStatus(
            session_id=session_id,
            prompt_tokens=prompt,
            candidates_tokens=candidates,
            thoughts_tokens=thoughts,
            total_tokens=prompt + candidates + thoughts,
            spent_gbp=spent,
            reserved_gbp=reserved,
            cap_gbp=cap,
            remaining_gbp=max(0.0, cap - spent - reserved),
            global_spent_gbp=global_spent,
            global_cap_gbp=settings.global_budget_cap_gbp,
            is_exceeded=spent >= cap or global_spent >= settings.global_budget_cap_gbp,
        )

    def check_budget(self, session_id: str) -> BudgetStatus:
        with self._transaction("DEFERRED") as db:
            return self._status(db, session_id)

    def reserve(self, session_id: str, estimate_gbp: float) -> str | None:
        """Return a reservation id, or None if a cap would be exceeded."""
        _check_amount(estimate_gbp)
        settings = get_settings()
        timeout = max(settings.classifier.timeout_sec, settings.agent.timeout_sec)
        ttl = timeout * RESERVATION_TTL_TIMEOUTS
        with self._transaction() as db:
            now = time.time()
            status = self._status(db, session_id)
            (global_reserved,) = db.execute(
                "SELECT TOTAL(amount_gbp) FROM reservations WHERE expires_at > ?", (now,)
            ).fetchone()
            if (
                status.is_exceeded
                or status.spent_gbp + status.reserved_gbp + estimate_gbp > status.cap_gbp
                or status.global_spent_gbp + global_reserved + estimate_gbp > status.global_cap_gbp
            ):
                return None
            reservation_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO reservations (id, session_id, amount_gbp, expires_at)"
                " VALUES (?, ?, ?, ?)",
                (reservation_id, session_id, estimate_gbp, now + ttl),
            )
            return reservation_id

    def commit(self, reservation_id: str, usage: TokenUsage, rates: ModelRates) -> CommitResult:
        """Replace a reservation with the actual cost. Unknown id: KeyError.

        rates are those of the model called.
        A database error is retried once, so a billed call is not lost to it.
        """
        try:
            return self._commit(reservation_id, usage, rates)
        except sqlite3.Error:
            logger.warning("commit failed; retrying once")
            return self._commit(reservation_id, usage, rates)

    def _commit(self, reservation_id: str, usage: TokenUsage, rates: ModelRates) -> CommitResult:
        cost = token_cost_gbp(
            rates, usage.prompt_tokens, usage.candidates_tokens + usage.thoughts_tokens
        )
        with self._transaction() as db:
            row = db.execute(
                "SELECT session_id FROM reservations WHERE id = ?", (reservation_id,)
            ).fetchone()
            if row is None:
                raise KeyError("unknown or already settled reservation")
            session_id = row[0]
            db.execute(
                "INSERT INTO usage (session_id, prompt_tokens, candidates_tokens,"
                " thoughts_tokens, cost_gbp) VALUES (?, ?, ?, ?, ?)",
                (
                    session_id,
                    usage.prompt_tokens,
                    usage.candidates_tokens,
                    usage.thoughts_tokens,
                    cost,
                ),
            )
            db.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))
            budget = self._status(db, session_id)
        return CommitResult(cost_gbp=cost, budget=budget)

    def release(self, reservation_id: str) -> None:
        with self._transaction() as db:
            db.execute("DELETE FROM reservations WHERE id = ?", (reservation_id,))

    def set_session_cap(self, session_id: str, cap_gbp: float) -> BudgetStatus:
        """Can only lower the cap."""
        _check_amount(cap_gbp)
        with self._transaction() as db:
            cap = min(cap_gbp, self._status(db, session_id).cap_gbp)
            db.execute(
                "INSERT OR REPLACE INTO session_caps (session_id, cap_gbp) VALUES (?, ?)",
                (session_id, cap),
            )
            return self._status(db, session_id)

    def reset(self, session_id: str) -> BudgetStatus:
        """Clear session spend and restore the default cap. Never resets global.

        Calls still in flight keep their reservations and are charged when they end.
        """
        with self._transaction() as db:
            db.execute("UPDATE usage SET cleared = 1 WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM session_caps WHERE session_id = ?", (session_id,))
            return self._status(db, session_id)

    def record(self, entry: LedgerEntry) -> None:
        with self._transaction() as db:
            db.execute("INSERT INTO entries (entry) VALUES (?)", (entry.model_dump_json(),))

    def recent_entries(self, limit: int = 50) -> list[LedgerEntry]:
        """Oldest first."""
        if limit <= 0:
            return []
        with self._transaction("DEFERRED") as db:
            rows = db.execute(
                "SELECT entry FROM entries ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [LedgerEntry.model_validate_json(row[0]) for row in reversed(rows)]


_tracker: SessionBudgetTracker | None = None


def get_tracker() -> SessionBudgetTracker:
    """Process-wide tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SessionBudgetTracker()
    return _tracker
