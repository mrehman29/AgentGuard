"""Budget ledger (Track A): SQLite, shared by every process on the same database.

Spend is reserved before a model call, then committed with the actual usage or
released. Signatures, including get_tracker(), are frozen.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
import logging
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Callable, Iterator, TypeVar
import uuid
import weakref

import anyio

from core.config import get_settings, token_cost_gbp
from core.schemas import (
    BudgetStatus,
    CommitCancelled,
    CommitResult,
    LedgerEntry,
    ModelRates,
    TokenUsage,
)
from core.storage import (
    RetentionStats,
    acquire_ledger_lock,
    apply_retention,
    backup_database,
    migrate_database,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

BUSY_TIMEOUT_MS = 5000
# A reservation left by a process that died stops counting after this many
# of the longer model timeouts. It can still be committed if its call finishes late.
RESERVATION_TTL_TIMEOUTS = 12


def session_key(client_id: str | None, session_id: str) -> str:
    """Compose a session key. Bare key when client_id is None."""
    if client_id is None:
        return session_id
    return f"{client_id}:{session_id}"


def split_session_key(key: str) -> tuple[str | None, str]:
    """Split session key into (client_id, session_id). Returns (None, key) for bare keys."""
    if ":" in key:
        client, sess = key.split(":", 1)
        return client, sess
    return None, key


def to_nano(val: float) -> int:
    """Convert GBP float to integer nano-pounds to eliminate float summation drift."""
    return round(val * 1_000_000_000)


def _check_amount(value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError("amount must be a finite number, zero or more")


async def _wait_out(
    future: asyncio.Future, on_cancel: Callable[[], None] | None = None
) -> asyncio.CancelledError | None:
    """Wait until future is done, however often the caller is cancelled meanwhile.

    Returns the first cancellation, which the caller must raise. The shield stops
    an anyio cancel scope re-delivering it on every turn of the loop; a plain
    Task.cancel() still arrives, and the wait resumes.
    """
    try:
        await asyncio.wait([future])  # never cancels the future
    except asyncio.CancelledError as cancelled:
        if on_cancel is not None:
            on_cancel()
        with anyio.CancelScope(shield=True):
            while not future.done():
                try:
                    await asyncio.wait([future])
                except asyncio.CancelledError:
                    pass  # the first one is raised
        return cancelled
    return None


class SessionBudgetTracker:
    def __init__(self) -> None:
        self._path = get_settings().sqlite_db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Held shared while this tracker exists, so restore can refuse.
        lock_fd = acquire_ledger_lock(self._path, timeout=BUSY_TIMEOUT_MS / 1000)
        weakref.finalize(self, os.close, lock_fd)
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agentguard-db")
        # Reads take no write lock, so they must not queue behind blocked writes.
        self._read_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="agentguard-db-read"
        )
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
                migrate_database(connection)
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

    def _status(self, db: sqlite3.Connection, session_key_str: str) -> BudgetStatus:
        settings = get_settings()
        client_id, _ = split_session_key(session_key_str)

        prompt, candidates, thoughts, spent = db.execute(
            "SELECT COALESCE(SUM(prompt_tokens), 0), COALESCE(SUM(candidates_tokens), 0),"
            " COALESCE(SUM(thoughts_tokens), 0), TOTAL(cost_gbp)"
            " FROM usage WHERE session_id = ? AND cleared = 0",
            (session_key_str,),
        ).fetchone()

        (reserved,) = db.execute(
            "SELECT TOTAL(amount_gbp) FROM reservations WHERE session_id = ? AND expires_at > ?",
            (session_key_str, time.time()),
        ).fetchone()

        (global_spent,) = db.execute("SELECT TOTAL(cost_gbp) FROM usage").fetchone()

        row = db.execute(
            "SELECT cap_gbp FROM session_caps WHERE session_id = ?", (session_key_str,)
        ).fetchone()
        cap = settings.session_budget_cap_gbp
        if row is not None:
            cap = min(cap, row[0])

        # Client-level accounting if client_id is present
        client_spent_gbp: float | None = None
        client_reserved_gbp: float | None = None
        client_cap_gbp: float | None = None

        if client_id is not None:
            (c_spent,) = db.execute(
                "SELECT TOTAL(cost_gbp) FROM usage WHERE client_id = ? AND cleared = 0",
                (client_id,),
            ).fetchone()
            (c_res,) = db.execute(
                "SELECT TOTAL(amount_gbp) FROM reservations WHERE client_id = ? AND expires_at > ?",
                (client_id, time.time()),
            ).fetchone()
            client_spent_gbp = c_spent
            client_reserved_gbp = c_res
            client_cap_gbp = settings.client_budget_cap_gbp

        # Integer comparisons for caps to prevent float rounding errors
        session_remaining = max(0.0, cap - spent - reserved)
        if client_id is not None and client_cap_gbp is not None and client_spent_gbp is not None and client_reserved_gbp is not None:
            client_remaining = max(0.0, client_cap_gbp - client_spent_gbp - client_reserved_gbp)
            remaining_gbp = min(session_remaining, client_remaining)
        else:
            remaining_gbp = session_remaining

        session_exceeded = to_nano(spent) >= to_nano(cap)
        global_exceeded = to_nano(global_spent) >= to_nano(settings.global_budget_cap_gbp)
        client_exceeded = (
            client_id is not None
            and client_cap_gbp is not None
            and client_spent_gbp is not None
            and to_nano(client_spent_gbp) >= to_nano(client_cap_gbp)
        )
        is_exceeded = session_exceeded or global_exceeded or client_exceeded

        return BudgetStatus(
            session_id=session_key_str,
            prompt_tokens=prompt,
            candidates_tokens=candidates,
            thoughts_tokens=thoughts,
            total_tokens=prompt + candidates + thoughts,
            spent_gbp=spent,
            reserved_gbp=reserved,
            cap_gbp=cap,
            remaining_gbp=remaining_gbp,
            global_spent_gbp=global_spent,
            global_cap_gbp=settings.global_budget_cap_gbp,
            is_exceeded=is_exceeded,
            client_id=client_id,
            client_spent_gbp=client_spent_gbp,
            client_reserved_gbp=client_reserved_gbp,
            client_cap_gbp=client_cap_gbp,
        )

    def check_budget(self, session_id: str) -> BudgetStatus:
        with self._transaction("DEFERRED") as db:
            return self._status(db, session_id)

    def reserve(
        self,
        session_id: str,
        estimate_gbp: float,
        *,
        rates: ModelRates,
        ttl_sec: float,
    ) -> str | None:
        """Return a reservation id, or None if a cap would be exceeded.

        rates price the later commit. After ttl_sec the reservation stops counting
        against the caps, but can still be committed.
        """
        _check_amount(estimate_gbp)
        _check_amount(ttl_sec)

        client_id, _ = split_session_key(session_id)

        with self._transaction() as db:
            now = time.time()
            status = self._status(db, session_id)
            (global_reserved,) = db.execute(
                "SELECT TOTAL(amount_gbp) FROM reservations WHERE expires_at > ?", (now,)
            ).fetchone()

            # Integer nano-pound comparison prevents float drift
            est_nano = to_nano(estimate_gbp)
            sess_overshoot = (
                to_nano(status.spent_gbp) + to_nano(status.reserved_gbp) + est_nano
                > to_nano(status.cap_gbp)
            )
            glob_overshoot = (
                to_nano(status.global_spent_gbp) + to_nano(global_reserved) + est_nano
                > to_nano(status.global_cap_gbp)
            )
            client_overshoot = False
            if (
                client_id is not None
                and status.client_cap_gbp is not None
                and status.client_spent_gbp is not None
                and status.client_reserved_gbp is not None
            ):
                client_overshoot = (
                    to_nano(status.client_spent_gbp) + to_nano(status.client_reserved_gbp) + est_nano
                    > to_nano(status.client_cap_gbp)
                )

            if status.is_exceeded or sess_overshoot or glob_overshoot or client_overshoot:
                return None

            reservation_id = uuid.uuid4().hex
            db.execute(
                "INSERT INTO reservations (id, session_id, amount_gbp, expires_at, client_id, "
                "input_cost_per_1m_usd, output_cost_per_1m_usd, usd_to_gbp) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    reservation_id,
                    session_id,
                    estimate_gbp,
                    now + ttl_sec,
                    client_id,
                    rates.input_cost_per_1m_usd,
                    rates.output_cost_per_1m_usd,
                    rates.usd_to_gbp,
                ),
            )
            return reservation_id

    def commit(self, reservation_id: str, usage: TokenUsage) -> CommitResult:
        """Replace a reservation with the actual cost at its stored rates.

        Unknown id: KeyError. A database error is retried once, so a billed call
        is not lost to it.
        """
        try:
            return self._commit(reservation_id, usage)
        except sqlite3.Error:
            logger.warning("commit failed; retrying once")
            return self._commit(reservation_id, usage)

    def _commit(self, reservation_id: str, usage: TokenUsage) -> CommitResult:
        with self._transaction() as db:
            row = db.execute(
                "SELECT session_id, client_id, amount_gbp, input_cost_per_1m_usd, "
                "output_cost_per_1m_usd, usd_to_gbp FROM reservations WHERE id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise KeyError("unknown or already settled reservation")

            session_id, client_id, amount_gbp, in_usd, out_usd, usd_gbp = row

            if (in_usd, out_usd, usd_gbp) != (None, None, None):
                stored_rates = ModelRates(
                    provider="",
                    model="",
                    input_cost_per_1m_usd=in_usd,
                    output_cost_per_1m_usd=out_usd,
                    usd_to_gbp=usd_gbp,
                )
                cost = token_cost_gbp(
                    stored_rates, usage.prompt_tokens, usage.candidates_tokens + usage.thoughts_tokens
                )
            else:
                # NULL rates: made before schema v2. Charge what was reserved.
                cost = amount_gbp

            now = time.time()
            db.execute(
                "INSERT INTO usage (session_id, client_id, prompt_tokens, candidates_tokens, "
                "thoughts_tokens, cost_gbp, cleared, created_at) VALUES (?, ?, ?, ?, ?, ?, 0, ?)",
                (
                    session_id,
                    client_id,
                    usage.prompt_tokens,
                    usage.candidates_tokens,
                    usage.thoughts_tokens,
                    cost,
                    now,
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

        For a client's session this also lowers the client's spend, so only an
        administrator may call it for another party (enforced by the routes, #33).
        Calls still in flight keep their reservations and are charged when they end.
        """
        with self._transaction() as db:
            db.execute("UPDATE usage SET cleared = 1 WHERE session_id = ?", (session_id,))
            db.execute("DELETE FROM session_caps WHERE session_id = ?", (session_id,))
            return self._status(db, session_id)

    def reset_client(self, client_id: str) -> None:
        """Clear client's spend, summary rows and session caps; keep live reservations.

        Never alters global spend.
        """
        with self._transaction() as db:
            db.execute("UPDATE usage SET cleared = 1 WHERE client_id = ?", (client_id,))
            # An exact prefix: LIKE would treat _ and % as wildcards and ignore case.
            prefix = f"{client_id}:"
            db.execute(
                "DELETE FROM session_caps WHERE substr(session_id, 1, ?) = ?",
                (len(prefix), prefix),
            )

    def record(self, entry: LedgerEntry) -> None:
        client_id, _ = split_session_key(entry.session_id)
        created_at = entry.time.timestamp()
        with self._transaction() as db:
            db.execute(
                "INSERT INTO entries (entry, created_at, client_id) VALUES (?, ?, ?)",
                (entry.model_dump_json(), created_at, client_id),
            )

    def recent_entries(self, limit: int = 50, client_id: str | None = None) -> list[LedgerEntry]:
        """Oldest first. If client_id is given, returns only that client's entries."""
        if limit <= 0:
            return []
        with self._transaction("DEFERRED") as db:
            if client_id is None:
                rows = db.execute(
                    "SELECT entry FROM entries ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT entry FROM entries WHERE client_id = ? ORDER BY id DESC LIMIT ?",
                    (client_id, limit),
                ).fetchall()
        return [LedgerEntry.model_validate_json(row[0]) for row in reversed(rows)]

    def apply_retention(
        self, retention_days: int, *, stop: threading.Event | None = None
    ) -> RetentionStats:
        """Prune in short transactions. Setting stop ends it before the next one."""
        connection = self._connect()
        try:
            return apply_retention(connection, retention_days, stop=stop)
        finally:
            connection.close()

    def backup(self, dest_backup_path: Path | str) -> Path:
        return backup_database(self._path, dest_backup_path)

    # Async forms. SQLite never runs on the event loop. A write that has started
    # is always waited for, and a cancellation is raised only after it ends.

    async def _write_out(
        self,
        func: Callable[..., T],
        *args: Any,
        on_cancel: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> tuple[T, asyncio.CancelledError | None]:
        """Run a write to its end. Returns its result and any cancellation received.

        A failed write raises its error, or the cancellation if there was one.
        """
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, partial(func, *args, **kwargs))
        cancelled = await _wait_out(future, on_cancel)
        if cancelled is not None and (error := future.exception()) is not None:
            logger.error("ledger write failed in a cancelled call: %s", type(error).__name__)
            raise cancelled from error
        return future.result(), cancelled

    async def _write(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        result, cancelled = await self._write_out(func, *args, **kwargs)
        if cancelled is not None:
            raise cancelled
        return result

    async def _read(self, func: Callable[..., T], *args: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._read_executor, partial(func, *args))

    async def areserve(
        self,
        session_id: str,
        estimate_gbp: float,
        *,
        rates: ModelRates,
        ttl_sec: float,
    ) -> str | None:
        """Reserve. A cancelled call releases what it reserved before raising."""
        reservation_id, cancelled = await self._write_out(
            self.reserve, session_id, estimate_gbp, rates=rates, ttl_sec=ttl_sec
        )
        if cancelled is None:
            return reservation_id
        if reservation_id is not None:
            try:
                await self.arelease(reservation_id)
            except asyncio.CancelledError:
                pass  # released or logged; the first cancellation is raised
            except Exception:
                logger.exception("could not release a reservation; it will expire")
        raise cancelled

    async def acommit(self, reservation_id: str, usage: TokenUsage) -> CommitResult:
        """Commit. If cancelled meanwhile, raises CommitCancelled carrying the result."""
        result, cancelled = await self._write_out(self.commit, reservation_id, usage)
        if cancelled is not None:
            # The same message: an anyio cancel scope recognises its own by it.
            raise CommitCancelled(result, *cancelled.args) from cancelled
        return result

    async def arelease(self, reservation_id: str) -> None:
        await self._write(self.release, reservation_id)

    async def acheck_budget(self, session_id: str) -> BudgetStatus:
        return await self._read(self.check_budget, session_id)

    async def aset_session_cap(self, session_id: str, cap_gbp: float) -> BudgetStatus:
        return await self._write(self.set_session_cap, session_id, cap_gbp)

    async def areset(self, session_id: str) -> BudgetStatus:
        return await self._write(self.reset, session_id)

    async def areset_client(self, client_id: str) -> None:
        await self._write(self.reset_client, client_id)

    async def arecord(self, entry: LedgerEntry) -> None:
        await self._write(self.record, entry)

    async def arecent_entries(
        self, limit: int = 50, client_id: str | None = None
    ) -> list[LedgerEntry]:
        return await self._read(self.recent_entries, limit, client_id)

    async def aapply_retention(self, retention_days: int) -> RetentionStats:
        """Prune. A cancelled call stops it after the transaction in progress."""
        stop = threading.Event()
        result, cancelled = await self._write_out(
            self.apply_retention, retention_days, on_cancel=stop.set, stop=stop
        )
        if cancelled is not None:
            raise cancelled
        return result

    async def abackup(self, dest_backup_path: Path | str) -> Path:
        return await self._write(self.backup, dest_backup_path)


_tracker: SessionBudgetTracker | None = None
_tracker_lock = threading.Lock()


def get_tracker() -> SessionBudgetTracker:
    """Process-wide tracker. The first call opens and migrates the database."""
    global _tracker
    if _tracker is None:
        with _tracker_lock:
            if _tracker is None:
                _tracker = SessionBudgetTracker()
    return _tracker
