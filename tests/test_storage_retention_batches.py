"""Retention works in short transactions (#41): bounded, exact, interruptible, shared.

No test here depends on the speed of the disk: work is counted in SQLite's own
instructions, and interleavings are placed in the gap between two transactions.
"""

import asyncio
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager

import pytest

from core import budget_store, storage
from core.config import PROJECT_ROOT
from storage_helpers import USAGE, query, reserve, use_database

DAYS = 5
PAUSE = getattr(storage, "RETENTION_PAUSE_SEC", 0)  # the product's own value
OLD_USAGE = "SELECT COUNT(*) FROM usage WHERE summary = 0 AND created_at > 0 AND created_at < ?"


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for name in ("SESSION_BUDGET_CAP_GBP", "GLOBAL_BUDGET_CAP_GBP", "CLIENT_BUDGET_CAP_GBP",
                 "LEDGER_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    small(monkeypatch, batch=3, pause=0)


def small(monkeypatch, *, batch=None, pause=None):
    if batch is not None:
        monkeypatch.setattr(storage, "RETENTION_BATCH_ROWS", batch, raising=False)
    if pause is not None:
        monkeypatch.setattr(storage, "RETENTION_PAUSE_SEC", pause, raising=False)


def between_transactions(monkeypatch, action):
    """Run action(stop) each time a prune has let go of the lock and means to go on."""
    monkeypatch.setattr(storage, "_let_writers_in", action, raising=False)


def connect(path, factory=sqlite3.Connection):
    return sqlite3.connect(path, timeout=10, isolation_level=None, factory=factory)


def prune(path, **kwargs):
    conn = connect(path)
    try:
        return storage.apply_retention(conn, DAYS, **kwargs)
    finally:
        conn.close()


def fill(path, usage, *, summaries=(), old_entries=0):
    """usage: (session_id, client_id, cleared, count, age_days); summaries: (session_id, client_id, cleared)."""
    conn = connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    storage.migrate_database(conn)
    now = time.time()
    conn.execute("BEGIN")
    insert = (
        "INSERT INTO usage (session_id, client_id, cleared, prompt_tokens, candidates_tokens,"
        " thoughts_tokens, cost_gbp, created_at, summary) VALUES (?, ?, ?, 3, 2, 1, ?, ?, ?)"
    )
    for session, client, cleared, count, age in usage:
        conn.executemany(insert, (
            (session, client, cleared, 0.0000137 * (1 + n % 7), now - age * 86400, 0)
            for n in range(count)
        ))
    conn.executemany(insert, ((s, c, cleared, 0.0000091, 0.0, 1) for s, c, cleared in summaries))
    conn.executemany(
        "INSERT INTO entries (entry, created_at) VALUES ('{}', ?)",
        [(now - 10 * 86400,)] * old_entries,
    )
    conn.execute("COMMIT")
    conn.close()
    return path


def mixed(path):
    """Several groups, one larger than a batch, with recent, legacy and doubled summary rows."""
    return fill(
        path,
        [("big", None, 0, 11, 10), ("big", None, 1, 4, 10), ("big", None, 0, 2, 0),
         ("k:a", "k", 0, 5, 10), ("k:b", "k", 0, 2, 10), ("new", None, 0, 2, 0)],
        summaries=[("k:a", "k", 0), *[("twice", None, 1)] * 7, *[("k:c", "k", 0)] * 2],
        old_entries=5,
    )


def totals(path):
    """Everything a cap is computed from: per group, so also per session, client and overall."""
    groups = query(
        path,
        "SELECT session_id, client_id, cleared, SUM(prompt_tokens), SUM(candidates_tokens),"
        " SUM(thoughts_tokens), TOTAL(cost_gbp) FROM usage"
        " GROUP BY session_id, client_id, cleared ORDER BY 1, 2, 3",
    )
    overall = query(path, "SELECT TOTAL(cost_gbp) FROM usage")[0][0]
    return [(*g[:6], budget_store.to_nano(g[6])) for g in groups], budget_store.to_nano(overall)


def old_rows(path):
    return query(path, OLD_USAGE, (time.time() - DAYS * 86400,))[0][0]


def assert_finished(path):
    assert old_rows(path) == 0
    assert query(path, "SELECT COUNT(*) FROM entries WHERE created_at > 0 AND created_at < ?",
                 (time.time() - DAYS * 86400,)) == [(0,)]
    assert query(
        path,
        "SELECT session_id FROM usage WHERE summary = 1"
        " GROUP BY session_id, client_id, cleared HAVING COUNT(*) > 1",
    ) == []


# --- How long one transaction holds the write lock ---


def work_per_transaction(path):
    """(SQLite instructions, rows changed) inside each write transaction of one prune."""
    conn = connect(path)
    done, started, result = [0], [0, 0], []

    def count():
        done[0] += 10
        return 0

    def trace(sql):
        if sql.startswith("BEGIN"):
            started[:] = done[0], conn.total_changes
        elif sql.startswith("COMMIT"):
            result.append((done[0] - started[0], conn.total_changes - started[1]))

    conn.set_progress_handler(count, 10)
    conn.set_trace_callback(trace)
    try:
        storage.apply_retention(conn, DAYS)
    finally:
        conn.close()
    return result


def ledger_of(shape, n, path):
    if shape == "many sessions":
        names = [f"s{i}" for i in range(n)]
        return fill(path, [(s, None, 0, 1, 10) for s in names],
                    summaries=[(s, None, 0) for s in names], old_entries=n)
    if shape == "one session":
        return fill(path, [("s", None, 0, n, 10), ("s", None, 0, n, 0)], old_entries=n)
    if shape == "one client":
        return fill(path, [(f"k:{i}", "k", 0, 2, 10) for i in range(n // 2)]
                    + [("k:recent", "k", 0, n, 0)])
    return fill(path, [], summaries=[(f"s{i}", None, 0) for i in range(n)] * 2)  # doubled


@pytest.mark.parametrize("shape", ["many sessions", "one session", "one client", "doubled summaries"])
def test_a_transaction_does_not_grow_with_the_ledger(tmp_path, monkeypatch, shape):
    batch = 50
    small(monkeypatch, batch=batch)
    largest = []
    for n in (500, 4000):
        path = ledger_of(shape, n, tmp_path / f"{n}.db")
        before = totals(path)
        transactions = work_per_transaction(path)
        assert totals(path) == before
        assert_finished(path)
        assert len(transactions) >= n // batch
        # A batch changes at most its entries, its rows and one summary for each.
        assert max(changed for _, changed in transactions) <= 3 * batch
        largest.append(max(work for work, _ in transactions))
    assert largest[1] <= 1.5 * largest[0], largest


def test_the_whole_prune_grows_with_the_rows_not_their_square(tmp_path, monkeypatch):
    small(monkeypatch, batch=500)
    per_row = []
    for n in (2000, 16000):
        path = ledger_of("many sessions", n, tmp_path / f"{n}.db")
        per_row.append(sum(work for work, _ in work_per_transaction(path)) / n)
    assert per_row[1] <= 1.5 * per_row[0], per_row


class StatementLog(sqlite3.Connection):
    """Keeps each statement with its parameters: the plan can depend on them."""

    def execute(self, sql, parameters=()):
        self.statements = [*getattr(self, "statements", []), (sql, parameters)]
        return super().execute(sql, parameters)


def test_every_retention_statement_has_a_bounded_plan(tmp_path):
    path = mixed(tmp_path / "ledger.db")
    conn = connect(path, StatementLog)
    try:
        storage.apply_retention(conn, DAYS)
        plans = {}
        for sql, parameters in list(conn.statements):
            if sql.startswith(("BEGIN", "COMMIT", "PRAGMA")):
                continue
            rows = sqlite3.Connection.execute(conn, f"EXPLAIN QUERY PLAN {sql}", parameters)
            plans.setdefault(" ".join(sql.split()[:3]), set()).add(tuple(r[3] for r in rows))
    finally:
        conn.close()
    by_rowid = ("SEARCH usage USING INTEGER PRIMARY KEY (rowid=?)",)
    # Every statement apply_retention runs. A new one must be added with its plan.
    assert plans == {
        "DELETE FROM entries": {(
            "SEARCH entries USING INTEGER PRIMARY KEY (rowid=?)", "LIST SUBQUERY 1",
            "SEARCH entries USING COVERING INDEX entries_created_at (created_at>? AND created_at<?)",
        )},
        # The old rows, found through the age index and cut off by LIMIT.
        "SELECT session_id, client_id,": {
            ("SEARCH usage USING INDEX usage_created_at (created_at>? AND created_at<?)",),
            # Groups with two summaries: reads every summary, outside any transaction.
            ("SCAN usage USING COVERING INDEX usage_summary",),
        },
        "SELECT id, prompt_tokens,": {
            ("SEARCH usage USING INDEX usage_summary (session_id=? AND cleared=? AND client_id=?)",),
        },
        "UPDATE usage SET": {by_rowid},
        "INSERT INTO usage": {()},
        "DELETE FROM usage": {by_rowid},
    }


# --- The gap between transactions ---


def test_every_committed_batch_leaves_all_totals_exact(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    use_database(monkeypatch, path)
    tracker = budget_store.get_tracker()
    keys = ["big", "k:a", "k:new", "twice", "new"]
    status = lambda: [
        (s.total_tokens, *(None if v is None else budget_store.to_nano(v)
                           for v in (s.spent_gbp, s.client_spent_gbp, s.global_spent_gbp)))
        for s in map(tracker.check_budget, keys)
    ]
    expected, caps, remaining = totals(path), status(), []

    def check(stop):
        connection = sqlite3.connect(path, timeout=0, isolation_level=None)
        connection.execute("BEGIN IMMEDIATE")  # nothing holds the write lock
        connection.execute("ROLLBACK")
        connection.close()
        assert totals(path) == expected and status() == caps
        remaining.append(old_rows(path))

    between_transactions(monkeypatch, check)
    stats = tracker.apply_retention(DAYS)
    assert remaining[:7] == [19, 16, 13, 10, 7, 4, 1]  # 22 old rows, 3 at a time
    assert len(remaining) > 7  # and the doubled summaries after them
    assert (stats.usage_pruned, stats.entries_pruned) == (22, 5)
    assert totals(path) == expected and status() == caps
    assert_finished(path)


def test_calls_and_a_reset_between_batches_are_counted(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    use_database(monkeypatch, path)
    tracker = budget_store.get_tracker()
    spent = tracker.check_budget("x").global_spent_gbp
    costs = []

    def work(stop):
        costs.append(tracker.commit(reserve(tracker, "big", 0.00001), USAGE).cost_gbp)
        if len(costs) == 2:
            tracker.reset("big")  # half of its old rows are summarised, half are not

    between_transactions(monkeypatch, work)
    tracker.apply_retention(DAYS)
    assert len(costs) > 2
    after = tracker.check_budget("big")
    assert budget_store.to_nano(after.global_spent_gbp) == budget_store.to_nano(spent + sum(costs))
    assert budget_store.to_nano(after.spent_gbp) == budget_store.to_nano(sum(costs[2:]))
    assert_finished(path)


def test_summaries_doubled_by_resets_are_merged_without_old_rows(tmp_path):
    path = fill(tmp_path / "ledger.db", [], summaries=[("s", None, 1)] * 8 + [("k:s", "k", 0)] * 2)
    before = totals(path)
    prune(path)
    assert query(path, "SELECT COUNT(*) FROM usage") == [(2,)]
    assert totals(path) == before


def test_another_prune_may_run_between_any_two_transactions(tmp_path, monkeypatch):
    for turn in range(30):
        path = mixed(tmp_path / f"{turn}.db")
        expected, pauses, other = totals(path), [], []

        def overlap(stop):
            pauses.append(None)
            if len(pauses) == turn + 1:
                between_transactions(monkeypatch, lambda stop: None)
                other.append(prune(path))

        between_transactions(monkeypatch, overlap)
        first = prune(path)
        assert totals(path) == expected
        assert_finished(path)
        if not other:
            break
        assert first.usage_pruned + other[0].usage_pruned == 22
        assert first.entries_pruned + other[0].entries_pruned == 5
    assert turn > 7  # reached the merging of doubled summaries


def test_a_backup_between_any_two_transactions_is_exact(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    expected, copies = totals(path), []

    def copy(stop):
        copies.append(storage.backup_database(path, tmp_path / f"copy{len(copies)}.bak"))

    between_transactions(monkeypatch, copy)
    prune(path)
    assert len(copies) > 7
    for backup in copies:
        assert query(backup, "PRAGMA integrity_check") == [("ok",)]
        assert totals(backup) == expected
    prune(copies[0])  # a copy taken half way can be restored and finished
    assert_finished(copies[0])


@pytest.mark.parametrize("writers", [1, 4])
def test_a_waiting_writer_gets_the_lock_between_batches(tmp_path, monkeypatch, writers):
    """SQLite's waiters sleep between tries; without a long enough gap they starve."""
    assert PAUSE > 0.1  # the longest of those sleeps
    batches = 16
    path = fill(tmp_path / "ledger.db", [("s", None, 0, batches, 10)])
    use_database(monkeypatch, path)
    tracker = budget_store.get_tracker()
    small(monkeypatch, batch=1, pause=PAUSE)
    holding, started = threading.Event(), []
    real = getattr(storage, "_prune_usage_batch", None)

    def slow(*args):
        started.append(None)
        holding.set()
        time.sleep(0.2)  # a batch holding the lock
        return real(*args)

    monkeypatch.setattr(storage, "_prune_usage_batch", slow, raising=False)
    pruning = threading.Thread(target=prune, args=(path,))
    pruning.start()
    assert holding.wait(10), "retention did not work in batches"
    admitted = []

    def call(n):
        held = reserve(tracker, f"w{n}", 0.00001)
        admitted.append(len(started))
        tracker.commit(held, USAGE)

    threads = [threading.Thread(target=call, args=(n,)) for n in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    pruning.join(timeout=30)
    assert len(admitted) == writers and max(admitted) <= 5, admitted
    assert len(started) >= batches
    assert query(path, "SELECT COUNT(*) FROM usage WHERE session_id LIKE 'w%'") == [(writers,)]


@contextmanager
def lock_held_for(path, seconds):
    """Another process holds the write lock, and lets go after the given time."""
    holder = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
    holder.execute("BEGIN IMMEDIATE")
    release = threading.Timer(seconds, lambda: holder.execute("ROLLBACK"))
    release.start()
    try:
        yield
    finally:
        release.cancel()
        release.join()
        holder.close()


def test_a_prune_outwaits_a_lock_longer_than_its_connections_timeout(tmp_path):
    path = mixed(tmp_path / "ledger.db")
    conn = sqlite3.connect(path, timeout=0.05, isolation_level=None)
    with lock_held_for(path, 0.5):
        assert storage.apply_retention(conn, DAYS).usage_pruned == 22
    assert conn.execute("PRAGMA busy_timeout").fetchone() == (50,)  # put back
    conn.close()
    assert_finished(path)


def test_a_prune_gives_up_on_a_lock_it_never_gets(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    expected = totals(path)
    monkeypatch.setattr(storage, "RETENTION_LOCK_WAIT_SEC", 0.1, raising=False)
    with lock_held_for(path, 60), pytest.raises(sqlite3.OperationalError, match="locked"):
        prune(path)
    assert totals(path) == expected and old_rows(path) == 22


async def test_a_cancelled_prune_stops_waiting_for_the_lock(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    use_database(monkeypatch, path)
    tracker = budget_store.get_tracker()
    with lock_held_for(path, 60):
        task = asyncio.create_task(tracker.aapply_retention(DAYS))
        await asyncio.sleep(0.2)
        assert not task.done()
        task.cancel()
        async with asyncio.timeout(3):  # well inside the lock wait
            with pytest.raises(asyncio.CancelledError):
                await task
    assert old_rows(path) == 22


def test_a_prune_gets_in_beside_a_writer_that_hardly_rests(tmp_path, monkeypatch):
    """The lock is free for 3 ms in every 33: SQLite's own waiters mostly miss that."""
    path = mixed(tmp_path / "ledger.db")
    small(monkeypatch, pause=0.005)  # longer than the gap, so each batch needs a new one
    stop, writes = threading.Event(), []

    def write():
        conn = sqlite3.connect(path, timeout=30, isolation_level=None)
        while not stop.is_set():
            conn.execute("BEGIN IMMEDIATE")
            time.sleep(0.03)
            conn.execute("COMMIT")
            writes.append(None)
            time.sleep(0.003)
        conn.close()

    writer = threading.Thread(target=write)
    writer.start()
    try:
        while len(writes) < 2:
            time.sleep(0.01)
        began = len(writes)
        assert prune(path).usage_pruned == 22
        # About twenty transactions, each after one or a few of the writer's.
        assert len(writes) - began < 200
    finally:
        stop.set()
        writer.join(timeout=60)
    assert_finished(path)


# --- Interruption ---


class Failing(sqlite3.Connection):
    """Fails at one statement, as SQLite does when the disk fills up."""

    fail_at = -1
    how = "left open"
    calls = 0

    def execute(self, sql, parameters=()):
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            if self.how == "rolled back" and self.in_transaction:
                super().execute("ROLLBACK")  # SQLite did this itself
            raise sqlite3.OperationalError("database or disk is full")
        if sql == "ROLLBACK" and self.how == "rollback fails":
            super().execute(sql)
            raise sqlite3.OperationalError("disk I/O error")
        return super().execute(sql, parameters)


@pytest.mark.parametrize("how", ["left open", "rolled back", "rollback fails"])
def test_an_error_at_any_statement_is_safe_and_the_next_run_finishes(tmp_path, how):
    kept_progress = 0
    for position in range(200):
        path = mixed(tmp_path / f"{position}.db")
        expected = totals(path)
        conn = connect(path, Failing)
        conn.fail_at, conn.how = position, how
        try:
            storage.apply_retention(conn, DAYS)
            break  # there is no statement this late
        except sqlite3.OperationalError as exc:
            assert str(exc) == "database or disk is full"  # the first error, not the rollback's
            assert not conn.in_transaction
        finally:
            conn.close()
        assert totals(path) == expected
        kept_progress += 0 < old_rows(path) < 22
        prune(path)
        assert totals(path) == expected
        assert_finished(path)
    assert position > 40 and kept_progress > 20


def test_a_real_interrupt_is_reported_as_itself(tmp_path):
    path = mixed(tmp_path / "ledger.db")
    expected = totals(path)
    conn = connect(path)
    interrupts = []

    def interrupt_once():
        interrupts.append(None)
        return len(interrupts) == 1

    conn.set_trace_callback(
        lambda sql: sql.startswith("INSERT INTO usage")
        and conn.set_progress_handler(interrupt_once, 1)
    )
    with pytest.raises(sqlite3.OperationalError, match="^interrupted$"):
        storage.apply_retention(conn, DAYS)
    assert interrupts and not conn.in_transaction
    conn.close()
    assert totals(path) == expected


@pytest.mark.parametrize("how", ["rolled back", "rollback fails"])
def test_a_failed_upgrade_reports_the_first_error(tmp_path, how):
    path = tmp_path / "old.db"
    conn = connect(path, Failing)
    conn.how = how
    conn.fail_at = 12  # inside the upgrade transaction
    with pytest.raises(sqlite3.OperationalError, match="^database or disk is full$"):
        storage.migrate_database(conn)
    assert not conn.in_transaction
    conn.close()


@pytest.mark.parametrize("moment", ["in the gap", "in a transaction"])
async def test_a_cancelled_prune_stops_after_the_transaction_in_progress(tmp_path, monkeypatch, moment):
    path = mixed(tmp_path / "ledger.db")
    use_database(monkeypatch, path)
    tracker = budget_store.get_tracker()
    expected = totals(path)
    reached, go_on = threading.Event(), threading.Event()
    if moment == "in the gap":
        small(monkeypatch, pause=20)  # only the cancellation can end this wait in time
        monkeypatch.setattr(
            storage, "_let_writers_in",
            lambda stop, real=getattr(storage, "_let_writers_in", None): (reached.set(), real(stop)),
            raising=False,
        )
    else:
        real = getattr(storage, "_prune_usage_batch", None)

        def held(*args):
            reached.set()
            assert go_on.wait(10)
            return real(*args)

        monkeypatch.setattr(storage, "_prune_usage_batch", held, raising=False)
    task = asyncio.create_task(tracker.aapply_retention(DAYS))
    async with asyncio.timeout(10):
        while not reached.is_set():
            assert not task.done(), "retention did not work in batches"
            await asyncio.sleep(0.005)
    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()
    go_on.set()
    done, _ = await asyncio.wait([task], timeout=10)
    if not done:
        small(monkeypatch, pause=0)  # let the prune end, so this test can fail
        await asyncio.wait([task], timeout=60)
    assert done, "the cancelled prune went on"
    assert task.cancelled()
    assert old_rows(path) == 19  # the one batch of three, and no more
    assert totals(path) == expected
    between_transactions(monkeypatch, lambda stop: None)
    prune(path)
    assert totals(path) == expected
    assert_finished(path)


def cli(*args):
    return subprocess.Popen(
        [sys.executable, "-m", "core.storage", *args],
        cwd=PROJECT_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )


def many_batches(path):
    """Twenty batches of the product's size: seconds of pauses for the command to be seen in."""
    return fill(path, [(f"s{n}", None, 0, 100, 10) for n in range(100)])


def test_a_killed_prune_keeps_what_it_committed_and_the_next_run_finishes(tmp_path):
    path = many_batches(tmp_path / "ledger.db")
    expected = totals(path)
    process = cli("prune", "--db", str(path), "--days", str(DAYS))
    try:
        deadline = time.monotonic() + 30
        while old_rows(path) == 10000:
            assert process.poll() is None and time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        process.kill()
        process.communicate(timeout=30)
    assert 0 < old_rows(path) < 10000
    assert query(path, "PRAGMA integrity_check") == [("ok",)]
    assert totals(path) == expected
    again = cli("prune", "--db", str(path), "--days", str(DAYS))
    _, errors = again.communicate(timeout=60)
    assert again.returncode == 0, errors
    assert totals(path) == expected
    assert_finished(path)


def test_two_prune_commands_and_a_backup_at_once(tmp_path):
    path = many_batches(tmp_path / "ledger.db")
    expected = totals(path)
    command = ("prune", "--db", str(path), "--days", str(DAYS))
    running = [cli(*command), cli(*command)]
    try:
        deadline = time.monotonic() + 30
        while old_rows(path) == 10000:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        backup = cli("backup", str(path), str(tmp_path / "copy.bak"))
        running.append(backup)
        outputs = [process.communicate(timeout=60) for process in running]
    finally:
        for process in running:
            process.kill()
    assert [process.returncode for process in running] == [0, 0, 0], outputs
    pruned = [int(out.split(" usage rows")[0].split()[-1]) for out, _ in outputs[:2]]
    assert sum(pruned) == 10000
    assert totals(path) == totals(tmp_path / "copy.bak") == expected
    assert_finished(path)


def test_a_ledger_from_before_gains_the_summary_index(tmp_path, monkeypatch):
    path = mixed(tmp_path / "ledger.db")
    conn = connect(path)
    conn.execute("DROP INDEX IF EXISTS usage_summary")
    conn.close()
    use_database(monkeypatch, path)
    budget_store.get_tracker()
    assert query(path, "SELECT 1 FROM sqlite_master WHERE name = 'usage_summary'") == [(1,)]
    assert list(tmp_path.glob("*.bak*")) == []
