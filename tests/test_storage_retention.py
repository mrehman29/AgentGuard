"""Retention, client accounting and cap arithmetic (#41 AC3 and the #33 contract)."""

import sqlite3
import subprocess
import sys
import time
from datetime import UTC, datetime

import pytest

from core import budget_store, storage
from core.config import PROJECT_ROOT
from core.schemas import LedgerEntry
from storage_helpers import RATES, USAGE, query, reserve, use_database

OLD = 10 * 86400


@pytest.fixture(autouse=True)
def database(monkeypatch, tmp_path):
    for name in ("SESSION_BUDGET_CAP_GBP", "GLOBAL_BUDGET_CAP_GBP", "CLIENT_BUDGET_CAP_GBP",
                 "LEDGER_RETENTION_DAYS"):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "ledger.db"
    use_database(monkeypatch, path)
    return path


@pytest.fixture
def tracker():
    return budget_store.get_tracker()


def spend(tracker, session_id, amount=0.001):
    return tracker.commit(reserve(tracker, session_id, amount), USAGE)


def age(path, seconds=OLD):
    conn = sqlite3.connect(path)
    for table in ("usage", "entries"):
        conn.execute(
            f"UPDATE {table} SET created_at = created_at - ? WHERE created_at > 0", (seconds,)
        )
    conn.commit()
    conn.close()


def add_usage(path, rows):
    """rows: (session_id, client_id, cost, created_at)"""
    conn = sqlite3.connect(path)
    conn.executemany(
        "INSERT INTO usage (session_id, client_id, prompt_tokens, candidates_tokens,"
        " thoughts_tokens, cost_gbp, created_at) VALUES (?, ?, 1, 1, 1, ?, ?)", rows
    )
    conn.commit()
    conn.close()


def entry(session_id, label="screen"):
    return LedgerEntry(
        time="2026-01-01T00:00:00Z", source="web", session_id=session_id,
        label=label, outcome="OK", cost_gbp=0,
    )


def test_an_expired_reservation_survives_pruning_and_a_late_commit(tracker, monkeypatch):
    held = reserve(tracker, "late", 0.001, ttl_sec=0.01)
    time.sleep(0.05)
    assert tracker.check_budget("late").reserved_gbp == 0
    tracker.apply_retention(1)
    assert tracker.commit(held, USAGE).cost_gbp > 0
    assert tracker.check_budget("late").spent_gbp > 0


def test_caps_are_unchanged_across_three_thousand_sessions(tracker, database, monkeypatch):
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.0001")
    monkeypatch.setenv("SESSION_BUDGET_CAP_GBP", "0.3")
    monkeypatch.setenv("GLOBAL_BUDGET_CAP_GBP", "1000")
    old = time.time() - OLD
    costs = [0.0000137, 0.0000091, 0.0000203]
    rows = []
    for index in range(3000):
        client = f"c{index % 100}" if index % 2 else None
        session = budget_store.session_key(client, f"s{index}")
        rows += [(session, client, cost * (1 + index % 7), old) for cost in costs]
    add_usage(database, rows)
    # One session exactly at its cap, on values whose float sum is inexact.
    tracker.set_session_cap("edge", 0.3)
    add_usage(database, [("edge", None, 0.1, old), ("edge", None, 0.2, old)])
    sessions = ["edge"] + [row[0] for row in rows[::3]]

    def results():
        return [
            (s.is_exceeded, *(None if v is None else budget_store.to_nano(v) for v in (
                s.spent_gbp, s.remaining_gbp, s.client_spent_gbp, s.client_cap_gbp,
                s.global_spent_gbp)))
            for s in map(tracker.check_budget, sessions)
        ]

    before = results()
    assert before[0][0] and any(not r[0] for r in before)
    assert any(r[0] and r[3] is not None for r in before)  # a client at its cap
    stats = tracker.apply_retention(5)
    assert stats.usage_pruned == 9002
    assert query(database, "SELECT COUNT(*) FROM usage") == [(3001,)]
    assert results() == before


def test_repeated_pruning_keeps_one_summary_per_group_and_legacy_rows(tracker, database):
    add_usage(database, [("s", None, 0.01, 0.0)])  # a legacy row
    for _ in range(4):
        spend(tracker, "s")
        spend(tracker, "k:s")
        age(database)
        before = tracker.check_budget("s"), tracker.check_budget("k:s")
        stats = tracker.apply_retention(5)
        assert stats.usage_pruned == 2 and stats.cost_archived_gbp > 0
        assert (tracker.check_budget("s"), tracker.check_budget("k:s")) == before
        assert query(database, "SELECT COUNT(*) FROM usage WHERE summary = 1") == [(2,)]
    assert query(database, "SELECT cost_gbp FROM usage WHERE summary = 0") == [(0.01,)]
    assert tracker.apply_retention(5).usage_pruned == 0
    # A reset clears the summary; the next prune keeps cleared spend global only.
    tracker.reset("s")
    spend(tracker, "s")
    age(database)
    before = tracker.check_budget("s")
    tracker.apply_retention(5)
    assert tracker.check_budget("s") == before


class StatementLog(sqlite3.Connection):
    """Keeps each statement with its parameters: the plan can depend on them."""

    def execute(self, sql, parameters=()):
        self.statements = [*getattr(self, "statements", []), (sql, parameters)]
        return super().execute(sql, parameters)


def many_sessions(path, client, sessions=4000, rows_each=5):
    old = time.time() - OLD
    prefix = f"{client}:" if client else ""
    add_usage(path, [(f"{prefix}s{n}", client, 0.000001, old)
                     for n in range(sessions) for _ in range(rows_each)])
    return sessions * rows_each


@pytest.mark.parametrize("client", [None, "big"], ids=["local rows", "one large client"])
def test_retention_finds_each_group_by_its_session(tracker, database, client):
    many_sessions(database, client, sessions=20)
    conn = sqlite3.connect(database, isolation_level=None, factory=StatementLog)
    try:
        assert storage.apply_retention(conn, 5).usage_pruned == 100
        plans = {}
        for sql, parameters in list(conn.statements):
            if sql.startswith(("BEGIN", "COMMIT")):
                continue
            rows = sqlite3.Connection.execute(conn, f"EXPLAIN QUERY PLAN {sql}", parameters)
            plans.setdefault(" ".join(sql.split()[:3]), set()).update(row[3] for row in rows)
    finally:
        conn.close()
    # Every statement in the write lock. Only the grouping may read all of usage,
    # and it runs once; a new statement must be added here with its plan.
    assert plans == {
        "DELETE FROM entries": {
            "SEARCH entries USING INDEX entries_created_at (created_at>? AND created_at<?)"
        },
        "SELECT session_id, client_id,": {
            "SCAN usage", "USE TEMP B-TREE FOR GROUP BY",
        },
        "DELETE FROM usage": {
            "SEARCH usage USING INDEX usage_session (session_id=? AND cleared=?)"
        },
        "INSERT INTO usage": set(),
    }


@pytest.mark.parametrize("client", [None, "big"], ids=["local rows", "one large client"])
def test_retention_work_grows_with_the_rows_not_their_square(tracker, database, client):
    """Counts SQLite's own instructions, so the speed of the disk does not matter."""
    rows = many_sessions(database, client)
    budget = 400 * rows  # a pass per session would need thousands per row
    conn = sqlite3.connect(database, isolation_level=None)

    def prune():
        done = [0]

        def count():
            done[0] += 1000
            return done[0] > budget  # non-zero stops the statement

        conn.set_progress_handler(count, 1000)
        return storage.apply_retention(conn, 5)

    try:
        assert prune().usage_pruned == rows
        # The next run merges new old rows into the 4,000 summaries now there.
        more = many_sessions(database, client, rows_each=1)
        assert prune().usage_pruned == more
        assert conn.execute("SELECT COUNT(*), SUM(summary) FROM usage").fetchone() == (4000, 4000)
    finally:
        conn.close()


def test_old_entries_are_pruned_and_recent_ones_kept(tracker):
    tracker.record(entry("s", "old"))
    recent = entry("s", "new").model_copy(update={"time": datetime.now(UTC)})
    tracker.record(recent)
    assert tracker.apply_retention(0).entries_pruned == 0
    assert tracker.apply_retention(30).entries_pruned == 1
    assert [e.label for e in tracker.recent_entries()] == ["new"]


def test_retention_keeps_the_client_cap(tracker, database, monkeypatch):
    monkeypatch.setenv("CLIENT_BUDGET_CAP_GBP", "0.002")
    add_usage(database, [("k:a", "k", 0.001, time.time() - OLD), ("k:b", "k", 0.001, time.time() - OLD)])
    before = tracker.check_budget("k:new")
    assert before.is_exceeded and before.client_spent_gbp == 0.002
    tracker.apply_retention(5)
    assert tracker.check_budget("k:new") == before
    assert reserve(tracker, "k:new", 0.0001) is None


def test_reset_client_removes_only_that_clients_caps(tracker):
    untouched = ["a_b", "axb:s", "A_B:s", "a%:s"]
    for key in [*untouched, "a_b:s", "a_b:t"]:
        tracker.set_session_cap(key, 0.001)
    tracker.reset_client("a_b")
    assert [tracker.check_budget(k).cap_gbp for k in ("a_b:s", "a_b:t")] == [0.05, 0.05]
    assert [tracker.check_budget(k).cap_gbp for k in untouched] == [0.001] * 4
    tracker.reset_client("a%")
    assert [tracker.check_budget(k).cap_gbp for k in untouched] == [0.001, 0.001, 0.001, 0.05]


def test_client_reset_keeps_live_reservations_and_global_spend(tracker):
    spend(tracker, "k:a")
    live = reserve(tracker, "k:b", 0.001)
    total = tracker.check_budget("x").global_spent_gbp
    tracker.reset_client("k")
    status = tracker.check_budget("k:b")
    assert (status.client_spent_gbp, status.client_reserved_gbp) == (0, 0.001)
    assert status.global_spent_gbp == total > 0
    tracker.commit(live, USAGE)


def test_session_reset_lowers_client_spend_but_not_global(tracker, monkeypatch):
    """Intended: only an administrator may reset (enforced by #33's routes)."""
    spend(tracker, "k:a")
    spend(tracker, "k:b")
    before = tracker.check_budget("k:a")
    tracker.reset("k:a")
    after = tracker.check_budget("k:a")
    assert after.client_spent_gbp == pytest.approx(before.client_spent_gbp / 2)
    assert after.global_spent_gbp == before.global_spent_gbp


def test_ledger_filtered_by_client(tracker):
    for session in ("k:a", "other:a", "k", "k:*", "k_:a"):
        tracker.record(entry(session))
    assert [e.session_id for e in tracker.recent_entries(client_id="k")] == ["k:a", "k:*"]
    assert [e.session_id for e in tracker.recent_entries(1, client_id="k")] == ["k:*"]
    assert len(tracker.recent_entries()) == 5


@pytest.mark.parametrize(
    "estimate,admitted",
    [(0.2, True), (0.2 + 1e-9, False), (0.2 + 4e-7, False)],
)
def test_cap_on_a_rounding_boundary(tracker, database, monkeypatch, estimate, admitted):
    monkeypatch.setenv("SESSION_BUDGET_CAP_GBP", "0.5")
    tracker.set_session_cap("s", 0.3)
    add_usage(database, [("s", None, 0.1, time.time())])
    assert 0.1 + 0.2 > 0.3  # float arithmetic alone would refuse the first case
    assert tracker.check_budget("s").cap_gbp == 0.3
    assert (reserve(tracker, "s", estimate) is not None) is admitted


def test_reserve_requires_rates_and_a_lifetime(tracker):
    with pytest.raises(TypeError):
        tracker.reserve("s", 0.001, ttl_sec=60)
    with pytest.raises(TypeError):
        tracker.reserve("s", 0.001, rates=RATES)
    with pytest.raises(ValueError):
        tracker.reserve("s", 0.001, rates=RATES, ttl_sec=float("nan"))


async def test_async_reserve_has_no_fallback(tracker, monkeypatch):
    calls = []

    def broken(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("a real bug")

    monkeypatch.setattr(tracker, "reserve", broken)
    with pytest.raises(TypeError, match="a real bug"):
        await tracker.areserve("s", 0.001, rates=RATES, ttl_sec=60)
    assert len(calls) == 1


def cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "core.storage", *args],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30,
    )


def test_prune_command(tracker, database):
    spend(tracker, "s")
    age(database)
    result = cli("prune", "--db", str(database), "--days", "5")
    assert result.returncode == 0 and "1 usage rows" in result.stdout, result.stderr
    off = cli("prune", "--db", str(database), "--days", "0")
    assert off.returncode == 0 and "Retention is off" in off.stdout


@pytest.mark.parametrize("command", [["prune", "--days", "5"], ["migrate"]])
def test_commands_refuse_a_path_that_is_not_a_ledger(tmp_path, command):
    missing = tmp_path / "mistyped.db"
    result = cli(*command, "--db", str(missing))
    assert result.returncode == 1 and "does not exist" in result.stderr
    assert not missing.exists()
    other = tmp_path / "other.db"
    conn = sqlite3.connect(other)
    conn.execute("CREATE TABLE unrelated (x)")
    conn.close()
    result = cli(*command, "--db", str(other))
    assert result.returncode == 1 and "Not an AgentGuard ledger" in result.stderr
    assert query(other, "SELECT name FROM sqlite_master") == [("unrelated",)]
