"""Cancellation and ledger failures: fail closed, charge once, record once (#41 AC5)."""

import asyncio
import gc
import sqlite3
import time

import anyio
import pytest

from app import agent_loop
from app.main import reset_budget
from core import budget_store, guard_core, pipeline
from core.config import get_settings, token_cost_gbp
from core.providers import ModelReply
from core.schemas import CommitCancelled, Decision, LedgerEntry, SandboxResult
from storage_helpers import RATES, USAGE, Gate, reserve, use_database, write_lock

VERDICT = (
    '{"decision": "ALLOW", "risk_score": 1, "owasp_tag": "SAFE: Compliant Action",'
    ' "reason": "fake"}'
)
CODE = {"code": "print(1)"}
LOCKED = sqlite3.OperationalError("database is locked")


@pytest.fixture(autouse=True)
def database(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")
    path = tmp_path / "ledger.db"
    use_database(monkeypatch, path)
    return path


@pytest.fixture
def tracker():
    return budget_store.get_tracker()


class Model:
    def __init__(self):
        self.calls = []
        self.delay = 0.0

    async def classify(self, system, payload, schema):
        self.calls.append("classify")
        return ModelReply(USAGE, text=VERDICT, finished=True)

    async def agent_turn(self, system, prompt, tools):
        self.calls.append("agent")
        await asyncio.sleep(self.delay)
        return ModelReply(USAGE, text="done", finished=True)


@pytest.fixture
def model(monkeypatch):
    fake = Model()
    monkeypatch.setattr(guard_core, "get_provider", lambda settings: fake)
    monkeypatch.setattr(agent_loop, "get_provider", lambda settings: fake)
    return fake


@pytest.fixture
def sandbox(monkeypatch):
    ran = []

    class Sandbox:
        provider_name = "docker"

        async def execute(self, request):
            ran.append(request.code)
            return SandboxResult(
                provider="docker", stdout="", stderr="", truncated=False,
                exit_code=0, duration_ms=1, timed_out=False,
            )

    monkeypatch.setattr(pipeline, "get_sandbox_provider", lambda override=None: Sandbox())
    return ran


def cost() -> float:
    settings = get_settings()
    return token_cost_gbp(settings.rates_for(settings.classifier), 400, 80)


def rows(tracker):
    return [(e.label, e.outcome, e.cost_gbp) for e in tracker.recent_entries()]


def entry(session_id="s"):
    return LedgerEntry(
        time="2026-11-28T00:00:00Z", source="web", session_id=session_id,
        label="step 1/1", outcome="ALLOW", cost_gbp=0.001,
    )


def execute(session_id="s"):
    return pipeline.execute_guarded("intent", "execute_python", dict(CODE), session_id)


def guard_call(session_id="s"):
    return guard_core.evaluate_action("execute_python", dict(CODE), session_id)


async def agent_run(queue=None):
    """The agent's own workflow task, with the prompt screen already allowed."""
    queue = queue or asyncio.Queue()
    await agent_loop._run_standard_workflow(
        queue, "run", "do it", "s", None, None, None, get_settings(), budget_store.get_tracker()
    )
    return [queue.get_nowait() for _ in range(queue.qsize())]


@pytest.fixture
def allowed_screen(monkeypatch):
    async def allow(*args, **kwargs):
        return guard_core.GuardEvaluation(
            decision=Decision.ALLOW, risk_score=0, owasp_tag="SAFE: Compliant Action",
            reason="ok", stage="OWASP_LLM", latency_ms=0, gates_passed=[], cost_gbp=0,
        )

    monkeypatch.setattr(guard_core, "evaluate_action", allow)


async def cancelled(task) -> asyncio.CancelledError:
    with pytest.raises(asyncio.CancelledError) as caught:
        async with asyncio.timeout(10):
            await task
    assert task.cancelled()
    return caught.value


# --- the shared wait ---


async def test_cancelled_reserve_releases_its_hold(tracker, monkeypatch):
    gate = Gate(monkeypatch, tracker, "reserve")
    task = asyncio.create_task(tracker.areserve("s", 0.01, rates=RATES, ttl_sec=60))
    await gate.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    task.cancel()
    gate.open.set()
    await cancelled(task)
    assert tracker.check_budget("s").reserved_gbp == 0


def write_calls(tracker, held):
    return {
        "reserve": lambda: tracker.areserve("s", 0.001, rates=RATES, ttl_sec=60),
        "commit": lambda: tracker.acommit(held, USAGE),
        "release": lambda: tracker.arelease(held),
        "record": lambda: tracker.arecord(entry()),
        "set_session_cap": lambda: tracker.aset_session_cap("s", 0.01),
        "reset": lambda: tracker.areset("s"),
    }


WRITES = ("reserve", "commit", "release", "record", "set_session_cap", "reset")


@pytest.mark.parametrize("method", WRITES)
async def test_a_failed_write_does_not_swallow_the_cancellation(tracker, monkeypatch, method):
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, method, error=LOCKED)
    task = asyncio.create_task(write_calls(tracker, held)[method]())
    await gate.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    gate.open.set()
    error = await cancelled(task)
    assert not isinstance(error, CommitCancelled)


@pytest.mark.parametrize("method", WRITES)
async def test_a_timeout_during_a_failing_write_is_still_a_timeout(tracker, monkeypatch, method):
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, method, error=LOCKED)
    asyncio.get_running_loop().call_later(0.15, gate.open.set)
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            await write_calls(tracker, held)[method]()


@pytest.mark.parametrize("method", WRITES)
async def test_a_cancelled_write_ends_before_the_cancellation_is_raised(
    tracker, monkeypatch, method
):
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, method)
    task = asyncio.create_task(write_calls(tracker, held)[method]())
    await gate.wait_entered()
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0.02)
        assert not task.done()
    gate.open.set()
    await cancelled(task)


async def test_a_failed_release_after_a_cancelled_reserve_is_logged(tracker, monkeypatch, caplog):
    def broken(*_):
        raise LOCKED

    monkeypatch.setattr(tracker, "release", broken)
    gate = Gate(monkeypatch, tracker, "reserve")
    unhandled = []
    asyncio.get_running_loop().set_exception_handler(lambda loop, context: unhandled.append(context))
    task = asyncio.create_task(tracker.areserve("s", 0.01, rates=RATES, ttl_sec=60))
    await gate.wait_entered()
    task.cancel()
    gate.open.set()
    await cancelled(task)
    del task
    gc.collect()
    await asyncio.sleep(0.02)
    assert "could not release a reservation" in caplog.text
    assert unhandled == []


async def test_waiting_under_a_cancel_scope_does_not_spin(tracker, monkeypatch):
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, "commit")
    seen = []

    async def call():
        try:
            await tracker.acommit(held, USAGE)
        except CommitCancelled as exc:
            seen.append(exc)
            raise

    async with asyncio.timeout(15):
        async with anyio.create_task_group() as group:
            group.start_soon(call)
            await gate.wait_entered()
            group.cancel_scope.cancel()
            asyncio.get_running_loop().call_later(1.0, gate.open.set)
            started = time.process_time()
    used = time.process_time() - started
    print(f"CPU used over a 1.0 s cancelled wait: {used:.3f} s")
    assert used < 0.3
    # The scope recognised its own cancellation, and the caller saw the cost.
    assert group.cancel_scope.cancelled_caught
    assert [e.result.cost_gbp for e in seen] == [pytest.approx(cost())]


async def test_the_cancelling_scope_absorbs_commit_cancelled(tracker, monkeypatch):
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, "commit")
    scope = anyio.CancelScope()

    async def call():
        with scope:
            await tracker.acommit(held, USAGE)
        return "continued after the scope"

    async with asyncio.timeout(10):
        task = asyncio.create_task(call())
        await gate.wait_entered()
        scope.cancel()
        asyncio.get_running_loop().call_later(0.05, gate.open.set)
        assert await task == "continued after the scope"
    assert scope.cancelled_caught


async def test_commit_under_a_timeout_raises_commit_cancelled_not_timeout(tracker, monkeypatch):
    """Why product code commits outside timeout blocks: 3.11 converts only the exact class."""
    held = reserve(tracker, "s", 0.001)
    gate = Gate(monkeypatch, tracker, "commit")
    asyncio.get_running_loop().call_later(0.15, gate.open.set)
    with pytest.raises(CommitCancelled) as caught:
        async with asyncio.timeout(0.05):
            await tracker.acommit(held, USAGE)
    assert caught.value.result.cost_gbp == pytest.approx(cost())


# --- through the guard, the pipeline and the agent ---


@pytest.mark.parametrize("path", [guard_call, execute])
async def test_cancel_during_a_commit_records_the_cost(
    tracker, monkeypatch, model, sandbox, path
):
    gate = Gate(monkeypatch, tracker, "commit")
    task = asyncio.create_task(path())
    await gate.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    gate.open.set()
    await cancelled(task)
    [row] = rows(tracker)
    assert row[1:] == ("BLOCK error", pytest.approx(cost()))
    assert tracker.check_budget("s").spent_gbp == pytest.approx(cost())
    assert tracker.check_budget("s").reserved_gbp == 0 and sandbox == []


@pytest.mark.parametrize("path", [guard_call, execute])
async def test_a_second_cancel_while_recording_keeps_the_record(
    tracker, monkeypatch, model, sandbox, path
):
    commit = Gate(monkeypatch, tracker, "commit")
    record = Gate(monkeypatch, tracker, "record")
    task = asyncio.create_task(path())
    await commit.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    commit.open.set()
    await record.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    record.open.set()
    await cancelled(task)
    assert [r[2] for r in rows(tracker)] == [pytest.approx(cost())]


@pytest.mark.parametrize("how", ["task", "anyio"])
async def test_cancel_between_the_charged_verdict_and_its_record(
    tracker, monkeypatch, model, sandbox, how
):
    """The pipeline reads the budget after the verdict and before the record."""
    real_check, real_commit = tracker.check_budget, tracker.commit
    gates = []

    def commit(*args):
        result = real_commit(*args)
        gates.append(Gate(monkeypatch, tracker, "check_budget"))
        return result

    monkeypatch.setattr(tracker, "commit", commit)
    scope = anyio.CancelScope()

    async def call():
        with scope:
            await execute()

    async with asyncio.timeout(15):
        task = asyncio.create_task(call())
        while not gates:
            await asyncio.sleep(0.005)
        await gates[0].wait_entered()
        if how == "task":
            task.cancel()
        else:
            scope.cancel()
        await asyncio.wait([task])
        gates[0].open.set()
    assert task.cancelled() if how == "task" else scope.cancelled_caught
    assert rows(tracker) == [("step 1/1", "ALLOW", pytest.approx(cost()))]
    assert real_check("s").spent_gbp == pytest.approx(cost()) and sandbox == []


@pytest.mark.parametrize("path", [guard_call, execute])
async def test_guard_cancelled_during_a_failing_commit_does_not_return(
    tracker, monkeypatch, model, sandbox, path
):
    gate = Gate(monkeypatch, tracker, "commit", error=LOCKED)
    task = asyncio.create_task(path())
    await gate.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    gate.open.set()
    await cancelled(task)
    assert [r[1:] for r in rows(tracker)] == [("BLOCK error", 0)]
    assert tracker.check_budget("s").reserved_gbp == 0 and sandbox == []


async def test_two_cancels_during_an_agent_turn_keep_its_record(
    tracker, monkeypatch, model, allowed_screen
):
    model.delay = 30
    release = Gate(monkeypatch, tracker, "release")
    task = asyncio.create_task(agent_run())
    async with asyncio.timeout(10):
        while "agent" not in model.calls:
            await asyncio.sleep(0.005)
    task.cancel()
    await release.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    release.open.set()
    await cancelled(task)
    assert rows(tracker) == [("agent turn", "ERROR", 0)]
    assert tracker.check_budget("s").reserved_gbp == 0


async def test_a_cancelled_agent_turn_ends_cancelled(tracker, model, allowed_screen):
    model.delay = 30
    queue = asyncio.Queue()
    task = asyncio.create_task(agent_run(queue))
    async with asyncio.timeout(10):
        while "agent" not in model.calls:
            await asyncio.sleep(0.005)
    task.cancel()
    await cancelled(task)
    assert rows(tracker) == [("agent turn", "ERROR", 0)]
    assert tracker.check_budget("s").reserved_gbp == 0


@pytest.mark.parametrize("method,charged", [("reserve", False), ("commit", True), ("record", True)])
async def test_agent_turn_cancelled_during_a_ledger_write(
    tracker, monkeypatch, model, allowed_screen, method, charged
):
    gate = Gate(monkeypatch, tracker, method)
    task = asyncio.create_task(agent_run())
    await gate.wait_entered()
    task.cancel()
    await asyncio.sleep(0.02)
    gate.open.set()
    await cancelled(task)
    spent = tracker.check_budget("s").spent_gbp
    settings = get_settings()
    expected = token_cost_gbp(settings.rates_for(settings.agent), 400, 80) if charged else 0
    assert spent == pytest.approx(expected) and tracker.check_budget("s").reserved_gbp == 0
    [row] = rows(tracker)
    # A record already being written when the cancel arrived keeps its outcome.
    assert row == ("agent turn", "OK" if method == "record" else "ERROR", pytest.approx(expected))


@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
async def test_a_slow_commit_is_not_inside_the_model_timeout(
    tracker, monkeypatch, model, sandbox, role
):
    monkeypatch.setenv(f"{role}_TIMEOUT_SEC", "0.1")
    real = tracker.commit

    def slow(*args):
        time.sleep(0.3)
        return real(*args)

    monkeypatch.setattr(tracker, "commit", slow)
    async with asyncio.timeout(10):
        if role == "CLASSIFIER":
            assert (await execute()).guard.decision is Decision.ALLOW
            assert sandbox == ["print(1)"]
        else:
            async def allow(*args, **kwargs):
                return guard_core.error_block("x").model_copy(update={"decision": Decision.ALLOW})

            monkeypatch.setattr(guard_core, "evaluate_action", allow)
            events = await agent_run()
            assert events[-1].type == "run_finished"
    assert tracker.check_budget("s").spent_gbp > 0


@pytest.mark.parametrize("method", ["reserve", "commit"])
async def test_ledger_failure_in_the_guard_blocks_and_runs_nothing(
    tracker, monkeypatch, model, sandbox, method
):
    def broken(*args, **kwargs):
        raise LOCKED

    monkeypatch.setattr(tracker, method, broken)
    result = await execute()
    assert (result.guard.decision, result.guard.cost_gbp) == (Decision.BLOCK, 0)
    assert rows(tracker) == [("step 1/1", "BLOCK error", 0)]
    assert tracker.check_budget("s").spent_gbp == tracker.check_budget("s").reserved_gbp == 0
    assert sandbox == [] and model.calls == ([] if method == "reserve" else ["classify"])


@pytest.mark.parametrize("method", ["reserve", "commit"])
async def test_ledger_failure_in_the_agent_turn_records_and_runs_nothing(
    tracker, monkeypatch, model, sandbox, allowed_screen, method
):
    def broken(*args, **kwargs):
        raise LOCKED

    monkeypatch.setattr(tracker, method, broken)
    async with asyncio.timeout(10):
        events = [event async for event in agent_loop.run_agent_workflow("do it", session_id="s")]
    assert events[-1].type == "run_error"
    assert rows(tracker) == [("agent turn", "ERROR", 0)]
    assert tracker.check_budget("s").spent_gbp == tracker.check_budget("s").reserved_gbp == 0
    assert sandbox == [] and model.calls == ([] if method == "reserve" else ["agent"])


async def test_a_real_lock_timeout_blocks_and_runs_nothing(
    database, monkeypatch, model, sandbox
):
    monkeypatch.setattr(budget_store, "BUSY_TIMEOUT_MS", 100)
    tracker = budget_store.get_tracker()
    async with asyncio.timeout(10):
        with write_lock(database):
            result = await execute()
    assert result.guard.decision is Decision.BLOCK
    assert "ledger unavailable" in result.guard.reason
    assert tracker.check_budget("s").spent_gbp == 0 and sandbox == [] and model.calls == []


async def test_cancelled_budget_reset_is_still_recorded(tracker, monkeypatch):
    class Request:
        headers = {"x-requested-with": "AgentGuard"}

    gate = Gate(monkeypatch, tracker, "reset")
    task = asyncio.create_task(reset_budget("s", Request()))
    await gate.wait_entered()
    task.cancel()
    gate.open.set()
    await cancelled(task)
    assert rows(tracker) == [("reset", "OK", 0)]
