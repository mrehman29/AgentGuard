"""Contract tests. They replace the guard, ledger and sandbox, so they stay
valid when the tracks swap the stubs for real code."""

import asyncio
import inspect
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core import budget_store, guard_core, pipeline
from core.config import ConfigError, get_settings, token_cost_gbp
from core.schemas import (
    BudgetStatus,
    Decision,
    GuardEvaluation,
    OwaspTag,
    PipelineStepResult,
    RunEvent,
    SandboxRequest,
    SandboxResult,
    Stage,
    ledger_outcome,
)
from sandbox import get_sandbox_provider
from sandbox.base import SandboxProvider


def verdict(decision=Decision.ALLOW, stage=Stage.OWASP_LLM, tag=OwaspTag.SAFE, cost=0.001):
    return GuardEvaluation(
        decision=decision,
        risk_score=0,
        owasp_tag=tag,
        reason="test",
        stage=stage,
        latency_ms=1,
        gates_passed=[Stage.BUDGET, Stage.REGEX],
        cost_gbp=cost,
    )


class FakeTracker:
    def __init__(self, fail_record=False, fail_check=False):
        self.entries = []
        self.fail_record = fail_record
        self.fail_check = fail_check
    def record(self, entry):
        if self.fail_record:
            raise OSError("disk full")
        self.entries.append(entry)

    def check_budget(self, session_id):
        if self.fail_check:
            raise OSError("locked")
        return BudgetStatus(
            session_id=session_id, prompt_tokens=0, candidates_tokens=0, thoughts_tokens=0,
            total_tokens=0, spent_gbp=0, reserved_gbp=0, cap_gbp=1, remaining_gbp=1,
            global_spent_gbp=0, global_cap_gbp=1, is_exceeded=False,
        )


class RecordingProvider(SandboxProvider):
    provider_name = "docker"

    def __init__(self, error=None):
        self.requests = []
        self.error = error

    async def execute(self, request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return SandboxResult(
            provider="docker", stdout="ok", stderr="", truncated=False,
            exit_code=0, duration_ms=1, timed_out=False,
        )

    async def health_check(self):
        return True


@pytest.fixture
def env(monkeypatch):
    state = SimpleNamespace(
        tracker=FakeTracker(), provider=RecordingProvider(), guard=verdict(), guard_error=None, calls=[]
    )

    async def fake_evaluate(tool_name, args, session_id, **kwargs):
        state.calls.append((tool_name, args, kwargs))
        args["code"] = "tampered"  # must not reach the sandbox
        if state.guard_error:
            raise state.guard_error
        return state.guard

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate)
    monkeypatch.setattr(budget_store, "get_tracker", lambda: state.tracker)
    monkeypatch.setattr(pipeline, "get_sandbox_provider", lambda override=None: state.provider)
    return state


async def run(args=None, tool="execute_python", **kwargs):
    return await pipeline.execute_guarded(
        "intent", tool, {"code": "print(1)"} if args is None else args, "demo", **kwargs
    )


async def test_allow_executes_the_evaluated_code_and_records_once(env):
    result = await run()
    assert [r.code for r in env.provider.requests] == ["print(1)"]
    assert result.sandbox.stdout == "ok"
    assert [(e.outcome, e.cost_gbp, e.label) for e in env.tracker.entries] == [
        ("ALLOW", 0.001, "step 1/1")
    ]
    assert env.calls[0][2]["record"] is False


@pytest.mark.parametrize("decision", [Decision.FLAG, Decision.BLOCK])
async def test_only_allow_executes(env, decision):
    env.guard = verdict(decision, tag=OwaspTag.LLM06)
    result = await run()
    assert env.provider.requests == []
    assert result.sandbox is None
    assert len(env.tracker.entries) == 1


@pytest.mark.parametrize(
    "tool,args",
    [
        ("execute_python", {}),
        ("execute_python", {"code": 5}),
        ("execute_python", {"code": "x" * 20_001}),
        ("execute_python", {"code": "1", "timeout_sec": 60}),
        ("execute_python", {"code": "1", "extra": True}),
        ("execute_python", None),
        ("shell", {"code": "1"}),
    ],
)
async def test_invalid_requests_block_without_evaluation(env, tool, args):
    result = await pipeline.execute_guarded("intent", tool, args, "demo")
    assert result.guard.decision is Decision.BLOCK
    assert result.guard.owasp_tag is OwaspTag.NA
    assert env.calls == [] and env.provider.requests == []
    assert [e.outcome for e in env.tracker.entries] == ["BLOCK error"]


async def test_unknown_provider_blocks(env, monkeypatch):
    monkeypatch.setattr(pipeline, "get_sandbox_provider", get_sandbox_provider)
    result = await run(provider="nope")
    assert result.guard.decision is Decision.BLOCK
    assert env.calls == []


async def test_guard_exception_blocks(env):
    env.guard_error = RuntimeError("boom")
    result = await run()
    assert result.guard.decision is Decision.BLOCK
    assert env.provider.requests == []


@pytest.mark.parametrize("tracker", [FakeTracker(fail_record=True), FakeTracker(fail_check=True)])
async def test_ledger_failure_blocks(env, tracker):
    env.tracker = tracker
    result = await run()
    assert result.guard.decision is Decision.BLOCK
    assert result.guard.cost_gbp == 0.001
    assert env.provider.requests == []
    assert [e.outcome for e in env.tracker.entries] == ([] if tracker.fail_record else ["BLOCK error"])


async def test_sandbox_exception_reports_exit_125(env):
    env.provider = RecordingProvider(error=ConnectionError("down"))
    result = await run()
    assert result.guard.decision is Decision.ALLOW
    assert (result.sandbox.exit_code, result.sandbox.timed_out) == (125, False)
    assert [e.outcome for e in env.tracker.entries] == ["ALLOW"]


def test_ledger_outcomes():
    assert ledger_outcome(verdict()) == "ALLOW"
    assert ledger_outcome(verdict(Decision.BLOCK, Stage.REGEX, OwaspTag.LLM01)) == "BLOCK regex"
    assert ledger_outcome(verdict(Decision.BLOCK, Stage.BUDGET, OwaspTag.NA)) == "BLOCK budget"
    assert ledger_outcome(verdict(Decision.BLOCK, tag=OwaspTag.NA)) == "BLOCK error"
    assert ledger_outcome(verdict(Decision.BLOCK, tag=OwaspTag.LLM02)) == "BLOCK owasp"


def test_contracts_are_immutable_and_bounded():
    with pytest.raises(ValidationError):
        verdict().decision = Decision.BLOCK
    with pytest.raises(ValidationError):
        SandboxRequest(code="1", timeout_sec=6)


def test_run_event_required_fields():
    event = RunEvent(type="run_started", run_id="r", session_id="demo", preset=None, total_steps=1)
    assert '"preset":null' in event.model_dump_json(exclude_unset=True)
    with pytest.raises(ValidationError):
        RunEvent(type="step_finished", run_id="r", index=0)


def test_provider_factory():
    assert get_sandbox_provider("docker").provider_name == "docker"
    assert get_sandbox_provider("cloudrun").provider_name == "cloudrun"
    for bad in ("", "nope"):
        with pytest.raises(ValueError):
            get_sandbox_provider(bad)


@pytest.mark.parametrize("value", ["abc", "-1", "nan", "inf"])
def test_bad_numeric_setting_is_rejected(monkeypatch, value):
    monkeypatch.setenv("SESSION_BUDGET_CAP_GBP", value)
    with pytest.raises(ConfigError):
        get_settings()


def test_api_key_is_not_in_repr(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    assert "not-a-real-key" not in repr(get_settings())


def test_token_cost(monkeypatch):
    for name, value in {"INPUT_COST_PER_1M_USD": "0.30", "OUTPUT_COST_PER_1M_USD": "2.50", "USD_TO_GBP": "0.80"}.items():
        monkeypatch.setenv(name, value)
    settings = get_settings()
    rates = settings.rates_for(settings.classifier)
    assert token_cost_gbp(rates, 1_000_000, 1_000_000) == pytest.approx(2.24)


async def test_cancellation_is_recorded_and_propagates(env):
    env.guard_error = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await run()
    assert [e.outcome for e in env.tracker.entries] == ["BLOCK error"]
    assert env.provider.requests == []


@pytest.mark.parametrize(
    "intent,args",
    [(None, {"code": "1"}), ("i", {1: "x"}), ("i", {"code": "\ud800"}), ("\ud800", {"code": "1"})],
)
async def test_unusable_caller_values_block_and_serialise(env, intent, args):
    result = await pipeline.execute_guarded(intent, "execute_python", args, "demo")
    assert result.guard.decision is Decision.BLOCK
    assert env.provider.requests == [] and env.calls == []
    result.model_dump_json()


async def test_ndjson_keeps_nested_defaults(env):
    env.guard = GuardEvaluation(
        decision=Decision.BLOCK, risk_score=100, owasp_tag=OwaspTag.NA, reason="r",
        stage=Stage.BUDGET, latency_ms=0,
    )
    result = await run()
    line = RunEvent(type="step_finished", run_id="r", index=1, result=result).to_ndjson()
    data = json.loads(line)
    assert line.endswith("\n") and "message" not in data
    assert data["result"]["guard"]["cost_gbp"] == 0
    assert data["result"]["guard"]["gates_passed"] == []


async def test_source_and_events_pass_through(env):
    callback = lambda *_: None  # noqa: E731
    await run(source="mcp", label="execute", on_event=callback)
    assert env.calls[0][2] == {"source": "mcp", "label": "execute", "on_event": callback, "record": False}
    assert env.tracker.entries[0].source == "mcp"


@pytest.mark.parametrize(
    "function,parameters",
    [
        (guard_core.fast_rule_check, "tool_name args"),
        (guard_core.evaluate_owasp_policy, "tool_name args session_id"),
        (guard_core.evaluate_action, "tool_name args session_id source label on_event record"),
        (pipeline.execute_guarded, "intent tool_name args session_id provider source label on_event"),
        (budget_store.SessionBudgetTracker.reserve, "self session_id estimate_gbp"),
        (budget_store.SessionBudgetTracker.commit, "self reservation_id usage rates"),
        (budget_store.SessionBudgetTracker.set_session_cap, "self session_id cap_gbp"),
        (budget_store.SessionBudgetTracker.recent_entries, "self limit"),
    ],
)
def test_frozen_signatures(function, parameters):
    assert list(inspect.signature(function).parameters) == parameters.split()


@pytest.mark.parametrize(
    "name",
    ["USD_TO_GBP", "INPUT_COST_PER_1M_USD", "OUTPUT_COST_PER_1M_USD",
     "BUDGET_RESERVE_INPUT_TOKENS", "BUDGET_RESERVE_OUTPUT_TOKENS", "CLASSIFIER_TIMEOUT_SEC"],
)
def test_zero_that_disables_the_breaker_is_rejected(monkeypatch, name):
    monkeypatch.setenv(name, "0")
    with pytest.raises(ConfigError):
        get_settings()


def test_zero_cap_is_allowed(monkeypatch):
    monkeypatch.setenv("SESSION_BUDGET_CAP_GBP", "0")
    assert get_settings().session_budget_cap_gbp == 0


@pytest.mark.parametrize("tool_name", [None, "\ud800"])
async def test_unusable_tool_name_blocks_and_serialises(env, tool_name):
    result = await pipeline.execute_guarded("i", tool_name, {"code": "1"}, "demo")
    assert result.guard.decision is Decision.BLOCK
    assert env.provider.requests == [] and env.calls == []
    result.model_dump_json()


def test_preset_is_a_config_id():
    event = RunEvent(type="run_started", run_id="r", session_id="demo", preset="safe", total_steps=1)
    assert json.loads(event.to_ndjson())["preset"] == "safe"
