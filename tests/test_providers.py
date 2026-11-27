"""Model provider contract, role settings and failures. No real model is called.

Every provider is driven over the wire against a local fake server. A new
provider adds a fake of its API to FAKES.
"""

import ast
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from dotenv import dotenv_values
from starlette.testclient import TestClient

from app import agent_loop
from app.main import app
from core import budget_store, guard_core
from core.config import PROJECT_ROOT, PROVIDER_NAMES, ConfigError, get_settings, token_cost_gbp
from core.providers import PROVIDERS, ModelError, ToolSpec, gemini, get_provider
from core.schemas import Decision, OwaspVerdict, TokenUsage

KEY = "secret-key-for-tests"
VERDICT = json.dumps(
    {"decision": "ALLOW", "risk_score": 1, "owasp_tag": "SAFE: Compliant Action", "reason": "ok"}
)
TOOL = ToolSpec(
    "execute_python",
    "Run code.",
    {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
)
# 100 input, 40 output of which 15 are reasoning.
USAGE = TokenUsage(prompt_tokens=100, candidates_tokens=25, thoughts_tokens=15)


class FakeGemini:
    @staticmethod
    def reply(text=None, tool_call=None, finished=True, usage=True) -> dict:
        part = (
            {"functionCall": {"name": tool_call[0], "args": tool_call[1]}}
            if tool_call
            else {"text": text}
        )
        body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [part]},
                    "finishReason": "STOP" if finished else "MAX_TOKENS",
                }
            ]
        }
        if usage:
            body["usageMetadata"] = {
                "promptTokenCount": 100,
                "candidatesTokenCount": 25,
                "thoughtsTokenCount": 15,
            }
        return body


FAKES = {"gemini": FakeGemini}


@pytest.fixture
def server():
    """Set .body, .status or .delay. Requests are kept in .requests."""
    state = SimpleNamespace(body={}, status=200, delay=0.0, requests=[])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers["Content-Length"])
            state.requests.append(
                SimpleNamespace(
                    path=self.path,
                    headers=dict(self.headers),
                    body=json.loads(self.rfile.read(length)),
                )
            )
            time.sleep(state.delay)
            data = state.body if isinstance(state.body, bytes) else json.dumps(state.body).encode()
            try:
                self.send_response(state.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the client gave up

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{http.server_port}"
    yield state
    http.shutdown()
    http.server_close()
    thread.join(5)


@pytest.fixture(autouse=True)
def settings_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.lower().endswith("_proxy") or name.startswith(("GOOGLE_", "GEMINI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "guard.db"))
    monkeypatch.setattr(budget_store, "_tracker", None)
    monkeypatch.setattr(gemini, "_client_cache", None)


@pytest.fixture(params=PROVIDER_NAMES)
def contract(request, monkeypatch, server):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", request.param)
    monkeypatch.setenv("CLASSIFIER_BASE_URL", server.url)
    monkeypatch.setenv("CLASSIFIER_API_KEY", KEY)
    return FAKES[request.param]


async def classify():
    return await get_provider(get_settings().classifier).classify("rules", "data", OwaspVerdict)


async def agent_turn():
    return await get_provider(get_settings().agent).agent_turn("rules", "do it", [TOOL])


OPERATIONS = [classify, agent_turn]


def reserved(session_id="demo"):
    return budget_store.get_tracker().check_budget(session_id).reserved_gbp


# --- the contract ---


def test_every_configurable_provider_exists_and_is_tested():
    assert set(PROVIDERS) == set(PROVIDER_NAMES) == set(FAKES)


async def test_classification(contract, server):
    server.body = contract.reply(text=VERDICT)
    reply = await classify()
    assert (reply.text, reply.tool_call, reply.finished) == (VERDICT, None, True)
    assert reply.usage == USAGE
    assert len(server.requests) == 1


async def test_agent_tool_call(contract, server):
    server.body = contract.reply(tool_call=("execute_python", {"code": "1"}))
    reply = await agent_turn()
    assert (reply.tool_call.name, reply.tool_call.args) == ("execute_python", {"code": "1"})
    assert reply.text is None and reply.usage == USAGE
    assert len(server.requests) == 1


async def test_agent_text(contract, server):
    server.body = contract.reply(text="hello")
    reply = await agent_turn()
    assert (reply.text, reply.tool_call, reply.finished) == ("hello", None, True)


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_unfinished_reply_is_marked_and_keeps_usage(contract, server, operation):
    server.body = contract.reply(text="{", finished=False)
    reply = await operation()
    assert reply.finished is False and reply.usage == USAGE


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("case", ["no usage", "empty", "not json", "http error"])
async def test_unbillable_reply_is_a_safe_error(contract, server, caplog, operation, case):
    caplog.set_level(logging.DEBUG)
    leak = f"detail {KEY}"
    if case == "no usage":
        server.body = contract.reply(text=VERDICT, usage=False)
    elif case == "not json":
        server.body = leak.encode()
    elif case == "http error":
        server.status = 429
        server.body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": leak}}
    with pytest.raises(ModelError) as caught:
        await operation()
    error = caught.value
    assert error.__suppress_context__ or error.__context__ is None
    assert KEY not in f"{error!s} {error!r} {caplog.text}"
    if case == "http error":
        assert "code=429 status=RESOURCE_EXHAUSTED" in str(error)
    assert len(server.requests) == 1  # no retry


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_unreachable_server_is_a_model_error(contract, monkeypatch, operation):
    monkeypatch.setenv("CLASSIFIER_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(ModelError):
        await operation()


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_cancellation_passes_through(contract, server, operation):
    server.body, server.delay = contract.reply(text=VERDICT), 0.5
    task = asyncio.create_task(operation())
    while not server.requests:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("name", PROVIDER_NAMES)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_missing_key_makes_no_call(monkeypatch, server, name, operation):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", name)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
    with pytest.raises(ModelError, match="MissingApiKey"):
        await operation()
    assert server.requests == []


async def test_gemini_requests(contract, server):
    """What goes on the wire, including the tool schema."""
    server.body = contract.reply(text=VERDICT)
    await classify()
    await agent_turn()
    first, second = server.requests
    assert first.headers["x-goog-api-key"] == KEY and KEY not in first.path + second.path
    assert get_settings().classifier.model in first.path
    assert first.body["generationConfig"]["maxOutputTokens"] == 512
    [declaration] = second.body["tools"][0]["functionDeclarations"]
    assert declaration["name"] == "execute_python"
    assert declaration["parameters"]["required"] == ["code"]
    assert declaration["parameters"]["properties"]["code"]["type"] == "STRING"


def test_gemini_sdk_is_imported_only_by_its_provider():
    importers = set()
    for folder in ("app", "core", "sandbox"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [f"{node.module}.{alias.name}" for alias in node.names]
                else:
                    continue
                if any(n.startswith(("google.genai", "google.generativeai")) for n in names):
                    importers.add(path.relative_to(PROJECT_ROOT).as_posix())
    assert importers == {"core/providers/gemini.py"}


# --- settings ---


def test_existing_settings_serve_both_roles(monkeypatch):
    values = {
        "GEMINI_MODEL": "some-model",
        "INPUT_COST_PER_1M_USD": "1",
        "OUTPUT_COST_PER_1M_USD": "2",
        "CLASSIFIER_TIMEOUT_SEC": "7",
        "BUDGET_RESERVE_OUTPUT_TOKENS": "99",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = get_settings()
    assert settings.agent == settings.classifier
    model = settings.classifier
    assert (model.provider, model.base_url, model.model, model.api_key) == (
        "gemini", "", "some-model", KEY,
    )
    assert (model.input_cost_per_1m_usd, model.output_cost_per_1m_usd) == (1, 2)
    assert (model.timeout_sec, model.max_output_tokens) == (7, 99)


def set_role(monkeypatch, role, **values):
    for name, value in values.items():
        monkeypatch.setenv(f"{role}_{name.upper()}", str(value))


def test_roles_override_and_fall_back_in_order(monkeypatch):
    set_role(
        monkeypatch, "CLASSIFIER", model="c-model", base_url="https://c.example/v1",
        api_key="c-key", input_cost_per_1m_usd=3, max_output_tokens=100,
    )
    set_role(monkeypatch, "AGENT", model="a-model", output_cost_per_1m_usd=9, timeout_sec=30)
    settings = get_settings()
    classifier, agent = settings.classifier, settings.agent
    assert (classifier.model, classifier.api_key) == ("c-model", "c-key")
    assert (classifier.output_cost_per_1m_usd, classifier.timeout_sec) == (2.5, 5)
    assert (agent.model, agent.api_key, agent.base_url) == (
        "a-model", "c-key", "https://c.example/v1",
    )
    assert (agent.input_cost_per_1m_usd, agent.output_cost_per_1m_usd) == (3, 9)
    assert (agent.timeout_sec, agent.max_output_tokens) == (30, 100)


def test_agent_settings_do_not_change_the_classifier(monkeypatch):
    before = get_settings().classifier
    set_role(monkeypatch, "AGENT", model="a", api_key="a-key", base_url="http://a.example")
    assert get_settings().classifier == before


@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
def test_a_key_is_not_shared_with_another_url(monkeypatch, role):
    set_role(monkeypatch, role, base_url="https://other.example")
    with pytest.raises(ConfigError, match=f"{role}_API_KEY"):
        get_settings()


@pytest.mark.parametrize(
    "name,value",
    [
        ("PROVIDER", KEY),
        ("BASE_URL", f"https://user:{KEY}@host.example"),
        ("BASE_URL", f"https://host.example/v1?key={KEY}"),
        ("BASE_URL", f"https://host.example/v1#{KEY}"),
        ("BASE_URL", f"ftp://{KEY}"),
        ("BASE_URL", f"host.example/{KEY}"),
        ("BASE_URL", f"https://host.example:{KEY}"),
        ("BASE_URL", f"https://[{KEY}"),
        ("API_KEY", f"{KEY} two"),
        ("API_KEY", f"{KEY}é"),
        ("INPUT_COST_PER_1M_USD", KEY),
        ("OUTPUT_COST_PER_1M_USD", "0"),
        ("TIMEOUT_SEC", "-1"),
        ("MAX_OUTPUT_TOKENS", "1.5"),
    ],
)
@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
def test_bad_role_setting_names_itself_without_values(monkeypatch, role, name, value):
    monkeypatch.setenv(f"{role}_{name}", value)
    with pytest.raises(ConfigError) as caught:
        get_settings()
    assert f"{role}_{name}" in str(caught.value) and KEY not in str(caught.value)


def test_bad_gemini_key_is_named_not_shown(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", f"{KEY} x")
    with pytest.raises(ConfigError) as caught:
        get_settings()
    assert "GEMINI_API_KEY" in str(caught.value) and KEY not in str(caught.value)


def test_keys_are_not_in_any_repr(monkeypatch):
    set_role(monkeypatch, "AGENT", api_key="agent-secret")
    settings = get_settings()
    text = repr(settings) + repr(get_provider(settings.agent)) + str(settings.classifier)
    assert KEY not in text and "agent-secret" not in text


def test_server_refuses_to_start_with_a_bad_setting(monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "nope")
    with pytest.raises(ConfigError, match="AGENT_PROVIDER"):
        with TestClient(app, base_url="http://127.0.0.1"):
            pass


def test_stdio_server_refuses_to_start_with_a_bad_setting(monkeypatch):
    monkeypatch.setenv("AGENT_API_KEY", f"{KEY} x")
    script = str(PROJECT_ROOT / "app" / "mcp_stdio.py")
    code = (
        f"import runpy, sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); import core.config;"
        " core.config.dotenv_values = lambda *a, **k: {};"
        f" runpy.run_path({script!r}, run_name='__main__')"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], input="", capture_output=True, text=True, timeout=30
    )
    assert done.returncode != 0 and done.stdout == ""
    assert "AGENT_API_KEY" in done.stderr and KEY not in done.stderr


def test_settings_of_an_existing_env_file_load(monkeypatch):
    """The example file, whose active lines are the v1.0 settings, with a key filled in."""
    values = dotenv_values(PROJECT_ROOT / ".env.example")
    assert not any(name.startswith(("AGENT_", "CLASSIFIER_M", "CLASSIFIER_A")) for name in values)
    for name, value in values.items():
        monkeypatch.setenv(name, value or "")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    settings = get_settings()
    assert settings.agent == settings.classifier
    assert (settings.classifier.model, settings.classifier.api_key) == (
        "gemini-3.5-flash-lite", KEY,
    )


# --- the two callers: costs and failures ---


@pytest.fixture
def rates(contract, monkeypatch):
    """Different rates and output limits per role."""
    set_role(monkeypatch, "CLASSIFIER", input_cost_per_1m_usd=1, output_cost_per_1m_usd=2)
    set_role(
        monkeypatch, "AGENT", model="agent-model", input_cost_per_1m_usd=10,
        output_cost_per_1m_usd=20, max_output_tokens=1000,
    )
    return contract


async def run_agent(monkeypatch):
    """One standard run with the prompt screen allowed. Returns the events."""

    async def allow(**_):
        return guard_core.GuardEvaluation(
            decision=Decision.ALLOW, risk_score=0, owasp_tag="SAFE: Compliant Action",
            reason="ok", stage="OWASP_LLM", latency_ms=0,
        )

    monkeypatch.setattr(guard_core, "evaluate_action", allow)
    return [event async for event in agent_loop.run_agent_workflow("do it")]


def agent_entries():
    return [
        (e.outcome, e.cost_gbp)
        for e in budget_store.get_tracker().recent_entries()
        if e.label == "agent turn"
    ]


async def test_classifier_is_charged_at_its_own_rates(rates, server):
    server.body = rates.reply(text=VERDICT)
    result = await guard_core.evaluate_action("execute_python", {"code": "1"}, "demo")
    assert result.decision is Decision.ALLOW
    expected = (100 * 1 + 40 * 2) / 1_000_000 * 0.8
    assert result.cost_gbp == pytest.approx(expected)
    assert budget_store.get_tracker().check_budget("demo").spent_gbp == pytest.approx(expected)
    assert get_settings().classifier.model in server.requests[0].path


async def test_agent_is_reserved_and_charged_at_its_own_rates(rates, server, monkeypatch):
    server.body = rates.reply(text="hello")
    tracker = budget_store.get_tracker()
    estimates = []
    reserve = tracker.reserve

    def spy(session_id, estimate):
        estimates.append(estimate)
        return reserve(session_id, estimate)

    monkeypatch.setattr(tracker, "reserve", spy)
    events = await run_agent(monkeypatch)
    assert "agent_reply" in [e.type for e in events]
    settings = get_settings()
    assert estimates == [pytest.approx(token_cost_gbp(settings, 610, 1000, settings.agent))]
    expected = (100 * 10 + 40 * 20) / 1_000_000 * 0.8
    assert agent_entries() == [("OK", pytest.approx(expected))]
    assert tracker.check_budget("demo").spent_gbp == pytest.approx(expected)
    assert "agent-model" in server.requests[0].path
    assert reserved() == 0


@pytest.mark.parametrize(
    "case,kind",
    [("http error", "ServerError"), ("no usage", "ValueError"), ("not json", None),
     ("timeout", "timed out")],
)
async def test_failed_calls_block_and_release(contract, server, monkeypatch, case, kind):
    if case == "http error":
        server.status, server.body = 500, {"error": {"code": 500, "status": "INTERNAL"}}
    elif case == "no usage":
        server.body = contract.reply(text=VERDICT, usage=False)
    elif case == "not json":
        server.body = b"nonsense"
    else:
        monkeypatch.setenv("CLASSIFIER_TIMEOUT_SEC", "0.2")
        server.body, server.delay = contract.reply(text=VERDICT), 1.0
    result = await guard_core.evaluate_action("execute_python", {"code": "1"}, "demo")
    assert result.decision is Decision.BLOCK and result.cost_gbp == 0
    if kind:
        assert kind in result.reason
    assert reserved() == 0
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error"
    assert agent_entries() == [("ERROR", 0)]
    assert reserved() == 0


async def test_agent_has_its_own_timeout(contract, server, monkeypatch):
    set_role(monkeypatch, "AGENT", timeout_sec=0.2)
    server.body, server.delay = contract.reply(text="late"), 1.0
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error"
    assert agent_entries() == [("ERROR", 0)]
    assert reserved() == 0


async def test_unusable_agent_output_is_charged_and_runs_nothing(contract, monkeypatch, server):
    async def unusable(prompt, settings):
        return USAGE, None, None

    monkeypatch.setattr(agent_loop, "model_caller", unusable)
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error"
    assert not any(e.type == "step_started" and e.index == 1 for e in events)
    settings = get_settings()
    expected = token_cost_gbp(settings, 100, 40, settings.agent)
    assert agent_entries() == [("ERROR", pytest.approx(expected))]
    assert reserved() == 0


def test_reservations_outlive_the_longer_timeout(monkeypatch):
    set_role(monkeypatch, "AGENT", timeout_sec=100)
    tracker = budget_store.get_tracker()
    tracker.reserve("demo", 0.001)
    later = time.time() + 100 * budget_store.RESERVATION_TTL_TIMEOUTS - 5
    monkeypatch.setattr(budget_store.time, "time", lambda: later)
    assert reserved() > 0
