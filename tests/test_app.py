"""Tests for AgentGuard web API, agent loop, and MCP server (Track C, review cycle 2)."""

import asyncio
import base64
import hashlib
import json
import os
import re
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

# Ensure default environment never writes to data/guard_state.db
_DEFAULT_TEST_DB = os.path.join(tempfile.gettempdir(), "guard_test_init.db")
os.environ.setdefault("SQLITE_DB_PATH", _DEFAULT_TEST_DB)

import pytest
from starlette.testclient import TestClient

from app import agent_loop
from app.agent_loop import default_model_caller
from app.main import app, lifespan
from core import budget_store, guard_core
from core.config import PROJECT_ROOT, get_settings
from core.schemas import (
    BudgetStatus,
    CommitResult,
    Decision,
    GuardEvaluation,
    LedgerEntry,
    OwaspTag,
    PipelineStepResult,
    SandboxResult,
    Stage,
    TokenUsage,
)


def mock_guard_eval(decision=Decision.ALLOW, stage=Stage.OWASP_LLM, cost=0.0):
    return GuardEvaluation(
        decision=decision,
        risk_score=0 if decision is Decision.ALLOW else 100,
        owasp_tag=OwaspTag.SAFE if decision is Decision.ALLOW else OwaspTag.LLM01,
        reason="mock guard reason",
        stage=stage,
        latency_ms=2,
        gates_passed=[Stage.BUDGET, Stage.REGEX]
        if decision is not Decision.ALLOW
        else [Stage.BUDGET, Stage.REGEX, Stage.OWASP_LLM],
        cost_gbp=cost,
    )


class MemoryTracker:
    def __init__(self, reserve_fail=False, is_exceeded=False):
        self.entries: list[LedgerEntry] = []
        self.caps: dict[str, float] = {}
        self.reserve_fail = reserve_fail
        self.is_exceeded_flag = is_exceeded
        self.reservations: list[str] = []
        self.released: list[str] = []
        self.commits: list[tuple[str, TokenUsage]] = []

    def check_budget(self, session_id: str) -> BudgetStatus:
        return BudgetStatus(
            session_id=session_id,
            prompt_tokens=10,
            candidates_tokens=10,
            thoughts_tokens=0,
            total_tokens=20,
            spent_gbp=0.001 if not self.is_exceeded_flag else 0.05,
            reserved_gbp=0.0,
            cap_gbp=self.caps.get(session_id, 0.05),
            remaining_gbp=0.049 if not self.is_exceeded_flag else 0.0,
            global_spent_gbp=0.001,
            global_cap_gbp=0.50,
            is_exceeded=self.is_exceeded_flag,
        )

    def reserve(self, session_id: str, estimate_gbp: float) -> str | None:
        if self.reserve_fail or self.is_exceeded_flag:
            return None
        res_id = f"res_{len(self.reservations) + 1}"
        self.reservations.append(res_id)
        return res_id

    def commit(self, reservation_id: str, usage: TokenUsage, model=None) -> CommitResult:
        self.commits.append((reservation_id, usage))
        return CommitResult(cost_gbp=0.0001, budget=self.check_budget("stub"))

    def release(self, reservation_id: str) -> None:
        self.released.append(reservation_id)

    def set_session_cap(self, session_id: str, cap_gbp: float) -> BudgetStatus:
        self.caps[session_id] = cap_gbp
        return self.check_budget(session_id)

    def reset(self, session_id: str) -> BudgetStatus:
        self.caps.pop(session_id, None)
        return self.check_budget(session_id)

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)

    def recent_entries(self, limit: int = 50) -> list[LedgerEntry]:
        return self.entries[-limit:] if limit > 0 else []


@pytest.fixture(autouse=True)
def isolate_sqlite_db(tmp_path, monkeypatch):
    test_db = str(tmp_path / "guard_test.db")
    monkeypatch.setenv("SQLITE_DB_PATH", test_db)


@pytest.fixture
def client_env(monkeypatch):
    tracker = MemoryTracker()
    monkeypatch.setattr(budget_store, "get_tracker", lambda: tracker)
    return tracker


def test_index_page(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.get("/")
        assert res.status_code == 200
        assert "AgentGuard" in res.text
        assert "Red-Team" in res.text
        assert "Run through AgentGuard" in res.text
        for card in (
            "1 · AGENT INTENT",
            "2 · GATE VERDICT",
            "3 · SANDBOX EXECUTION",
            "4 · SESSION TOKEN P&amp;L",
            "CONNECT YOUR AGENT (MCP)",
        ):
            assert card in res.text


STATIC_DIR = PROJECT_ROOT / "app" / "static"


def test_console_assets_are_local_and_served(client_env):
    """No external URLs, and everything the console references exists (C-03)."""
    sources = {
        name: (STATIC_DIR / name).read_text(encoding="utf-8")
        for name in ("index.html", "fonts.css", "style.css", "console.js")
    }
    for name, text in sources.items():
        assert not re.search(r"((https?|wss?)://|[\"'(]//[a-z])", text, re.IGNORECASE), name

    referenced = set(re.findall(r"/static/[\w./-]+\.\w+", sources["index.html"]))
    for sheet in ("fonts.css", "style.css"):
        referenced |= {
            f"/static/{path}"
            for path in re.findall(r'url\("(?!data:)([^"]+)"\)', sources[sheet])
        }
    assert {"/static/fonts.css", "/static/style.css", "/static/console.js"} <= referenced
    # Icons whose names the script builds from a state.
    for kind in ("idle", "checking", "allow", "flag", "block", "failed"):
        referenced.add(f"/static/icons/verdict-{kind}.svg")
    for kind in ("waiting", "running", "passed", "flagged", "stopped", "skipped"):
        referenced.add(f"/static/icons/chip-{kind}.svg")
    referenced |= {f"/static/icons/pill-{kind}.svg" for kind in ("online", "offline")}

    with TestClient(app, base_url="http://127.0.0.1") as client:
        for path in sorted(referenced):
            assert client.get(path).status_code == 200, path


FONT_HASHES = {
    ("Inter", "693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3"),
    ("JetBrains Mono", "2c32b9b3ee358c119e210f6f5195f9bd34894d78a785ff2e95d60e718e400af4"),
    ("JetBrains Mono", "9c38cb2d0d2d93c1ee6e21fa78db76f13ea7e15e15cc64214c7ca89b6aaa35c4"),
}


def test_fonts_are_served_as_data_uris(client_env):
    """The fonts travel inside a text stylesheet, byte for byte."""
    with TestClient(app, base_url="http://127.0.0.1") as client:
        page = client.get("/").text
        res = client.get("/static/fonts.css")
    assert 'href="/static/fonts.css"' in page
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/css")

    faces = re.findall(
        r'font-family: "([^"]+)";[^}]*?url\("data:font/woff2;base64,([A-Za-z0-9+/=]+)"\)',
        res.text,
    )
    decoded = [(family, base64.b64decode(data, validate=True)) for family, data in faces]
    assert all(data.startswith(b"wOF2") for _, data in decoded)
    assert {(family, hashlib.sha256(data).hexdigest()) for family, data in decoded} == FONT_HASHES
    assert len(decoded) == len(FONT_HASHES)


def test_font_licences_are_kept():
    for name, copyright_line in (
        ("Inter-OFL.txt", "Copyright 2016 The Inter Project Authors"),
        (
            "jetbrains-mono-OFL.txt",
            "Copyright 2020 The JetBrains Mono Project Authors "
            "(https://github.com/JetBrains/JetBrainsMono)",
        ),
    ):
        text = (STATIC_DIR / "fonts" / name).read_text(encoding="utf-8")
        assert text.splitlines()[0] == copyright_line
        assert "SIL OPEN FONT LICENSE Version 1.1" in text


def test_static_assets_are_all_text():
    """The upload path cannot carry binary files."""
    for path in STATIC_DIR.rglob("*"):
        if path.is_file():
            data = path.read_bytes()
            assert b"\0" not in data, path.name
            data.decode("utf-8")


def test_console_never_builds_markup_from_text():
    """Model, sandbox and ledger text is only written with textContent."""
    script = (STATIC_DIR / "console.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script, sink
    assert "textContent" in script


def test_console_leaves_preset_text_and_limits_to_the_config():
    """Preset text and caps come from /api/config (C-16)."""
    sources = "".join(
        (STATIC_DIR / name).read_text(encoding="utf-8") for name in ("index.html", "console.js")
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        presets = client.get("/api/config").json()["presets"]
    for preset in presets:
        assert preset["prompt"] not in sources
        assert preset["description"] not in sources
    assert not re.search(r"£0\.\d", sources)


def test_health_check(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert "provider" in data
        assert "providers" in data
        assert "docker" in data["providers"]
        assert "cloudrun" in data["providers"]


def test_get_config(client_env):
    with TestClient(app, base_url="http://127.0.0.1:8765") as client:
        res = client.get("/api/config")
        assert res.status_code == 200
        data = res.json()
        assert "presets" in data
        assert len(data["presets"]) == 3
        preset_map = {p["id"]: p for p in data["presets"]}
        assert "safe" in preset_map
        assert "injection" in preset_map
        assert "loop" in preset_map

        assert preset_map["safe"]["title"] == "Safe data query"
        assert preset_map["safe"]["description"] == "Compound growth of £1,000. Expected: ALLOW"

        assert preset_map["injection"]["title"] == "Prompt injection"
        assert preset_map["injection"]["description"] == "Override attempt. Expected: BLOCK, LLM01"
        assert "[SYSTEM OVERRIDE" in preset_map["injection"]["prompt"]

        assert preset_map["loop"]["title"] == "Runaway cost loop"
        settings = get_settings()
        cap_formatted = f"£{settings.preset3_session_cap_gbp:.4f}"
        assert preset_map["loop"]["description"] == f"Cap {cap_formatted}. Expected: budget BLOCK"

        assert "session_cap_gbp" in data
        assert "global_cap_gbp" in data
        assert "providers" in data
        assert "default_provider" in data
        assert "mcp" in data
        assert "command" in data["mcp"]
        assert "script_path" in data["mcp"]
        assert data["mcp"]["url"] == "http://127.0.0.1:8765/mcp"
        assert data["mcp"]["args"] == [data["mcp"]["script_path"]]


def test_budget_endpoints(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.get("/api/budget/demo")
        assert res.status_code == 200
        assert res.json()["session_id"] == "demo"

        # Reset without X-Requested-With header is rejected with 403
        bad_reset = client.post("/api/budget/demo/reset")
        assert bad_reset.status_code == 403

        reset_res = client.post("/api/budget/demo/reset", headers={"X-Requested-With": "AgentGuard"})
        assert reset_res.status_code == 200
        assert reset_res.json()["session_id"] == "demo"
        assert any(e.label == "reset" and e.outcome == "OK" for e in client_env.entries)


def test_ledger_recent(client_env):
    client_env.record(
        LedgerEntry(
            time=datetime.now(UTC),
            source="web",
            session_id="demo",
            label="screen",
            outcome="ALLOW",
            cost_gbp=0.0001,
        )
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.get("/api/ledger/recent?limit=10")
        assert res.status_code == 200
        entries = res.json()
        assert len(entries) >= 1
        assert entries[-1]["label"] == "screen"
        assert entries[-1]["outcome"] == "ALLOW"


# C-02: Security tests (Host and Origin validation)
def test_host_header_validation(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Invalid host should be rejected
        res = client.get("/health", headers={"host": "evil.example:8765"})
        assert res.status_code == 400


def test_origin_header_validation(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Evil origin on POST is rejected with 403 (C-02, C-13)
        res = client.post(
            "/api/budget/demo/reset",
            headers={"origin": "http://evil.example", "X-Requested-With": "AgentGuard"},
        )
        assert res.status_code == 403

        # Legitimate origin is accepted
        good_res = client.post(
            "/api/budget/demo/reset",
            headers={"origin": "http://127.0.0.1:8765", "X-Requested-With": "AgentGuard"},
        )
        assert good_res.status_code == 200


# C-14: Input validation and bounds
def test_input_validation_bounds(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Empty prompt without preset
        r1 = client.post("/api/run", json={"prompt": ""})
        assert r1.status_code == 400

        # Prompt with surrogates rejected
        r2 = client.post(
            "/api/run",
            content=b'{"prompt": "bad\\ud800prompt"}',
            headers={"content-type": "application/json"},
        )
        assert r2.status_code == 422

        # Prompt exceeding 20,000 characters rejected
        r3 = client.post("/api/run", json={"prompt": "x" * 20_001})
        assert r3.status_code == 422

        # Negative budget cap rejected
        r4 = client.post("/api/run", json={"prompt": "test", "budget_cap_gbp": -1.0})
        assert r4.status_code == 422

        # NaN budget cap rejected
        r5 = client.post(
            "/api/run",
            content=b'{"prompt": "test", "budget_cap_gbp": "NaN"}',
            headers={"content-type": "application/json"},
        )
        assert r5.status_code == 422

        # Ledger limit > 200 rejected
        r6 = client.get("/api/ledger/recent?limit=250")
        assert r6.status_code == 422


# C-01: default_model_caller with disabled AFC
async def test_default_model_caller_disables_afc(monkeypatch):
    from google import genai
    from google.genai import types

    call_count = 0
    captured_config = None

    class FakeResponse:
        candidates = None
        usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=45,
            candidates_token_count=20,
            thoughts_token_count=0,
        )
        function_calls = [
            types.FunctionCall(name="execute_python", args={"code": "print(42)"})
        ]
        text = None

    class FakeAioModels:
        async def generate_content(self, model, contents, config):
            nonlocal call_count, captured_config
            call_count += 1
            captured_config = config
            return FakeResponse()

    class FakeAioClient:
        models = FakeAioModels()
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass

    class FakeGenaiClient:
        def __init__(self, api_key):
            self.aio = FakeAioClient()

    monkeypatch.setattr(genai, "Client", FakeGenaiClient)

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    settings = get_settings()

    usage, text, tool_call = await default_model_caller("Calculate 6*7", settings)

    assert call_count == 1
    assert captured_config is not None
    assert captured_config.automatic_function_calling.disable is True
    assert usage.prompt_tokens == 45
    assert usage.candidates_tokens == 20
    assert tool_call == {"tool_name": "execute_python", "args": {"code": "print(42)"}}
    assert text is None


# C-10: Real-time budget_reserved event emission before step_finished
async def test_budget_reserved_event_timing(client_env, monkeypatch):
    events_received: list[str] = []

    async def slow_evaluate_action(tool_name, args, session_id, **kwargs):
        cb = kwargs.get("on_event")
        if cb:
            cb("budget_reserved", client_env.check_budget(session_id))
        await asyncio.sleep(0.05)
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_model_caller(prompt, settings):
        return (
            TokenUsage(prompt_tokens=10, candidates_tokens=5, thoughts_tokens=0),
            "Done",
            None,
        )

    monkeypatch.setattr(guard_core, "evaluate_action", slow_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_model_caller)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test timing"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        types = [l["type"] for l in lines]
        idx_reserved = types.index("budget_reserved")
        idx_step_finished = types.index("step_finished")
        assert idx_reserved < idx_step_finished


# C-24: Refused reservation gates check
def test_run_refused_reservation_exhausted_cap(client_env, monkeypatch):
    client_env.is_exceeded_flag = True

    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        refused_step = lines[3]
        # Since cap is already exhausted, gates_passed must be empty (C-24)
        assert refused_step["result"]["guard"]["stage"] == "BUDGET"
        assert refused_step["result"]["guard"]["gates_passed"] == []
        assert "already exhausted" in refused_step["result"]["guard"]["reason"]


def test_run_refused_reservation_would_exceed_cap(client_env, monkeypatch):
    client_env.reserve_fail = True
    client_env.is_exceeded_flag = False

    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        refused_step = lines[3]
        # When would exceed, gates_passed must include REGEX (C-24)
        assert refused_step["result"]["guard"]["stage"] == "BUDGET"
        assert refused_step["result"]["guard"]["gates_passed"] == ["REGEX"]
        assert "reservation would exceed cap" in refused_step["result"]["guard"]["reason"]


# C-08: Reservation release on cancellation / error
def test_run_reservation_released_on_model_error(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_failing_model(prompt, settings):
        raise RuntimeError("Model server crashed")

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_failing_model)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test error"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        assert any(l["type"] == "run_error" for l in lines)
        assert len(client_env.released) == 1
        assert any(e.label == "agent turn" and e.outcome == "ERROR" for e in client_env.entries)


# C-09: Fail-closed guard error on prompt screen
def test_run_prompt_screen_guard_error_handled(client_env, monkeypatch):
    async def broken_evaluate_action(*args, **kwargs):
        raise OSError("Classifier connection refused")

    monkeypatch.setattr(guard_core, "evaluate_action", broken_evaluate_action)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test guard error"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        assert lines[2]["type"] == "step_finished"
        assert lines[2]["result"]["guard"]["decision"] == "BLOCK"
        assert lines[2]["result"]["guard"]["owasp_tag"] == "N/A"
        assert lines[2]["result"]["guard"]["reason"] == "Guard error. Failing closed."
        assert any(e.outcome == "BLOCK error" for e in client_env.entries)


# C-12: Unique loop-N sessions across runs and loop step intent format
def test_preset_3_unique_session_ids(client_env, monkeypatch):
    async def fake_evaluate_action(*args, **kwargs):
        return mock_guard_eval(decision=Decision.BLOCK, stage=Stage.BUDGET)

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res1 = client.post("/api/run", json={"preset": "loop"})
        res2 = client.post("/api/run", json={"preset": "loop"})
        lines1 = [json.loads(l) for l in res1.text.strip().split("\n") if l]
        lines2 = [json.loads(l) for l in res2.text.strip().split("\n") if l]
        sess1 = lines1[0]["session_id"]
        sess2 = lines2[0]["session_id"]
        assert sess1.startswith("loop-")
        assert sess2.startswith("loop-")
        assert sess1 != sess2
        # Verify loop step intent format matches Figma: "Loop step n of N."
        assert lines1[1]["type"] == "step_started"
        assert lines1[1]["intent"] == "Loop step 1 of 6."


def test_step_intents_standard_run(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_model_caller(prompt, settings):
        return (
            TokenUsage(prompt_tokens=10, candidates_tokens=5, thoughts_tokens=0),
            None,
            {"tool_name": "execute_python", "args": {"code": "print(42)"}},
        )

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_model_caller)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "Calculate compound growth"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        # Step 0 (prompt screen) intent
        assert lines[1]["type"] == "step_started"
        assert lines[1]["intent"] == "Screen the agent prompt for security policy compliance."
        assert lines[2]["type"] == "step_finished"
        assert lines[2]["result"]["intent"] == "Screen the agent prompt for security policy compliance."

        # Step 1 (tool execution) intent
        assert lines[3]["type"] == "step_started"
        assert lines[3]["intent"] == "Execute the proposed Python code in the sandbox environment."
        assert lines[4]["type"] == "step_finished"
        assert lines[4]["result"]["intent"] == "Execute the proposed Python code in the sandbox environment."

        # Assert committed amounts (C-20)
        assert len(client_env.commits) == 1
        res_id, usage = client_env.commits[0]
        assert usage.prompt_tokens == 10
        assert usage.candidates_tokens == 5
        agent_entries = [e for e in client_env.entries if e.label == "agent turn" and e.outcome == "OK"]
        assert len(agent_entries) == 1
        assert agent_entries[0].cost_gbp == 0.0001


# C-20: Cancellation test (reservation released, agent turn ERROR)
def test_run_cancellation_releases_reservation(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_cancelled_model(prompt, settings):
        raise asyncio.CancelledError()

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_cancelled_model)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test cancel"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        assert any(l["type"] == "run_error" for l in lines)
        assert any("cancelled" in l.get("message", "").lower() for l in lines if l["type"] == "run_error")
        assert len(client_env.released) == 1
        assert any(e.label == "agent turn" and e.outcome == "ERROR" for e in client_env.entries)


# C-11, C-20: Nested-field assertions on streamed events
def test_streamed_event_nested_fields(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW, cost=0.0001)

    async def fake_model_caller(prompt, settings):
        return (
            TokenUsage(prompt_tokens=10, candidates_tokens=5, thoughts_tokens=0),
            None,
            {"tool_name": "execute_python", "args": {"code": "print(1)"}},
        )

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_model_caller)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test nested fields"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        step_finished_events = [l for l in lines if l["type"] == "step_finished"]
        assert len(step_finished_events) >= 2

        # Verify step 0 (prompt screen) nested fields (C-11)
        step0 = step_finished_events[0]
        res0 = step0["result"]
        assert "intent" in res0
        assert "guard" in res0
        assert "decision" in res0["guard"]
        assert "risk_score" in res0["guard"]
        assert "owasp_tag" in res0["guard"]
        assert "reason" in res0["guard"]
        assert "stage" in res0["guard"]
        assert "latency_ms" in res0["guard"]
        assert "gates_passed" in res0["guard"]
        assert "cost_gbp" in res0["guard"]
        assert "budget" in res0
        assert "spent_gbp" in res0["budget"]
        assert "cap_gbp" in res0["budget"]
        assert "remaining_gbp" in res0["budget"]
        assert "is_exceeded" in res0["budget"]

        # Verify step 1 (sandbox execution) nested fields (C-11)
        step1 = step_finished_events[1]
        res1 = step1["result"]
        assert "guard" in res1
        assert "sandbox" in res1
        assert "exit_code" in res1["sandbox"]
        assert "stdout" in res1["sandbox"]
        assert "stderr" in res1["sandbox"]
        assert "duration_ms" in res1["sandbox"]


# N-1: Model text with lone surrogates ends with run_error
def test_run_model_text_with_surrogates_ends_with_run_error(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_surrogate_model(prompt, settings):
        return (
            TokenUsage(prompt_tokens=10, candidates_tokens=5, thoughts_tokens=0),
            "Bad text with surrogate \ud800",
            None,
        )

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_surrogate_model)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test surrogates"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        assert any(l["type"] == "run_error" for l in lines)


# D-2: Model tool code with lone surrogates ends with run_error
def test_run_tool_code_with_surrogates_ends_with_run_error(client_env, monkeypatch):
    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    async def fake_surrogate_tool_model(prompt, settings):
        return (
            TokenUsage(prompt_tokens=10, candidates_tokens=5, thoughts_tokens=0),
            None,
            {"tool_name": "execute_python", "args": {"code": "print('\ud800')"}},
        )

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
    monkeypatch.setattr(agent_loop, "model_caller", fake_surrogate_tool_model)

    with TestClient(app, base_url="http://127.0.0.1") as client:
        res = client.post("/api/run", json={"prompt": "test tool surrogates"})
        assert res.status_code == 200
        lines = [json.loads(line) for line in res.text.strip().split("\n") if line]
        assert any(l["type"] == "run_error" for l in lines)


# C-21: Route exact /mcp without 307 redirect
async def test_mcp_exact_route_no_redirect(client_env):
    import httpx
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            res = await client.post("/mcp", headers={"content-type": "application/json"}, json={})
            # Must not be a 307 redirect
            assert res.status_code != 307


# MCP streamable HTTP tests
async def test_mcp_streamable_http_tools_and_calls(client_env, monkeypatch):
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async def fake_evaluate_action(tool_name, args, session_id, **kwargs):
        return mock_guard_eval(decision=Decision.ALLOW)

    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", follow_redirects=True) as http_client:
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    tool_names = [t.name for t in tools_result.tools]
                    assert "guard_check_action" in tool_names
                    assert "guard_execute_sandbox" in tool_names
                    assert "guard_get_budget_status" in tool_names

                    # Test guard_get_budget_status
                    budget_res = await session.call_tool("guard_get_budget_status", {"session_id": "test_sess"})
                    assert budget_res.structuredContent["session_id"] == "test_sess"

                    # Test guard_check_action with invalid JSON
                    check_bad = await session.call_tool("guard_check_action", {"tool_name": "execute_python", "args_json": "not json"})
                    assert check_bad.structuredContent["decision"] == "BLOCK"
                    assert any(e.source == "mcp" and e.label == "check" and e.outcome == "BLOCK error" for e in client_env.entries)

                    # Test guard_check_action with valid JSON
                    check_good = await session.call_tool("guard_check_action", {"tool_name": "execute_python", "args_json": '{"code": "print(1)"}'})
                    assert check_good.structuredContent["decision"] == "ALLOW"

                    # Test guard_check_action internal error sanitization
                    async def broken_evaluate(*args, **kwargs):
                        raise RuntimeError("DB dropped")
                    monkeypatch.setattr(guard_core, "evaluate_action", broken_evaluate)
                    check_err = await session.call_tool("guard_check_action", {"tool_name": "execute_python", "args_json": '{"code": "print(1)"}'})
                    assert check_err.structuredContent["decision"] == "BLOCK"
                    assert check_err.structuredContent["reason"] == "Guard error. Failing closed."

                    # Test guard_execute_sandbox
                    monkeypatch.setattr(guard_core, "evaluate_action", fake_evaluate_action)
                    exec_res = await session.call_tool("guard_execute_sandbox", {"code": "print('hello')", "session_id": "test_sess"})
                    assert "guard" in exec_res.structuredContent
                    assert any(e.source == "mcp" and e.label == "execute" for e in client_env.entries)


# C-14, D-1: Validate session_id in MCP tools and ensure 'invalid' placeholder is recorded
async def test_mcp_invalid_session_ids_record_placeholder(client_env):
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    long_id = "s" * 1000
    newline_id = "a\nweb  demo  screen  ALLOW\n\n\n"
    trailing_newline_id = "demo\n"
    max_len_trailing_newline_id = ("x" * 64) + "\n"

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", follow_redirects=True) as http_client:
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Test guard_check_action with long id
                    res_check = await session.call_tool("guard_check_action", {"tool_name": "execute_python", "args_json": "{}", "session_id": long_id})
                    assert res_check.structuredContent["decision"] == "BLOCK"
                    assert any(e.source == "mcp" and e.session_id == "invalid" and e.label == "check" for e in client_env.entries)
                    assert not any(long_id in e.session_id for e in client_env.entries)

                    # Test guard_execute_sandbox with newline id
                    res_exec = await session.call_tool("guard_execute_sandbox", {"code": "print(1)", "session_id": newline_id})
                    assert res_exec.structuredContent["decision"] == "BLOCK"
                    assert any(e.source == "mcp" and e.session_id == "invalid" and e.label == "execute" for e in client_env.entries)
                    assert not any(newline_id in e.session_id for e in client_env.entries)

                    # Test trailing newline id (demo\n)
                    res_trail = await session.call_tool("guard_check_action", {"tool_name": "execute_python", "args_json": "{}", "session_id": trailing_newline_id})
                    assert res_trail.structuredContent["decision"] == "BLOCK"
                    assert not any(trailing_newline_id in e.session_id for e in client_env.entries)

                    # Test 64 chars + newline
                    res_64trail = await session.call_tool("guard_execute_sandbox", {"code": "print(1)", "session_id": max_len_trailing_newline_id})
                    assert res_64trail.structuredContent["decision"] == "BLOCK"
                    assert not any(max_len_trailing_newline_id in e.session_id for e in client_env.entries)

                    # Test guard_get_budget_status with newline id returns error
                    res_budget = await session.call_tool("guard_get_budget_status", {"session_id": newline_id})
                    assert "error" in res_budget.structuredContent
                    assert any(e.source == "mcp" and e.session_id == "invalid" and e.label == "budget" for e in client_env.entries)
                    assert not any(newline_id in e.session_id for e in client_env.entries)

                    # Test guard_get_budget_status with trailing newline id returns error
                    res_budget_trail = await session.call_tool("guard_get_budget_status", {"session_id": trailing_newline_id})
                    assert "error" in res_budget_trail.structuredContent
                    assert not any(trailing_newline_id in e.session_id for e in client_env.entries)


# C-14, D-1: Validate session_id in budget endpoints
def test_budget_endpoints_invalid_session_ids(client_env):
    with TestClient(app, base_url="http://127.0.0.1") as client:
        # Long session_id
        res_get = client.get(f"/api/budget/{'x' * 100}")
        assert res_get.status_code == 422

        # Newline session_id in middle
        res_get2 = client.get("/api/budget/a%0Atest")
        assert res_get2.status_code == 422

        res_reset = client.post("/api/budget/a%0Atest/reset", headers={"X-Requested-With": "AgentGuard"})
        assert res_reset.status_code == 422

        # Trailing newline session_id (demo\n)
        res_get3 = client.get("/api/budget/demo%0A")
        assert res_get3.status_code == 422

        res_reset2 = client.post("/api/budget/demo%0A/reset", headers={"X-Requested-With": "AgentGuard"})
        assert res_reset2.status_code == 422

        # 64 chars + \n (65 chars total)
        long_trail = ("x" * 64) + "%0A"
        res_get4 = client.get(f"/api/budget/{long_trail}")
        assert res_get4.status_code == 422

        res_reset3 = client.post(f"/api/budget/{long_trail}/reset", headers={"X-Requested-With": "AgentGuard"})
        assert res_reset3.status_code == 422

        # Ensure raw newline was never written to ledger
        assert not any("\n" in e.session_id for e in client_env.entries)


# C-15, C-20: MCP stdio tests using sys.executable, explicit cwd, and temporary SQLITE_DB_PATH
async def test_mcp_stdio_subprocess(tmp_path):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    db_path = str(tmp_path / "stdio_test.db")
    env = dict(os.environ, SQLITE_DB_PATH=db_path)

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(PROJECT_ROOT / "app" / "mcp_stdio.py")],
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            tool_names = [t.name for t in tools_result.tools]
            assert "guard_check_action" in tool_names
            assert "guard_execute_sandbox" in tool_names
            assert "guard_get_budget_status" in tool_names

            res = await session.call_tool("guard_check_action", {"tool_name": "test", "args_json": "{invalid"})
            assert res.structuredContent["decision"] == "BLOCK"

    # Read the temporary database and assert the mcp entry is recorded in it (C-20)
    con = sqlite3.connect(db_path)
    rows = con.execute("SELECT entry FROM entries").fetchall()
    con.close()
    assert len(rows) > 0
    assert any("mcp" in r[0] for r in rows)


async def test_mcp_invalid_args_with_valid_session_id_records_caller_session(client_env):
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", follow_redirects=True) as http_client:
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # Invalid JSON with valid session_id="caller_session"
                    res = await session.call_tool(
                        "guard_check_action",
                        {"tool_name": "execute_python", "args_json": "not-json", "session_id": "caller_session"},
                    )
                    assert res.structuredContent["decision"] == "BLOCK"
                    assert any(
                        e.source == "mcp" and e.session_id == "caller_session" and e.label == "check" and e.outcome == "BLOCK error"
                        for e in client_env.entries
                    )

                    # Invalid provider with valid session_id="caller_exec_session"
                    res_exec = await session.call_tool(
                        "guard_execute_sandbox",
                        {"code": "print(1)", "session_id": "caller_exec_session", "provider": "x" * 100},
                    )
                    assert res_exec.structuredContent["decision"] == "BLOCK"
                    assert any(
                        e.source == "mcp" and e.session_id == "caller_exec_session" and e.label == "execute" and e.outcome == "BLOCK error"
                        for e in client_env.entries
                    )


# 1. MCP server tracker failure handling
async def test_mcp_tracker_failure_returns_sanitized_block(monkeypatch):
    import httpx
    from mcp.client.session import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    def broken_get_tracker():
        raise RuntimeError("database locked")

    monkeypatch.setattr(budget_store, "get_tracker", broken_get_tracker)

    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", follow_redirects=True) as http_client:
            async with streamable_http_client("http://127.0.0.1/mcp", http_client=http_client) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()

                    # guard_check_action with valid arguments
                    res_valid = await session.call_tool(
                        "guard_check_action",
                        {"tool_name": "execute_python", "args_json": '{"code": "print(1)"}', "session_id": "test_s"},
                    )
                    assert res_valid.structuredContent["decision"] == "BLOCK"
                    assert res_valid.structuredContent["reason"] == "Guard error. Failing closed."

                    # guard_check_action with invalid arguments
                    res_inv = await session.call_tool(
                        "guard_check_action",
                        {"tool_name": "execute_python", "args_json": "invalid-json", "session_id": "test_s"},
                    )
                    assert res_inv.structuredContent["decision"] == "BLOCK"
                    assert res_inv.structuredContent["reason"] == "Guard error. Failing closed."

                    # guard_execute_sandbox with valid code -> returns BLOCK and runs nothing
                    res_exec_valid = await session.call_tool(
                        "guard_execute_sandbox",
                        {"code": "print(1)", "session_id": "test_s"},
                    )
                    assert res_exec_valid.structuredContent["decision"] == "BLOCK"
                    assert res_exec_valid.structuredContent["reason"] == "Guard error. Failing closed."

                    # guard_execute_sandbox with invalid arguments
                    res_exec_inv = await session.call_tool(
                        "guard_execute_sandbox",
                        {"code": "print(1)", "session_id": "bad\n", "provider": "x" * 100},
                    )
                    assert res_exec_inv.structuredContent["decision"] == "BLOCK"
                    assert res_exec_inv.structuredContent["reason"] == "Guard error. Failing closed."

                    # guard_get_budget_status with valid session
                    res_budget_valid = await session.call_tool(
                        "guard_get_budget_status",
                        {"session_id": "test_s"},
                    )
                    assert res_budget_valid.structuredContent.get("error") == "Guard error. Failing closed."

                    # guard_get_budget_status with invalid session
                    res_budget_inv = await session.call_tool(
                        "guard_get_budget_status",
                        {"session_id": "bad\n"},
                    )
                    assert "error" in res_budget_inv.structuredContent


def test_preset3_default_cap():
    assert get_settings().preset3_session_cap_gbp == 0.0017
