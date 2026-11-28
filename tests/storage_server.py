"""The real app under real uvicorn with fake models and a fake sandbox.

Run as a script: storage_server.py PORT. It measures its own event loop lag and
notes any synchronous ledger method that runs on an event loop thread. Both are
read from GET /__test__/state. No .env is read and no model is called.
"""

import asyncio
import functools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.config

core.config.dotenv_values = lambda *args, **kwargs: {}

import uvicorn

from app import agent_loop
from app.main import app
from core import budget_store, guard_core, pipeline
from core.providers import ModelReply, ToolCall
from core.schemas import SandboxResult, TokenUsage

SYNC_METHODS = (
    "_create_schema", "check_budget", "reserve", "commit", "release", "set_session_cap",
    "reset", "reset_client", "record", "recent_entries", "apply_retention", "backup",
)
VERDICT = (
    '{"decision": "ALLOW", "risk_score": 1, "owasp_tag": "SAFE: Compliant Action",'
    ' "reason": "fake"}'
)
state = {"max_lag_ms": 0.0, "violations": [], "calls": {}, "agent_calls": 0}


def watch(name, method):
    @functools.wraps(method)
    def watched(*args, **kwargs):
        state["calls"][name] = state["calls"].get(name, 0) + 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            state["violations"].append(name)
        return method(*args, **kwargs)

    return watched


class FakeModel:
    def __init__(self, settings):
        self.settings = settings

    async def classify(self, system, payload, schema):
        await asyncio.sleep(0.5)
        if "fail-classify" in payload:
            raise RuntimeError("fake classifier failure")
        # Exactly what was reserved, so spend plus holds never moves.
        reserve_input = core.config.get_settings().budget_reserve_input_tokens
        usage = TokenUsage(
            prompt_tokens=reserve_input + len(payload.encode()),
            candidates_tokens=self.settings.max_output_tokens,
        )
        return ModelReply(usage, text=VERDICT, finished=True)

    async def agent_turn(self, system, prompt, tools):
        state["agent_calls"] += 1
        await asyncio.sleep(60 if "slow-agent" in prompt else 0.05)
        if "fail-agent" in prompt:
            raise RuntimeError("fake agent failure")
        usage = TokenUsage(prompt_tokens=10, candidates_tokens=5)
        return ModelReply(usage, tool_call=ToolCall("execute_python", {"code": "print(1)"}), finished=True)


class FakeSandbox:
    provider_name = "docker"

    async def execute(self, request):
        return SandboxResult(
            provider="docker", stdout="1\n", stderr="", truncated=False,
            exit_code=0, duration_ms=1, timed_out=False,
        )


@app.get("/__test__/state")
async def read_state(reset: bool = False):
    seen = {**state, "violations": list(state["violations"]), "calls": dict(state["calls"])}
    if reset:
        state["max_lag_ms"] = 0.0
    return seen


async def watch_lag():
    loop = asyncio.get_running_loop()
    while True:
        start = loop.time()
        await asyncio.sleep(0.005)
        state["max_lag_ms"] = max(state["max_lag_ms"], (loop.time() - start - 0.005) * 1000)


async def main(port: int) -> int:
    for name in SYNC_METHODS:
        method = getattr(budget_store.SessionBudgetTracker, name)
        setattr(budget_store.SessionBudgetTracker, name, watch(name, method))
    guard_core.get_provider = FakeModel
    agent_loop.get_provider = FakeModel
    pipeline.get_sandbox_provider = lambda override=None: FakeSandbox()
    monitor = asyncio.create_task(watch_lag())
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    try:
        await server.serve()
    finally:
        monitor.cancel()
    return 0 if server.started else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main(int(sys.argv[1]))))
