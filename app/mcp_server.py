"""AgentGuard MCP server (PLAN.md sections 2, 5, 9, C.1).

Exposes FastMCP tools for policy checking, sandboxed execution, and budget status.
"""

import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from core import budget_store, guard_core
from core.pipeline import execute_guarded
from core.schemas import LedgerEntry, error_block

logger = logging.getLogger(__name__)

# DNS rebinding protection enabled with loopback allowed hosts & origins (C-02)
_transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=[
        "127.0.0.1",
        "127.0.0.1:*",
        "localhost",
        "localhost:*",
        "testserver",
        "testserver:*",
    ],
    allowed_origins=[
        "http://127.0.0.1",
        "http://127.0.0.1:*",
        "http://localhost",
        "http://localhost:*",
        "http://testserver",
        "http://testserver:*",
        "https://127.0.0.1",
        "https://127.0.0.1:*",
        "https://localhost",
        "https://localhost:*",
        "https://testserver",
        "https://testserver:*",
    ],
)

mcp = FastMCP(
    "AgentGuard",
    transport_security=_transport_security,
    streamable_http_path="/",
)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


def _validate_session_id(val: Any) -> str:
    if not isinstance(val, str) or not SESSION_ID_PATTERN.fullmatch(val):
        raise ValueError("session_id must match ^[A-Za-z0-9_.-]{1,64}$")
    return val


def _safe_session_id(val: Any) -> str:
    if isinstance(val, str) and SESSION_ID_PATTERN.fullmatch(val):
        return val
    return "invalid"


def _validate_arg_str(val: str, name: str, max_len: int = 20_000) -> str:
    if not isinstance(val, str):
        raise TypeError(f"{name} must be text")
    if len(val) > max_len:
        raise ValueError(f"{name} exceeds maximum length of {max_len}")
    try:
        val.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} contains invalid UTF-8 surrogates.") from exc
    return val


@mcp.tool()
async def guard_check_action(
    tool_name: str, args_json: str, session_id: str = "default"
) -> dict[str, Any]:
    """Screen an action before execution without running code."""
    try:
        tracker = budget_store.get_tracker()
    except Exception:
        logger.exception("MCP guard_check_action budget tracker unavailable")
        return error_block("Guard error. Failing closed.").model_dump()

    try:
        tool_name = _validate_arg_str(tool_name, "tool_name", max_len=64)
        args_json = _validate_arg_str(args_json, "args_json", max_len=20_000)
        session_id = _validate_session_id(session_id)
        args = json.loads(args_json)
        if not isinstance(args, dict):
            raise ValueError("args_json must be a JSON object")
    except Exception as exc:
        try:
            await tracker.arecord(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="mcp",
                    session_id=_safe_session_id(session_id),
                    label="check",
                    outcome="BLOCK error",
                    cost_gbp=0.0,
                ),
            )
        except Exception:
            pass
        return error_block(f"Invalid request: {exc}.").model_dump()

    try:
        eval_result = await guard_core.evaluate_action(
            tool_name=tool_name,
            args=args,
            session_id=session_id,
            source="mcp",
            label="check",
            record=True,
        )
        return eval_result.model_dump()
    except Exception as exc:
        logger.exception("MCP guard_check_action failed")
        try:
            await tracker.arecord(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="mcp",
                    session_id=session_id,
                    label="check",
                    outcome="BLOCK error",
                    cost_gbp=0.0,
                ),
            )
        except Exception:
            pass
        return error_block("Guard error. Failing closed.").model_dump()


@mcp.tool()
async def guard_execute_sandbox(
    code: str, session_id: str = "default", provider: str | None = None
) -> dict[str, Any]:
    """Execute Python code in the sandbox through the guarded pipeline."""
    try:
        tracker = budget_store.get_tracker()
    except Exception:
        logger.exception("MCP guard_execute_sandbox budget tracker unavailable")
        return error_block("Guard error. Failing closed.").model_dump()

    try:
        code = _validate_arg_str(code, "code", max_len=20_000)
        session_id = _validate_session_id(session_id)
        if provider is not None:
            provider = _validate_arg_str(provider, "provider", max_len=32)
    except Exception as exc:
        try:
            await tracker.arecord(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="mcp",
                    session_id=_safe_session_id(session_id),
                    label="execute",
                    outcome="BLOCK error",
                    cost_gbp=0.0,
                ),
            )
        except Exception:
            pass
        return error_block(f"Invalid request: {exc}.").model_dump()

    try:
        result = await execute_guarded(
            intent="MCP sandbox execution",
            tool_name="execute_python",
            args={"code": code},
            session_id=session_id,
            provider=provider,
            source="mcp",
            label="execute",
        )
        return result.model_dump()
    except Exception:
        logger.exception("MCP guard_execute_sandbox failed")
        return error_block("Guard error. Failing closed.").model_dump()


@mcp.tool()
async def guard_get_budget_status(session_id: str = "default") -> dict[str, Any]:
    """Get the current budget status for a session."""
    try:
        tracker = budget_store.get_tracker()
    except Exception:
        logger.exception("MCP guard_get_budget_status budget tracker unavailable")
        return {"error": "Guard error. Failing closed."}

    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        try:
            await tracker.arecord(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="mcp",
                    session_id="invalid",
                    label="budget",
                    outcome="BLOCK error",
                    cost_gbp=0.0,
                ),
            )
        except Exception:
            pass
        return {"error": "Invalid session_id", "session_id": "invalid"}

    try:
        status = await tracker.acheck_budget(session_id)
        return status.model_dump()
    except Exception:
        logger.exception("MCP check_budget failed")
        return {"error": "Guard error. Failing closed."}
