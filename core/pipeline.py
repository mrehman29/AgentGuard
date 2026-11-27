"""The only path from a request to code execution (PLAN.md section 2).

Only ALLOW executes. Every guard error ends in BLOCK. Frozen after Phase 0.
"""

import asyncio
import copy
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError

from core import budget_store, guard_core
from core.schemas import (
    EXECUTABLE_TOOL,
    MAX_OUTPUT_CHARS,
    BudgetStatus,
    Decision,
    GuardEvaluation,
    LedgerEntry,
    PipelineStepResult,
    SandboxRequest,
    SandboxResult,
    Source,
    error_block,
    ledger_outcome,
)
from sandbox import get_sandbox_provider

logger = logging.getLogger(__name__)

SANDBOX_ERROR_EXIT_CODE = 125


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _sandbox_request(tool_name: str, args: dict) -> SandboxRequest:
    if tool_name != EXECUTABLE_TOOL:
        raise ValueError(f"only the {EXECUTABLE_TOOL} tool can be executed")
    unknown = set(args) - set(SandboxRequest.model_fields)
    if unknown:
        raise ValueError("unexpected tool arguments")
    return SandboxRequest.model_validate(args, strict=True)


def _unavailable_budget(session_id: str) -> BudgetStatus:
    """Shown when the ledger cannot be read: nothing may be spent."""
    return BudgetStatus(
        session_id=session_id,
        prompt_tokens=0,
        candidates_tokens=0,
        thoughts_tokens=0,
        total_tokens=0,
        spent_gbp=0.0,
        reserved_gbp=0.0,
        cap_gbp=0.0,
        remaining_gbp=0.0,
        global_spent_gbp=0.0,
        global_cap_gbp=0.0,
        is_exceeded=True,
    )


def _check_caller_values(intent: str, tool_name: str, session_id: str, args: dict) -> None:
    """Reject values that could not be put in the result or serialised."""
    if not all(isinstance(value, str) for value in (intent, tool_name, session_id)):
        raise TypeError("intent, tool_name and session_id must be text")
    if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
        raise TypeError("tool arguments must be an object")
    values = [intent, tool_name, session_id, args]
    json.dumps(values, ensure_ascii=False, allow_nan=False).encode()


def _safe_text(value: object) -> str:
    return str(value).encode("utf-8", "replace").decode()


def _record(session_id: str, source: Source, label: str, guard: GuardEvaluation) -> None:
    budget_store.get_tracker().record(
        LedgerEntry(
            time=datetime.now(UTC),
            source=source,
            session_id=session_id,
            label=label,
            outcome=ledger_outcome(guard),
            cost_gbp=guard.cost_gbp,
        )
    )


async def execute_guarded(
    intent: str,
    tool_name: str,
    args: dict,
    session_id: str,
    provider: str | None = None,
    source: Source = "web",
    label: str = "step 1/1",
    on_event: Callable[[str, Any], None] | None = None,
) -> PipelineStepResult:
    """Evaluate one action, record it once, and execute it only if allowed."""
    start = time.perf_counter()
    request = None
    sandbox_provider = None
    guard: GuardEvaluation
    tool_args: dict = {}

    try:
        # Snapshot first: what is evaluated is exactly what runs.
        _check_caller_values(intent, tool_name, session_id, args)
        tool_args = copy.deepcopy(args)
        request = _sandbox_request(tool_name, tool_args)
        sandbox_provider = get_sandbox_provider(provider)
    except (ValidationError, ValueError, TypeError) as exc:
        # UnicodeError and ValidationError are ValueErrors with unsafe messages.
        if isinstance(exc, (ValidationError, UnicodeError)):
            reason = "Invalid arguments."
        else:
            reason = _safe_text(f"Invalid request: {exc}.")
        guard = error_block(reason, latency_ms=_elapsed_ms(start))
    except Exception:
        logger.exception("request setup failed")
        guard = error_block("Guard error: request setup failed.", latency_ms=_elapsed_ms(start))
    else:
        try:
            guard = await guard_core.evaluate_action(
                tool_name,
                copy.deepcopy(tool_args),
                session_id,
                source=source,
                label=label,
                on_event=on_event,
                record=False,
            )
        except asyncio.CancelledError:
            cancelled = error_block("Guard error: cancelled.", latency_ms=_elapsed_ms(start))
            try:
                _record(session_id, source, label, cancelled)
            except Exception:
                logger.exception("ledger update failed")
            raise
        except Exception:
            logger.exception("guard evaluation failed")
            guard = error_block("Guard error: evaluation failed.", latency_ms=_elapsed_ms(start))

    intent, tool_name = _safe_text(intent), _safe_text(tool_name)
    session_id = _safe_text(session_id)
    try:
        json.dumps(tool_args, ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError):
        tool_args = {}

    # Read first, then record once, so the entry always matches the verdict.
    def ledger_error(previous: GuardEvaluation) -> GuardEvaluation:
        logger.exception("ledger update failed")
        return error_block(
            "Guard error: ledger unavailable.",
            gates_passed=list(previous.gates_passed),
            latency_ms=previous.latency_ms,
            cost_gbp=previous.cost_gbp,
        )

    try:
        budget_store.get_tracker().check_budget(session_id)
    except Exception:
        guard = ledger_error(guard)
    try:
        _record(session_id, source, label, guard)
    except Exception:
        guard = ledger_error(guard)

    result = None
    if guard.decision is Decision.ALLOW and request is not None and sandbox_provider is not None:
        run_start = time.perf_counter()
        try:
            result = await sandbox_provider.execute(request)
        except Exception as exc:
            logger.exception("sandbox execution failed")
            message = f"Sandbox error: {type(exc).__name__}."
            result = SandboxResult(
                provider=sandbox_provider.provider_name,
                stdout="",
                stderr=message[:MAX_OUTPUT_CHARS],
                truncated=False,
                exit_code=SANDBOX_ERROR_EXIT_CODE,
                duration_ms=_elapsed_ms(run_start),
                timed_out=False,
            )

    try:
        budget = budget_store.get_tracker().check_budget(session_id)
    except Exception:
        logger.exception("budget read failed")
        budget = _unavailable_budget(session_id)

    return PipelineStepResult(
        intent=intent,
        tool_name=tool_name,
        tool_args=tool_args,
        guard=guard,
        sandbox=result,
        budget=budget,
    )
