"""Agent loop implementation (PLAN.md sections 2, 5, 9, C.2).

Screens prompts before execution, reserves and commits model spend,
and executes all proposed tools through core.pipeline.execute_guarded.
"""

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from typing import Any

from google import genai
from google.genai import types

from core import budget_store, guard_core
from core.config import Settings, get_settings, token_cost_gbp
from core.pipeline import execute_guarded
from core.schemas import (
    Decision,
    GuardEvaluation,
    LedgerEntry,
    OwaspTag,
    PipelineStepResult,
    RunEvent,
    Stage,
    TokenUsage,
    error_block,
)

logger = logging.getLogger(__name__)

MAX_PROMPT_CHARS = 20_000


def _validate_text(text: str, name: str = "text") -> str:
    """Validate text and reject surrogates."""
    if not isinstance(text, str):
        raise TypeError(f"{name} must be text")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} contains invalid UTF-8 surrogates.") from exc
    return text


async def default_model_caller(
    prompt: str, settings: Settings
) -> tuple[TokenUsage, str | None, dict[str, Any] | None]:
    """Call Gemini model using google.genai with AFC disabled.

    Returns (TokenUsage, optional_text_reply, optional_tool_call).
    """
    _validate_text(prompt, "prompt")
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    client = genai.Client(api_key=settings.gemini_api_key)

    execute_python_func = types.FunctionDeclaration(
        name="execute_python",
        description="Execute Python code in the secure sandbox environment.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "code": types.Schema(
                    type=types.Type.STRING,
                    description="The Python code to execute.",
                ),
                "timeout_sec": types.Schema(
                    type=types.Type.INTEGER,
                    description="Timeout in seconds (1-5).",
                ),
            },
            required=["code"],
        ),
    )
    tool = types.Tool(function_declarations=[execute_python_func])

    config = types.GenerateContentConfig(
        tools=[tool],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.0,
        max_output_tokens=settings.budget_reserve_output_tokens,
        system_instruction=(
            "You are a helpful Python data assistant. When asked to perform calculations, "
            "data processing, or queries, generate and execute Python code using execute_python."
        ),
    )

    async with client.aio as aio_client:
        response = await asyncio.wait_for(
            aio_client.models.generate_content(
                model=settings.gemini_model,
                contents=prompt,
                config=config,
            ),
            timeout=settings.classifier_timeout_sec,
        )

    usage_meta = response.usage_metadata
    if usage_meta is None or getattr(usage_meta, "prompt_token_count", None) is None:
        raise RuntimeError("Model response missing token usage metadata.")

    prompt_tokens = usage_meta.prompt_token_count or 0
    cand_tokens = usage_meta.candidates_token_count or 0
    thought_tokens = usage_meta.thoughts_token_count or 0
    usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        candidates_tokens=cand_tokens,
        thoughts_tokens=thought_tokens,
    )

    if response.function_calls:
        fc = response.function_calls[0]
        return usage, None, {"tool_name": fc.name, "args": dict(fc.args or {})}

    text = response.text or ""
    return usage, text, None


# Injectable/monkeypatchable caller for testing
model_caller: Callable[
    [str, Settings], Any
] = default_model_caller


async def _run_loop_workflow(
    queue: asyncio.Queue[RunEvent | None],
    run_id: str,
    provider: str | None,
    settings: Settings,
    tracker: budget_store.SessionBudgetTracker,
) -> None:
    """Preset 3 runaway loop in a unique loop session."""
    active_session = f"loop-{uuid.uuid4().hex[:6]}"
    tracker.set_session_cap(active_session, settings.preset3_session_cap_gbp)
    total_steps = settings.preset3_max_steps

    await queue.put(
        RunEvent(
            type="run_started",
            run_id=run_id,
            session_id=active_session,
            preset="loop",
            total_steps=total_steps,
        )
    )

    for step_idx in range(1, total_steps + 1):
        intent = f"Loop step {step_idx} of {total_steps}."
        tool_name = "execute_python"
        tool_args = {
            "code": f"total = sum(i * {step_idx} for i in range(100))\nprint(f'Step {step_idx}: {{total}}')"
        }

        await queue.put(
            RunEvent(
                type="step_started",
                run_id=run_id,
                index=step_idx,
                intent=intent,
                tool_name=tool_name,
                tool_args=tool_args,
            )
        )

        def on_loop_event(ev_type: str, data: Any) -> None:
            if ev_type == "budget_reserved":
                queue.put_nowait(
                    RunEvent(
                        type="budget_reserved",
                        run_id=run_id,
                        index=step_idx,
                        budget=data,
                    )
                )

        step_result = await execute_guarded(
            intent=intent,
            tool_name=tool_name,
            args=tool_args,
            session_id=active_session,
            provider=provider,
            source="web",
            label=f"step {step_idx}/{total_steps}",
            on_event=on_loop_event,
        )

        await queue.put(
            RunEvent(
                type="step_finished",
                run_id=run_id,
                index=step_idx,
                result=step_result,
            )
        )

        if step_result.guard.decision is not Decision.ALLOW:
            break

    await queue.put(
        RunEvent(
            type="run_finished",
            run_id=run_id,
            budget=tracker.check_budget(active_session),
        )
    )


async def _run_standard_workflow(
    queue: asyncio.Queue[RunEvent | None],
    run_id: str,
    prompt: str,
    session_id: str,
    provider: str | None,
    budget_cap_gbp: float | None,
    preset_id: str | None,
    settings: Settings,
    tracker: budget_store.SessionBudgetTracker,
) -> None:
    """Standard run workflow (presets safe, injection, or custom)."""
    active_session = session_id or "demo"
    if budget_cap_gbp is not None:
        tracker.set_session_cap(active_session, budget_cap_gbp)

    total_steps = 1
    await queue.put(
        RunEvent(
            type="run_started",
            run_id=run_id,
            session_id=active_session,
            preset=preset_id,
            total_steps=total_steps,
        )
    )

    # Step 0: prompt screen
    prompt_args = {"prompt": prompt}
    await queue.put(
        RunEvent(
            type="step_started",
            run_id=run_id,
            index=0,
            intent="Screen the agent prompt for security policy compliance.",
            tool_name="agent_prompt",
            tool_args=prompt_args,
        )
    )

    def on_screen_event(ev_type: str, data: Any) -> None:
        if ev_type == "budget_reserved":
            queue.put_nowait(
                RunEvent(
                    type="budget_reserved",
                    run_id=run_id,
                    index=0,
                    budget=data,
                )
            )

    try:
        guard = await guard_core.evaluate_action(
            tool_name="agent_prompt",
            args=prompt_args,
            session_id=active_session,
            source="web",
            label="screen",
            on_event=on_screen_event,
            record=True,
        )
    except Exception as exc:
        logger.exception("Prompt screen guard evaluation failed")
        guard = error_block("Guard error. Failing closed.", latency_ms=0)
        try:
            tracker.record(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="web",
                    session_id=active_session,
                    label="screen",
                    outcome="BLOCK error",
                    cost_gbp=0.0,
                )
            )
        except Exception:
            pass

    step_0_result = PipelineStepResult(
        intent="Screen the agent prompt for security policy compliance.",
        tool_name="agent_prompt",
        tool_args=prompt_args,
        guard=guard,
        sandbox=None,
        budget=tracker.check_budget(active_session),
    )
    await queue.put(
        RunEvent(
            type="step_finished",
            run_id=run_id,
            index=0,
            result=step_0_result,
        )
    )

    # If prompt blocked, stop
    if guard.decision is not Decision.ALLOW:
        await queue.put(
            RunEvent(
                type="run_finished",
                run_id=run_id,
                budget=tracker.check_budget(active_session),
            )
        )
        return

    # Agent model turn
    budget_before = tracker.check_budget(active_session)
    estimated_input_tokens = max(
        len(prompt.encode("utf-8", "replace")) // 4 + 100,
        settings.budget_reserve_input_tokens,
    )
    reserve_cost = token_cost_gbp(
        settings,
        estimated_input_tokens,
        settings.budget_reserve_output_tokens,
    )
    reservation_id = tracker.reserve(active_session, reserve_cost)

    if reservation_id is None:
        # Refused reservation -> check if already exceeded (gates=[]) or would exceed (gates=[REGEX])
        is_already_exceeded = budget_before.is_exceeded
        gates_passed = [] if is_already_exceeded else [Stage.REGEX]
        reason = (
            "Circuit breaker open: budget cap already exhausted."
            if is_already_exceeded
            else "Circuit breaker open: agent turn reservation would exceed cap."
        )

        try:
            tracker.record(
                LedgerEntry(
                    time=datetime.now(UTC),
                    source="web",
                    session_id=active_session,
                    label="agent turn",
                    outcome="BLOCK budget",
                    cost_gbp=0.0,
                )
            )
        except Exception:
            pass

        refused_guard = GuardEvaluation(
            decision=Decision.BLOCK,
            risk_score=100,
            owasp_tag=OwaspTag.NA,
            reason=reason,
            stage=Stage.BUDGET,
            latency_ms=0,
            gates_passed=gates_passed,
            cost_gbp=0.0,
        )
        refused_result = PipelineStepResult(
            intent="Evaluate agent model turn.",
            tool_name="agent_turn",
            tool_args={},
            guard=refused_guard,
            sandbox=None,
            budget=tracker.check_budget(active_session),
        )
        await queue.put(
            RunEvent(
                type="step_finished",
                run_id=run_id,
                index=0,
                result=refused_result,
            )
        )
        await queue.put(
            RunEvent(
                type="run_finished",
                run_id=run_id,
                budget=tracker.check_budget(active_session),
            )
        )
        return

    # Settle reservation with try/finally to prevent reservation leaks
    committed = False
    try:
        usage, text_reply, tool_call = await model_caller(prompt, settings)
        commit_res = tracker.commit(reservation_id, usage)
        committed = True
        tracker.record(
            LedgerEntry(
                time=datetime.now(UTC),
                source="web",
                session_id=active_session,
                label="agent turn",
                outcome="OK",
                cost_gbp=commit_res.cost_gbp,
            )
        )
    except BaseException as exc:
        if not committed:
            try:
                tracker.release(reservation_id)
            except Exception:
                pass
            try:
                tracker.record(
                    LedgerEntry(
                        time=datetime.now(UTC),
                        source="web",
                        session_id=active_session,
                        label="agent turn",
                        outcome="ERROR",
                        cost_gbp=0.0,
                    )
                )
            except Exception:
                pass
        err_msg = (
            "Agent model call cancelled."
            if isinstance(exc, asyncio.CancelledError)
            else "Agent model call failed. Nothing ran."
        )
        await queue.put(RunEvent(type="run_error", run_id=run_id, message=err_msg))
        return

    # Check for lone surrogates in model outputs (N-1)
    has_surrogates = False
    try:
        if text_reply:
            text_reply.encode("utf-8")
        if tool_call:
            json.dumps(tool_call, ensure_ascii=False).encode("utf-8")
    except (UnicodeEncodeError, UnicodeError):
        has_surrogates = True

    if has_surrogates:
        await queue.put(
            RunEvent(
                type="run_error",
                run_id=run_id,
                message="Guard error. Failing closed.",
            )
        )
        return

    if not tool_call:
        await queue.put(RunEvent(type="agent_reply", run_id=run_id, text=text_reply or ""))
        await queue.put(
            RunEvent(
                type="run_finished",
                run_id=run_id,
                budget=tracker.check_budget(active_session),
            )
        )
        return

    # Step 1: execute proposed tool
    tool_name = tool_call.get("tool_name", "execute_python")
    tool_args = tool_call.get("args", {})
    intent = "Execute the proposed Python code in the sandbox environment."

    await queue.put(
        RunEvent(
            type="step_started",
            run_id=run_id,
            index=1,
            intent=intent,
            tool_name=tool_name,
            tool_args=tool_args,
        )
    )

    def on_tool_event(ev_type: str, data: Any) -> None:
        if ev_type == "budget_reserved":
            queue.put_nowait(
                RunEvent(
                    type="budget_reserved",
                    run_id=run_id,
                    index=1,
                    budget=data,
                )
            )

    tool_result = await execute_guarded(
        intent=intent,
        tool_name=tool_name,
        args=tool_args,
        session_id=active_session,
        provider=provider,
        source="web",
        label="step 1/1",
        on_event=on_tool_event,
    )

    await queue.put(
        RunEvent(
            type="step_finished",
            run_id=run_id,
            index=1,
            result=tool_result,
        )
    )

    await queue.put(
        RunEvent(
            type="run_finished",
            run_id=run_id,
            budget=tracker.check_budget(active_session),
        )
    )


async def run_agent_workflow(
    prompt: str,
    session_id: str = "demo",
    provider: str | None = None,
    budget_cap_gbp: float | None = None,
    preset: str | None = None,
) -> AsyncIterator[RunEvent]:
    """Execute the full agent workflow and yield RunEvents via Queue streaming."""
    _validate_text(prompt, "prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"Prompt exceeds maximum length of {MAX_PROMPT_CHARS} characters.")

    _validate_text(session_id, "session_id")
    if provider is not None:
        _validate_text(provider, "provider")

    settings = get_settings()
    tracker = budget_store.get_tracker()
    run_id = f"run_{uuid.uuid4().hex[:8]}"

    # Map preset
    preset_id: str | None = None
    if preset is not None:
        p_str = str(preset).strip().lower()
        if p_str in ("1", "safe"):
            preset_id = "safe"
        elif p_str in ("2", "injection"):
            preset_id = "injection"
        elif p_str in ("3", "loop"):
            preset_id = "loop"

    queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()

    async def workflow_runner() -> None:
        try:
            if preset_id == "loop":
                await _run_loop_workflow(queue, run_id, provider, settings, tracker)
            else:
                await _run_standard_workflow(
                    queue,
                    run_id,
                    prompt,
                    session_id,
                    provider,
                    budget_cap_gbp,
                    preset_id,
                    settings,
                    tracker,
                )
        except Exception as exc:
            logger.exception("Workflow execution error")
            await queue.put(
                RunEvent(type="run_error", run_id=run_id, message="Guard error. Failing closed.")
            )
        finally:
            await queue.put(None)

    task = asyncio.create_task(workflow_runner())

    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield event
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
