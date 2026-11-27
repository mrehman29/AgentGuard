"""Guard core (Track A). Phase 0 stub: signatures are frozen, bodies are not.

The stub classifier blocks everything unless AGENTGUARD_STUB_DECISION is set,
so a half-built tree never executes code by default.
"""

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from core import budget_store
from core.schemas import (
    Decision,
    GuardEvaluation,
    LedgerEntry,
    OwaspTag,
    Source,
    Stage,
    ledger_outcome,
)


def fast_rule_check(tool_name: str, args: dict) -> GuardEvaluation | None:
    return None


async def evaluate_owasp_policy(tool_name: str, args: dict, session_id: str) -> GuardEvaluation:
    decision = Decision(os.environ.get("AGENTGUARD_STUB_DECISION", "BLOCK"))
    allowed = decision is Decision.ALLOW
    return GuardEvaluation(
        decision=decision,
        risk_score=0 if allowed else 100,
        owasp_tag=OwaspTag.SAFE if allowed else OwaspTag.NA,
        reason="Stub classifier: no model was called.",
        stage=Stage.OWASP_LLM,
        latency_ms=0,
        gates_passed=[Stage.BUDGET, Stage.REGEX, Stage.OWASP_LLM]
        if allowed
        else [Stage.BUDGET, Stage.REGEX],
    )


async def evaluate_action(
    tool_name: str,
    args: dict,
    session_id: str,
    source: Source = "web",
    label: str = "screen",
    on_event: Callable[[str, Any], None] | None = None,
    record: bool = True,
) -> GuardEvaluation:
    """Gates 1-3. on_event("budget_reserved", BudgetStatus) is synchronous."""
    guard = fast_rule_check(tool_name, args) or await evaluate_owasp_policy(
        tool_name, args, session_id
    )
    if record:
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
    return guard
