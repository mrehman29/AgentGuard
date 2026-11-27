"""Frozen contracts shared by all tracks (PLAN.md section 5).

Change only through the coordinator.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 16 * 1024
EXECUTABLE_TOOL = "execute_python"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Decision(StrEnum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    BLOCK = "BLOCK"


class Stage(StrEnum):
    BUDGET = "BUDGET"
    REGEX = "REGEX"
    OWASP_LLM = "OWASP_LLM"


class OwaspTag(StrEnum):
    LLM01 = "LLM01: Prompt Injection"
    LLM02 = "LLM02: Sensitive Information Disclosure"
    LLM05 = "LLM05: Improper Output Handling"
    LLM06 = "LLM06: Excessive Agency"
    SAFE = "SAFE: Compliant Action"
    NA = "N/A"


Source = Literal["web", "mcp"]
Outcome = Literal[
    "ALLOW", "FLAG", "BLOCK regex", "BLOCK owasp", "BLOCK budget", "BLOCK error", "OK", "ERROR"
]


class OwaspVerdict(Frozen):
    """The only model schema sent to Gemini."""

    decision: Decision
    risk_score: int = Field(ge=0, le=100)
    owasp_tag: OwaspTag
    reason: str


class GuardEvaluation(OwaspVerdict):
    stage: Stage
    latency_ms: int = Field(ge=0)
    gates_passed: list[Stage] = Field(default_factory=list)
    cost_gbp: float = Field(default=0.0, ge=0)


class TokenUsage(Frozen):
    prompt_tokens: int = Field(default=0, ge=0)
    candidates_tokens: int = Field(default=0, ge=0)
    thoughts_tokens: int = Field(default=0, ge=0)


class BudgetStatus(Frozen):
    session_id: str
    prompt_tokens: int
    candidates_tokens: int
    thoughts_tokens: int
    total_tokens: int
    spent_gbp: float
    reserved_gbp: float
    cap_gbp: float
    remaining_gbp: float
    global_spent_gbp: float
    global_cap_gbp: float
    is_exceeded: bool


class CommitResult(Frozen):
    cost_gbp: float = Field(ge=0)
    budget: BudgetStatus


class SandboxRequest(Frozen):
    code: str = Field(max_length=MAX_CODE_CHARS)
    timeout_sec: int = Field(default=5, ge=1, le=5)


class SandboxResult(Frozen):
    provider: str
    stdout: str = Field(max_length=MAX_OUTPUT_CHARS)
    stderr: str = Field(max_length=MAX_OUTPUT_CHARS)
    truncated: bool
    exit_code: int
    duration_ms: int = Field(ge=0)
    timed_out: bool


class PipelineStepResult(Frozen):
    intent: str
    tool_name: str
    tool_args: dict[str, Any]
    guard: GuardEvaluation
    sandbox: SandboxResult | None = None
    budget: BudgetStatus


class LedgerEntry(Frozen):
    time: datetime
    source: Source
    session_id: str
    label: str  # screen | agent turn | step n/m | check | execute | reset
    outcome: Outcome
    cost_gbp: float = Field(ge=0)


RunEventType = Literal[
    "run_started",
    "step_started",
    "budget_reserved",
    "step_finished",
    "agent_reply",
    "run_finished",
    "run_error",
]

# Fields each event type must set (PLAN.md section 9).
RUN_EVENT_FIELDS: dict[str, tuple[str, ...]] = {
    "run_started": ("session_id", "preset", "total_steps"),
    "step_started": ("index", "intent", "tool_name", "tool_args"),
    "budget_reserved": ("index", "budget"),
    "step_finished": ("index", "result"),
    "agent_reply": ("text",),
    "run_finished": ("budget",),
    "run_error": ("message",),
}


class RunEvent(Frozen):
    """One NDJSON line from POST /api/run.

    Serialise with model_dump_json(exclude_unset=True). `preset` and
    `total_steps` may be null but must be passed for run_started.
    """

    type: RunEventType
    run_id: str
    session_id: str | None = None
    preset: int | None = None
    total_steps: int | None = None
    index: int | None = None
    intent: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    budget: BudgetStatus | None = None
    result: PipelineStepResult | None = None
    text: str | None = None
    message: str | None = None

    @model_validator(mode="after")
    def _required_fields(self) -> "RunEvent":
        nullable = {"preset", "total_steps"}
        missing = [
            name
            for name in RUN_EVENT_FIELDS[self.type]
            if name not in self.model_fields_set
            or (name not in nullable and getattr(self, name) is None)
        ]
        if missing:
            raise ValueError(f"{self.type} event is missing: {', '.join(missing)}")
        return self


def ledger_outcome(guard: GuardEvaluation) -> Outcome:
    """Ledger outcome for a guard decision."""
    if guard.decision is Decision.ALLOW:
        return "ALLOW"
    if guard.decision is Decision.FLAG:
        return "FLAG"
    if guard.stage is Stage.REGEX:
        return "BLOCK regex"
    if guard.stage is Stage.BUDGET:
        return "BLOCK budget"
    return "BLOCK error" if guard.owasp_tag is OwaspTag.NA else "BLOCK owasp"


def error_block(
    reason: str, gates_passed: list[Stage] | None = None, latency_ms: int = 0, cost_gbp: float = 0.0
) -> GuardEvaluation:
    """Fail-closed verdict for guard errors."""
    return GuardEvaluation(
        decision=Decision.BLOCK,
        risk_score=100,
        owasp_tag=OwaspTag.NA,
        reason=reason,
        stage=Stage.OWASP_LLM,
        latency_ms=latency_ms,
        gates_passed=gates_passed or [],
        cost_gbp=cost_gbp,
    )
