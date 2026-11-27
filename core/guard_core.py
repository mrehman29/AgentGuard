"""Guard core (Track A). Phase 0 stub: signatures are frozen, bodies are not.

The stub classifier blocks everything unless AGENTGUARD_STUB_DECISION is set,
so a half-built tree never executes code by default.
"""

import os
import re
import time
import unicodedata
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


# Separator between a command and its arguments, in shell text or an argv list.
_SEP = r"[\s\"',]+"
# Any options before the one being looked for.
_OPTIONS = rf"(?:{_SEP}-[a-z-]+){{0,6}}?{_SEP}"
_NETWORK_MODULES = (
    "socket", "requests", "urllib", "urllib3", "httpx", "aiohttp",
    "http.client", "ftplib", "smtplib", "telnetlib", "paramiko",
)


def _imports(modules: str) -> str:
    """An import statement of one of the modules, not prose that mentions it."""
    earlier = r"(?:[\w.]+(?:[ \t]+as[ \t]+\w+)?[ \t]*,[ \t]*){0,10}"
    return (
        rf"(?:^|[;:])[ \t]*(?:import[ \t]+{earlier}(?:{modules})\b"
        rf"|from[ \t]+(?:{modules})(?:\.\w+)*[ \t]+import\b)"
    )


def _sig(pattern: str, *triggers: str) -> tuple[tuple[str, ...], re.Pattern[str]]:
    """A signature, searched only if the casefolded text contains one of its triggers."""
    return triggers, re.compile(pattern, re.IGNORECASE | re.MULTILINE)


_FROM_OS = r"\bfrom[ \t]+os[ \t]+import\b(?:[ \t]*\([^)]{0,200}?|[^\n#;(]{0,200}?)"

# Checked in order; the first hit decides. A cheap pre-filter, not a boundary.
_RULES = (
    (
        "instruction override",
        OwaspTag.LLM01,
        (
            _sig(r"\bsystem[\s_-]{0,3}override\b", "override"),
            _sig(
                r"\b(?:ignore|disregard|forget|override|bypass)\b[\s\w,'\"-]{0,40}?"
                r"\b(?:previous|prior|above|earlier|all|your|system|safety)\b[\s\w-]{0,20}?"
                r"\b(?:instructions?|rules?|prompts?|guidelines?|polic(?:y|ies))\b",
                "instruction", "rule", "prompt", "guideline", "polic",
            ),
        ),
    ),
    (
        "destructive command",
        OwaspTag.LLM06,
        (
            # rm with both recursive and force flags, in any order or grouping.
            _sig(
                rf"\brm\b(?={_OPTIONS}(?:-[a-z]*r|--recursive\b))"
                rf"(?={_OPTIONS}(?:-[a-z]*f|--force\b))",
                "rm",
            ),
            _sig(r"\brmtree\s*\(", "rmtree"),
            _sig(r"\bmkfs\b", "mkfs"),
            _sig(r"\bdd\s+(?:if|of)=/", "dd"),
            _sig(r">\s*/dev/(?:sd[a-z]|nvme\d)", "/dev/"),
            # Fork bomb, with or without spaces.
            _sig(r":\s*\(\s*\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "{"),
        ),
    ),
    (
        "credential access",
        OwaspTag.LLM06,
        (
            _sig(r"/etc/(?:passwd|shadow|sudoers)\b", "/etc/"),
            _sig(r"/proc/(?:self|\d+)/environ\b", "/proc/"),
            _sig(r"\.ssh/", ".ssh/"),
            _sig(r"\bid_(?:rsa|ed25519|ecdsa)\b", "id_"),
            _sig(r"\.aws/credentials\b", ".aws/"),
            _sig(r"\.git-credentials\b", ".git-credentials"),
            _sig(r"\bapplication_default_credentials\b", "application_default"),
            _sig(r"(?<![\w.])\.env\b", ".env"),
            _sig(r"\bos\s*\.\s*(?:environb?|getenvb?|putenv)\b", "environ", "getenv", "putenv"),
            _sig(_FROM_OS + r"\b(?:environb?|getenvb?)\b", "environ", "getenv"),
        ),
    ),
    (
        "dynamic execution",
        OwaspTag.LLM06,
        (
            # Not retrieval(x) or model.eval().
            _sig(r"(?<![\w.])(?:eval|exec)\s*\(", "eval", "exec"),
            _sig(r"\bbuiltins\s*\.\s*(?:eval|exec)\b", "builtins"),
            _sig(r"\b__import__\s*\(", "__import__"),
            _sig(r"\bos\s*\.\s*(?:system|popen|exec[lv]p?e?|spawn[lv]p?e?)\s*\(", "os"),
            _sig(_FROM_OS + r"\b(?:system|popen)\b", "system", "popen"),
            _sig(
                r"\bsubprocess\s*\.\s*"
                r"(?:run|call|Popen|check_call|check_output|getoutput|getstatusoutput)\b",
                "subprocess",
            ),
            _sig(_imports("subprocess"), "subprocess"),
        ),
    ),
    (
        "network access",
        OwaspTag.LLM06,
        (
            _sig(_imports("|".join(map(re.escape, _NETWORK_MODULES))), *_NETWORK_MODULES),
            _sig(
                r"\b(?:curl|wget)[ \t]+(?:-{1,2}[a-z]|(?:https?|ftp)://|[\w-]+\.[a-z]{2,}\b)",
                "curl", "wget",
            ),
            _sig(r"\brequests\s*\.\s*(?:get|post|put|patch|delete|head|request)\s*\(", "requests"),
            _sig(r"\burlopen\s*\(", "urlopen"),
            _sig(r"\bsocket\s*\.\s*(?:socket|create_connection)\s*\(", "socket"),
            _sig(r"/dev/tcp/", "/dev/tcp/"),
            _sig(r"\bnc[ \t]+-[a-z]*e\b", "nc"),
        ),
    ),
)
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))


def _strings(value: Any) -> list[str]:
    """Every string in the arguments, including keys, in order."""
    found: list[str] = []
    seen: set[int] = set()
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            found.append(item)
        elif isinstance(item, (dict, list, tuple)) and id(item) not in seen:
            seen.add(id(item))
            if isinstance(item, dict):
                children = [part for pair in item.items() for part in pair]
            else:
                children = list(item)
            stack.extend(reversed(children))
    return found


def fast_rule_check(tool_name: str, args: dict) -> GuardEvaluation | None:
    """Signature rules over the arguments. No model call."""
    start = time.perf_counter()
    text = unicodedata.normalize("NFKC", "\n".join(_strings(args))).translate(_INVISIBLE)
    folded = text.casefold()
    for name, tag, signatures in _RULES:
        if any(
            any(trigger in folded for trigger in triggers) and pattern.search(text)
            for triggers, pattern in signatures
        ):
            return GuardEvaluation(
                decision=Decision.BLOCK,
                risk_score=100,
                owasp_tag=tag,
                reason=f"Matched a signature rule: {name}.",
                stage=Stage.REGEX,
                latency_ms=_elapsed_ms(start),
                gates_passed=[Stage.BUDGET],
            )
    return None


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


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
