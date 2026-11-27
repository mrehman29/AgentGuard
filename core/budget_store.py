"""Budget ledger (Track A). Phase 0 stub: in memory, spends nothing.

Signatures, including get_tracker(), are frozen.
"""

from core.config import get_settings
from core.schemas import BudgetStatus, CommitResult, LedgerEntry, TokenUsage


class SessionBudgetTracker:
    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def check_budget(self, session_id: str) -> BudgetStatus:
        settings = get_settings()
        return BudgetStatus(
            session_id=session_id,
            prompt_tokens=0,
            candidates_tokens=0,
            thoughts_tokens=0,
            total_tokens=0,
            spent_gbp=0.0,
            reserved_gbp=0.0,
            cap_gbp=settings.session_budget_cap_gbp,
            remaining_gbp=settings.session_budget_cap_gbp,
            global_spent_gbp=0.0,
            global_cap_gbp=settings.global_budget_cap_gbp,
            is_exceeded=False,
        )

    def reserve(self, session_id: str, estimate_gbp: float) -> str | None:
        """Return a reservation id, or None if a cap would be exceeded."""
        return "stub-reservation"

    def commit(self, reservation_id: str, usage: TokenUsage) -> CommitResult:
        return CommitResult(cost_gbp=0.0, budget=self.check_budget("stub"))

    def release(self, reservation_id: str) -> None:
        return None

    def set_session_cap(self, session_id: str, cap_gbp: float) -> BudgetStatus:
        """Can only lower the cap."""
        return self.check_budget(session_id)

    def reset(self, session_id: str) -> BudgetStatus:
        """Clear session spend and restore the default cap. Never resets global."""
        return self.check_budget(session_id)

    def record(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    def recent_entries(self, limit: int = 50) -> list[LedgerEntry]:
        """Oldest first."""
        return self._entries[-limit:] if limit > 0 else []


_tracker: SessionBudgetTracker | None = None


def get_tracker() -> SessionBudgetTracker:
    """Process-wide tracker."""
    global _tracker
    if _tracker is None:
        _tracker = SessionBudgetTracker()
    return _tracker
