"""Frozen sandbox contract (PLAN.md section 5). Change only through the coordinator."""

from abc import ABC, abstractmethod

from core.schemas import SandboxRequest, SandboxResult


class SandboxProvider(ABC):
    provider_name: str

    @abstractmethod
    async def execute(self, request: SandboxRequest) -> SandboxResult:
        """Run the code. Unreachable or refused: exit_code 125, timed_out False."""

    @abstractmethod
    async def health_check(self) -> bool: ...
