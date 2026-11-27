"""Cloud Run provider (Track B). Phase 0 stub: reports itself unreachable."""

from core.schemas import SandboxRequest, SandboxResult
from sandbox.base import SandboxProvider


class CloudRunSandboxProvider(SandboxProvider):
    provider_name = "cloudrun"

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        return SandboxResult(
            provider=self.provider_name,
            stdout="",
            stderr="stub: Cloud Run provider is not implemented",
            truncated=False,
            exit_code=125,
            duration_ms=0,
            timed_out=False,
        )

    async def health_check(self) -> bool:
        return False
