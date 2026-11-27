"""Docker provider (Track B). Phase 0 stub: executes nothing."""

from core.schemas import SandboxRequest, SandboxResult
from sandbox.base import SandboxProvider


class DockerSandboxProvider(SandboxProvider):
    provider_name = "docker"

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        return SandboxResult(
            provider=self.provider_name,
            stdout="stub: nothing was executed\n",
            stderr="",
            truncated=False,
            exit_code=0,
            duration_ms=0,
            timed_out=False,
        )

    async def health_check(self) -> bool:
        return True
