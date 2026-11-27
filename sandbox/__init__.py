"""Sandbox provider registry (Track B)."""

from core.config import get_settings
from sandbox.base import SandboxProvider
from sandbox.cloudrun_provider import CloudRunSandboxProvider
from sandbox.docker_provider import DockerSandboxProvider

PROVIDERS: dict[str, type[SandboxProvider]] = {
    "docker": DockerSandboxProvider,
    "cloudrun": CloudRunSandboxProvider,
}

__all__ = ["PROVIDERS", "SandboxProvider", "get_sandbox_provider"]


def get_sandbox_provider(override: str | None = None) -> SandboxProvider:
    """Return the named provider, or the configured one. Unknown name: ValueError."""
    name = get_settings().sandbox_provider if override is None else override
    if name not in PROVIDERS:
        raise ValueError(f"unknown sandbox provider: {name!r}")
    return PROVIDERS[name]()
