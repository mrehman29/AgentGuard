"""Docker provider (PLAN.md B.2): one throwaway, isolated container per request."""

import asyncio
import logging
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

from core.config import get_settings
from core.schemas import MAX_OUTPUT_CHARS, SandboxRequest, SandboxResult
from sandbox.base import SandboxProvider

logger = logging.getLogger(__name__)

TIMEOUT_EXIT_CODE = 124
UNAVAILABLE_EXIT_CODE = 125
HOST_GRACE_SEC = 2  # host deadline = timeout_sec + this
BACKSTOP_GRACE_SEC = 5  # in-container limit, in case this process dies first
DOCKER_COMMAND_TIMEOUT_SEC = 5
KILL_ATTEMPTS = 5
KILL_WAIT_SEC = 0.5
READ_CHUNK_BYTES = 64 * 1024

ISOLATION_FLAGS = (
    "--network", "none",
    "--read-only",
    "--tmpfs", "/tmp:size=32m",
    "--ipc", "none",  # no writable /dev/shm
    "--memory", "256m",
    "--memory-swap", "256m",
    "--cpus", "0.5",
    "--pids-limit", "64",
    "--cap-drop", "ALL",
    "--security-opt", "no-new-privileges",
    "--user", "65534:65534",
    "--pull", "never",  # run only the locally built image
    "--log-driver", "none",  # output is not also written to the host disk
)

# The docker client copies proxy settings, passwords included, from its config
# into containers. Setting each variable to empty overrides that.
BLANK_PROXY_FLAGS = tuple(
    flag
    for kind in ("HTTP", "HTTPS", "FTP", "NO", "ALL")
    for name in (f"{kind}_PROXY", f"{kind.lower()}_proxy")
    for flag in ("--env", f"{name}=")
)


class _Capture:
    """Keeps the first bytes of a stream and discards the rest."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.truncated = False

    async def drain(self, stream: asyncio.StreamReader) -> None:
        while chunk := await stream.read(READ_CHUNK_BYTES):
            room = MAX_OUTPUT_CHARS - len(self.data)
            self.data += chunk[:room]
            self.truncated = self.truncated or len(chunk) > room

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


async def _feed(stream: asyncio.StreamWriter, data: bytes) -> None:
    try:
        with suppress(BrokenPipeError, ConnectionResetError):
            stream.write(data)
            await stream.drain()
    finally:
        stream.close()


async def _docker(*args: str) -> tuple[int, str]:
    """Run a short docker command. Returns (exit code, stderr)."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as exc:
        return UNAVAILABLE_EXIT_CODE, type(exc).__name__
    try:
        _, stderr = await asyncio.wait_for(process.communicate(), DOCKER_COMMAND_TIMEOUT_SEC)
    except TimeoutError:
        return UNAVAILABLE_EXIT_CODE, "docker command timed out"
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
    return process.returncode, stderr.decode("utf-8", errors="replace")


async def _remove(name: str) -> None:
    code, stderr = await _docker("rm", "-f", name)
    if code != 0 and "No such container" not in stderr and "already in progress" not in stderr:
        logger.error("could not remove container %s: %s", name, stderr[:500])


async def _destroy(name: str, process: asyncio.subprocess.Process, io: asyncio.Future) -> None:
    """Kill the container by name, then the docker client, and leave nothing behind."""
    # The container may not exist yet, so repeat while the client is still running.
    for _ in range(KILL_ATTEMPTS):
        code, _ = await _docker("rm", "-f", name)
        done, _pending = await asyncio.wait({io}, timeout=KILL_WAIT_SEC)
        if done or code == UNAVAILABLE_EXIT_CODE:  # finished, or docker is not answering
            break
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.kill()
    await asyncio.wait({io}, timeout=DOCKER_COMMAND_TIMEOUT_SEC)
    if not io.done():
        io.cancel()
    elif not io.cancelled():
        io.exception()  # mark as retrieved
    await _remove(name)


class DockerSandboxProvider(SandboxProvider):
    provider_name = "docker"

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        start = time.monotonic()
        name = f"agentguard-{uuid4().hex}"
        image = get_settings().sandbox_docker_image
        out, err = _Capture(), _Capture()
        timed_out = False

        with tempfile.TemporaryDirectory(prefix="agentguard-") as tmp:
            # Written by docker once the container exists: this tells a daemon
            # failure apart from code that exits with the same status.
            cidfile = Path(tmp) / "cid"
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker", "run", "--rm", "--name", name, "--cidfile", str(cidfile),
                    *ISOLATION_FLAGS,
                    *BLANK_PROXY_FLAGS,
                    "-i", image,
                    "timeout", "--signal=KILL", str(request.timeout_sec + BACKSTOP_GRACE_SEC),
                    "python3", "-I", "-u", "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )  # fmt: skip
            except OSError as exc:
                return self._unavailable(f"cannot run docker ({type(exc).__name__}).", start)

            code = request.code.encode("utf-8", errors="surrogatepass")
            io = asyncio.gather(
                _feed(process.stdin, code),
                out.drain(process.stdout),
                err.drain(process.stderr),
                process.wait(),
            )
            completed = False
            try:
                done, _pending = await asyncio.wait(
                    {io}, timeout=request.timeout_sec + HOST_GRACE_SEC
                )
                if done:
                    io.result()
                    completed = True
                else:
                    timed_out = True
            finally:
                # Also runs when the caller cancels; shield so cleanup finishes.
                if not completed:
                    await asyncio.shield(_destroy(name, process, io))
                elif process.returncode < 0 or process.returncode == UNAVAILABLE_EXIT_CODE:
                    await asyncio.shield(_remove(name))
            created = cidfile.is_file() and cidfile.stat().st_size > 0

        exit_code = process.returncode
        if not created and (timed_out or exit_code != 0):
            detail = err.text().strip() or "the container was not created."
            return self._unavailable(detail, start)
        if exit_code is not None and exit_code < 0:
            exit_code = 128 - exit_code
        return SandboxResult(
            provider=self.provider_name,
            stdout=out.text(),
            stderr=err.text(),
            truncated=out.truncated or err.truncated,
            exit_code=TIMEOUT_EXIT_CODE if timed_out else exit_code,
            duration_ms=int((time.monotonic() - start) * 1000),
            timed_out=timed_out,
        )

    async def health_check(self) -> bool:
        """True when the daemon answers and the sandbox image exists."""
        code, _ = await _docker("image", "inspect", get_settings().sandbox_docker_image)
        return code == 0

    def _unavailable(self, detail: str, start: float) -> SandboxResult:
        return SandboxResult(
            provider=self.provider_name,
            stdout="",
            stderr=f"Docker sandbox unavailable: {detail}"[:MAX_OUTPUT_CHARS],
            truncated=False,
            exit_code=UNAVAILABLE_EXIT_CODE,
            duration_ms=int((time.monotonic() - start) * 1000),
            timed_out=False,
        )
