"""HTTP sandbox worker that runs inside the sandbox image (PLAN.md B.1).

Standalone: the image holds only this file, so it defines its own models and
limits. tests/test_sandboxes.py keeps them equal to the frozen contracts.
The container must hold no secret: the code run here is untrusted.
"""

import ctypes
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MAX_CODE_CHARS = 20_000
MAX_OUTPUT_CHARS = 16 * 1024
TIMEOUT_EXIT_CODE = 124
READ_CHUNK_BYTES = 64 * 1024
CLEANUP_DEADLINE_SEC = 2.0

PR_SET_DUMPABLE = 4
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38

logger = logging.getLogger(__name__)

# One execution at a time: cleanup kills every descendant of this process.
# A second request is refused, never queued: running code can reach this port
# and must not be able to leave work behind for after its own run.
_run_lock = threading.Lock()


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str = Field(max_length=MAX_CODE_CHARS)
    timeout_sec: int = Field(default=5, ge=1, le=5)


class ExecuteResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    stdout: str = Field(max_length=MAX_OUTPUT_CHARS)
    stderr: str = Field(max_length=MAX_OUTPUT_CHARS)
    truncated: bool
    exit_code: int
    duration_ms: int = Field(ge=0)
    timed_out: bool


class _Capture:
    """Keeps the first bytes of a stream and discards the rest."""

    def __init__(self) -> None:
        self.data = bytearray()
        self.truncated = False

    def drain(self, stream) -> None:
        while chunk := stream.read(READ_CHUNK_BYTES):
            room = MAX_OUTPUT_CHARS - len(self.data)
            self.data += chunk[:room]
            self.truncated = self.truncated or len(chunk) > room

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


def _harden() -> None:
    """Linux process settings. Untrusted code runs under the same user id."""
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    for option, value in (
        (PR_SET_DUMPABLE, 0),  # no ptrace or /proc/<pid>/mem access to the worker
        (PR_SET_NO_NEW_PRIVS, 1),  # set-uid binaries gain nothing
        (PR_SET_CHILD_SUBREAPER, 1),  # orphans stay our descendants, so we can kill them
    ):
        if prctl(option, value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), f"prctl({option}) failed")


def _descendants() -> list[int]:
    children: dict[int, list[int]] = {}
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as stat:
                parent = int(stat.read().rpartition(b")")[2].split()[1])
        except OSError:
            continue
        children.setdefault(parent, []).append(int(name))
    found = list(children.get(os.getpid(), []))
    for pid in found:  # grows while iterating: breadth first
        found.extend(children.get(pid, []))
    return found


def _kill_leftovers() -> None:
    """Kill processes that left the process group, for example with setsid()."""
    deadline = time.monotonic() + CLEANUP_DEADLINE_SEC
    while pids := _descendants():
        if time.monotonic() > deadline:
            # A dirty instance must not serve another request.
            logger.error("could not kill sandbox processes; stopping the worker")
            os._exit(1)
        for pid in pids:
            with suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        for pid in pids:
            with suppress(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)
        time.sleep(0.01)


def _feed(stream, data: bytes) -> None:
    with suppress(BrokenPipeError):
        with stream:
            stream.write(data)


def _run(request: ExecuteRequest) -> ExecuteResponse:
    start = time.monotonic()
    workdir = tempfile.mkdtemp(prefix="agentguard-run-")
    out, err = _Capture(), _Capture()
    timed_out = False
    try:
        process = subprocess.Popen(
            [sys.executable, "-I", "-u", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=workdir,
            env={},
            start_new_session=True,
        )
        threads = [
            threading.Thread(target=out.drain, args=(process.stdout,), daemon=True),
            threading.Thread(target=err.drain, args=(process.stderr,), daemon=True),
            threading.Thread(
                target=_feed,
                args=(process.stdin, request.code.encode("utf-8", errors="surrogatepass")),
                daemon=True,
            ),
        ]
        try:
            for thread in threads:
                thread.start()
            process.wait(timeout=request.timeout_sec)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            _kill_leftovers()
            for thread in threads:
                thread.join(CLEANUP_DEADLINE_SEC)
            process.stdout.close()
            process.stderr.close()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    exit_code = process.returncode
    if exit_code < 0:  # killed by a signal: report it as a shell would
        exit_code = 128 - exit_code
    return ExecuteResponse(
        provider="cloudrun",
        stdout=out.text(),
        stderr=err.text(),
        truncated=out.truncated or err.truncated,
        exit_code=TIMEOUT_EXIT_CODE if timed_out else exit_code,
        duration_ms=int((time.monotonic() - start) * 1000),
        timed_out=timed_out,
    )


@asynccontextmanager
async def _lifespan(_: FastAPI):
    _harden()
    yield


app = FastAPI(lifespan=_lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/execute")
def execute(request: ExecuteRequest) -> ExecuteResponse:
    if not _run_lock.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="the sandbox is busy")
    try:
        return _run(request)
    finally:
        _run_lock.release()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
