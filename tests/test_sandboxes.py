"""Sandbox tests (Track B). Docker tests are skipped without a daemon and image."""

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from core import schemas
from core.schemas import SandboxRequest
from sandbox import worker

SANDBOX_DIR = Path(worker.__file__).parent


def _same_fields(model, contract):
    def shape(cls):
        return {
            name: (f.annotation, f.default, f.is_required(), f.metadata)
            for name, f in cls.model_fields.items()
        }

    assert shape(model) == shape(contract)


def test_worker_models_match_the_contracts():
    _same_fields(worker.ExecuteRequest, schemas.SandboxRequest)
    _same_fields(worker.ExecuteResponse, schemas.SandboxResult)
    assert worker.MAX_CODE_CHARS == schemas.MAX_CODE_CHARS
    assert worker.MAX_OUTPUT_CHARS == schemas.MAX_OUTPUT_CHARS


@pytest.fixture(scope="module")
def worker_url():
    """The real worker, in its own process, on a free local port."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "worker:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=SANDBOX_DIR,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "AGENTGUARD_CANARY": "leaked"},
    )
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health", trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("worker did not start")
        yield url
    finally:
        process.terminate()
        process.wait(timeout=10)


def post(url, code, timeout_sec=5):
    response = httpx.post(
        f"{url}/execute",
        json={"code": code, "timeout_sec": timeout_sec},
        timeout=20,
        trust_env=False,
    )
    response.raise_for_status()
    return response.json()


def alive(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rpartition(")")[2].split()[0] != "Z"


def test_worker_runs_code(worker_url):
    result = post(worker_url, "print(sum(range(101)))")
    assert (result["stdout"], result["exit_code"], result["timed_out"]) == ("5050\n", 0, False)
    assert result["truncated"] is False


def test_worker_reports_errors_and_signals(worker_url):
    failed = post(worker_url, "raise SystemExit(3)")
    assert (failed["exit_code"], failed["timed_out"]) == (3, False)
    killed = post(worker_url, "import os; os.kill(os.getpid(), 9)")
    assert (killed["exit_code"], killed["timed_out"]) == (137, False)


def test_worker_times_out_and_survives(worker_url):
    result = post(worker_url, "print('started')\nwhile True: pass", timeout_sec=1)
    assert (result["exit_code"], result["timed_out"]) == (124, True)
    assert result["stdout"] == "started\n"
    assert post(worker_url, "print(1)")["stdout"] == "1\n"


def test_worker_isolates_environment_and_directory(worker_url):
    code = "import os; print(sorted(os.environ)); print(os.getcwd()); open('f','w').write('x')"
    first = post(worker_url, code)["stdout"].splitlines()
    second = post(worker_url, code)["stdout"].splitlines()
    assert "AGENTGUARD_CANARY" not in first[0] and "PATH" not in first[0]
    assert first[1] != second[1]
    assert not Path(first[1]).exists() and not Path(second[1]).exists()


def test_worker_memory_is_not_readable(worker_url):
    code = "import os; p = os.getppid()\nfor n in ('environ', 'mem'): open(f'/proc/{p}/{n}', 'rb')"
    result = post(worker_url, code)
    assert result["exit_code"] == 1
    assert "PermissionError" in result["stderr"]


def test_worker_caps_output(worker_url):
    code = "import sys\nfor _ in range(2000): sys.stdout.write('a'*65536); sys.stderr.write('\\xe9'*100)"
    result = post(worker_url, code)
    assert result["truncated"] is True
    assert result["stdout"] == "a" * schemas.MAX_OUTPUT_CHARS
    assert len(result["stderr"]) <= schemas.MAX_OUTPUT_CHARS


@pytest.mark.parametrize("detach", ["False", "True"])
@pytest.mark.parametrize("ending", ["", "\nwhile True: pass"])
def test_worker_kills_background_processes(worker_url, detach, ending):
    """Children are killed even in a new session (outside the process group)."""
    code = (
        "import subprocess, sys\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'],"
        f" start_new_session={detach})\n"
        "print(p.pid)" + ending
    )
    start = time.monotonic()
    result = post(worker_url, code, timeout_sec=1)
    assert time.monotonic() - start < 5
    assert not alive(int(result["stdout"]))


def test_worker_kills_double_forked_daemon(worker_url):
    code = (
        "import os, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    if os.fork() == 0:\n"
        "        print(os.getpid()); time.sleep(60)\n"
        "    os._exit(0)\n"
        "time.sleep(0.5)\n"
    )
    result = post(worker_url, code)
    assert not alive(int(result["stdout"]))


@pytest.mark.parametrize(
    "body", [{}, {"code": "x" * 20_001}, {"code": "1", "timeout_sec": 6}, {"code": 5}]
)
def test_worker_rejects_invalid_requests(worker_url, body):
    assert httpx.post(f"{worker_url}/execute", json=body, trust_env=False).status_code == 422
