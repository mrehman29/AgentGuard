"""Sandbox tests (Track B). Docker tests are skipped without a daemon and image."""

import asyncio
import os
import socket
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import httpx
import pytest

from core import schemas
from core.schemas import SandboxRequest
from sandbox import docker_provider, worker
from sandbox.docker_provider import DockerSandboxProvider

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


def _docker_ready():
    try:
        return asyncio.run(DockerSandboxProvider().health_check())
    except Exception:
        return False


needs_docker = pytest.mark.skipif(not _docker_ready(), reason="no Docker daemon or sandbox image")


@pytest.fixture
def containers(monkeypatch):
    """Records the names the provider uses and checks that none survive."""
    names = []
    real = docker_provider.uuid4

    def recording_uuid4():
        value = real()
        names.append(f"agentguard-{value.hex}")
        return value

    monkeypatch.setattr(docker_provider, "uuid4", recording_uuid4)
    yield names
    listed = subprocess.run(
        ["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True, check=True
    ).stdout.split()
    assert not set(names) & set(listed)


def running(name):
    done = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], capture_output=True, text=True
    )
    return done.stdout.strip() == "true"


async def docker_run(code, timeout_sec=5):
    request = SandboxRequest(code=code, timeout_sec=timeout_sec)
    return await DockerSandboxProvider().execute(request)


@needs_docker
async def test_docker_runs_code(containers):
    result = await docker_run("print(sum(range(101))); print('caf\u00e9')")
    assert result.stdout == "5050\ncaf\u00e9\n"
    assert (result.provider, result.exit_code, result.timed_out) == ("docker", 0, False)
    assert result.truncated is False and result.duration_ms > 0
    assert len(containers) == 1


@needs_docker
@pytest.mark.parametrize("code", [1, 3, 125])
async def test_docker_reports_the_exit_code_of_the_code(containers, code):
    result = await docker_run(f"import sys; print('x'); sys.exit({code})")
    assert (result.exit_code, result.stdout, result.timed_out) == (code, "x\n", False)


@needs_docker
async def test_docker_has_no_network(containers):
    code = """
import os, socket
print(sorted(os.listdir('/sys/class/net')))
for target in (('1.1.1.1', 443), ('172.17.0.1', 22), ('169.254.169.254', 80)):
    try:
        socket.create_connection(target, timeout=1)
        print('CONNECTED', target)
    except OSError:
        print('blocked')
try:
    socket.getaddrinfo('example.com', 443)
    print('RESOLVED')
except OSError:
    print('blocked')
"""
    result = await docker_run(code)
    assert result.stdout.splitlines() == ["['lo']"] + ["blocked"] * 4


@needs_docker
async def test_docker_filesystem_is_read_only_except_tmp(containers):
    code = """
for path in ('/app/x', '/app/worker.py', '/etc/x', '/x', '/usr/local/lib/x', '/var/tmp/x', '/dev/shm/x'):
    try:
        open(path, 'w')
        print('WROTE', path)
    except OSError:
        print('blocked')
open('/tmp/ok', 'w').write('1')
try:
    open('/tmp/big', 'wb').write(bytes(64 * 1024 * 1024))
    print('NO LIMIT')
except OSError:
    print('tmp limited')
"""
    result = await docker_run(code)
    assert result.stdout.splitlines() == ["blocked"] * 7 + ["tmp limited"]


@needs_docker
async def test_docker_is_unprivileged_and_limited(containers, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "canary-not-a-key")
    code = """
import os
print(os.getuid(), os.getgid(), os.getgroups())
status = dict(l.split(':', 1) for l in open('/proc/self/status'))
print(int(status['CapEff'], 16), int(status['CapBnd'], 16), status['NoNewPrivs'].strip())
for name in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max'):
    print(open('/sys/fs/cgroup/' + name).read().strip())
print(any('canary' in v for v in os.environ.values()), os.path.exists('/var/run/docker.sock'))
"""
    result = await docker_run(code)
    assert result.stdout.splitlines() == [
        "65534 65534 [65534]",
        "0 0 1",
        str(256 * 1024 * 1024),
        "0",
        "50000 100000",
        "64",
        "False False",
    ]


@needs_docker
async def test_docker_limits_processes_and_memory(containers):
    forks = await docker_run(
        "import os, time\nn = 0\ntry:\n    while n < 200:\n"
        "        if os.fork() == 0:\n            time.sleep(30); os._exit(0)\n        n += 1\n"
        "except OSError:\n    pass\nprint(n)"
    )
    assert int(forks.stdout) < 64
    memory = await docker_run("x = bytearray(600 * 1024 * 1024); print('NO LIMIT')")
    assert memory.exit_code != 0 and "NO LIMIT" not in memory.stdout


@needs_docker
async def test_docker_timeout_kills_the_container(containers):
    start = time.monotonic()
    result = await docker_run("print('started')\nwhile True: pass", timeout_sec=1)
    assert (result.exit_code, result.timed_out) == (124, True)
    assert result.stdout == "started\n"
    assert 3000 <= result.duration_ms and time.monotonic() - start < 8


@needs_docker
async def test_docker_caps_output_with_bounded_memory(containers):
    tracemalloc.start()
    try:
        result = await docker_run(
            "import sys\nwhile True:\n    sys.stdout.write('a' * 65536)\n"
            "    sys.stderr.write('b' * 65536)",
            timeout_sec=1,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result.truncated and result.timed_out
    assert result.stdout == "a" * schemas.MAX_OUTPUT_CHARS
    assert result.stderr == "b" * schemas.MAX_OUTPUT_CHARS
    assert peak < 8 * 1024 * 1024


@needs_docker
@pytest.mark.parametrize("delay", [0.05, 1.5])
async def test_docker_cancellation_removes_the_container(containers, delay):
    task = asyncio.create_task(docker_run("import time; time.sleep(30)"))
    await asyncio.sleep(delay)
    if delay > 1:
        assert running(containers[0])
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@needs_docker
async def test_docker_container_ends_itself_if_the_host_dies(containers, monkeypatch):
    """Backstop: with host cleanup out of reach, the container still stops."""
    monkeypatch.setattr(docker_provider, "HOST_GRACE_SEC", 30)
    monkeypatch.setattr(docker_provider, "BACKSTOP_GRACE_SEC", 1)
    result = await docker_run("import time; time.sleep(30)", timeout_sec=1)
    assert (result.exit_code, result.timed_out) == (137, False)
    assert result.duration_ms < 8000


@needs_docker
async def test_docker_concurrent_requests(containers):
    results = await asyncio.gather(*(docker_run(f"print({i})") for i in range(4)))
    assert [r.stdout for r in results] == ["0\n", "1\n", "2\n", "3\n"]


async def _assert_unavailable():
    result = await docker_run("print(1)")
    assert (result.exit_code, result.timed_out, result.stdout) == (125, False, "")
    assert result.stderr.startswith("Docker sandbox unavailable")
    assert await DockerSandboxProvider().health_check() is False


@needs_docker
async def test_docker_missing_image_is_unavailable(monkeypatch):
    monkeypatch.setenv("SANDBOX_DOCKER_IMAGE", "agentguard-missing:latest")
    await _assert_unavailable()


async def test_docker_unreachable_daemon_is_unavailable(monkeypatch):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", "unix:///nonexistent/docker.sock")
    await _assert_unavailable()


async def test_docker_missing_binary_is_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    await _assert_unavailable()


async def test_docker_hung_client_is_unavailable_and_bounded(monkeypatch, tmp_path):
    fake = tmp_path / "docker"
    fake.write_text("#!/bin/sh\nexec /bin/sleep 60\n")
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(docker_provider, "DOCKER_COMMAND_TIMEOUT_SEC", 0.2)
    monkeypatch.setattr(docker_provider, "KILL_WAIT_SEC", 0.1)
    start = time.monotonic()
    result = await docker_run("print(1)", timeout_sec=1)
    assert (result.exit_code, result.timed_out) == (125, False)
    assert time.monotonic() - start < 10
