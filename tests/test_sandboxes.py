"""Sandbox tests (Track B). Docker tests are skipped without a daemon and image."""

import asyncio
import base64
import errno
import json
import os
import socket
import subprocess
import sys
import time
import tracemalloc
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import httpx
import pytest

from core import schemas
from core.config import get_settings
from core.schemas import SandboxRequest
from sandbox import cloudrun_provider, docker_provider, get_sandbox_provider, worker
from sandbox.cloudrun_provider import CloudRunSandboxProvider
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
    """The real worker, in its own process, on a free local port.

    Test client and code both connect from loopback, so this one opts out of the
    refusal of local requests.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = subprocess.Popen(
        [sys.executable, "-c", "import worker; worker.serve('127.0.0.1')"],
        cwd=SANDBOX_DIR,
        env={
            **os.environ,
            "PORT": str(port),
            "PYTHONDONTWRITEBYTECODE": "1",
            "AGENTGUARD_CANARY": "leaked",
            worker.ALLOW_LOCAL_ENV: "1",
        },
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


def test_worker_refuses_requests_while_busy(worker_url):
    """Ordinary concurrency: a second request during a run is refused."""
    code = (
        "import urllib.request as u, urllib.error\n"
        f"r = u.Request('{worker_url}/execute', data=b'{{\"code\": \"print(1)\"}}',"
        " headers={'Content-Type': 'application/json'})\n"
        "try:\n    u.build_opener(u.ProxyHandler({})).open(r, timeout=3); print('RAN')\n"
        "except urllib.error.HTTPError as e:\n    print(e.code)\n"
    )
    assert post(worker_url, code)["stdout"] == "429\n"
    assert post(worker_url, "print(1)")["stdout"] == "1\n"


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


def jwt(exp):
    body = base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode()).decode().rstrip("=")
    return f"header.{body}.signature"


@pytest.fixture
def auth(monkeypatch):
    """Mocked gcloud: counts token requests and starts with an empty cache."""
    issued = []

    async def fake_token():
        issued.append(jwt(int(time.time()) + 3600 + len(issued)))
        return issued[-1]

    monkeypatch.setattr(cloudrun_provider, "_gcloud_identity_token", fake_token)
    monkeypatch.setattr(cloudrun_provider, "_cached_token", None)
    return issued


@pytest.fixture
def service(monkeypatch):
    """A scripted service: set `replies` to (status, body, delay) tuples."""

    class State:
        replies = []
        seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            length = int(self.headers.get("Content-Length", 0))
            State.seen.append((self.path, self.headers["Authorization"], self.rfile.read(length)))
            status, body, delay = State.replies.pop(0)
            time.sleep(delay)
            try:
                self.send_response(status)
                self.send_header("Location", "/elsewhere")
                self.end_headers()
                self.wfile.write(body)
            except OSError:
                pass  # the client gave up

        do_POST = do_GET

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", f"http://127.0.0.1:{server.server_port}/")
    yield State
    server.shutdown()
    server.server_close()


def reply(status=200, delay=0, **changes):
    result = dict(
        provider="cloudrun", stdout="ok", stderr="", truncated=False,
        exit_code=0, duration_ms=3, timed_out=False,
    )
    return status, json.dumps(result | changes).encode(), delay


async def cloud_run(code="print(1)", timeout_sec=5):
    request = SandboxRequest(code=code, timeout_sec=timeout_sec)
    return await CloudRunSandboxProvider().execute(request)


def assert_cloud_unavailable(result, detail=""):
    assert (result.provider, result.exit_code, result.timed_out) == ("cloudrun", 125, False)
    assert result.stderr.startswith("Cloud Run sandbox unavailable") and detail in result.stderr
    assert "signature" not in result.stderr


async def test_cloudrun_runs_code_on_the_worker(worker_url, auth, monkeypatch):
    monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", worker_url)
    provider = get_sandbox_provider("cloudrun")
    assert await provider.health_check() is True
    result = await provider.execute(SandboxRequest(code="print(sum(range(101)))"))
    assert (result.provider, result.stdout, result.exit_code) == ("cloudrun", "5050\n", 0)
    timed = await cloud_run("while True: pass", timeout_sec=1)
    assert (timed.exit_code, timed.timed_out) == (124, True)
    capped = await cloud_run("print('\\x00' * 100_000)")
    assert capped.truncated and len(capped.stdout) == schemas.MAX_OUTPUT_CHARS
    assert len(auth) == 1  # cached across providers and requests


async def test_cloudrun_sends_the_token_and_request(service, auth):
    service.replies = [reply(provider="spoofed")]
    result = await cloud_run("print(2)", timeout_sec=3)
    path, header, body = service.seen[-1]
    assert (path, header) == ("/execute", f"Bearer {auth[0]}")
    assert json.loads(body) == {"code": "print(2)", "timeout_sec": 3}
    assert (result.provider, result.stdout) == ("cloudrun", "ok")


@pytest.mark.parametrize("status", [401, 403])
async def test_cloudrun_refreshes_a_refused_token_once(service, auth, status):
    service.replies = [reply(status), reply()]
    assert (await cloud_run()).exit_code == 0
    assert [seen[1] for seen in service.seen] == [f"Bearer {token}" for token in auth]
    assert len(auth) == 2
    service.replies = [reply(status), reply(status)]
    assert_cloud_unavailable(await cloud_run(), f"HTTP {status}")
    assert service.replies == [] and cloudrun_provider._cached_token is None


@pytest.mark.parametrize(
    "scripted",
    [
        reply(500),
        reply(302),
        (200, b"not json", 0),
        (200, b"{}", 0),
        reply(stdout="x" * (schemas.MAX_OUTPUT_CHARS + 1)),
        (200, b"x" * 1_000_000, 0),
    ],
)
async def test_cloudrun_bad_responses_are_unavailable(service, auth, scripted):
    service.replies = [scripted]
    assert_cloud_unavailable(await cloud_run())
    assert len(service.seen) == 1  # never run twice


async def test_cloudrun_deadline(service, auth, monkeypatch):
    monkeypatch.setattr(cloudrun_provider, "TOKEN_TIMEOUT_SEC", 0)
    monkeypatch.setattr(cloudrun_provider, "REQUEST_GRACE_SEC", 0)
    monkeypatch.setattr(cloudrun_provider, "HEALTH_TIMEOUT_SEC", 0.2)
    service.replies = [reply(delay=2)]
    result = await cloud_run(timeout_sec=1)
    assert_cloud_unavailable(result, "timed out")
    assert 900 <= result.duration_ms < 1900
    service.replies = [reply(delay=1)]
    assert await CloudRunSandboxProvider().health_check() is False


@pytest.mark.parametrize(
    "url", ["", "http://example.com", "ftp://127.0.0.1", "example.com", "https://", "https://host:abc"]
)
async def test_cloudrun_rejects_bad_urls_without_a_token(auth, monkeypatch, url):
    monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", url)
    assert_cloud_unavailable(await cloud_run(), "CLOUDRUN_SANDBOX_URL")
    assert await CloudRunSandboxProvider().health_check() is False
    assert auth == []


async def test_cloudrun_unreachable(auth, monkeypatch):
    with socket.socket() as closed:
        closed.bind(("127.0.0.1", 0))
        monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", f"http://127.0.0.1:{closed.getsockname()[1]}")
        assert_cloud_unavailable(await cloud_run(), "ConnectError")
        assert await CloudRunSandboxProvider().health_check() is False


def fake_gcloud(tmp_path, monkeypatch, script):
    gcloud = tmp_path / "gcloud"
    gcloud.write_text(f"#!/bin/sh\n{script}\n")
    gcloud.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))


async def test_gcloud_token_command(tmp_path, monkeypatch):
    fake_gcloud(tmp_path, monkeypatch, 'echo "$1.$2.signature"')
    assert await cloudrun_provider._gcloud_identity_token() == "auth.print-identity-token.signature"


@pytest.mark.parametrize("script", ["echo secret.signature >&2; exit 1", "exit 0", "exec /bin/sleep 30"])
async def test_gcloud_failures_are_unavailable(tmp_path, monkeypatch, script):
    fake_gcloud(tmp_path, monkeypatch, script)
    monkeypatch.setattr(cloudrun_provider, "TOKEN_TIMEOUT_SEC", 0.3)
    monkeypatch.setattr(cloudrun_provider, "_cached_token", None)
    monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", "https://sandbox.invalid")
    assert_cloud_unavailable(await cloud_run())


async def test_gcloud_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cloudrun_provider, "_cached_token", None)
    monkeypatch.setenv("CLOUDRUN_SANDBOX_URL", "https://sandbox.invalid")
    assert_cloud_unavailable(await cloud_run(), "gcloud was not found")


async def test_token_is_reused_until_near_expiry(auth, monkeypatch):
    first = await cloudrun_provider._identity_token()
    assert await cloudrun_provider._identity_token() == first
    later = time.time() + 3600
    monkeypatch.setattr(cloudrun_provider.time, "time", lambda: later)
    assert await cloudrun_provider._identity_token() != first
    assert cloudrun_provider._expiry("opaque") == later + 300


def test_configured_provider_is_selected(monkeypatch):
    for name in ("docker", "cloudrun"):
        monkeypatch.setenv("SANDBOX_PROVIDER", name)
        assert get_sandbox_provider().provider_name == name


# B-1: code running in the worker must not be able to submit work to the worker.

# Runs inside the sandbox. It fires requests at its own worker without reading the
# replies, then exits. A queued copy leaves a marker and does the same again.
ATTACK = r"""
import json, socket
if not globals().get('ROOT'):
    open('/tmp/queued-code-ran', 'a').write('x')
own = socket.gethostbyname(socket.gethostname())
payload = json.dumps({'code': 'S=%r\nexec(S)' % S, 'timeout_sec': 2}).encode()
def request(method, path, body=b'', extra=''):
    head = f'{method} {path} HTTP/1.1\r\nHost: x\r\n{extra}Content-Type: application/json\r\n'
    return (head + f'Content-Length: {len(body)}\r\n\r\n').encode() + body
spoof = ('X-Forwarded-For: 203.0.113.9:4444\r\nForwarded: for=203.0.113.9\r\n'
         'X-Real-IP: 203.0.113.9\r\n')
held, sent = [], 0
def fire(target, data, source=None, family=socket.AF_INET):
    global sent
    try:
        s = socket.socket(family)
        s.settimeout(0.5)
        if source:
            s.bind((source, 0))
        s.connect((target, 8080))
        s.sendall(data)
        held.append(s)  # kept open, never read
        sent += 1
    except OSError:
        pass
for _ in range(FILLERS):
    fire('127.0.0.1', request('GET', '/health'))
for i in range(COPIES):
    target = ('127.0.0.1', '127.0.0.2', '0.0.0.0', own)[i % 4]
    source = (None, own, '127.0.0.3')[i % 3]
    fire(target, request('POST', '/execute', payload, spoof if i % 2 else ''), source)
# Several requests on one keep-alive connection.
fire('127.0.0.1', request('GET', '/health') * 20 + request('POST', '/execute', payload) * 5)
fire(own, request('POST', '/execute', payload, spoof) * 5)
fire('::1', request('POST', '/execute', payload), family=socket.AF_INET6)
print(sent)
"""


@pytest.fixture
def image_worker():
    """The worker as the image starts it. Requests from the test arrive from outside."""
    name = f"agentguard-test-worker-{os.urandom(4).hex()}"
    image = get_settings().sandbox_docker_image
    subprocess.run(
        ["docker", "run", "-d", "--rm", "--pull", "never", "--name", name,
         "-p", "127.0.0.1::8080", image],
        check=True, capture_output=True,
    )  # fmt: skip
    try:
        port = subprocess.run(
            ["docker", "port", name, "8080"], check=True, capture_output=True, text=True
        ).stdout.strip().rsplit(":", 1)[1]
        url = f"http://127.0.0.1:{port}"
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health", trust_env=False).status_code == 200:
                    break
            except httpx.TransportError:
                time.sleep(0.1)
        else:
            pytest.fail("worker container did not start")
        yield url, name
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


@needs_docker
@pytest.mark.parametrize("fillers,copies", [(0, 100), (100, 20), (300, 50), (800, 50), (0, 200)])
def test_code_cannot_submit_work_to_its_own_worker(image_worker, fillers, copies):
    url, _ = image_worker
    # Queued copies are the same program without ROOT.
    attack = f"FILLERS, COPIES = {fillers}, {copies}\n" + ATTACK
    code = f"ROOT = True\nS = {attack!r}\nexec(S)"
    result = post(url, code)
    assert result["exit_code"] == 0, result["stderr"]
    assert int(result["stdout"]) >= fillers + copies  # the requests really were sent
    time.sleep(3)
    check = "import os; print(os.path.exists('/tmp/queued-code-ran'))"
    for _ in range(50):
        response = httpx.post(f"{url}/execute", json={"code": check}, timeout=20, trust_env=False)
        if response.status_code != 429:
            break
        time.sleep(0.2)
    else:
        pytest.fail("the worker is still busy with queued code")
    assert response.json()["stdout"] == "False\n"


@needs_docker
def test_code_in_the_worker_gets_403_on_every_local_route(image_worker):
    """Replies are read here: each route connects and is refused."""
    url, _ = image_worker
    code = r"""
import socket
own = socket.gethostbyname(socket.gethostname())
def status(target, source=None, path='/execute', headers=''):
    s = socket.socket()
    s.settimeout(2)
    try:
        if source == 'freebind':
            s.setsockopt(socket.IPPROTO_IP, 15, 1)  # IP_FREEBIND: needs no privilege
            s.bind(('203.0.113.9', 0))
        elif source:
            s.bind((source, 0))
        s.connect((target, 8080))
        s.sendall(f'POST {path} HTTP/1.1\r\nHost: x\r\n{headers}Content-Length: 0\r\n\r\n'.encode())
        return s.recv(100).split()[1].decode()
    except OSError:
        return 'no connection'
spoof = 'X-Forwarded-For: 203.0.113.9:4444\r\nX-Forwarded-Proto: https\r\n'
for target in ('127.0.0.1', '127.1.2.3', '0.0.0.0', own):
    for source in (None, own, '127.0.0.3'):
        for headers in ('', spoof):
            result = status(target, source, headers=headers)
            if result not in ('403', 'no connection'):
                print('OPEN', target, source, result)
print(status('127.0.0.1'), status(own), status(own, path='/health'), status('127.0.0.1', headers=spoof))
print(status(own, 'freebind'), status('127.0.0.1', 'freebind'))
"""
    result = post(url, code)
    assert result["exit_code"] == 0, result["stderr"]
    assert result["stdout"].splitlines() == [
        "403 403 403 403",
        "no connection no connection",
    ]


@needs_docker
def test_image_worker_bounds_the_request_body(image_worker):
    url, _ = image_worker
    port = int(url.rsplit(":", 1)[1])

    def status(headers):
        with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
            # No body is sent: the answer must come without it.
            client.sendall(f"POST /execute HTTP/1.1\r\nHost: x\r\n{headers}\r\n\r\n".encode())
            return int(client.recv(100).split()[1])

    assert status(f"Content-Length: {worker.MAX_BODY_BYTES + 1}") == 413
    assert status("Content-Length: 1000000000") == 413
    assert status("Transfer-Encoding: chunked") == 411
    # The largest valid request still fits.
    body = json.dumps({"code": "#" + "\U0001f600" * (schemas.MAX_CODE_CHARS - 1)})
    assert len(body) <= worker.MAX_BODY_BYTES
    response = httpx.post(
        f"{url}/execute", content=body, headers={"Content-Type": "application/json"},
        timeout=20, trust_env=False,
    )
    assert (response.status_code, response.json()["exit_code"]) == (200, 0)


@needs_docker
def test_image_never_opts_out():
    image = get_settings().sandbox_docker_image
    config = json.loads(
        subprocess.run(
            ["docker", "image", "inspect", "--format", "{{json .Config}}", image],
            check=True, capture_output=True, text=True,
        ).stdout
    )
    assert worker.ALLOW_LOCAL_ENV not in json.dumps(config)
    assert config["Cmd"] == ["python3", "worker.py"] and not config.get("Entrypoint")
    assert worker.ALLOW_LOCAL_ENV not in (SANDBOX_DIR / "Dockerfile").read_text()


async def gate(client, headers=(), kind="http"):
    """Runs one request through the app. Returns the status, or the message type."""
    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": kind, "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/health", "raw_path": b"/health", "query_string": b"",
        "root_path": "", "headers": list(headers), "client": client,
        "server": ("192.0.2.1", 8080),
    }  # fmt: skip
    await worker.app(scope, receive, send)
    return sent[0].get("status", sent[0]["type"])


def own_address():
    """A non-loopback address of this machine. No packet is sent."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            probe.connect(("203.0.113.1", 9))
        except OSError:
            pytest.skip("no non-loopback address")
        return probe.getsockname()[0]


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes", " 1"])
async def test_local_requests_are_refused_by_default(monkeypatch, value):
    monkeypatch.delenv(worker.ALLOW_LOCAL_ENV, raising=False)
    if value is not None:
        monkeypatch.setenv(worker.ALLOW_LOCAL_ENV, value)
    assert await gate(("127.0.0.1", 5000)) == 403


@pytest.mark.parametrize(
    "client",
    [
        None,
        ("127.0.0.1", 1),
        ("127.255.0.9", 1),
        ("::1", 1),
        ("::ffff:127.0.0.1", 1),
        ("0.0.0.0", 1),
        ("::", 1),
        ("224.0.0.1", 1),
        ("fe80::1%nosuchlink", 1),
        ("testclient", 1),
        ("", 1),
        ("127.0.0.1",),
    ],
)
async def test_local_or_unclear_peers_are_refused(monkeypatch, client):
    monkeypatch.delenv(worker.ALLOW_LOCAL_ENV, raising=False)
    assert await gate(client) == 403


async def test_own_address_is_refused_and_others_pass(monkeypatch):
    monkeypatch.delenv(worker.ALLOW_LOCAL_ENV, raising=False)
    own = own_address()
    assert await gate((own, 1)) == 403
    assert await gate((f"::ffff:{own}", 1)) == 403
    # Wherever the front end connects from, if it is not our address.
    for outside in ("203.0.113.9", "169.254.1.1"):
        assert await gate((outside, 1)) == 200
    assert await gate(("203.0.113.9", 1), kind="websocket") == "websocket.close"


@pytest.mark.parametrize("failure", [errno.EACCES, errno.EAFNOSUPPORT, errno.EINVAL, None])
async def test_peer_check_fails_closed(monkeypatch, failure):
    """Any answer from the kernel other than "not our address" refuses."""
    monkeypatch.delenv(worker.ALLOW_LOCAL_ENV, raising=False)

    def broken(*args):
        raise OSError(failure, "probe failed")

    monkeypatch.setattr(worker.socket, "socket", broken)
    assert await gate(("203.0.113.9", 1)) == 403


async def test_serve_turns_proxy_headers_off(monkeypatch):
    calls = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append(kwargs))
    worker.serve()
    assert calls[0]["proxy_headers"] is False and calls[0]["host"] == "0.0.0.0"
