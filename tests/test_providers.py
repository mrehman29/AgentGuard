"""Model provider contract, role settings and failures. No real model is called.

Every provider is driven over the wire against a local fake server. A new
provider adds a fake of its API to FAKES.
"""

import ast
import asyncio
import json
import logging
import os
import shutil
import socket
import sqlite3
import ssl
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import httpx2
import pytest
from dotenv import dotenv_values
from google.genai import errors, types
from starlette.testclient import TestClient

import core.config
from app import agent_loop
from app.main import app
from core import budget_store, guard_core, pipeline
from core.config import PROJECT_ROOT, PROVIDER_NAMES, ConfigError, get_settings
from core.providers import (
    PROVIDERS,
    ModelError,
    ModelReply,
    ToolCall,
    ToolSpec,
    gemini,
    get_provider,
)
from core.schemas import Decision, ModelRates, OwaspVerdict, SandboxResult, TokenUsage

KEY = "secret-key-for-tests"
GOLDEN = PROJECT_ROOT / "tests" / "data" / "gemini_requests.json"
VERDICT = json.dumps(
    {"decision": "ALLOW", "risk_score": 1, "owasp_tag": "SAFE: Compliant Action", "reason": "ok"}
)
TOOL = ToolSpec(
    "execute_python",
    "Run code.",
    {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
)
# 100 input, 40 output of which 15 are reasoning.
USAGE = TokenUsage(prompt_tokens=100, candidates_tokens=25, thoughts_tokens=15)


class FakeGemini:
    @staticmethod
    def reply(text=None, tool_call=None, finished=True, usage=True) -> dict:
        if tool_call:
            call = {"name": tool_call[0], "args": tool_call[1]}
            part = {"functionCall": {k: v for k, v in call.items() if v is not None}}
        else:
            part = {"text": text}
        body = {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [part]},
                    "finishReason": "STOP" if finished else "MAX_TOKENS",
                }
            ]
        }
        if usage:
            body["usageMetadata"] = {
                "promptTokenCount": 100,
                "candidatesTokenCount": 25,
                "thoughtsTokenCount": 15,
            }
        return body


FAKES = {"gemini": FakeGemini}


@contextmanager
def fake_server(tls: ssl.SSLContext | None = None):
    """Set .body, .status, .delay, .location or .hook. Requests are kept in .requests."""
    state = SimpleNamespace(
        body={}, status=200, delay=0.0, location=None, hook=None, requests=[]
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.record(json.loads(raw))
            if state.hook:
                state.hook()
            time.sleep(state.delay)
            data = state.body if isinstance(state.body, bytes) else json.dumps(state.body).encode()
            try:
                self.send_response(state.status)
                if state.location:
                    self.send_header("Location", state.location)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass  # the client gave up

        def do_CONNECT(self):
            """Seen only when this server stands in for a proxy: record and refuse."""
            self.record(None)
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def record(self, body):
            state.requests.append(
                SimpleNamespace(
                    method=self.command, path=self.path, headers=dict(self.headers), body=body
                )
            )

        def log_message(self, *args):
            pass

    class Server(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass  # refused TLS handshakes are expected

    http = Server(("127.0.0.1", 0), Handler)
    if tls:
        http.socket = tls.wrap_socket(http.socket, server_side=True)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    state.url = f"{'https' if tls else 'http'}://127.0.0.1:{http.server_port}"
    try:
        yield state
    finally:
        http.shutdown()
        http.server_close()
        thread.join(5)


@pytest.fixture
def server():
    with fake_server() as state:
        yield state


@pytest.fixture(autouse=True)
def settings_env(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.lower().endswith("_proxy") or name.startswith(("GOOGLE_", "GEMINI_")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("SQLITE_DB_PATH", str(tmp_path / "guard.db"))
    monkeypatch.setattr(budget_store, "_tracker", None)
    monkeypatch.setattr(gemini, "_clients", {})


@pytest.fixture(params=PROVIDER_NAMES)
def contract(request, monkeypatch, server):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", request.param)
    monkeypatch.setenv("CLASSIFIER_BASE_URL", server.url)
    monkeypatch.setenv("CLASSIFIER_API_KEY", KEY)
    set_role(monkeypatch, "CLASSIFIER", input_cost_per_1m_usd=0.3, output_cost_per_1m_usd=2.5)
    return FAKES[request.param]


async def classify():
    return await get_provider(get_settings().classifier).classify("rules", "data", OwaspVerdict)


async def agent_turn():
    return await get_provider(get_settings().agent).agent_turn("rules", "do it", [TOOL])


OPERATIONS = [classify, agent_turn]


def reserved(session_id="demo"):
    return budget_store.get_tracker().check_budget(session_id).reserved_gbp


# --- the contract ---


def test_every_configurable_provider_exists_and_is_tested():
    assert set(PROVIDERS) == set(PROVIDER_NAMES) == set(FAKES)


async def test_classification(contract, server):
    server.body = contract.reply(text=VERDICT)
    reply = await classify()
    assert (reply.text, reply.tool_call, reply.finished) == (VERDICT, None, True)
    assert reply.usage == USAGE
    assert len(server.requests) == 1


async def test_agent_tool_call(contract, server):
    server.body = contract.reply(tool_call=("execute_python", {"code": "1"}))
    reply = await agent_turn()
    assert (reply.tool_call.name, reply.tool_call.args) == ("execute_python", {"code": "1"})
    assert reply.text is None and reply.usage == USAGE
    assert len(server.requests) == 1


async def test_agent_text(contract, server):
    server.body = contract.reply(text="hello")
    reply = await agent_turn()
    assert (reply.text, reply.tool_call, reply.finished) == ("hello", None, True)


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_unfinished_reply_is_marked_and_keeps_usage(contract, server, operation):
    server.body = contract.reply(text="{", finished=False)
    reply = await operation()
    assert reply.finished is False and reply.usage == USAGE


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("case", ["no usage", "empty", "not json", "http error"])
async def test_unbillable_reply_is_a_safe_error(contract, server, caplog, operation, case):
    caplog.set_level(logging.DEBUG)
    leak = f"detail {KEY}"
    if case == "no usage":
        server.body = contract.reply(text=VERDICT, usage=False)
    elif case == "not json":
        server.body = leak.encode()
    elif case == "http error":
        server.status = 429
        server.body = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": leak}}
    with pytest.raises(ModelError) as caught:
        await operation()
    error = caught.value
    assert error.__suppress_context__ or error.__context__ is None
    assert KEY not in f"{error!s} {error!r} {caplog.text}"
    if case == "http error":
        assert "code=429 status=RESOURCE_EXHAUSTED" in str(error)
        assert (error.code, error.status) == (429, "RESOURCE_EXHAUSTED")
    else:
        assert (error.code, error.status) == (None, None)
    if case == "no usage":
        assert error.kind == "MissingUsage"
    assert len(server.requests) == 1  # no retry


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_impossible_usage_is_a_safe_error(contract, server, operation):
    server.body = contract.reply(text="x")
    server.body["usageMetadata"]["candidatesTokenCount"] = -1
    with pytest.raises(ModelError) as caught:
        await operation()
    assert caught.value.kind == "InvalidUsage"


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_unreachable_server_is_a_model_error(contract, monkeypatch, operation):
    monkeypatch.setenv("CLASSIFIER_BASE_URL", "http://127.0.0.1:1")
    with pytest.raises(ModelError):
        await operation()


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_cancellation_passes_through(contract, server, operation):
    server.body, server.delay = contract.reply(text=VERDICT), 0.5
    task = asyncio.create_task(operation())
    while not server.requests:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("name", PROVIDER_NAMES)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_missing_key_makes_no_call(monkeypatch, server, name, operation):
    monkeypatch.setenv("CLASSIFIER_PROVIDER", name)
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GOOGLE_API_KEY", "ambient-key")
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
    with pytest.raises(ModelError, match="MissingApiKey"):
        await operation()
    assert server.requests == []


async def test_gemini_requests(contract, server):
    """What goes on the wire, including the tool schema."""
    server.body = contract.reply(text=VERDICT)
    await classify()
    await agent_turn()
    first, second = server.requests
    assert first.headers["x-goog-api-key"] == KEY and KEY not in first.path + second.path
    assert get_settings().classifier.model in first.path
    assert first.body["generationConfig"]["maxOutputTokens"] == 512
    [declaration] = second.body["tools"][0]["functionDeclarations"]
    assert declaration["name"] == "execute_python"
    assert declaration["parameters"]["required"] == ["code"]
    assert declaration["parameters"]["properties"]["code"]["type"] == "STRING"


def test_tool_schema_with_an_unsupported_keyword_is_refused(server, monkeypatch):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
    tool = ToolSpec("t", "d", {"type": "object", "properties": {"a": {"type": "array", "items": {}}}})
    with pytest.raises(ModelError, match="ValueError"):
        asyncio.run(get_provider(get_settings().agent).agent_turn("rules", "x", [tool]))
    assert server.requests == []


async def test_requests_equal_those_of_v1(server, monkeypatch):
    """The fixture was captured by running main at 02369a4 against a fake server."""
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    monkeypatch.setenv("GEMINI_API_KEY", golden["classify"]["api_key"])
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
    server.body = FakeGemini.reply(text=VERDICT)
    result = await guard_core.evaluate_owasp_policy(
        "execute_python", {"code": "print(sum(range(101)))"}, "demo"
    )
    assert result.decision is Decision.ALLOW
    await agent_loop.default_model_caller("Calculate 6*7", get_settings())
    for name, request in zip(("classify", "agent_turn"), server.requests, strict=True):
        sent = (request.path, request.headers["x-goog-api-key"], request.body)
        assert sent == tuple(golden[name][part] for part in ("path", "api_key", "body")), name


def imports_gemini_sdk(source: str) -> bool:
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names += [f"{node.module}.{alias.name}" for alias in node.names]
    return any(name.startswith(("google.genai", "google.generativeai")) for name in names)


@pytest.mark.parametrize(
    "source",
    [
        "import google.genai",
        "import google.genai as g",
        "from google import genai",
        "from google import auth, genai",
        "from google.genai import types",
        "from google.genai import types, errors",
        "def f():\n    from google.genai.errors import APIError",
        "import google.generativeai",
    ],
)
def test_the_import_scan_sees_every_form(source):
    assert imports_gemini_sdk(source)


def test_the_import_scan_ignores_other_google_packages():
    assert not imports_gemini_sdk("import google.auth\nfrom google.oauth2 import id_token")


def test_gemini_sdk_is_imported_only_by_its_provider():
    importers = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for folder in ("app", "core", "sandbox")
        for path in (PROJECT_ROOT / folder).rglob("*.py")
        if imports_gemini_sdk(path.read_text(encoding="utf-8"))
    }
    assert importers == {"core/providers/gemini.py"}


# --- transport ---


BACKENDS = ["", "httpx2", "httpx"]  # GOOGLE_GENAI_HTTP_CLIENT


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_redirects_are_not_followed(contract, server, monkeypatch, operation, backend):
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", backend)
    with fake_server() as other:
        other.body = contract.reply(text=VERDICT)
        server.status, server.location = 307, f"{other.url}/elsewhere"
        with pytest.raises(ModelError):
            await operation()
        assert len(server.requests) == 1 and other.requests == []


async def test_the_sdks_synchronous_client_follows_no_redirect(contract, server):
    with fake_server() as other:
        server.status, server.location = 307, f"{other.url}/elsewhere"
        client = gemini._client(KEY, server.url)
        with pytest.raises(errors.APIError):
            await asyncio.to_thread(client.models.generate_content, model="m", contents="x")
        assert len(server.requests) == 1 and other.requests == []


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_requests_go_through_httpx2(contract, server, monkeypatch, operation, backend):
    monkeypatch.setenv("GOOGLE_GENAI_HTTP_CLIENT", backend)
    sent = []
    send = httpx2.AsyncClient.send

    async def spy(self, request, **kwargs):
        sent.append((self.follow_redirects, self.timeout))
        return await send(self, request, **kwargs)

    monkeypatch.setattr(httpx2.AsyncClient, "send", spy)
    server.body = contract.reply(text=VERDICT)
    await operation()
    assert sent == [(False, httpx2.Timeout(None))]


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_vertex_variables_are_ignored(contract, server, monkeypatch, operation):
    server.body = contract.reply(text=VERDICT)
    await operation()
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "a-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "a-location")
    gemini._clients.clear()  # the environment is read when a client is made
    await operation()
    before, after = server.requests
    assert after.path == before.path and after.body == before.body
    assert after.headers["x-goog-api-key"] == KEY


async def test_two_roles_reuse_two_clients(contract, server, monkeypatch):
    set_role(monkeypatch, "AGENT", api_key="agent-key")
    server.body = contract.reply(text=VERDICT)
    seen = []
    for operation in (classify, agent_turn, classify, agent_turn):
        await operation()
        seen.append(tuple(map(id, gemini._clients.values())))
    assert len(gemini._clients) == 2 and seen[1] == seen[2] == seen[3]
    keys = [r.headers["x-goog-api-key"] for r in server.requests]
    assert keys == [KEY, "agent-key", KEY, "agent-key"]


async def test_client_cache_is_bounded():
    first = gemini._client("key-0", "")
    for number in range(1, gemini.MAX_CLIENTS + 2):
        gemini._client(f"key-{number}", "")
    assert len(gemini._clients) == gemini.MAX_CLIENTS
    assert first not in gemini._clients.values()


def test_clients_of_closed_loops_are_dropped():
    async def make(key):
        gemini._client(key, "")

    asyncio.run(make("one"))
    asyncio.run(make("two"))
    assert [key for key, _, _ in gemini._clients] == ["two"]


# A range that some HTTP libraries cannot parse.
IPV6_RANGE = "localhost,127.0.0.1,::1/128,fd00::/8"


@pytest.mark.parametrize(
    "variables",
    [
        {"NO_PROXY": IPV6_RANGE},
        {"NO_PROXY": "::1/128"},
        {"HTTP_PROXY": "not a url", "HTTPS_PROXY": "ftp://x", "ALL_PROXY": "socks9://[bad"},
    ],
)
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_plain_http_to_this_machine_never_fails_on_proxy_variables(
    server, monkeypatch, operation, variables
):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", server.url)
    server.body = FakeGemini.reply(text=VERDICT)
    with fake_server() as proxy:
        monkeypatch.setenv("HTTP_PROXY", proxy.url)
        for name, value in variables.items():
            monkeypatch.setenv(name, value)
        assert (await operation()).text == VERDICT
        assert proxy.requests == [] and len(server.requests) == 1


@pytest.mark.parametrize("variable", ["HTTP_PROXY", "ALL_PROXY"])
@pytest.mark.parametrize("through_sdk_variable", [False, True])
@pytest.mark.parametrize("operation", OPERATIONS)
async def test_plain_http_to_this_machine_never_uses_a_proxy(
    server, monkeypatch, operation, through_sdk_variable, variable
):
    name = "GOOGLE_GEMINI_BASE_URL" if through_sdk_variable else "CLASSIFIER_BASE_URL"
    monkeypatch.setenv(name, server.url)
    if not through_sdk_variable:
        set_role(
            monkeypatch, "CLASSIFIER", api_key=KEY, input_cost_per_1m_usd=1,
            output_cost_per_1m_usd=1,
        )
    server.body = FakeGemini.reply(text=VERDICT)
    with fake_server() as proxy:
        monkeypatch.setenv(variable, proxy.url)
        await operation()
        assert proxy.requests == [] and len(server.requests) == 1


@pytest.fixture(scope="module")
def certificate(tmp_path_factory):
    """A self-signed certificate for 127.0.0.1."""
    if shutil.which("openssl") is None:
        pytest.skip("openssl is not available")
    folder = tmp_path_factory.mktemp("tls")
    cert, key = folder / "cert.pem", folder / "key.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(key), "-out", str(cert), "-subj", "/CN=127.0.0.1",
            "-addext", "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    return SimpleNamespace(path=str(cert), context=context)


@pytest.fixture
def https(certificate, monkeypatch):
    with fake_server(certificate.context) as state:
        set_role(
            monkeypatch, "CLASSIFIER", base_url=state.url, api_key=KEY,
            input_cost_per_1m_usd=1, output_cost_per_1m_usd=1,
        )
        state.body = FakeGemini.reply(text=VERDICT)
        yield state


def trust_in_system_store(monkeypatch, cafile):
    """As if the certificate had been added to the operating system's store."""
    paths = ssl.get_default_verify_paths()._replace(openssl_cafile=cafile)
    monkeypatch.setattr(ssl, "get_default_verify_paths", lambda: paths)


@pytest.fixture
def no_certificate_variables(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_trusts_the_system_store(
    https, certificate, no_certificate_variables, monkeypatch, operation
):
    trust_in_system_store(monkeypatch, certificate.path)
    assert (await operation()).text == VERDICT
    assert len(https.requests) == 1


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_trusts_ssl_cert_dir(
    https, certificate, no_certificate_variables, monkeypatch, tmp_path, operation
):
    digest = subprocess.run(
        ["openssl", "x509", "-hash", "-noout", "-in", certificate.path],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    shutil.copy(certificate.path, tmp_path / f"{digest}.0")
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert (await operation()).text == VERDICT


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_loads_ssl_cert_file_and_ssl_cert_dir_together(
    https, certificate, monkeypatch, tmp_path, operation
):
    """As the SDK's default does: httpx2 alone would stop at the file."""
    other = tmp_path / "other.pem"
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
            "-keyout", str(tmp_path / "other.key"), "-out", str(other), "-subj", "/CN=other",
        ],
        check=True, capture_output=True, timeout=30,
    )
    digest = subprocess.run(
        ["openssl", "x509", "-hash", "-noout", "-in", certificate.path],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip()
    shutil.copy(certificate.path, tmp_path / f"{digest}.0")
    monkeypatch.setenv("SSL_CERT_FILE", str(other))
    monkeypatch.setenv("SSL_CERT_DIR", str(tmp_path))
    assert (await operation()).text == VERDICT


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_with_an_ipv6_range_in_no_proxy(https, certificate, monkeypatch, operation):
    monkeypatch.setenv("SSL_CERT_FILE", certificate.path)
    with fake_server() as proxy:
        monkeypatch.setenv("HTTPS_PROXY", proxy.url)
        monkeypatch.setenv("NO_PROXY", IPV6_RANGE)
        assert (await operation()).text == VERDICT
        assert proxy.requests == [] and len(https.requests) == 1
        # The range alone excludes nothing here: the proxy is used, without an error.
        monkeypatch.setenv("NO_PROXY", "::1/128")
        gemini._clients.clear()
        with pytest.raises(ModelError):
            await operation()
        assert [r.method for r in proxy.requests] == ["CONNECT"]


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_trusts_ssl_cert_file(https, certificate, monkeypatch, operation):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("SSL_CERT_DIR", raising=False)
    # Neither the system store nor anything else knows this certificate yet.
    with pytest.raises(ModelError, match="ConnectError"):
        await operation()
    assert https.requests == []
    monkeypatch.setenv("SSL_CERT_FILE", certificate.path)
    gemini._clients.clear()  # the certificate is read when a client is made
    assert (await operation()).text == VERDICT


@pytest.mark.parametrize("operation", OPERATIONS)
async def test_https_honours_proxy_settings(https, certificate, monkeypatch, operation):
    monkeypatch.setenv("SSL_CERT_FILE", certificate.path)
    with fake_server() as proxy:
        monkeypatch.setenv("HTTPS_PROXY", proxy.url)
        with pytest.raises(ModelError):
            await operation()
        [connect] = proxy.requests
        assert connect.method == "CONNECT" and KEY not in str(connect.headers)
        assert https.requests == []
        monkeypatch.setenv("NO_PROXY", "127.0.0.1")
        gemini._clients.clear()  # proxies are read when a client is made
        assert (await operation()).text == VERDICT
        assert len(proxy.requests) == 1 and len(https.requests) == 1


# --- settings ---


def test_existing_settings_serve_both_roles(monkeypatch):
    values = {
        "GEMINI_MODEL": "some-model",
        "INPUT_COST_PER_1M_USD": "1",
        "OUTPUT_COST_PER_1M_USD": "2",
        "CLASSIFIER_TIMEOUT_SEC": "7",
        "BUDGET_RESERVE_OUTPUT_TOKENS": "99",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    settings = get_settings()
    assert settings.agent == settings.classifier
    model = settings.classifier
    assert (model.provider, model.base_url, model.model, model.api_key) == (
        "gemini", "", "some-model", KEY,
    )
    assert (model.input_cost_per_1m_usd, model.output_cost_per_1m_usd) == (1, 2)
    assert (model.timeout_sec, model.max_output_tokens) == (7, 99)


def set_role(monkeypatch, role, **values):
    for name, value in values.items():
        monkeypatch.setenv(f"{role}_{name.upper()}", str(value))


def test_roles_override_and_fall_back_in_order(monkeypatch):
    set_role(
        monkeypatch, "CLASSIFIER", model="c-model", base_url="https://c.example/v1",
        api_key="c-key", input_cost_per_1m_usd=3, output_cost_per_1m_usd=4,
        max_output_tokens=100,
    )
    set_role(
        monkeypatch, "AGENT", model="a-model", input_cost_per_1m_usd=8,
        output_cost_per_1m_usd=9, timeout_sec=30,
    )
    settings = get_settings()
    classifier, agent = settings.classifier, settings.agent
    assert (classifier.model, classifier.api_key) == ("c-model", "c-key")
    assert (classifier.output_cost_per_1m_usd, classifier.timeout_sec) == (4, 5)
    assert (agent.model, agent.api_key, agent.base_url) == (
        "a-model", "c-key", "https://c.example/v1",
    )
    assert (agent.input_cost_per_1m_usd, agent.output_cost_per_1m_usd) == (8, 9)
    assert (agent.timeout_sec, agent.max_output_tokens) == (30, 100)
    assert settings.rates_for(agent) == ModelRates(
        provider="gemini", model="a-model", input_cost_per_1m_usd=8,
        output_cost_per_1m_usd=9, usd_to_gbp=0.8,
    )


def test_agent_with_the_classifiers_model_inherits_its_rates(monkeypatch):
    set_role(
        monkeypatch, "CLASSIFIER", model="c", input_cost_per_1m_usd=3, output_cost_per_1m_usd=4
    )
    set_role(monkeypatch, "AGENT", model="c", api_key="other-key")
    agent = get_settings().agent
    assert (agent.input_cost_per_1m_usd, agent.output_cost_per_1m_usd) == (3, 4)


@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
@pytest.mark.parametrize(
    "change",
    [{"model": "another"}, {"base_url": "https://other.example", "api_key": "k"}],
)
@pytest.mark.parametrize("missing", ["input_cost_per_1m_usd", "output_cost_per_1m_usd"])
def test_a_different_model_or_url_needs_both_rates(monkeypatch, role, change, missing):
    rates = {"input_cost_per_1m_usd": 1, "output_cost_per_1m_usd": 2}
    del rates[missing]
    set_role(monkeypatch, role, **change, **rates)
    with pytest.raises(ConfigError, match=f"{role}_{missing.upper()}"):
        get_settings()


def test_agent_settings_do_not_change_the_classifier(monkeypatch):
    before = get_settings().classifier
    set_role(
        monkeypatch, "AGENT", model="a", api_key="a-key", base_url="https://a.example",
        input_cost_per_1m_usd=1, output_cost_per_1m_usd=1,
    )
    assert get_settings().classifier == before


@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
def test_a_key_is_not_shared_with_another_url(monkeypatch, role):
    set_role(monkeypatch, role, base_url="https://other.example")
    with pytest.raises(ConfigError, match=f"{role}_API_KEY"):
        get_settings()


URL_NAMES = ["CLASSIFIER_BASE_URL", "AGENT_BASE_URL", "GOOGLE_GEMINI_BASE_URL"]


def set_url(monkeypatch, name, url):
    monkeypatch.setenv(name, url)
    if name != "GOOGLE_GEMINI_BASE_URL":
        set_role(
            monkeypatch, name.split("_")[0], api_key="k", input_cost_per_1m_usd=1,
            output_cost_per_1m_usd=1,
        )


@pytest.mark.parametrize("name", URL_NAMES)
@pytest.mark.parametrize(
    "url",
    [
        "https://models.example/v1", "http://localhost:8000", "http://127.0.0.1:8000/v1",
        "http://127.8.8.8", "http://[::1]:8000",
    ],
)
def test_allowed_urls(monkeypatch, name, url):
    set_url(monkeypatch, name, url)
    get_settings()


@pytest.mark.parametrize("name", URL_NAMES)
@pytest.mark.parametrize(
    "url",
    [
        f"http://models.example/{KEY}", "http://10.0.0.5", "http://0.0.0.0",
        "http://localhost.example", "http://127.0.0.1.example",
        f"http://127.0.0.1\\@models.example/{KEY}", f"https://user:{KEY}@host.example",
        f"https://host.example/?key={KEY}", f"ftp://{KEY}", f" https://{KEY}.example/ a",
        f"https://{KEY}.example/v1?", f"https://{KEY}.example/v1#", "http://127.0.0.1:8000#",
        "http://127.0.0.1:8000?",
    ],
)
def test_refused_urls_are_not_echoed(monkeypatch, name, url):
    set_url(monkeypatch, name, url)
    with pytest.raises(ConfigError) as caught:
        get_settings()
    assert name in str(caught.value) and KEY not in str(caught.value)


def test_sdk_url_is_not_checked_when_no_role_would_use_it(monkeypatch):
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", "http://models.example")
    set_url(monkeypatch, "CLASSIFIER_BASE_URL", "https://models.example")
    get_settings()


def test_sdk_url_in_the_env_file_only_is_ignored(monkeypatch):
    monkeypatch.setattr(
        core.config, "dotenv_values",
        lambda *a, **k: {"GOOGLE_GEMINI_BASE_URL": "http://models.example"},
    )
    get_settings()  # the SDK reads the process environment only


@pytest.mark.parametrize(
    "name,value",
    [
        ("PROVIDER", KEY),
        ("BASE_URL", f"https://user:{KEY}@host.example"),
        ("BASE_URL", f"https://host.example/v1?key={KEY}"),
        ("BASE_URL", f"https://host.example/v1#{KEY}"),
        ("BASE_URL", f"ftp://{KEY}"),
        ("BASE_URL", f"host.example/{KEY}"),
        ("BASE_URL", f"https://host.example:{KEY}"),
        ("BASE_URL", f"https://[{KEY}"),
        ("API_KEY", f"{KEY} two"),
        ("API_KEY", f"{KEY}é"),
        ("INPUT_COST_PER_1M_USD", KEY),
        ("OUTPUT_COST_PER_1M_USD", "0"),
        ("TIMEOUT_SEC", "-1"),
        ("MAX_OUTPUT_TOKENS", "1.5"),
    ],
)
@pytest.mark.parametrize("role", ["CLASSIFIER", "AGENT"])
def test_bad_role_setting_names_itself_without_values(monkeypatch, role, name, value):
    monkeypatch.setenv(f"{role}_{name}", value)
    with pytest.raises(ConfigError) as caught:
        get_settings()
    assert f"{role}_{name}" in str(caught.value) and KEY not in str(caught.value)


def test_bad_gemini_key_is_named_not_shown(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", f"{KEY} x")
    with pytest.raises(ConfigError) as caught:
        get_settings()
    assert "GEMINI_API_KEY" in str(caught.value) and KEY not in str(caught.value)


def test_keys_are_not_in_any_repr(monkeypatch):
    set_role(monkeypatch, "AGENT", api_key="agent-secret")
    settings = get_settings()
    text = repr(settings) + repr(get_provider(settings.agent)) + str(settings.classifier)
    assert KEY not in text and "agent-secret" not in text


def test_server_refuses_to_start_with_a_bad_setting(monkeypatch):
    monkeypatch.setenv("AGENT_PROVIDER", "nope")
    with pytest.raises(ConfigError, match="AGENT_PROVIDER"):
        with TestClient(app, base_url="http://127.0.0.1"):
            pass


def run_product(code: str) -> subprocess.CompletedProcess:
    """Python code in a new process that does not read the project .env."""
    prelude = (
        f"import runpy, sys; sys.path.insert(0, {str(PROJECT_ROOT)!r}); import core.config;"
        " core.config.dotenv_values = lambda *a, **k: {}; "
    )
    return subprocess.run(
        [sys.executable, "-c", prelude + code],
        input="", capture_output=True, text=True, timeout=30, cwd=PROJECT_ROOT,
    )


BAD_STARTUP = [
    ("AGENT_API_KEY", f"{KEY} x"),
    ("CLASSIFIER_BASE_URL", f"http://{KEY}.example"),
    ("AGENT_MODEL", "unpriced-model"),
]


@pytest.mark.parametrize("name,value", BAD_STARTUP)
def test_stdio_server_exits_on_a_bad_setting(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    script = str(PROJECT_ROOT / "app" / "mcp_stdio.py")
    done = run_product(f"runpy.run_path({script!r}, run_name='__main__')")
    assert done.returncode != 0 and done.stdout == ""
    assert "ConfigError" in done.stderr and name.split("_")[0] in done.stderr
    assert KEY not in done.stderr


@pytest.mark.parametrize("name,value", BAD_STARTUP)
def test_uvicorn_exits_on_a_bad_setting(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    done = run_product(
        f"sys.argv = ['uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '{port}'];"
        " runpy.run_module('uvicorn', run_name='__main__', alter_sys=True)"
    )
    output = done.stdout + done.stderr
    assert done.returncode != 0 and "Application startup failed" in output
    assert "ConfigError" in output and KEY not in output


def test_settings_of_an_existing_env_file_load(monkeypatch):
    """The example file, whose active lines are the v1.0 settings, with a key filled in."""
    values = dotenv_values(PROJECT_ROOT / ".env.example")
    assert not any(name.startswith(("AGENT_", "CLASSIFIER_M", "CLASSIFIER_A")) for name in values)
    for name, value in values.items():
        monkeypatch.setenv(name, value or "")
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    settings = get_settings()
    assert settings.agent == settings.classifier
    assert (settings.classifier.model, settings.classifier.api_key) == (
        "gemini-3.5-flash-lite", KEY,
    )


# --- the two callers: costs and failures ---


@pytest.fixture
def rates(contract, monkeypatch):
    """Different rates and output limits per role."""
    set_role(monkeypatch, "CLASSIFIER", input_cost_per_1m_usd=1, output_cost_per_1m_usd=2)
    set_role(
        monkeypatch, "AGENT", model="agent-model", input_cost_per_1m_usd=10,
        output_cost_per_1m_usd=20, max_output_tokens=1000,
    )
    return contract


CLASSIFIER_COST = (100 * 1 + 40 * 2) / 1_000_000 * 0.8
AGENT_COST = (100 * 10 + 40 * 20) / 1_000_000 * 0.8


@pytest.fixture
def sandbox(monkeypatch):
    """Records every execution."""
    ran = []

    class Sandbox:
        provider_name = "docker"

        async def execute(self, request):
            ran.append(request.code)
            return SandboxResult(
                provider="docker", stdout="", stderr="", truncated=False, exit_code=0,
                duration_ms=0, timed_out=False,
            )

    monkeypatch.setattr(pipeline, "get_sandbox_provider", lambda override=None: Sandbox())
    return ran


def allow_every_action(monkeypatch):
    """So that the only model call of a run is the agent's."""

    async def allow(*_, **__):
        return guard_core.GuardEvaluation(
            decision=Decision.ALLOW, risk_score=0, owasp_tag="SAFE: Compliant Action",
            reason="ok", stage="OWASP_LLM", latency_ms=0,
        )

    monkeypatch.setattr(guard_core, "evaluate_action", allow)


async def run_agent(monkeypatch):
    """One standard run with every action allowed. Returns the events."""
    allow_every_action(monkeypatch)
    return [event async for event in agent_loop.run_agent_workflow("do it")]


async def run_guarded():
    """One action through the real guard and pipeline."""
    return await pipeline.execute_guarded("sum", "execute_python", {"code": "1"}, "demo")


def entries(label):
    return [
        (e.outcome, e.cost_gbp)
        for e in budget_store.get_tracker().recent_entries()
        if e.label == label
    ]


def agent_entries():
    return entries("agent turn")


def spent():
    return budget_store.get_tracker().check_budget("demo").spent_gbp


def assert_no_key(caplog, *values):
    ledger = [e.model_dump_json() for e in budget_store.get_tracker().recent_entries()]
    text = " ".join([caplog.text, *ledger, *(v.model_dump_json() for v in values)])
    assert KEY not in text


@pytest.fixture
def broken_call(request, contract, server, monkeypatch):
    """Make the model call fail. Returns the error kind shown for it."""
    case = request.param
    leak = f"detail {KEY}"
    if case == "http error":
        server.status = 500
        server.body = {"error": {"code": 500, "status": "INTERNAL", "message": leak}}
        return "ServerError"
    if case == "redirect":
        server.status, server.location = 307, "http://127.0.0.1:1/"
        return "APIError"
    if case == "network error":
        monkeypatch.setenv("CLASSIFIER_BASE_URL", "http://127.0.0.1:1")
        return "ConnectError"
    if case == "bad body":
        server.body = leak.encode()
        return ""
    if case == "no usage":
        server.body = contract.reply(text=VERDICT, usage=False)
        return "MissingUsage"
    if case == "invalid usage":
        server.body = contract.reply(text=VERDICT)
        server.body["usageMetadata"]["promptTokenCount"] = -5
        return "InvalidUsage"
    if case == "timeout":
        monkeypatch.setenv("CLASSIFIER_TIMEOUT_SEC", "0.2")
        server.body, server.delay = contract.reply(text=VERDICT), 1.0
        return "timed out"
    assert case == "no key"
    monkeypatch.setenv("CLASSIFIER_API_KEY", "")
    monkeypatch.setenv("CLASSIFIER_BASE_URL", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    return "API key"


FAILURES = [
    "http error", "redirect", "network error", "bad body", "no usage", "invalid usage",
    "timeout", "no key",
]


@pytest.mark.parametrize("broken_call", FAILURES, indirect=True)
async def test_failed_classifier_call_blocks_and_releases(broken_call, sandbox, caplog):
    caplog.set_level(logging.DEBUG)
    result = await run_guarded()
    assert (result.guard.decision, result.guard.cost_gbp) == (Decision.BLOCK, 0)
    assert broken_call in result.guard.reason
    assert entries("step 1/1") == [("BLOCK error", 0)]
    assert spent() == reserved() == 0 and sandbox == []
    assert_no_key(caplog, result)


@pytest.mark.parametrize("broken_call", FAILURES, indirect=True)
async def test_failed_agent_call_runs_nothing_and_releases(
    broken_call, sandbox, caplog, monkeypatch
):
    caplog.set_level(logging.DEBUG)
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error"
    assert agent_entries() == [("ERROR", 0)]
    assert spent() == reserved() == 0 and sandbox == []
    assert "agent model call" in caplog.text or "agent turn failed" in caplog.text
    assert_no_key(caplog, *events)


async def test_agent_has_its_own_timeout(contract, server, monkeypatch):
    set_role(monkeypatch, "AGENT", timeout_sec=0.2)
    server.body, server.delay = contract.reply(text="late"), 1.0
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error"
    assert agent_entries() == [("ERROR", 0)]
    assert reserved() == 0


@pytest.mark.parametrize(
    "reply",
    [
        {"text": ""},
        {"text": "{}"},
        {"text": VERDICT, "finished": False},
        {"tool_call": ("execute_python", {"code": "1"})},
    ],
)
async def test_unusable_classifier_output_blocks_and_is_charged(
    rates, server, sandbox, reply, caplog
):
    caplog.set_level(logging.DEBUG)
    if "text" in reply:
        reply = {**reply, "text": f"{reply['text']} {KEY}"}
    server.body = rates.reply(**reply)
    result = await run_guarded()
    assert_no_key(caplog, result)
    assert result.guard.decision is Decision.BLOCK
    assert result.guard.cost_gbp == spent() == pytest.approx(CLASSIFIER_COST)
    assert entries("step 1/1") == [("BLOCK error", pytest.approx(CLASSIFIER_COST))]
    assert reserved() == 0 and sandbox == []


async def test_allowed_action_is_charged_at_classifier_rates(
    rates, server, sandbox, monkeypatch
):
    server.body = rates.reply(text=VERDICT)
    tracker = budget_store.get_tracker()
    estimates = []
    reserve = tracker.reserve

    def spy(session_id, estimate):
        estimates.append(estimate)
        return reserve(session_id, estimate)

    monkeypatch.setattr(tracker, "reserve", spy)
    result = await run_guarded()
    payload = json.dumps({"tool_name": "execute_python", "args": {"code": "1"}})
    assert estimates == [pytest.approx(((610 + len(payload)) * 1 + 512 * 2) / 1_000_000 * 0.8)]
    assert result.guard.decision is Decision.ALLOW and sandbox == ["1"]
    assert result.guard.cost_gbp == spent() == pytest.approx(CLASSIFIER_COST)
    assert get_settings().classifier.model in server.requests[0].path


CODE = {"code": "print(1)"}


@pytest.mark.parametrize("finished", [True, False])
@pytest.mark.parametrize(
    "tool_call",
    [
        ("execute_python", CODE), (None, CODE), ("", CODE), ("  ", CODE),
    ],
)
async def test_agent_tool_calls(
    rates, server, sandbox, monkeypatch, caplog, tool_call, finished
):
    caplog.set_level(logging.DEBUG)
    server.body = rates.reply(tool_call=tool_call, finished=finished)
    events = await run_agent(monkeypatch)
    assert_no_key(caplog, *events)
    accepted = finished and tool_call[0] == "execute_python"
    assert sandbox == (["print(1)"] if accepted else [])
    assert events[-1].type == ("run_finished" if accepted else "run_error")
    assert any(e.type == "step_started" and e.index == 1 for e in events) is accepted
    assert agent_entries() == [("OK" if accepted else "ERROR", pytest.approx(AGENT_COST))]
    assert spent() == pytest.approx(AGENT_COST) and reserved() == 0


async def test_arguments_that_are_not_an_object_run_nothing(
    rates, server, sandbox, monkeypatch
):
    """The SDK refuses the whole reply, so its usage is unknown: released."""
    server.body = rates.reply(tool_call=("execute_python", "rm"))
    events = await run_agent(monkeypatch)
    assert events[-1].type == "run_error" and sandbox == []
    assert agent_entries() == [("ERROR", 0)]
    assert spent() == reserved() == 0


@pytest.mark.parametrize("arguments", [None, "print(1)", ["print(1)"]])
async def test_caller_rejects_arguments_that_are_not_an_object(monkeypatch, arguments):
    class Provider:
        async def agent_turn(self, *_):
            return ModelReply(USAGE, tool_call=ToolCall("execute_python", arguments), finished=True)

    monkeypatch.setattr(agent_loop, "get_provider", lambda model: Provider())
    assert await agent_loop.default_model_caller("x", get_settings()) == (USAGE, None, None)


@pytest.mark.parametrize("finished", [True, False])
@pytest.mark.parametrize("text", ["partial answer", "", " \n "])
async def test_agent_text(rates, server, sandbox, monkeypatch, caplog, text, finished):
    caplog.set_level(logging.DEBUG)
    server.body = rates.reply(text=text, finished=finished)
    events = await run_agent(monkeypatch)
    assert_no_key(caplog, *events)
    shown = bool(text.strip())
    assert [e.text for e in events if e.type == "agent_reply"] == ([text] if shown else [])
    assert events[-1].type == ("run_finished" if shown else "run_error")
    assert agent_entries() == [("OK" if shown else "ERROR", pytest.approx(AGENT_COST))]
    assert spent() == pytest.approx(AGENT_COST) and reserved() == 0 and sandbox == []


async def test_agent_is_reserved_and_charged_at_its_own_rates(rates, server, monkeypatch):
    server.body = rates.reply(text="hello")
    tracker = budget_store.get_tracker()
    estimates = []
    reserve = tracker.reserve

    def spy(session_id, estimate):
        estimates.append(estimate)
        return reserve(session_id, estimate)

    monkeypatch.setattr(tracker, "reserve", spy)
    await run_agent(monkeypatch)
    assert estimates == [pytest.approx((610 * 10 + 1000 * 20) / 1_000_000 * 0.8)]
    assert agent_entries() == [("OK", pytest.approx(AGENT_COST))]
    assert spent() == pytest.approx(AGENT_COST)
    assert "agent-model" in server.requests[0].path


@pytest.mark.parametrize("role", ["classifier", "agent"])
async def test_rates_are_captured_before_the_reservation(rates, server, monkeypatch, role):
    def change_rates():
        for prefix in ("CLASSIFIER", "AGENT"):
            os.environ[f"{prefix}_INPUT_COST_PER_1M_USD"] = "500"
            os.environ[f"{prefix}_OUTPUT_COST_PER_1M_USD"] = "500"
        os.environ["USD_TO_GBP"] = "9"

    monkeypatch.setenv("USD_TO_GBP", "0.8")  # restored after the hook changes it
    server.hook = change_rates
    if role == "classifier":
        server.body = rates.reply(text=VERDICT)
        await run_guarded()
        expected = CLASSIFIER_COST
    else:
        server.body = rates.reply(text="hello")
        await run_agent(monkeypatch)
        expected = AGENT_COST
    assert get_settings().usd_to_gbp == 9
    assert spent() == pytest.approx(expected)


async def test_setting_broken_while_running_fails_closed(contract, server, sandbox, monkeypatch):
    tracker = budget_store.get_tracker()
    monkeypatch.setenv("AGENT_TIMEOUT_SEC", "soon")
    with pytest.raises(ConfigError):
        [event async for event in agent_loop.run_agent_workflow("do it")]
    result = await guard_core.evaluate_action("execute_python", {"code": "1"}, "demo")
    assert result.decision is Decision.BLOCK
    monkeypatch.delenv("AGENT_TIMEOUT_SEC")
    assert tracker.check_budget("demo").spent_gbp == reserved() == 0
    assert server.requests == [] and sandbox == []


async def test_cap_refuses_both_calls(contract, server, sandbox, monkeypatch):
    budget_store.get_tracker().set_session_cap("demo", 0.00001)
    result = await run_guarded()
    assert (result.guard.decision, result.guard.stage) == (Decision.BLOCK, "BUDGET")
    events = await run_agent(monkeypatch)
    assert agent_entries() == [("BLOCK budget", 0)]
    assert events[-1].type == "run_finished"
    assert server.requests == [] and sandbox == [] and reserved() == 0


async def test_cancelled_classifier_call_releases(contract, server, sandbox, caplog):
    caplog.set_level(logging.DEBUG)
    server.body, server.delay = contract.reply(text=VERDICT), 0.5
    task = asyncio.create_task(run_guarded())
    while not server.requests:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert entries("step 1/1") == [("BLOCK error", 0)]
    assert spent() == reserved() == 0 and sandbox == []
    assert_no_key(caplog)


async def test_cancelled_agent_call_releases(contract, server, sandbox, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    allow_every_action(monkeypatch)
    server.body, server.delay = contract.reply(text="late"), 0.5
    events = []

    async def consume():
        async for event in agent_loop.run_agent_workflow("do it"):
            events.append(event)

    task = asyncio.create_task(consume())
    while not server.requests:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert agent_entries() == [("ERROR", 0)]
    assert spent() == reserved() == 0 and sandbox == []
    assert_no_key(caplog, *events)


def reply_at_the_deadline(monkeypatch, text, first):
    """The model answers in the same loop turn as the deadline; first says which wins."""

    async def generate_content(**_):
        loop = asyncio.get_running_loop()
        pending = loop.create_future()
        response = types.GenerateContentResponse.model_validate(FakeGemini.reply(text=text))

        def arrive():
            time.sleep(0.1)  # past the deadline, without yielding to the loop
            if first == "reply":
                pending.set_result(response)
            else:  # queued ahead of the expired timer, so the task resumes after it
                loop.call_soon(pending.set_result, response)

        loop.call_soon(arrive)
        return await pending

    models = SimpleNamespace(generate_content=generate_content)
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    monkeypatch.setattr(gemini, "_client", lambda *_: client)
    monkeypatch.setenv("CLASSIFIER_TIMEOUT_SEC", "0.05")


@pytest.mark.parametrize("first", ["reply", "deadline"])
async def test_classifier_reply_arriving_with_the_deadline(
    rates, sandbox, monkeypatch, caplog, first
):
    caplog.set_level(logging.DEBUG)
    reply_at_the_deadline(monkeypatch, VERDICT, first)
    result = await run_guarded()
    charged = first == "reply"
    cost = pytest.approx(CLASSIFIER_COST if charged else 0)
    assert (result.guard.decision is Decision.ALLOW) is charged
    assert ("timed out" in result.guard.reason) is not charged
    assert entries("step 1/1") == [("ALLOW" if charged else "BLOCK error", cost)]
    assert spent() == cost and reserved() == 0 and sandbox == (["1"] if charged else [])
    assert_no_key(caplog, result)


@pytest.mark.parametrize("first", ["reply", "deadline"])
async def test_agent_reply_arriving_with_the_deadline(
    rates, sandbox, monkeypatch, caplog, first
):
    caplog.set_level(logging.DEBUG)
    reply_at_the_deadline(monkeypatch, "hello", first)
    events = await run_agent(monkeypatch)
    charged = first == "reply"
    cost = pytest.approx(AGENT_COST if charged else 0)
    assert events[-1].type == ("run_finished" if charged else "run_error")
    assert agent_entries() == [("OK" if charged else "ERROR", cost)]
    assert spent() == cost and reserved() == 0 and sandbox == []
    assert_no_key(caplog, *events)


@pytest.mark.parametrize("role", ["classifier", "agent"])
async def test_commit_that_fails_twice_blocks_and_releases(
    contract, server, sandbox, monkeypatch, caplog, role
):
    caplog.set_level(logging.DEBUG)
    attempts = []

    def broken(*args):
        attempts.append(args)
        raise sqlite3.OperationalError(f"database is locked {KEY}")

    monkeypatch.setattr(budget_store.get_tracker(), "_commit", broken)
    server.body = contract.reply(tool_call=("execute_python", CODE))
    if role == "classifier":
        server.body = contract.reply(text=VERDICT)
        result = await run_guarded()
        assert result.guard.decision is Decision.BLOCK
        assert_no_key(caplog, result)
    else:
        events = await run_agent(monkeypatch)
        assert events[-1].type == "run_error" and agent_entries() == [("ERROR", 0)]
        assert_no_key(caplog, *events)
    assert len(attempts) == 2 and len(server.requests) == 1
    assert spent() == reserved() == 0 and sandbox == []


@pytest.mark.parametrize("role", ["classifier", "agent"])
async def test_reservation_that_cannot_be_released_expires(
    contract, server, sandbox, monkeypatch, role
):
    def broken(*_):
        raise sqlite3.OperationalError("database is locked")

    tracker = budget_store.get_tracker()
    monkeypatch.setattr(tracker, "release", broken)
    server.status, server.body = 500, {}
    if role == "classifier":
        assert (await run_guarded()).guard.decision is Decision.BLOCK
    else:
        assert (await run_agent(monkeypatch))[-1].type == "run_error"
    assert reserved() > 0 and sandbox == []
    later = time.time() + 5 * budget_store.RESERVATION_TTL_TIMEOUTS + 1
    monkeypatch.setattr(budget_store.time, "time", lambda: later)
    assert reserved() == 0


def test_reservations_outlive_the_longer_timeout(monkeypatch):
    set_role(monkeypatch, "AGENT", timeout_sec=100)
    tracker = budget_store.get_tracker()
    tracker.reserve("demo", 0.001)
    later = time.time() + 100 * budget_store.RESERVATION_TTL_TIMEOUTS - 5
    monkeypatch.setattr(budget_store.time, "time", lambda: later)
    assert reserved() > 0
