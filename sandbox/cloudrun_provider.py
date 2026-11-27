"""Cloud Run provider (PLAN.md B.3): calls the worker with a Google identity token."""

import asyncio
import base64
import json
import shutil
import time
from contextlib import suppress
from pathlib import Path

import httpx

from core.config import get_settings
from core.schemas import MAX_OUTPUT_CHARS, SandboxRequest, SandboxResult
from sandbox.base import SandboxProvider

UNAVAILABLE_EXIT_CODE = 125
REQUEST_GRACE_SEC = 10  # allowed on top of timeout_sec for network and cold start
HEALTH_TIMEOUT_SEC = 5
TOKEN_TIMEOUT_SEC = 10
TOKEN_EXPIRY_MARGIN_SEC = 60
TOKEN_FALLBACK_LIFETIME_SEC = 300  # when the expiry cannot be read from the token
# Both output streams at their cap, fully JSON-escaped, plus the other fields.
MAX_RESPONSE_BYTES = 2 * 6 * MAX_OUTPUT_CHARS + 4096
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

_cached_token: tuple[str, float] | None = None  # (token, use until, epoch seconds)


class _Unavailable(Exception):
    """The message is shown to the user: never put a token or response body in it."""


def _expiry(token: str) -> float:
    try:
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(claims["exp"]) - TOKEN_EXPIRY_MARGIN_SEC
    except (IndexError, KeyError, TypeError, ValueError):
        return time.time() + TOKEN_FALLBACK_LIFETIME_SEC


async def _gcloud_identity_token() -> str:
    gcloud = shutil.which("gcloud") or str(Path.home() / "google-cloud-sdk/bin/gcloud")
    try:
        process = await asyncio.create_subprocess_exec(
            gcloud, "auth", "print-identity-token",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )  # fmt: skip
    except OSError:
        raise _Unavailable("gcloud was not found.") from None
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), TOKEN_TIMEOUT_SEC)
    except TimeoutError:
        raise _Unavailable("gcloud did not return an identity token in time.") from None
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
    token = stdout.decode("ascii", errors="replace").strip()
    if process.returncode != 0 or not token or not token.isascii() or len(token.split()) != 1:
        raise _Unavailable("no identity token; run 'gcloud auth login'.")
    return token


async def _identity_token() -> str:
    global _cached_token
    if _cached_token is None or time.time() >= _cached_token[1]:
        token = await _gcloud_identity_token()
        _cached_token = (token, _expiry(token))
    return _cached_token[0]


def _forget_token() -> None:
    global _cached_token
    _cached_token = None


def _base_url() -> httpx.URL:
    raw = get_settings().cloudrun_sandbox_url.rstrip("/")
    if not raw:
        raise _Unavailable("CLOUDRUN_SANDBOX_URL is not set.")
    try:
        url = httpx.URL(raw)
    except httpx.InvalidURL:
        raise _Unavailable("CLOUDRUN_SANDBOX_URL is not a valid URL.") from None
    local = url.scheme == "http" and url.host in LOOPBACK_HOSTS
    if not url.host or not (url.scheme == "https" or local):
        # The token must not cross a network in clear text.
        raise _Unavailable("CLOUDRUN_SANDBOX_URL must use https.")
    return url


async def _call(method: str, path: str, body: dict | None = None) -> bytes:
    """Authenticated request to the worker. Returns the body of a 200 response."""
    url = _base_url()
    target = url.copy_with(path=url.path.rstrip("/") + path)
    async with httpx.AsyncClient(timeout=None, trust_env=url.scheme == "https") as client:
        for attempt in (1, 2):
            token = await _identity_token()
            headers = {"Authorization": f"Bearer {token}", "Accept-Encoding": "identity"}
            async with client.stream(method, target, json=body, headers=headers) as response:
                if response.status_code in (401, 403):
                    # Refused before anything ran, so one retry with a new token is safe.
                    _forget_token()
                    if attempt == 1:
                        continue
                if response.status_code != 200:
                    raise _Unavailable(f"the service answered HTTP {response.status_code}.")
                data = bytearray()
                async for chunk in response.aiter_raw():
                    data += chunk
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise _Unavailable("the response was too large.")
                return bytes(data)


class CloudRunSandboxProvider(SandboxProvider):
    provider_name = "cloudrun"

    async def execute(self, request: SandboxRequest) -> SandboxResult:
        start = time.monotonic()
        try:
            # The deadline covers the whole exchange, including a slow response.
            async with asyncio.timeout(
                TOKEN_TIMEOUT_SEC + request.timeout_sec + REQUEST_GRACE_SEC
            ):
                data = await _call("POST", "/execute", request.model_dump())
            result = SandboxResult.model_validate_json(data)
        except _Unavailable as exc:
            return self._unavailable(str(exc), start)
        except TimeoutError:
            return self._unavailable("the request timed out.", start)
        except (httpx.HTTPError, ValueError) as exc:
            # Includes an invalid response. The type name alone leaks nothing.
            return self._unavailable(f"request failed ({type(exc).__name__}).", start)
        return result.model_copy(update={"provider": self.provider_name})

    async def health_check(self) -> bool:
        try:
            async with asyncio.timeout(HEALTH_TIMEOUT_SEC):
                await _call("GET", "/health")
        except (_Unavailable, TimeoutError, httpx.HTTPError, ValueError):
            return False
        return True

    def _unavailable(self, detail: str, start: float) -> SandboxResult:
        return SandboxResult(
            provider=self.provider_name,
            stdout="",
            stderr=f"Cloud Run sandbox unavailable: {detail}"[:MAX_OUTPUT_CHARS],
            truncated=False,
            exit_code=UNAVAILABLE_EXIT_CODE,
            duration_ms=int((time.monotonic() - start) * 1000),
            timed_out=False,
        )
