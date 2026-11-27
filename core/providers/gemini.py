"""Gemini through google-genai. No other product module imports the SDK."""

import asyncio
import os
import ssl
from typing import Any
from urllib.parse import urlsplit

import httpx2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from core.providers.base import ModelError, ModelProvider, ModelReply, ToolCall, ToolSpec
from core.schemas import TokenUsage

MAX_CLIENTS = 4
_SCHEMA_KEYWORDS = {"type", "description", "properties", "required"}

_clients: dict[tuple[str, str, asyncio.AbstractEventLoop], genai.Client] = {}


def _ssl_context() -> ssl.SSLContext:
    """What the SDK verifies with by default: SSL_CERT_FILE or SSL_CERT_DIR if set,
    otherwise the operating system's trust store.
    """
    cafile, capath = os.environ.get("SSL_CERT_FILE"), os.environ.get("SSL_CERT_DIR")
    if cafile or capath:
        return ssl.create_default_context(cafile=cafile, capath=capath)
    return httpx2.create_ssl_context()


def _new_client(api_key: str, base_url: str) -> genai.Client:
    # Our own clients, of the SDK's default library: the SDK's would follow
    # redirects, carrying the key header to another server. No time limit here,
    # because the caller sets one. Plain http is only allowed to this machine, so
    # no proxy variable may affect it.
    local = urlsplit(base_url).scheme == "http"
    options: dict[str, Any] = {
        "httpx_async_client": httpx2.AsyncClient(
            follow_redirects=False,
            timeout=None,
            trust_env=not local,
            verify=True if local else _ssl_context(),
        ),
        # Not used for model calls. Passed so that the SDK builds none of its own.
        "httpx_client": httpx2.Client(follow_redirects=False, timeout=None, trust_env=False),
    }
    if base_url:
        options["base_url"] = base_url
    # vertexai=False: the environment must not send the key to another service.
    return genai.Client(api_key=api_key, vertexai=False, http_options=types.HttpOptions(**options))


def _client(api_key: str, base_url: str) -> genai.Client:
    """One client per key, URL and event loop, so connections are reused. No retries.

    A dropped client is not closed: a call may still be using it.
    """
    # The variable the SDK would read itself; config has checked it.
    url = base_url or os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
    key = (api_key, url, asyncio.get_running_loop())
    client = _clients.get(key)
    if client is None:
        client = _new_client(api_key, url)
        for old in [old for old in _clients if old[2].is_closed()]:
            del _clients[old]
        while len(_clients) >= MAX_CLIENTS:
            del _clients[next(iter(_clients))]
        _clients[key] = client
    return client


def _usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None)
    if usage is None or usage.prompt_token_count is None:
        raise ModelError("MissingUsage")
    try:
        # Gemini's candidate count already excludes thought tokens.
        return TokenUsage(
            prompt_tokens=usage.prompt_token_count,
            candidates_tokens=usage.candidates_token_count or 0,
            thoughts_tokens=usage.thoughts_token_count or 0,
        )
    except ValidationError:
        raise ModelError("InvalidUsage") from None


def _finished(response: Any) -> bool:
    candidates = response.candidates
    return bool(candidates) and candidates[0].finish_reason == types.FinishReason.STOP


def _schema(schema: dict[str, Any]) -> types.Schema:
    """The subset of JSON Schema that tool parameters use. Anything else is refused."""
    if set(schema) - _SCHEMA_KEYWORDS:
        raise ValueError("unsupported keyword in a tool schema")
    properties = schema.get("properties")
    return types.Schema(
        type=types.Type(schema["type"].upper()),
        description=schema.get("description"),
        properties=None
        if properties is None
        else {name: _schema(value) for name, value in properties.items()},
        required=schema.get("required"),
    )


def _error(exc: Exception) -> ModelError:
    if isinstance(exc, errors.APIError):
        return ModelError(type(exc).__name__, exc.code, exc.status)
    return ModelError(type(exc).__name__)


class GeminiProvider(ModelProvider):
    async def classify(self, system: str, payload: str, schema: type[BaseModel]) -> ModelReply:
        response, usage = await self._generate(
            payload,
            lambda: types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
                # Includes thought tokens, so output cannot cost more than reserved.
                max_output_tokens=self.model.max_output_tokens,
            ),
        )
        try:
            return ModelReply(usage, text=response.text or "", finished=_finished(response))
        except Exception:
            return ModelReply(usage)

    async def agent_turn(self, system: str, prompt: str, tools: list[ToolSpec]) -> ModelReply:
        def config() -> types.GenerateContentConfig:
            declarations = [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=_schema(tool.parameters),
                )
                for tool in tools
            ]
            return types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.0,
                max_output_tokens=self.model.max_output_tokens,
                system_instruction=system,
            )

        response, usage = await self._generate(prompt, config)
        try:
            finished = _finished(response)
            if response.function_calls:
                call = response.function_calls[0]
                tool_call = ToolCall(call.name, dict(call.args or {}))
                return ModelReply(usage, tool_call=tool_call, finished=finished)
            return ModelReply(usage, text=response.text or "", finished=finished)
        except Exception:
            return ModelReply(usage)

    async def _generate(self, contents: str, config: Any) -> tuple[Any, TokenUsage]:
        """One request. Nothing is awaited after the response arrives."""
        # Without this the SDK would look for a key of its own in the environment.
        if not self.model.api_key:
            raise ModelError("MissingApiKey")
        try:
            client = _client(self.model.api_key, self.model.base_url)
            response = await client.aio.models.generate_content(
                model=self.model.model, contents=contents, config=config()
            )
            return response, _usage(response)
        except (TimeoutError, ModelError):
            raise
        except Exception as exc:
            raise _error(exc) from None
