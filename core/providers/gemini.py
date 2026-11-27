"""Gemini through google-genai. No other product module imports the SDK."""

import asyncio
from typing import Any

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

from core.providers.base import ModelError, ModelProvider, ModelReply, ToolCall, ToolSpec
from core.schemas import TokenUsage

_client_cache: tuple[tuple[str, str, asyncio.AbstractEventLoop], genai.Client] | None = None


def _client_options(base_url: str) -> dict[str, Any]:
    # Without a base URL the SDK chooses, honouring GOOGLE_GEMINI_BASE_URL.
    return {"http_options": types.HttpOptions(base_url=base_url)} if base_url else {}


def _make_client(api_key: str, base_url: str) -> genai.Client:
    """One client per key, URL and event loop, so connections are reused. No retries."""
    global _client_cache
    key = (api_key, base_url, asyncio.get_running_loop())
    if _client_cache is None or _client_cache[0] != key:
        client = genai.Client(api_key=api_key, vertexai=False, **_client_options(base_url))
        _client_cache = (key, client)
    return _client_cache[1]


def _usage(response: Any) -> TokenUsage:
    usage = response.usage_metadata
    if usage is None or usage.prompt_token_count is None:
        raise ValueError("the response has no token usage")
    # Gemini's candidate count already excludes thought tokens.
    return TokenUsage(
        prompt_tokens=usage.prompt_token_count,
        candidates_tokens=usage.candidates_token_count or 0,
        thoughts_tokens=usage.thoughts_token_count or 0,
    )


def _finished(response: Any) -> bool:
    candidates = response.candidates
    return bool(candidates) and candidates[0].finish_reason == types.FinishReason.STOP


def _schema(schema: dict[str, Any]) -> types.Schema:
    """The subset of JSON Schema that tool parameters use."""
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
    def _check_key(self) -> None:
        # Without this the SDK would look for a key of its own in the environment.
        if not self.model.api_key:
            raise ModelError("MissingApiKey")

    async def classify(self, system: str, payload: str, schema: type[BaseModel]) -> ModelReply:
        self._check_key()
        try:
            client = _make_client(self.model.api_key, self.model.base_url)
            response = await client.aio.models.generate_content(
                model=self.model.model,
                contents=payload,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.MINIMAL
                    ),
                    # Includes thought tokens, so output cannot cost more than reserved.
                    max_output_tokens=self.model.max_output_tokens,
                ),
            )
            usage = _usage(response)
        except TimeoutError:
            raise
        except Exception as exc:
            raise _error(exc) from None
        try:
            return ModelReply(usage, text=response.text or "", finished=_finished(response))
        except Exception:
            return ModelReply(usage)

    async def agent_turn(self, system: str, prompt: str, tools: list[ToolSpec]) -> ModelReply:
        self._check_key()
        try:
            declarations = [
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=_schema(tool.parameters),
                )
                for tool in tools
            ]
            client = genai.Client(
                api_key=self.model.api_key, **_client_options(self.model.base_url)
            )
            config = types.GenerateContentConfig(
                tools=[types.Tool(function_declarations=declarations)],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                temperature=0.0,
                max_output_tokens=self.model.max_output_tokens,
                system_instruction=system,
            )
            async with client.aio as aio_client:
                response = await aio_client.models.generate_content(
                    model=self.model.model, contents=prompt, config=config
                )
            usage = _usage(response)
        except TimeoutError:
            raise
        except Exception as exc:
            raise _error(exc) from None
        try:
            finished = _finished(response)
            if response.function_calls:
                call = response.function_calls[0]
                tool_call = ToolCall(call.name, dict(call.args or {}))
                return ModelReply(usage, tool_call=tool_call, finished=finished)
            return ModelReply(usage, text=response.text or "", finished=finished)
        except Exception:
            return ModelReply(usage)
