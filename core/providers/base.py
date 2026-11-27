"""The interface every model provider implements.

A provider makes one request per call, with no retries. The caller sets the time
limit and settles the budget reservation.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from core.config import ModelSettings
from core.schemas import TokenUsage


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema of an object


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ModelReply:
    """A billed reply. Output that cannot be used has no text and no tool call.

    usage.candidates_tokens excludes reasoning tokens, which go in
    usage.thoughts_tokens: output is charged as the sum of the two.
    """

    usage: TokenUsage
    text: str | None = None
    tool_call: ToolCall | None = None
    finished: bool = False  # the model stopped normally


class ModelError(Exception):
    """A call that gave no billable reply.

    Holds only an error type, a numeric code and a status name, so it is safe to
    log: never a message, which could echo the request or the key.
    """

    def __init__(self, kind: str, code: object = None, status: object = None) -> None:
        self.kind = kind if re.fullmatch(r"\w{1,64}", kind) else "Error"
        parts = [self.kind]
        if code is not None or status is not None:
            safe_code = code if isinstance(code, int) else "unknown"
            if not (isinstance(status, str) and re.fullmatch(r"[A-Za-z_ ]{1,64}", status)):
                status = "unknown"
            parts += [f"code={safe_code}", f"status={status}"]
        super().__init__(" ".join(parts))


class ModelProvider(ABC):
    """Both operations raise ModelError, or TimeoutError, and let cancellation through.

    A reply without token usage is a ModelError. Once usage is known, return a
    ModelReply even if the output is unusable, so that the call is charged.
    """

    def __init__(self, model: ModelSettings) -> None:
        self.model = model

    @abstractmethod
    async def classify(self, system: str, payload: str, schema: type[BaseModel]) -> ModelReply:
        """Ask for JSON matching schema, in text. The caller validates it.

        A provider without a JSON Schema mode may use a JSON object mode and
        describe the schema in its instructions.
        """

    @abstractmethod
    async def agent_turn(self, system: str, prompt: str, tools: list[ToolSpec]) -> ModelReply:
        """One turn: a tool call (the first, if several), or else text. Nothing is run."""
