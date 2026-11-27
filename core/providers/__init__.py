"""Model providers, chosen by the provider setting of a role."""

from core.config import ModelSettings
from core.providers.base import ModelError, ModelProvider, ModelReply, ToolCall, ToolSpec
from core.providers.gemini import GeminiProvider

# Keys must match core.config.PROVIDER_NAMES.
PROVIDERS: dict[str, type[ModelProvider]] = {"gemini": GeminiProvider}

__all__ = [
    "PROVIDERS",
    "ModelError",
    "ModelProvider",
    "ModelReply",
    "ToolCall",
    "ToolSpec",
    "get_provider",
]


def get_provider(model: ModelSettings) -> ModelProvider:
    return PROVIDERS[model.provider](model)
