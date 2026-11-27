"""Settings read from the environment and the project .env file.

Every estimate (rates, caps, timeouts, reserve sizes) comes from here.
"""

import ipaddress
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

from core.schemas import ModelRates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVIDER_NAMES = ("gemini",)


class ConfigError(ValueError):
    """A setting is missing or invalid."""


@dataclass(frozen=True)
class ModelSettings:
    """The model of one role. Rates are estimates in US dollars."""

    provider: str
    base_url: str  # empty: the provider's default
    model: str
    api_key: str = field(repr=False)
    input_cost_per_1m_usd: float
    output_cost_per_1m_usd: float
    timeout_sec: float
    max_output_tokens: int  # also the output size reserved before a call


@dataclass(frozen=True)
class Settings:
    classifier: ModelSettings
    agent: ModelSettings
    usd_to_gbp: float
    session_budget_cap_gbp: float
    global_budget_cap_gbp: float
    sandbox_provider: str
    sandbox_docker_image: str
    cloudrun_sandbox_url: str
    sqlite_db_path: Path
    budget_reserve_input_tokens: int
    preset3_session_cap_gbp: float
    preset3_max_steps: int

    def rates_for(self, model: ModelSettings) -> ModelRates:
        """What prices a call to model. Capture it once, before reserving."""
        return ModelRates(
            provider=model.provider,
            model=model.model,
            input_cost_per_1m_usd=model.input_cost_per_1m_usd,
            output_cost_per_1m_usd=model.output_cost_per_1m_usd,
            usd_to_gbp=self.usd_to_gbp,
        )


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_url(name: str, url: str) -> None:
    """A model API address: a key is sent to it."""
    try:
        parts = urlsplit(url)
        parts.port  # raises ValueError if it is not a port number
        valid = (
            bool(parts.hostname)
            and (parts.scheme == "https" or (parts.scheme == "http" and _is_loopback(parts.hostname)))
            and not any(char in parts.netloc for char in "@\\")
            and not parts.query
            and not parts.fragment
            and all("!" <= char <= "~" for char in url)
        )
    except ValueError:
        valid = False
    # The message must not include the value: it could hold a credential.
    if not valid:
        raise ConfigError(
            f"{name} must be an https URL (http only for a loopback host)"
            " without credentials, query or fragment"
        )


def get_settings() -> Settings:
    """Return current settings. Real environment variables override .env.

    .env values are not written to os.environ, so child processes do not
    inherit the API key.
    """
    values = {k: v for k, v in dotenv_values(PROJECT_ROOT / ".env").items() if v is not None}
    values.update(os.environ)

    def _text(name: str, default: str) -> str:
        return values.get(name, "").strip() or default

    def _number(name: str, default: float, cast: type = float, allow_zero: bool = False):
        raw = _text(name, str(default))
        try:
            value = cast(raw)
        except ValueError:
            raise ConfigError(f"{name} must be a number") from None
        # A zero rate or reserve would silently disable the cost breaker.
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ConfigError(f"{name} must be a finite number above zero")
        return value

    def _api_key(name: str) -> str:
        key = _text(name, "")
        # It is sent in an HTTP header. The message must not include it.
        if not all("!" <= char <= "~" for char in key):
            raise ConfigError(f"{name} must be printable ASCII without spaces")
        return key

    def _model(role: str, fallback: ModelSettings) -> ModelSettings:
        """The role's own settings, each falling back to the fallback's."""
        provider = _text(f"{role}_PROVIDER", fallback.provider).lower()
        if provider not in PROVIDER_NAMES:
            raise ConfigError(f"{role}_PROVIDER must be one of: {', '.join(PROVIDER_NAMES)}")
        same_provider = provider == fallback.provider
        own_url = _text(f"{role}_BASE_URL", "")
        if own_url:
            _check_url(f"{role}_BASE_URL", own_url)
        base_url = own_url or (fallback.base_url if same_provider else "")
        model = _text(f"{role}_MODEL", fallback.model if same_provider else "")
        if not model:
            raise ConfigError(f"{role}_MODEL must be set for this provider")
        api_key = _api_key(f"{role}_API_KEY")
        if not api_key:
            # A key is never sent to an endpoint it was not set for.
            if not same_provider or base_url != fallback.base_url:
                raise ConfigError(f"{role}_API_KEY must be set for a different provider or URL")
            api_key = fallback.api_key
        # Rates set for another model or endpoint could under-charge this one.
        if (provider, base_url, model) != (fallback.provider, fallback.base_url, fallback.model):
            for name in ("INPUT_COST_PER_1M_USD", "OUTPUT_COST_PER_1M_USD"):
                if not _text(f"{role}_{name}", ""):
                    raise ConfigError(
                        f"{role}_{name} must be set for a different model, provider or URL"
                    )
        # With no URL of its own, the Gemini SDK would use this process variable.
        sdk_url = os.environ.get("GOOGLE_GEMINI_BASE_URL", "")
        if provider == "gemini" and not base_url and sdk_url:
            _check_url("GOOGLE_GEMINI_BASE_URL", sdk_url)
        return ModelSettings(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            input_cost_per_1m_usd=_number(
                f"{role}_INPUT_COST_PER_1M_USD", fallback.input_cost_per_1m_usd
            ),
            output_cost_per_1m_usd=_number(
                f"{role}_OUTPUT_COST_PER_1M_USD", fallback.output_cost_per_1m_usd
            ),
            timeout_sec=_number(f"{role}_TIMEOUT_SEC", fallback.timeout_sec),
            max_output_tokens=_number(
                f"{role}_MAX_OUTPUT_TOKENS", fallback.max_output_tokens, int
            ),
        )

    gemini = ModelSettings(
        provider="gemini",
        base_url="",
        model=_text("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        api_key=_api_key("GEMINI_API_KEY"),
        input_cost_per_1m_usd=_number("INPUT_COST_PER_1M_USD", 0.30),
        output_cost_per_1m_usd=_number("OUTPUT_COST_PER_1M_USD", 2.50),
        timeout_sec=5.0,
        max_output_tokens=_number("BUDGET_RESERVE_OUTPUT_TOKENS", 512, int),
    )
    classifier = _model("CLASSIFIER", gemini)

    db_path = Path(_text("SQLITE_DB_PATH", "data/guard_state.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        classifier=classifier,
        agent=_model("AGENT", classifier),
        usd_to_gbp=_number("USD_TO_GBP", 0.80),
        session_budget_cap_gbp=_number("SESSION_BUDGET_CAP_GBP", 0.05, allow_zero=True),
        global_budget_cap_gbp=_number("GLOBAL_BUDGET_CAP_GBP", 0.50, allow_zero=True),
        sandbox_provider=_text("SANDBOX_PROVIDER", "docker"),
        sandbox_docker_image=_text("SANDBOX_DOCKER_IMAGE", "agentguard-sandbox:latest"),
        cloudrun_sandbox_url=_text("CLOUDRUN_SANDBOX_URL", ""),
        sqlite_db_path=db_path,
        budget_reserve_input_tokens=_number("BUDGET_RESERVE_INPUT_TOKENS", 610, int),
        preset3_session_cap_gbp=_number("PRESET3_SESSION_CAP_GBP", 0.0017, allow_zero=True),
        preset3_max_steps=_number("PRESET3_MAX_STEPS", 6, int),
    )


def token_cost_gbp(rates: ModelRates, input_tokens: int, output_tokens: int) -> float:
    """Cost of a model call. output_tokens includes thought tokens."""
    usd = (
        input_tokens * rates.input_cost_per_1m_usd + output_tokens * rates.output_cost_per_1m_usd
    ) / 1_000_000
    return usd * rates.usd_to_gbp
