"""Settings read from the environment and the project .env file.

Every estimate (rates, caps, timeouts, reserve sizes) comes from here.
"""

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigError(ValueError):
    """A setting is missing or invalid."""


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str = field(repr=False)
    gemini_model: str
    input_cost_per_1m_usd: float
    output_cost_per_1m_usd: float
    usd_to_gbp: float
    session_budget_cap_gbp: float
    global_budget_cap_gbp: float
    sandbox_provider: str
    sandbox_docker_image: str
    cloudrun_sandbox_url: str
    sqlite_db_path: Path
    classifier_timeout_sec: float
    budget_reserve_input_tokens: int
    budget_reserve_output_tokens: int
    preset3_session_cap_gbp: float
    preset3_max_steps: int


def _text(name: str, default: str) -> str:
    return os.environ.get(name, "").strip() or default


def _number(name: str, default: float, cast: type = float):
    raw = _text(name, str(default))
    try:
        value = cast(raw)
    except ValueError:
        raise ConfigError(f"{name} must be a number") from None
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} must be a finite number, zero or more")
    return value


def get_settings() -> Settings:
    """Return current settings. Real environment variables override .env."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    db_path = Path(_text("SQLITE_DB_PATH", "data/guard_state.db"))
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return Settings(
        gemini_api_key=_text("GEMINI_API_KEY", ""),
        gemini_model=_text("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        input_cost_per_1m_usd=_number("INPUT_COST_PER_1M_USD", 0.30),
        output_cost_per_1m_usd=_number("OUTPUT_COST_PER_1M_USD", 2.50),
        usd_to_gbp=_number("USD_TO_GBP", 0.80),
        session_budget_cap_gbp=_number("SESSION_BUDGET_CAP_GBP", 0.05),
        global_budget_cap_gbp=_number("GLOBAL_BUDGET_CAP_GBP", 0.50),
        sandbox_provider=_text("SANDBOX_PROVIDER", "docker"),
        sandbox_docker_image=_text("SANDBOX_DOCKER_IMAGE", "agentguard-sandbox:latest"),
        cloudrun_sandbox_url=_text("CLOUDRUN_SANDBOX_URL", ""),
        sqlite_db_path=db_path,
        classifier_timeout_sec=_number("CLASSIFIER_TIMEOUT_SEC", 5.0),
        budget_reserve_input_tokens=_number("BUDGET_RESERVE_INPUT_TOKENS", 610, int),
        budget_reserve_output_tokens=_number("BUDGET_RESERVE_OUTPUT_TOKENS", 512, int),
        preset3_session_cap_gbp=_number("PRESET3_SESSION_CAP_GBP", 0.002),
        preset3_max_steps=_number("PRESET3_MAX_STEPS", 6, int),
    )


def token_cost_gbp(settings: Settings, input_tokens: int, output_tokens: int) -> float:
    """Cost of a model call. output_tokens includes thought tokens."""
    usd = (
        input_tokens * settings.input_cost_per_1m_usd
        + output_tokens * settings.output_cost_per_1m_usd
    ) / 1_000_000
    return usd * settings.usd_to_gbp
