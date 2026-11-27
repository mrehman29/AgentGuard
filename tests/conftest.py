import os
import pytest
import core.config

# Pin default PRESET3_SESSION_CAP_GBP of 0.0017 without reading any .env or ambient environment
core.config.dotenv_values = lambda *args, **kwargs: {}
os.environ.pop("PRESET3_SESSION_CAP_GBP", None)

# Role settings from the shell would change which model and rates the tests use.
ROLE_SETTINGS = tuple(
    f"{role}_{name}"
    for role in ("CLASSIFIER", "AGENT")
    for name in (
        "PROVIDER", "BASE_URL", "MODEL", "API_KEY", "INPUT_COST_PER_1M_USD",
        "OUTPUT_COST_PER_1M_USD", "TIMEOUT_SEC", "MAX_OUTPUT_TOKENS",
    )
)


@pytest.fixture(autouse=True)
def pin_default_preset3_cap(monkeypatch):
    monkeypatch.delenv("PRESET3_SESSION_CAP_GBP", raising=False)
    for name in ROLE_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(core.config, "dotenv_values", lambda *args, **kwargs: {})
