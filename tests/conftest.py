import os
import pytest
import core.config

# Pin default PRESET3_SESSION_CAP_GBP of 0.0017 without reading any .env or ambient environment
core.config.dotenv_values = lambda *args, **kwargs: {}
os.environ.pop("PRESET3_SESSION_CAP_GBP", None)


@pytest.fixture(autouse=True)
def pin_default_preset3_cap(monkeypatch):
    monkeypatch.delenv("PRESET3_SESSION_CAP_GBP", raising=False)
    monkeypatch.setattr(core.config, "dotenv_values", lambda *args, **kwargs: {})
