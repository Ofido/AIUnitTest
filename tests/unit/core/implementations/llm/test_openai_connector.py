"""Test cases for the OpenAI LLM connector."""

import importlib
import sys
import time
import types

import pytest


@pytest.mark.asyncio
async def test_openai_missing_and_rate_limiter_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAI missing and rate limiter sleep behavior."""
    mod_name = "ai_unit_test.core.implementations.llm.openai_connector"
    # Ensure fresh import
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    # Insert a dummy 'openai' module that does NOT provide AsyncOpenAI/ChatCompletion to trigger ImportError handling
    dummy_openai = types.ModuleType("openai")
    if "openai" in sys.modules:
        orig_openai = sys.modules["openai"]
        removed = False
    else:
        orig_openai = None
        removed = True
    sys.modules["openai"] = dummy_openai
    try:
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        # After reload with a broken openai module, OPENAI_AVAILABLE should be False and AsyncOpenAI/ChatCompletion None
        assert getattr(mod, "OPENAI_AVAILABLE") is False  # noqa
        assert getattr(mod, "AsyncOpenAI") is None  # noqa
        assert getattr(mod, "ChatCompletion") is None  # noqa
    finally:
        # Restore original openai in sys.modules
        if orig_openai is not None:
            sys.modules["openai"] = orig_openai
        elif removed:
            del sys.modules["openai"]
        # Ensure module is reloaded from actual code for subsequent checks
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    # Import the real module now
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    RateLimiter = mod.RateLimiter
    # Case 1: when under limit, no sleep is awaited
    rl_ok = RateLimiter(requests_per_minute=10)
    sleep_called = []

    async def fake_sleep_ok(seconds: float) -> None:
        sleep_called.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep_ok)
    rl_ok.requests = []
    await rl_ok.acquire()
    assert sleep_called == []
    assert len(rl_ok.requests) == 1
    # Case 2: when at limit, sleep should be awaited with computed delay and request appended
    rl = RateLimiter(requests_per_minute=1)
    rl.requests = [time.time() - 59.0]
    slept = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    await rl.acquire()
    assert slept, "Expected asyncio.sleep to be called when rate limit exceeded"
    assert pytest.approx(slept[0], rel=0.1, abs=0.5) == 1.1
    assert len(rl.requests) == 2


@pytest.mark.asyncio
async def test_rate_limiter_removes_old_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test RateLimiter removes requests older than 60 seconds."""
    mod = importlib.import_module("ai_unit_test.core.implementations.llm.openai_connector")
    RateLimiter = mod.RateLimiter
    rl = RateLimiter(requests_per_minute=2)
    now = time.time()
    rl.requests = [now - 61, now - 30]
    sleep_called = []

    async def fake_sleep(seconds: float) -> None:
        sleep_called.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    await rl.acquire()
    # The old request should be removed, only one remains, so no sleep
    assert sleep_called == []
    assert len(rl.requests) == 2


def test_openai_connector_config_defaults() -> None:
    """Test OpenAIConnectorConfig default values."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnectorConfig

    cfg = OpenAIConnectorConfig()
    assert cfg.api_key is None
    assert cfg.timeout == 30
    assert cfg.rate_limit == 60
    assert cfg.models_cache_ttl == 300.0
    assert cfg.max_retries == 3
    assert cfg.retry_delay == 1.0
    assert isinstance(cfg.extra, dict)


def test_get_available_models_cache_and_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_available_models returns cache or fallback."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector, OpenAIConnectorConfig

    cfg = OpenAIConnectorConfig(api_key="dummy")
    # Patch OPENAI_AVAILABLE to True to avoid ConfigurationError
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    connector = OpenAIConnector(cfg)
    # No cache, should return fallback
    fallback = connector.get_available_models()
    assert "gpt-4o" in fallback
    # Set cache, should return cache
    connector._available_models_cache = ["model-x", "model-y"]
    assert connector.get_available_models() == ["model-x", "model-y"]


@pytest.mark.asyncio
async def test_refresh_available_models_force(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _refresh_available_models with force and API error fallback."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector, OpenAIConnectorConfig

    cfg = OpenAIConnectorConfig(api_key="dummy")
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    connector = OpenAIConnector(cfg)

    class DummyClient:
        class models:
            @staticmethod
            async def list() -> None:
                raise Exception("API error")

    connector.client = DummyClient()  # type: ignore
    models = await connector._refresh_available_models(force=True)
    assert "gpt-4o" in models


def test_connector_info(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_connector_info returns expected dict."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector, OpenAIConnectorConfig

    cfg = OpenAIConnectorConfig(api_key="dummy", model="gpt-4o", rate_limit=42)
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    connector = OpenAIConnector(cfg)
    info = connector.get_connector_info()
    assert info["provider"] == "openai"
    assert info["rate_limit"] == 42
    assert info["default_model"] == "gpt-4o"
    assert info["supports_streaming"] is True


@pytest.mark.asyncio
async def test_select_model_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test select_model returns priority model if available."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector, OpenAIConnectorConfig

    cfg = OpenAIConnectorConfig(api_key="dummy")
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    connector = OpenAIConnector(cfg)
    connector._available_models_cache = ["gpt-4o", "other-model"]
    connector._models_cache_ts = time.time()
    connector._models_cache_ttl = 300
    # Should pick gpt-4o for chat
    model = await connector.select_model("chat")
    assert model == "gpt-4o"
    # Should pick gpt-4o for streaming
    model = await connector.select_model("streaming")
    assert model == "gpt-4o"
    # Should pick fallback for embeddings
    connector._available_models_cache = ["text-embedding-3-large", "other-model"]
    model = await connector.select_model("embeddings")
    assert model == "text-embedding-3-large"
    # Should pick config model if set
    connector.config.model = "custom-model"
    model = await connector.select_model("chat")
    assert model == "custom-model"


def test_create_config_from_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test _create_config_from_dict parses config dict."""
    from ai_unit_test.core.implementations.llm.openai_connector import OpenAIConnector  # noqa

    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    config_dict = {
        "api_key": "key",
        "timeout": 10,
        "rate_limit": 5,
        "model": "gpt-4o",
        "extra1": "val1",
        "extra2": "val2",
    }
    connector = OpenAIConnector(config_dict)
    cfg = connector.config
    assert cfg.api_key == "key"
    assert cfg.timeout == 10
    assert cfg.rate_limit == 5
    assert cfg.model == "gpt-4o"
    assert cfg.extra["extra1"] == "val1"
    assert cfg.extra["extra2"] == "val2"


def test_connector_init_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test OpenAIConnector raises if api_key missing or openai unavailable."""
    from ai_unit_test.core.implementations.llm.openai_connector import (  # type: ignore
        ConfigurationError,
        OpenAIConnector,
        OpenAIConnectorConfig,
    )

    # Patch OPENAI_AVAILABLE to True
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", True)
    with pytest.raises(ConfigurationError):
        OpenAIConnector(OpenAIConnectorConfig(api_key=None))
    # Patch OPENAI_AVAILABLE to False
    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.OPENAI_AVAILABLE", False)
    with pytest.raises(ConfigurationError):
        OpenAIConnector(OpenAIConnectorConfig(api_key="dummy"))


def test_rate_limiter_init_and_acquire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test RateLimiter initialization and acquire logic."""
    from ai_unit_test.core.implementations.llm.openai_connector import RateLimiter

    rl = RateLimiter(requests_per_minute=2)
    assert rl.requests_per_minute == 2
    assert rl.requests == []
    # Simulate two requests, then a third triggers sleep
    now = time.time()
    rl.requests = [now - 10, now - 5]
    slept = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("ai_unit_test.core.implementations.llm.openai_connector.asyncio.sleep", fake_sleep)
    # Should sleep since limit reached
    import asyncio

    asyncio.run(rl.acquire())
    assert slept, "Should have slept due to rate limit"
