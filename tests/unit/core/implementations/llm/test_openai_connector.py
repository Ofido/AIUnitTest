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
        assert getattr(mod, "OPENAI_AVAILABLE") is False  # noqa B009
        assert getattr(mod, "AsyncOpenAI") is None  # noqa B009
        assert getattr(mod, "ChatCompletion") is None  # noqa B009
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
        """Fake sleep function for testing."""
        sleep_called.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep_ok)
    rl_ok.requests = []
    await rl_ok.acquire()
    assert sleep_called == []  # no sleep called
    assert len(rl_ok.requests) == 1
    # Case 2: when at limit, sleep should be awaited with computed delay and request appended
    rl = RateLimiter(requests_per_minute=1)
    # Simulate one recent request within the last 60 seconds to trigger waiting
    rl.requests = [time.time() - 59.0]
    slept = []

    async def fake_sleep(seconds: float) -> None:
        """Fake sleep function for testing."""
        slept.append(seconds)

    monkeypatch.setattr(mod.asyncio, "sleep", fake_sleep)
    await rl.acquire()
    assert slept, "Expected asyncio.sleep to be called when rate limit exceeded"
    # The computed sleep should be slightly above 1 second (60 - 59 + 0.1)
    assert pytest.approx(slept[0], rel=0.1, abs=0.5) == 1.1
    # After acquire, a new timestamp must be appended
    assert len(rl.requests) == 2
