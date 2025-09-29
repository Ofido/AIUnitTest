"""Test cases for the mock LLM connector."""

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_mock_connector_full_flow_and_failure_modes() -> None:  # noqa
    """Test the full flow and failure modes of the mock connector."""
    from ai_unit_test.core.implementations.llm.mock_connector import (  # type: ignore
        EmbeddingRequest,
        LLMConnectionError,
        LLMRequest,
        MockConnector,
        MockConnectorConfig,
    )

    def pick(obj: Any, candidates: list[str]) -> Any:  # noqa
        for name in candidates:
            if hasattr(obj, name):
                return getattr(obj, name)
        raise AttributeError("No candidate found: " + ", ".join(candidates))

    config = MockConnectorConfig(should_fail=False, response_delay=0)
    connector = MockConnector(config)

    # initialize should succeed when should_fail is False
    init = pick(connector, ["initialize", "init", "start"])
    await init()

    # pick generate-like method and test code-parsing path (valid function)
    gen = pick(connector, ["generate", "generate_response", "respond", "chat"])
    valid_code = "<source_code_chunk>\ndef foo():\n    return 1\n</source_code_chunk>"
    resp = await gen(LLMRequest(system_message="SYS", user_message=valid_code, model="mock-model"))
    # extract text-like content robustly
    if hasattr(resp, "text"):
        text = resp.text
    elif hasattr(resp, "content"):
        text = resp.content
    elif hasattr(resp, "tests"):
        text = "".join(resp.tests)
    else:
        text = str(resp)
    assert "test_foo" in text

    # test invalid source triggers dummy test fallback
    invalid_code = "<source_code_chunk>\nthis is not valid python $$$\n</source_code_chunk>"
    resp2 = await gen(LLMRequest(system_message="SYS", user_message=invalid_code, model="mock-model"))
    if hasattr(resp2, "text"):
        text2 = resp2.text
    elif hasattr(resp2, "content"):
        text2 = resp2.content
    elif hasattr(resp2, "tests"):
        text2 = "".join(resp2.tests)
    else:
        text2 = str(resp2)
    assert "test_dummy" in text2

    # test streaming generator yields expected chunks
    stream = pick(connector, ["generate_stream", "stream", "stream_generate", "stream_response", "stream_chat"])
    chunks = []
    agen = stream(LLMRequest(system_message="SYS", user_message="", model="mock-model"))
    # support both async iterator and async generator functions
    if hasattr(agen, "__aiter__"):
        async for part in agen:
            chunks.append(part)
    else:
        # if stream returned an awaitable producing an async iterator
        ait = await agen
        async for part in ait:
            chunks.append(part)
    joined = "".join(chunks)
    assert "Chunk 0" in joined
    assert "Chunk 1" in joined
    assert "Chunk 2" in joined

    # test embeddings
    embed = pick(connector, ["generate_embeddings", "embed", "embeddings"])
    emb_req = EmbeddingRequest(texts=["a", "b"], model="mock-model")
    emb_resp = await embed(emb_req)
    # robust check: either has embeddings attr or string representation
    has_embeddings = hasattr(emb_resp, "embeddings") or "embedding" in str(emb_resp).lower()
    assert has_embeddings

    # health check / is_healthy
    health = pick(connector, ["is_healthy", "health_check", "healthy"])
    assert await health() is True

    # list models and model info
    list_models = pick(connector, ["get_available_models", "models", "available_models"])
    models = list_models()
    assert isinstance(models, list)
    assert "mock-model" in models

    # set failure mode and ensure operations raise LLMConnectionError
    set_fail = pick(connector, ["set_failure_mode", "set_failure", "set_fail"])
    set_fail(True)
    assert await health() is False

    with pytest.raises(LLMConnectionError):
        await gen(LLMRequest(system_message="SYS", user_message=valid_code, model="mock-model"))

    # streaming should also raise when in failure mode
    with pytest.raises(LLMConnectionError):  # noqa
        # attempt to iterate stream and expect immediate failure
        agen2 = stream(LLMRequest(system_message="SYS", user_message="", model="mock-model"))
        async for _ in agen2:
            pass

    # embedding should raise in failure mode
    with pytest.raises(LLMConnectionError):
        await embed(emb_req)

    # restore non-failure and ensure operations work again
    set_fail(False)
    assert await health() is True
    _ = await gen(LLMRequest(system_message="SYS", user_message=valid_code, model="mock-model"))
    # collect stream again
    chunks2 = []
    agen3 = stream(LLMRequest(system_message="SYS", user_message="", model="mock-model"))
    async for p in agen3:
        chunks2.append(p)
    assert chunks2, "Expected chunks after clearing failure mode"
