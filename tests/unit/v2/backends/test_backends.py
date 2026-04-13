"""Tests for v2 backend adapters."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.v2.backends.base import BackendRegistry
from ai_unit_test.v2.backends.copilot_cli import CopilotCliBackend
from ai_unit_test.v2.backends.gemini_cli import GeminiCliBackend
from ai_unit_test.v2.models import ContextBundle, TargetSpec


def _make_context() -> ContextBundle:
    """Create a ContextBundle for testing."""
    return ContextBundle(
        target=TargetSpec(mode="explicit-file", files=["mod.py"]),
        source_snippets={"mod.py": "def hello(): pass"},
    )


class TestBackendRegistry:
    """Test suite for BackendRegistry."""

    def test_register_and_get(self) -> None:
        """Test registering and retrieving backends."""
        registry = BackendRegistry()
        backend = MagicMock()
        backend.name = "test-backend"
        registry.register(backend)

        assert registry.get("test-backend") is backend

    def test_get_missing_raises(self) -> None:
        """Test that getting a missing backend raises KeyError."""
        registry = BackendRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_names(self) -> None:
        """Test listing backend names."""
        registry = BackendRegistry()
        b1 = MagicMock()
        b1.name = "beta"
        b2 = MagicMock()
        b2.name = "alpha"
        registry.register(b1)
        registry.register(b2)

        assert registry.list_names() == ["alpha", "beta"]


class TestCopilotCliBackend:
    """Test suite for CopilotCliBackend."""

    def test_name(self) -> None:
        """Test backend name."""
        assert CopilotCliBackend().name == "copilot-cli"

    @pytest.mark.asyncio
    async def test_propose_patch_json_response(self) -> None:
        """Test parsing a JSON response."""
        response = json.dumps({
            "plan_summary": "Add tests for hello",
            "files": {"test_mod.py": "def test_hello(): pass"},
        })
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(response.encode(), b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = CopilotCliBackend()
            result = await backend.propose_patch(_make_context())

        assert result.backend_name == "copilot-cli"
        assert "test_mod.py" in result.touched_files
        assert result.plan_summary == "Add tests for hello"

    @pytest.mark.asyncio
    async def test_propose_patch_fenced_python(self) -> None:
        """Test parsing fenced Python blocks as fallback."""
        response = "Some explanation\n```python\ndef test_x(): pass\n```\n"
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(response.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = CopilotCliBackend()
            result = await backend.propose_patch(_make_context())

        assert "def test_x" in result.patch_text

    @pytest.mark.asyncio
    async def test_propose_patch_unparseable(self) -> None:
        """Test handling of unparseable output."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"just text", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = CopilotCliBackend()
            result = await backend.propose_patch(_make_context())

        assert result.patch_text == "just text"
        assert "Could not parse" in result.plan_summary

    @pytest.mark.asyncio
    async def test_gh_not_found(self) -> None:
        """Test handling of missing gh CLI."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            backend = CopilotCliBackend()
            with pytest.raises(RuntimeError, match="gh CLI not found"):
                await backend.propose_patch(_make_context())

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Test availability check when gh is present."""
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))
        mock_proc.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            assert await CopilotCliBackend.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        """Test availability check when gh is missing."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            assert await CopilotCliBackend.is_available() is False


class TestGeminiCliBackend:
    """Test suite for GeminiCliBackend."""

    def test_name(self) -> None:
        """Test backend name."""
        assert GeminiCliBackend().name == "gemini-cli"

    @pytest.mark.asyncio
    async def test_propose_patch_json_response(self) -> None:
        """Test parsing a JSON response."""
        response = json.dumps({
            "plan_summary": "Add tests",
            "files": {"test_mod.py": "def test_y(): pass"},
        })
        mock_proc = AsyncMock()
        mock_proc.communicate = AsyncMock(return_value=(response.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = GeminiCliBackend()
            result = await backend.propose_patch(_make_context())

        assert result.backend_name == "gemini-cli"
        assert "test_mod.py" in result.touched_files

    @pytest.mark.asyncio
    async def test_gemini_not_found(self) -> None:
        """Test handling of missing gemini CLI."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            backend = GeminiCliBackend()
            with pytest.raises(RuntimeError, match="gemini CLI not found"):
                await backend.propose_patch(_make_context())

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        """Test availability check when gemini is missing."""
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            assert await GeminiCliBackend.is_available() is False
