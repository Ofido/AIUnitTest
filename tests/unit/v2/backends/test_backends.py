"""Tests for v2 backend adapters."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.v2.backends.base import (
    BackendRegistry,
    BaseCliBackend,
    _extract_json_object,
)
from ai_unit_test.v2.backends.copilot_cli import CopilotCliBackend
from ai_unit_test.v2.backends.gemini_cli import GeminiCliBackend
from ai_unit_test.v2.models import ContextBundle, TargetSpec


def _make_context() -> ContextBundle:
    """Create a ContextBundle for testing."""
    return ContextBundle(
        target=TargetSpec(mode="explicit-file", files=["mod.py"]),
        source_snippets={"mod.py": "def hello(): pass"},
    )


class _FakeCliBackend(BaseCliBackend):
    """Concrete BaseCliBackend used to drive tests of the shared scaffold."""

    name = "fake-cli"

    def _subprocess_argv(self, prompt: str) -> list[str]:
        return ["true"]


def _mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


class TestBackendRegistry:
    """Test suite for BackendRegistry."""

    def test_register_and_get(self) -> None:
        registry = BackendRegistry()
        backend = MagicMock()
        backend.name = "test-backend"
        registry.register(backend)
        assert registry.get("test-backend") is backend

    def test_get_missing_raises(self) -> None:
        registry = BackendRegistry()
        with pytest.raises(KeyError):
            registry.get("nonexistent")

    def test_list_names(self) -> None:
        registry = BackendRegistry()
        b1 = MagicMock()
        b1.name = "beta"
        b2 = MagicMock()
        b2.name = "alpha"
        registry.register(b1)
        registry.register(b2)
        assert registry.list_names() == ["alpha", "beta"]


class TestExtractJsonObject:
    """Test suite for the balanced-brace JSON extractor."""

    def test_simple_object(self) -> None:
        assert _extract_json_object('prefix {"a": 1} suffix') == '{"a": 1}'

    def test_nested_object(self) -> None:
        text = 'log: {"a": {"b": 2}, "c": 3} done'
        assert _extract_json_object(text) == '{"a": {"b": 2}, "c": 3}'

    def test_string_with_escaped_braces(self) -> None:
        text = '{"raw": "look {nested}", "ok": true}'
        assert _extract_json_object(text) == text

    def test_string_with_escaped_quotes(self) -> None:
        text = '{"q": "he said \\"hi\\"", "n": 1}'
        assert _extract_json_object(text) == text

    def test_no_object(self) -> None:
        assert _extract_json_object("nothing here") is None


class TestBaseCliBackend:
    """Test suite for shared BaseCliBackend behavior."""

    @pytest.mark.asyncio
    async def test_propose_patch_pure_json(self) -> None:
        """Direct JSON output is parsed into a PatchCandidate."""
        response = json.dumps(
            {
                "plan_summary": "tests for hello",
                "files": {"test_mod.py": "def test_hello(): assert True"},
            }
        ).encode()

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=response)):
            result = await _FakeCliBackend().propose_patch(_make_context())

        assert result.touched_files == ["test_mod.py"]
        assert result.plan_summary == "tests for hello"
        assert "def test_hello" in result.patch_text

    @pytest.mark.asyncio
    async def test_propose_patch_strips_ansi_and_extracts_json(self) -> None:
        """Output with ANSI escapes and prose noise still parses."""
        payload = {
            "plan_summary": "ok",
            "files": {"test_x.py": "def test_x(): pass"},
        }
        noisy = (
            b"\x1b[1mWelcome\x1b[0m\n"
            b"Generating...\n"
            + json.dumps(payload).encode()
            + b"\n\x1b[32mDone\x1b[0m"
        )

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=noisy)):
            result = await _FakeCliBackend().propose_patch(_make_context())

        assert "test_x.py" in result.touched_files

    @pytest.mark.asyncio
    async def test_propose_patch_fenced_python_fallback(self) -> None:
        """Fenced python blocks fall back to a partial result with no touched_files."""
        response = b"Some explanation\n```python\ndef test_x(): pass\n```\n"

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=response)):
            result = await _FakeCliBackend().propose_patch(_make_context())

        assert "def test_x" in result.patch_text
        assert result.touched_files == []

    @pytest.mark.asyncio
    async def test_propose_patch_unparseable_dumps_artifact(self, tmp_path: Path) -> None:
        """Unparseable output is still returned but raw is persisted to artifacts dir."""
        backend = _FakeCliBackend()
        backend.artifacts_dir = tmp_path / "v2-backends"

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=b"just text")):
            result = await backend.propose_patch(_make_context())

        assert result.patch_text == "just text"
        assert "Could not parse" in result.plan_summary
        artifacts = list((tmp_path / "v2-backends").iterdir())
        assert len(artifacts) == 1
        assert "just text" in artifacts[0].read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_binary_not_found_raises(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError):
                await _FakeCliBackend().propose_patch(_make_context())

    @pytest.mark.asyncio
    async def test_nonzero_returncode_raises(self) -> None:
        with patch(
            "asyncio.create_subprocess_exec",
            return_value=_mock_proc(stderr=b"boom", returncode=1),
        ):
            with pytest.raises(RuntimeError, match="exited with code 1"):
                await _FakeCliBackend().propose_patch(_make_context())

    @pytest.mark.asyncio
    async def test_empty_output_raises(self) -> None:
        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=b"  ")):
            with pytest.raises(RuntimeError, match="empty output"):
                await _FakeCliBackend().propose_patch(_make_context())

    def test_prompt_includes_guardrails(self) -> None:
        """The shared prompt must mention the test-file-only and JSON-only contract."""
        prompt = _FakeCliBackend()._build_prompt(_make_context())
        assert "pytest" in prompt
        assert "test_*.py" in prompt
        assert "JSON" in prompt
        assert "repository-relative" in prompt


class TestCopilotCliBackend:
    """Test suite for CopilotCliBackend."""

    def test_name(self) -> None:
        assert CopilotCliBackend().name == "copilot-cli"

    def test_default_model(self) -> None:
        backend = CopilotCliBackend()
        assert backend.model

    def test_explicit_model_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_UNIT_TEST_COPILOT_MODEL", "from-env")
        assert CopilotCliBackend(model="explicit").model == "explicit"

    def test_env_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_UNIT_TEST_COPILOT_MODEL", "from-env")
        assert CopilotCliBackend().model == "from-env"

    def test_argv_uses_gh_models_run(self) -> None:
        argv = CopilotCliBackend(model="gpt-4o-mini")._subprocess_argv("prompt")
        assert argv == ["gh", "models", "run", "gpt-4o-mini"]

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        """Available when gh, gh-models extension and auth all OK."""

        async def fake_exec(*args: str, **_: object) -> AsyncMock:
            cmd = list(args)
            if cmd[:2] == ["gh", "extension"]:
                return _mock_proc(stdout=b"github/gh-models\tv0.0.1\n")
            return _mock_proc(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            assert await CopilotCliBackend.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_missing_extension(self) -> None:
        async def fake_exec(*args: str, **_: object) -> AsyncMock:
            cmd = list(args)
            if cmd[:2] == ["gh", "extension"]:
                return _mock_proc(stdout=b"some/other-ext\n")
            return _mock_proc(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            assert await CopilotCliBackend.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_missing_auth(self) -> None:
        async def fake_exec(*args: str, **_: object) -> AsyncMock:
            cmd = list(args)
            if cmd[:2] == ["gh", "extension"]:
                return _mock_proc(stdout=b"github/gh-models\n")
            if cmd[:3] == ["gh", "auth", "status"]:
                return _mock_proc(returncode=1)
            return _mock_proc(stdout=b"ok")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            assert await CopilotCliBackend.is_available() is False

    @pytest.mark.asyncio
    async def test_is_available_no_gh(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            assert await CopilotCliBackend.is_available() is False


class TestGeminiCliBackend:
    """Test suite for GeminiCliBackend."""

    def test_name(self) -> None:
        assert GeminiCliBackend().name == "gemini-cli"

    def test_argv_includes_json_and_yolo(self) -> None:
        argv = GeminiCliBackend(model="gemini-2.5-flash")._subprocess_argv("prompt")
        assert "--yolo" in argv
        assert "--output-format" in argv and "json" in argv
        assert "--model" in argv and "gemini-2.5-flash" in argv

    def test_env_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AI_UNIT_TEST_GEMINI_MODEL", "custom-gemini")
        assert GeminiCliBackend().model == "custom-gemini"

    @pytest.mark.asyncio
    async def test_unwraps_json_envelope(self) -> None:
        """Gemini's JSON envelope wrapping our payload is unwrapped before parsing."""
        inner = {
            "plan_summary": "from gemini",
            "files": {"test_mod.py": "def test_y(): pass"},
        }
        envelope = {"response": json.dumps(inner)}
        stdout = json.dumps(envelope).encode()

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=stdout)):
            result = await GeminiCliBackend().propose_patch(_make_context())

        assert result.plan_summary == "from gemini"
        assert "test_mod.py" in result.touched_files

    @pytest.mark.asyncio
    async def test_falls_back_to_base_parsing_when_no_envelope(self) -> None:
        """Without an envelope, the base parser still works."""
        payload = json.dumps(
            {
                "plan_summary": "direct",
                "files": {"test_mod.py": "def test(): pass"},
            }
        ).encode()

        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc(stdout=payload)):
            result = await GeminiCliBackend().propose_patch(_make_context())

        assert result.touched_files == ["test_mod.py"]

    @pytest.mark.asyncio
    async def test_gemini_not_found(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            with pytest.raises(RuntimeError, match="gemini CLI not found"):
                await GeminiCliBackend().propose_patch(_make_context())

    @pytest.mark.asyncio
    async def test_is_available_true(self) -> None:
        with patch("asyncio.create_subprocess_exec", return_value=_mock_proc()):
            assert await GeminiCliBackend.is_available() is True

    @pytest.mark.asyncio
    async def test_is_available_false(self) -> None:
        with patch("asyncio.create_subprocess_exec", side_effect=FileNotFoundError()):
            assert await GeminiCliBackend.is_available() is False
