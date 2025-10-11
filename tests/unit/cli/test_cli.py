"""Tests for cli.py."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from ai_unit_test.cli import *  # noqa


def test_cli_helpers_and_health(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test CLI helpers and health check."""
    import importlib
    from types import SimpleNamespace

    import typer

    cli = importlib.import_module("ai_unit_test.cli")

    outputs = []
    errors = []

    def fake_echo(message: str = "", **kwargs: Any) -> None:  # noqa
        outputs.append(str(message))

    def fake_logger_error(msg: str) -> None:
        errors.append(str(msg))

    monkeypatch.setattr(cli.typer, "echo", fake_echo)
    monkeypatch.setattr(cli.logger, "error", fake_logger_error)

    # Test handle_cli_exception
    cli.handle_cli_exception(Exception("boom"))
    assert any("CLI command failed: boom" in e for e in errors)
    assert any("❌ Error: boom" in o for o in outputs)

    outputs.clear()
    errors.clear()

    # Test _display_test_generation_results with duration, file_results and errors
    results = {
        "status": "partial_success",
        "files_processed": 5,
        "tests_generated": 3,
        "workflow_duration_seconds": 1.2345,
        "file_results": {
            "a.py": {"test_generated": True, "status": "ok"},
            "b.py": {"test_generated": False, "status": "failed"},
        },
        "errors": ["error_one"],
    }
    cli._display_test_generation_results(results)
    outtext = "\n".join(outputs)
    assert "📊 Test Generation Results" in outtext
    assert "Status: partial_success" in outtext
    assert "Files processed: 5" in outtext
    assert "Tests generated: 3" in outtext
    assert "Duration: 1.23s" in outtext
    assert "✅ a.py" in outtext
    assert "⚠️ b.py" in outtext
    assert "❌ Errors" in outtext
    assert "error_one" in outtext

    outputs.clear()

    # Test _display_index_creation_results when error
    idx_results = {"status": "error", "error": "index failure"}
    cli._display_index_creation_results(idx_results)
    outtext = "\n".join(outputs)
    assert "📚 Index Creation Results" in outtext
    assert "Status: error" in outtext
    assert "Error: index failure" in outtext

    outputs.clear()

    # Test _display_health_check_results with healthy and unhealthy checks
    checks = SimpleNamespace(
        db=SimpleNamespace(healthy=False, error="db fail"),
        cache=SimpleNamespace(healthy=True),
    )
    health = SimpleNamespace(status="unhealthy", checks=checks)
    cli._display_health_check_results(health)
    outtext = "\n".join(outputs)
    assert "🏥 Health Check Results" in outtext
    assert "Overall Status: unhealthy" in outtext
    assert "Db: Unhealthy" in outtext or "Db: Unhealthy" in outtext.replace("  ", "")
    assert "Cache: Healthy" in outtext
    assert "Error: db fail" in outtext

    outputs.clear()

    # Test health_check function raises Exit for unhealthy status and echoes start message
    class Orchestration:
        def run_health_check_workflow(self) -> SimpleNamespace:
            return health

    ctx = SimpleNamespace(obj={"orchestration_service": Orchestration()})

    # monkeypatch asyncio.run used inside health_check to return our health object
    monkeypatch.setattr(cli.asyncio, "run", MagicMock(return_value=health))

    # Ensure typer.Exit is raised for unhealthy
    try:
        import pytest

        with pytest.raises(typer.Exit):
            cli.health_check(ctx)
    finally:
        # also confirm the start message was echoed
        assert any("🏥 Running health check" in o for o in outputs) or any(
            "🏥 Running health check" in o for o in outputs
        )
