"""Test for __main__.py."""

import pytest

from ai_unit_test.__main__ import run_app


def test_run_app_variants(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Test different variants of the run_app function."""
    # success case: CLIManager.run executes normally
    events = []

    def fake_system_orchestrator() -> object:
        events.append("orchestrator_created")
        return object()

    class FakeCLI:
        def __init__(self, orchestrator: object) -> None:
            events.append(("cli_init", isinstance(orchestrator, object)))  # type: ignore

        def run(self) -> None:
            events.append("cli_run")

    monkeypatch.setattr("ai_unit_test.__main__.SystemOrchestrator", fake_system_orchestrator)
    monkeypatch.setattr("ai_unit_test.__main__.CLIManager", FakeCLI)

    run_app()
    assert events == ["orchestrator_created", ("cli_init", True), "cli_run"]

    # SystemExit from CLIManager.run is swallowed
    class CLIRaisesSystemExit:
        def __init__(self, orchestrator: object) -> None:
            pass

        def run(self) -> None:
            raise SystemExit

    monkeypatch.setattr("ai_unit_test.__main__.CLIManager", CLIRaisesSystemExit)
    # should not raise
    run_app()

    # other exceptions cause message to stderr and sys.exit(1)
    class CLIRaisesError:
        def __init__(self, orchestrator: object) -> None:
            pass

        def run(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr("ai_unit_test.__main__.CLIManager", CLIRaisesError)
    with pytest.raises(SystemExit) as exc:
        run_app()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "❌ Application crashed: boom" in captured.err
