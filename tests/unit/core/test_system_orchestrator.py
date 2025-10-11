"""Tests for system_orchestrator.py."""

from collections.abc import Callable
from pathlib import PosixPath
from typing import Any

import pytest

from ai_unit_test.core.system_orchestrator import *  # noqa


def test_system_orchestrator_init_signal_and_excepthook(
    tmp_path: PosixPath, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test."""
    import asyncio
    import signal
    import sys
    from pathlib import Path

    from ai_unit_test.core.system_orchestrator import (  # type: ignore
        AIUnitTestError,
        ConfigurationError,
        SystemOrchestrator,
    )

    # run in isolated temp dir
    monkeypatch.chdir(tmp_path)

    # instantiate with verbose and a log file to exercise file handler branch
    orchestrator = SystemOrchestrator()
    orchestrator.setup_comprehensive_logging(verbose=True, log_file=Path("app.log"))

    # verify log files/dirs created
    assert (tmp_path / "app.log").exists()
    logs_dir = tmp_path / "logs"
    assert (logs_dir / "ai_unit_test_errors.log").exists()
    assert (logs_dir / "ai_unit_test_performance.log").exists()

    # Prepare to intercept sys.exit calls
    def fake_exit(code: int = 0) -> None:
        raise SystemExit(code)

    monkeypatch.setattr(sys, "exit", fake_exit)

    # Capture the real handler registered by signal.signal
    registered_handlers = {}
    original_signal = signal.signal

    def capture_signal(signum: int, handler: Callable[[int, Any], Any]) -> Callable[[int, Any], Any] | int | None:
        registered_handlers[signum] = handler
        return original_signal(signum, handler)

    monkeypatch.setattr(signal, "signal", capture_signal)

    # Register real handlers
    orchestrator.setup_signal_handlers()

    # Simulate absence of event loop
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: (_ for _ in ()).throw(RuntimeError()))

    # Ensure that sys.exit raises SystemExit
    monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))

    # Capture the real handler for SIGTERM and expect SystemExit(1)
    with pytest.raises(SystemExit) as si:
        registered_handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert si.value.code == 1

    # Test excepthook behavior for KeyboardInterrupt -> should call original __excepthook__ and exit 130
    called = []

    def fake_original_excepthook(t: type, v: BaseException, tb: object) -> None:
        called.append((t, v, tb))

    # Set global excepthook for the orchestrator handler
    monkeypatch.setattr(sys, "excepthook", orchestrator.global_exception_handler)
    monkeypatch.setattr(sys, "__excepthook__", fake_original_excepthook)

    # Test excepthook behavior for KeyboardInterrupt -> should call __excepthook__ and exit with 130
    with pytest.raises(SystemExit) as si2:
        sys.excepthook(KeyboardInterrupt, KeyboardInterrupt(), None)
    assert si2.value.code == 130
    assert len(called) == 1

    # Test excepthook behavior for ConfigurationError
    with pytest.raises(SystemExit) as si3:
        sys.excepthook(ConfigurationError, ConfigurationError("bad config"), None)
    assert si3.value.code == 1
    err = capsys.readouterr().err
    assert "Configuration Error" in err or "Configuration error" in err

    # Test excepthook behavior for AIUnitTestError
    with pytest.raises(SystemExit) as si4:
        sys.excepthook(AIUnitTestError, AIUnitTestError("app error"), None)
    assert si4.value.code == 1
    err = capsys.readouterr().err
    assert "Error:" in err or "Application error" in err

    # Test excepthook behavior for generic Exception -> prints fatal and exits with code 1
    with pytest.raises(SystemExit) as si5:
        sys.excepthook(Exception, Exception("boom"), None)
    assert si5.value.code == 1
    err = capsys.readouterr().err
    assert "Fatal error" in err or "Traceback" in err or "Error:" in err

    # cleanup: ensure handlers closed without error
    # call orchestrator cleanup sequence if available (close handlers)
    for h in getattr(orchestrator, "log_handlers", []):
        try:
            h.close()
        except Exception:  # nosec
            pass
