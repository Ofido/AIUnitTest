"""Tests for v2 target selectors."""

from pathlib import Path

import pytest

from ai_unit_test.v2.models import RunRequest
from ai_unit_test.v2.targeting.selectors import ExplicitFileSelector


class TestExplicitFileSelector:
    """Test suite for ExplicitFileSelector."""

    def test_select_existing_file(self, tmp_path: Path) -> None:
        """Test selecting an existing file."""
        source = tmp_path / "module.py"
        source.write_text("def hello(): pass")

        selector = ExplicitFileSelector()
        request = RunRequest(file_path=str(source), backend_name="test")
        targets = selector.select(request)

        assert len(targets) == 1
        assert targets[0].mode == "explicit-file"
        assert str(source) in targets[0].files
        assert targets[0].rationale is not None

    def test_select_nonexistent_file_raises(self) -> None:
        """Test that selecting a nonexistent file raises FileNotFoundError."""
        selector = ExplicitFileSelector()
        request = RunRequest(file_path="/nonexistent/module.py", backend_name="test")

        with pytest.raises(FileNotFoundError, match="Target file not found"):
            selector.select(request)
