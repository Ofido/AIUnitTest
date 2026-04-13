"""Tests for v2 context builder."""

from pathlib import Path

from ai_unit_test.v2.context.builder import FileContextBuilder
from ai_unit_test.v2.models import TargetSpec


class TestFileContextBuilder:
    """Test suite for FileContextBuilder."""

    def test_build_reads_source_file(self, tmp_path: Path) -> None:
        """Test that build reads the source file content."""
        source = tmp_path / "module.py"
        source.write_text("def hello(): pass")

        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(source)])
        bundle = builder.build(target)

        assert str(source) in bundle.source_snippets
        assert "def hello" in bundle.source_snippets[str(source)]

    def test_build_finds_related_tests(self, tmp_path: Path) -> None:
        """Test that build finds related test files."""
        source = tmp_path / "calculator.py"
        source.write_text("def add(a, b): return a + b")

        tests_dir = tmp_path / "tests"
        tests_dir.mkdir()
        test_file = tests_dir / "test_calculator.py"
        test_file.write_text("def test_add(): assert add(1, 2) == 3")

        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(source)])
        bundle = builder.build(target)

        assert str(test_file) in bundle.test_snippets

    def test_build_reads_pyproject(self, tmp_path: Path) -> None:
        """Test that build reads pyproject.toml if present."""
        source = tmp_path / "mod.py"
        source.write_text("x = 1")
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.pytest]")

        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(source)])
        bundle = builder.build(target)

        assert "pyproject.toml" in bundle.project_config

    def test_build_with_feedback_and_previous_patch(self, tmp_path: Path) -> None:
        """Test that build passes feedback and previous patch through."""
        source = tmp_path / "mod.py"
        source.write_text("x = 1")

        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(source)])
        bundle = builder.build(target, feedback=["error: syntax"], previous_patch="old code")

        assert bundle.validator_feedback == ["error: syntax"]
        assert bundle.previous_patch == "old code"

    def test_build_with_missing_source(self, tmp_path: Path) -> None:
        """Test that build handles missing source files gracefully."""
        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(tmp_path / "missing.py")])
        bundle = builder.build(target)

        assert len(bundle.source_snippets) == 0

    def test_is_test_file_custom_patterns(self, tmp_path: Path) -> None:
        """Test that custom test patterns are used."""
        builder = FileContextBuilder(project_root=tmp_path, test_patterns=["check_*.py"])
        assert builder._is_test_file(Path("check_math.py")) is True
        assert builder._is_test_file(Path("test_math.py")) is False

    def test_no_test_dir(self, tmp_path: Path) -> None:
        """Test behavior when no tests directory exists."""
        source = tmp_path / "mod.py"
        source.write_text("x = 1")

        builder = FileContextBuilder(project_root=tmp_path)
        target = TargetSpec(mode="explicit-file", files=[str(source)])
        bundle = builder.build(target)

        assert bundle.test_snippets == {}
