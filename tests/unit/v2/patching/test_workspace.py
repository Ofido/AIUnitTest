"""Tests for v2 patch applier."""

from pathlib import Path

from ai_unit_test.v2.models import PatchCandidate, RunRequest
from ai_unit_test.v2.patching.workspace import PatchApplier


class TestPatchApplier:
    """Test suite for PatchApplier."""

    def _make_request(self, allow_source: bool = False, dry_run: bool = False) -> RunRequest:
        """Create a RunRequest for testing."""
        return RunRequest(
            file_path="src/mod.py",
            backend_name="test",
            allow_source_edits=allow_source,
            dry_run=dry_run,
        )

    def test_apply_test_file(self, tmp_path: Path) -> None:
        """Test applying a patch to a test file."""
        test_file = tmp_path / "test_mod.py"
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="add test",
            patch_text=f"--- file: {test_file}\ndef test_hello(): pass\n",
            touched_files=[str(test_file)],
        )

        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request())

        assert result.success is True
        assert str(test_file) in result.applied_files
        assert test_file.read_text() == "def test_hello(): pass\n"

    def test_refuse_source_file_without_flag(self, tmp_path: Path) -> None:
        """Test that source files are refused without allow_source_edits."""
        source_file = tmp_path / "module.py"
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="edit source",
            patch_text=f"--- file: {source_file}\nmodified\n",
            touched_files=[str(source_file)],
        )

        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request(allow_source=False))

        assert result.success is False
        assert "Refused to write non-test file" in (result.error or "")

    def test_allow_source_file_with_flag(self, tmp_path: Path) -> None:
        """Test that source files are allowed with allow_source_edits."""
        source_file = tmp_path / "module.py"
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="edit source",
            patch_text=f"--- file: {source_file}\nmodified\n",
            touched_files=[str(source_file)],
        )

        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request(allow_source=True))

        assert result.success is True

    def test_dry_run(self, tmp_path: Path) -> None:
        """Test dry run does not write files."""
        test_file = tmp_path / "test_mod.py"
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="add test",
            patch_text=f"--- file: {test_file}\ncontent\n",
            touched_files=[str(test_file)],
        )

        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request(dry_run=True))

        assert result.success is True
        assert not test_file.exists()

    def test_rollback_on_failure(self, tmp_path: Path) -> None:
        """Test that rollback restores original file content."""
        test_file = tmp_path / "test_existing.py"
        test_file.write_text("original content")

        applier = PatchApplier()
        applier._snapshot_file(test_file)
        test_file.write_text("modified content")
        applier.rollback()

        assert test_file.read_text() == "original content"

    def test_rollback_removes_new_files(self, tmp_path: Path) -> None:
        """Test that rollback removes files that did not exist before."""
        new_file = tmp_path / "test_new.py"

        applier = PatchApplier()
        applier._snapshot_file(new_file)
        new_file.write_text("new content")
        applier.rollback()

        assert not new_file.exists()

    def test_parse_patch_text_multiple_files(self) -> None:
        """Test parsing patch text with multiple files."""
        patch = "--- file: test_a.py\ncontent_a\n--- file: test_b.py\ncontent_b\n"
        applier = PatchApplier()
        files = applier._parse_patch_text(patch)

        assert "test_a.py" in files
        assert "test_b.py" in files
        assert "content_a" in files["test_a.py"]

    def test_compute_diff(self) -> None:
        """Test unified diff computation."""
        diff = PatchApplier._compute_diff("old\n", "new\n", "test.py")
        assert "---" in diff
        assert "+++" in diff

    def test_is_test_file_patterns(self) -> None:
        """Test test file pattern matching."""
        applier = PatchApplier(test_patterns=["test_*.py", "*_test.py"])
        assert applier._is_test_file(Path("test_module.py")) is True
        assert applier._is_test_file(Path("module_test.py")) is True
        assert applier._is_test_file(Path("module.py")) is False
        assert applier._is_test_file(Path("conftest.py")) is False

    def test_empty_patch_text_fails(self) -> None:
        """Test that unparseable/empty patch text fails explicitly."""
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="bad output",
            patch_text="just some text without file markers",
            touched_files=[],
        )
        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request())

        assert result.success is False
        assert "no parseable files" in (result.error or "").lower()

    def test_whitespace_only_patch_fails(self) -> None:
        """Test that whitespace-only patch text fails."""
        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="empty",
            patch_text="   \n\n  ",
            touched_files=[],
        )
        applier = PatchApplier()
        result = applier.apply(candidate, self._make_request())

        assert result.success is False

    def test_rollback_preserves_empty_files(self, tmp_path: Path) -> None:
        """Test that rollback preserves pre-existing empty files."""
        empty_file = tmp_path / "test_empty.py"
        empty_file.write_text("", encoding="utf-8")

        candidate = PatchCandidate(
            backend_name="test",
            plan_summary="overwrite",
            patch_text=f"--- file: {empty_file}\ndef test(): pass\n",
            touched_files=[str(empty_file)],
        )
        applier = PatchApplier()
        applier.apply(candidate, self._make_request())

        assert empty_file.read_text() == "def test(): pass\n"

        applier.rollback()

        assert empty_file.exists(), "Rollback should not delete pre-existing empty files"
        assert empty_file.read_text() == ""
