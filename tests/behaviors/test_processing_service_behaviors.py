"""Tests for processing_service.py behaviors."""

from pathlib import Path
from typing import TYPE_CHECKING

from ai_unit_test.core.exceptions import TestGenerationError

if TYPE_CHECKING:
    import pytest


# Test functions will be added here


async def test_processing_service_various_behaviors(tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
    """Test various behaviors of the processing service."""
    import inspect

    import ai_unit_test.services.processing_service as mod

    service = mod.TestProcessingService({})

    # _detect_test_style: unittest class detection
    t1 = tmp_path / "test_unittest.py"
    t1.write_text("import unittest\n\nclass MyTest(unittest.TestCase):\n    pass\n")
    assert service._detect_test_style(t1) == "unittest_class"

    # _detect_test_style: empty file -> default pytest
    t_empty = tmp_path / "empty.py"
    t_empty.write_text("")
    assert service._detect_test_style(t_empty) == "pytest_function"

    # _detect_test_style: syntax error -> fallback pytest
    t_bad = tmp_path / "bad.py"
    t_bad.write_text("def f(:\n")
    assert service._detect_test_style(t_bad) == "pytest_function"

    # _group_consecutive_lines groups and sorts input
    chunks = service._group_consecutive_lines([3, 1, 2, 5])
    assert chunks == [[1, 2, 3], [5]]

    # _extract_source_code_for_lines: create source file and request lines including out of range
    src = tmp_path / "module.py"
    src.write_text("line1\nline2\nline3\nline4\n")
    extracted = service._extract_source_code_for_lines(src, [1, 2, 4, 10])
    assert "# Lines 1-2" in extracted
    assert "line1" in extracted
    assert "line2" in extracted
    assert "# Lines 4-4" in extracted
    assert "line4" in extracted
    assert "10" not in extracted  # out of range line ignored

    # _extract_source_code_for_lines returns empty string on empty content
    src_empty = tmp_path / "src_empty.py"
    src_empty.write_text("")
    assert service._extract_source_code_for_lines(src_empty, [1]) == ""

    # _generate_test_with_llm should raise when no llm_connector
    import pytest

    async_func = service._generate_test_with_llm
    # Correct argument order: source_code, test_code, file_name, coverage_lines, other_tests_content, test_style
    args = ("source", "testcode", "file.py", [1, 2], "", "pytest_function")

    # Test that it raises TestGenerationError
    with pytest.raises(TestGenerationError):
        await async_func(*args)

    # _process_single_file: when no test exists it should create one and if no extractable code -> skipped
    source_file = tmp_path / "to_process.py"
    source_file.write_text("def foo():\n    return 1\n")
    tests_folder = tmp_path / "tests"
    tests_folder.mkdir()
    # ensure no test file exists
    test_file_path = tests_folder / f"test_{source_file.name}"
    if test_file_path.exists():
        test_file_path.unlink()

    # Monkeypatch internal helpers to control behavior
    monkeypatch.setattr(service, "_detect_test_style", lambda p: "pytest_function")
    monkeypatch.setattr(service, "_extract_source_code_for_lines", lambda p, lines: "")
    # Call _process_single_file (async or sync)
    proc = service._process_single_file
    if inspect.iscoroutinefunction(proc):
        result = await proc(source_file, [1], str(tests_folder))
    else:
        result = proc(source_file, [1], str(tests_folder))  # type: ignore
    assert isinstance(result, mod.FileTestResult)
    assert result.status == "skipped"
    assert test_file_path.exists()
    # the created test file should be empty string content
    from ai_unit_test.file_helper import read_file_content

    content = read_file_content(test_file_path)
    assert content == ""

    # _find_relevant_test_context: when index_organizer is None returns empty string
    service.index_organizer = None
    finder = service._find_relevant_test_context
    if inspect.iscoroutinefunction(finder):
        res_none = await finder(source_file, str(tests_folder))
    else:
        res_none = finder(source_file, str(tests_folder))  # type: ignore[assignment]
    assert res_none == ""

    # when index_organizer present and find_relevant_tests returns multiple entries, limit to 3
    # Create a mock index organizer that has the required interface
    class MockIndexOrganizer:
        pass

    service.index_organizer = MockIndexOrganizer()  # type: ignore[assignment]
    monkeypatch.setattr(mod, "find_relevant_tests", lambda a, b: ["A", "B", "C", "D"])
    if inspect.iscoroutinefunction(finder):
        res_ctx = await finder(source_file, str(tests_folder))
    else:
        res_ctx = finder(source_file, str(tests_folder))  # type: ignore[assignment]
    assert res_ctx == "A\n\nB\n\nC"
