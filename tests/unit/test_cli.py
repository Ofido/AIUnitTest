import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from typer.testing import CliRunner

from ai_unit_test.chunking import Chunk
from ai_unit_test.cli import (
    _detect_test_style,
    _main,
    _process_missing_info,
    _resolve_paths_from_config,
    app,
    extract_from_pyproject,
    extract_test_patterns_from_pyproject,
    index,
    load_pyproject_config,
    search,
)


def test_load_pyproject_config_exists() -> None:
    """
    Tests that the pyproject.toml file is loaded correctly.
    """
    config = load_pyproject_config(Path("tests/unit/fake_pyproject.toml"))
    assert "tool" in config
    assert "pytest" in config["tool"]


def test_load_pyproject_config_not_exists() -> None:
    """
    Tests that an empty dictionary is returned when pyproject.toml does not exist.
    """
    config = load_pyproject_config(Path("non_existent_file.toml"))
    assert config == {}


def test_extract_from_pyproject() -> None:
    """
    Tests that the source folders, tests folder, and coverage file are extracted correctly.
    """
    config = load_pyproject_config(Path("tests/unit/fake_pyproject.toml"))
    folders, tests_folder, coverage_file = extract_from_pyproject(config)
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file == ".coverage.test"


@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("ai_unit_test.cli.chunk_test_file")
@patch("ai_unit_test.cli.extract_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
@patch("ai_unit_test.cli.read_file_content")
def test_main_auto_discovery(
    mock_read_file_content: MagicMock,
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_extract_from_pyproject: MagicMock,
    mock_chunk_test_file: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
) -> None:
    """
    Tests the _main function with auto-discovery enabled.
    """
    mock_extract_from_pyproject.return_value = (["src"], "tests", ".coverage")
    mock_collect_missing_lines.return_value = {Path("src/main.py"): [1]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.return_value = "test_code"
    mock_chunk_test_file.return_value = [
        Chunk(
            name="main",
            type="function",
            source_code="def main(): pass",
            start_line=1,
            end_line=1,
            chunk_id="",
            content_hash="",
            text_preview="",
        )
    ]
    mock_update_test_with_llm.return_value = "updated_test_code"

    asyncio.run(_main(auto=True, folders=["src"], tests_folder="tests", coverage_file=".coverage"))

    mock_load_pyproject_config.assert_called_once()
    mock_collect_missing_lines.assert_called_once_with(".coverage")
    mock_find_test_file.assert_called_once_with(str(Path("src/main.py")), "tests")
    mock_read_file_content.assert_any_call(Path("tests/test_main.py"))
    mock_update_test_with_llm.assert_called_once()
    assert mock_write_file_content.call_args[0][0] == Path("tests/test_main.py")
    assert "updated_test_code" in mock_write_file_content.call_args[0][1]


@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("ai_unit_test.cli.chunk_test_file")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
@patch("ai_unit_test.cli.read_file_content")
def test_main_explicit_args(
    mock_read_file_content: MagicMock,
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_chunk_test_file: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
) -> None:
    """
    Tests the _main function with explicit arguments.
    """
    mock_collect_missing_lines.return_value = {Path("src/main.py"): [1]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.return_value = "test_code"
    mock_chunk_test_file.return_value = [
        Chunk(
            name="main",
            type="function",
            source_code="def main(): pass",
            start_line=1,
            end_line=1,
            chunk_id="",
            content_hash="",
            text_preview="",
        )
    ]
    mock_update_test_with_llm.return_value = "updated_test_code"

    asyncio.run(
        _main(
            folders=["src"],
            tests_folder="tests",
            coverage_file=".coverage",
            auto=False,
        )
    )

    mock_collect_missing_lines.assert_called_once_with(".coverage")
    mock_find_test_file.assert_called_once_with(str(Path("src/main.py")), "tests")
    mock_read_file_content.assert_any_call(Path("tests/test_main.py"))
    mock_update_test_with_llm.assert_called_once()
    assert mock_write_file_content.call_args[0][0] == Path("tests/test_main.py")
    assert "updated_test_code" in mock_write_file_content.call_args[0][1]


@patch("ai_unit_test.cli.logger.error")
@patch("ai_unit_test.cli.extract_function_source")
def test_func_command_function_not_found(
    mock_extract_function_source: MagicMock,
    mock_logger_error: MagicMock,
) -> None:
    """
    Tests the func command when the function is not found.
    """
    runner = CliRunner()
    mock_extract_function_source.return_value = None

    result = runner.invoke(
        app,
        [
            "func",
            "src/simple_math.py",
            "non_existent_function",
            "--tests-folder",
            "tests",
        ],
    )
    assert result.exit_code == 1
    mock_extract_function_source.assert_called_once_with("src/simple_math.py", "non_existent_function")
    mock_logger_error.assert_called_once_with("Function 'non_existent_function' not found in 'src/simple_math.py'.")


@patch("pathlib.Path.is_dir", return_value=True)
def test_extract_from_pyproject_fallback_to_tests_directory(mock_is_dir: MagicMock) -> None:
    """
    Tests that the function falls back to the 'tests' directory when no tests folder is specified.
    """
    data = {
        "tool": {
            "coverage": {"run": {"source": ["src"], "data_file": ".coverage.test"}},
            "pytest": {"ini_options": {"testpaths": [None]}},
        }
    }
    folders, tests_folder, coverage_file = extract_from_pyproject(data)
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file == ".coverage.test"


@patch("ai_unit_test.cli.load_pyproject_config", return_value={})
@patch("ai_unit_test.cli.extract_from_pyproject", return_value=([], None, None))
@patch("ai_unit_test.cli.logger")
def test_resolve_paths_from_config_no_folders_or_tests_folder(
    mock_logger: MagicMock,
    mock_extract_from_pyproject: MagicMock,
    mock_load_pyproject_config: MagicMock,
) -> None:
    """
    Tests the _resolve_paths_from_config function when no folders or tests_folder are provided.
    """
    with pytest.raises(SystemExit) as excinfo:
        _resolve_paths_from_config(None, None, ".coverage", False)
    assert excinfo.value.code == 1
    mock_logger.error.assert_called_once_with(
        "Source code folders not defined (--folders) and not found in pyproject.toml."
    )


@patch("ai_unit_test.cli.logger.debug")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("ai_unit_test.cli.extract_from_pyproject")
def test_resolve_paths_from_config_with_coverage_file(
    mock_extract_from_pyproject: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_logger_debug: MagicMock,
) -> None:
    """
    Tests the _resolve_paths_from_config function when a coverage file is provided.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = (["src"], "tests", ".coverage.test")

    folders, tests_folder, coverage_file = _resolve_paths_from_config(None, None, ".coverage.test", False)

    mock_logger_debug.assert_any_call("Using source folders from pyproject.toml: ['src']")
    mock_logger_debug.assert_any_call("Using tests folder from pyproject.toml: tests")
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file == ".coverage.test"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_multiple_tests(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with multiple test cases.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase):\n"
        "    def test_example(self):\n"
        "        pass\n"
        "    def test_another_example(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_multiple.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_multiple_bases(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with multiple base classes.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase, AnotherBase):\n"
        "    def test_example(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_multiple_bases.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_inheritance(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with inheritance.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class BaseTest(unittest.TestCase):\n"
        "    pass\n"
        "class TestExample(BaseTest):\n"
        "    def test_example(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_inheritance.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.logger.info")
@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.chunk_test_file")
@patch("ai_unit_test.cli.read_file_content")
@pytest.mark.asyncio
async def test_process_missing_info_updates_test_file(
    mock_read_file_content: MagicMock,
    mock_chunk_test_file: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
    mock_logger_info: MagicMock,
) -> None:
    """
    Tests the _process_missing_info function when a test file is found and updated.
    """
    func_path_str = "src/example.py"
    test_path_str = "tests/example.py"
    func_name = "example_function"
    start_line = 140
    end_line = 145
    missing_info = {Path(func_path_str): [140, 141, 142]}
    mock_find_test_file.return_value = Path(test_path_str)
    mock_chunk_test_file.return_value = [
        Chunk(
            name=func_name,
            type="function",
            source_code="def example_function(): pass",
            start_line=start_line,
            end_line=end_line,
            chunk_id="",
            content_hash="",
            text_preview="",
        )
    ]
    mock_read_file_content.return_value = "existing_test_code"
    mock_update_test_with_llm.return_value = "updated_test_code"

    await _process_missing_info(missing_info, "tests")

    mock_logger_info.assert_has_calls(
        calls=[
            call(f"Processing source file: {func_path_str}"),
            call(
                f"Updating {test_path_str} for chunk '{func_name}' "
                f"(lines {start_line}-{end_line}) with uncovered lines: [140, 141, 142]"
            ),
            call(f"✅ Test file updated successfully: {test_path_str}"),
        ]
    )
    assert mock_write_file_content.call_args[0][0] == Path(test_path_str)
    assert "updated_test_code" in mock_write_file_content.call_args[0][1]


@patch("ai_unit_test.cli.logger.info")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("ai_unit_test.cli._process_missing_info")
@patch("pathlib.Path.exists", return_value=True)
def test_main_no_missing_info(
    mock_path_exists: MagicMock,
    mock_process_missing_info: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_logger_info: MagicMock,
) -> None:
    """
    Tests the _main function when there are no files with missing coverage.
    """
    mock_collect_missing_lines.return_value = {}

    asyncio.run(_main(folders=["src"], tests_folder="tests", coverage_file=".coverage", auto=False))

    mock_collect_missing_lines.assert_called_once_with(".coverage")
    mock_process_missing_info.assert_not_called()
    mock_logger_info.assert_has_calls(
        [
            call("Starting AI Unit Test generation process."),
            call("Using source folders: ['src']"),
            call("Using tests folder: tests"),
            call("Using coverage file: .coverage"),
            call("No files with missing coverage 🎉"),
        ]
    )


def test_command_without_arguments() -> None:
    """
    Tests the command when no arguments are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app)
    assert result.exit_code == 2


@patch("ai_unit_test.cli.logger.info")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("ai_unit_test.cli._process_missing_info")
@patch("pathlib.Path.exists", return_value=True)
def test_main_with_missing_info(
    mock_path_exists: MagicMock,
    mock_process_missing_info: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_logger_info: MagicMock,
) -> None:
    """
    Tests the _main function when there are files with missing coverage.
    """
    folder_str = "src"
    tests_folder = "tests"
    missing_info = {Path("src/example.py"): [140, 141, 142]}
    mock_collect_missing_lines.return_value = missing_info

    asyncio.run(_main(folders=[folder_str], tests_folder=tests_folder, coverage_file=".coverage", auto=False))

    mock_collect_missing_lines.assert_called_once_with(".coverage")
    mock_process_missing_info.assert_called_once_with(missing_info, tests_folder)
    mock_logger_info.assert_has_calls(
        [
            call("Starting AI Unit Test generation process."),
            call(f"Using source folders: ['{folder_str}']"),
            call(f"Using tests folder: {tests_folder}"),
            call("Using coverage file: .coverage"),
            call("👉 Found 1 files with missing coverage."),
        ]
    )


@patch("ai_unit_test.cli.logger.error")
def test_main_no_folders_and_no_coverage_file_provided(mock_logger_error: MagicMock) -> None:
    """
    Tests the main command when no folders and no coverage file are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["main"])
    assert result.exit_code == 1
    mock_logger_error.assert_called_with("Coverage file not found: .coverage")


@patch("ai_unit_test.cli.semantic_search")
@patch("ai_unit_test.cli.logger.info")
def test_search_results_found(mock_logger_info: MagicMock, mock_semantic_search: MagicMock) -> None:
    """
    Tests the search function when results are found for the query.
    """
    query = "existing_query"
    index_dir = ".ai_unit_test_cache/faiss_index"
    k = 5
    threshold = 0.7

    # Mocking the semantic_search function to return a result
    mock_semantic_search.return_value = [
        ({"source_filepath": "file.py", "start_line": 1, "end_line": 2, "text_preview": "This is a preview."}, 0.9)
    ]

    search(query, index_dir, k, threshold)

    mock_semantic_search.assert_called_once_with(query, index_dir, k, threshold)
    mock_logger_info.assert_any_call("Found 1 results:")
    mock_logger_info.assert_any_call("  1. Similarity: 0.9000 | file.py:1-2")
    mock_logger_info.assert_any_call("      Preview: This is a preview.")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_from_pyproject_no_source_folders_debug(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the function handles the case when no source folders are specified in the pyproject.toml.
    """
    data = {
        "tool": {
            "coverage": {"run": {"source": []}},
            "pytest": {"ini_options": {"testpaths": ["tests"]}},
        }
    }
    folders, tests_folder, coverage_file = extract_from_pyproject(data)
    assert folders == []
    assert tests_folder == "tests"
    assert coverage_file is None
    mock_logger_debug.assert_any_call("Found source folders: []")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_from_pyproject_no_coverage_file_debug(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the function handles the case when no coverage file is specified in the pyproject.toml.
    """
    data = {
        "tool": {
            "coverage": {"run": {"source": ["src"], "data_file": None}},
            "pytest": {"ini_options": {"testpaths": ["tests"]}},
        }
    }
    folders, tests_folder, coverage_file = extract_from_pyproject(data)
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file is None
    mock_logger_debug.assert_any_call("Found coverage file path: None")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_from_pyproject_no_coverage_file_fallback(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the function correctly handles the case when no coverage file is specified.
    """
    data = {
        "tool": {
            "coverage": {"run": {"source": ["src"], "data_file": None}},
            "pytest": {"ini_options": {"testpaths": ["tests"]}},
        }
    }
    folders, tests_folder, coverage_file = extract_from_pyproject(data)
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file is None
    mock_logger_debug.assert_called_with("Found coverage file path: None")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_test_patterns_from_pyproject_empty(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the default patterns are returned when an empty list is provided in the pyproject.toml.
    """
    data: dict[str, Any] = {"tool": {"ai-unit-test": {"test-patterns": []}}}
    patterns = extract_test_patterns_from_pyproject(data)
    assert patterns == []
    mock_logger_debug.assert_any_call("Found test patterns: []")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_test_patterns_from_pyproject_invalid(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the function returns invalid patterns as-is (the casting allows this).
    """
    data: dict[str, Any] = {"tool": {"ai-unit-test": {"test-patterns": [123, None, True]}}}
    patterns = extract_test_patterns_from_pyproject(data)
    # The function uses cast() so it will return whatever is there
    assert len(patterns) == 3
    mock_logger_debug.assert_any_call("Found test patterns: [123, None, True]")


@patch("ai_unit_test.cli.logger.debug")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("ai_unit_test.cli.extract_from_pyproject")
def test_resolve_paths_from_config_with_no_coverage_file(
    mock_extract_from_pyproject: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_logger_debug: MagicMock,
) -> None:
    """
    Tests the _resolve_paths_from_config function when no coverage file is provided.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = (["src"], "tests", None)

    folders, tests_folder, coverage_file = _resolve_paths_from_config(None, None, "", True)

    mock_logger_debug.assert_any_call("Using source folders from pyproject.toml: ['src']")
    mock_logger_debug.assert_any_call("Using tests folder from pyproject.toml: tests")
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file == ""


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_no_methods(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with no methods.
    """
    mock_read_file_content.return_value = "import unittest\nclass TestExample(unittest.TestCase):\n    pass"
    result = _detect_test_style(Path("tests/test_example_no_methods.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_no_assert(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with no assert statement.
    """
    mock_read_file_content.return_value = "def test_example():\n    pass"
    result = _detect_test_style(Path("tests/test_example_no_assert.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_docstring(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with a docstring.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase):\n"
        '    """Example test case."""\n'
        "    def test_example(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_with_docstring.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_docstring(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with a docstring.
    """
    mock_read_file_content.return_value = 'def test_example():\n    """Example test."""\n    assert True'
    result = _detect_test_style(Path("tests/test_example_pytest_with_docstring.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_setup_and_teardown(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with setUp and tearDown methods.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase):\n"
        "    def setUp(self):\n"
        "        pass\n"
        "    def test_example(self):\n"
        "        pass\n"
        "    def tearDown(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_with_setup_teardown.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_setup_and_teardown(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with setup and teardown.
    """
    mock_read_file_content.return_value = (
        "def setup_function():\n"
        "    pass\n"
        "def test_example():\n"
        "    assert True\n"
        "def teardown_function():\n"
        "    pass"
    )
    result = _detect_test_style(Path("tests/test_example_pytest_with_setup_teardown.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.logger.debug")
def test_load_pyproject_config_logging_when_file_exists(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the debug log is correctly generated when the pyproject.toml file exists.
    """
    mock_path = Path("tests/unit/fake_pyproject.toml")
    load_pyproject_config(mock_path)
    mock_logger_debug.assert_any_call(f"Attempting to load pyproject config from: {mock_path}")
    mock_logger_debug.assert_any_call("pyproject.toml loaded successfully.")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_from_pyproject_no_tests_folder_and_no_source_folders(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the function handles the case when no tests folder and no
    source folders are specified in the pyproject.toml.
    """
    data = {
        "tool": {
            "coverage": {"run": {"source": []}},
            "pytest": {"ini_options": {"testpaths": [None]}},
        }
    }
    folders, tests_folder, coverage_file = extract_from_pyproject(data)
    assert folders == []
    assert tests_folder == "tests"
    assert coverage_file is None
    mock_logger_debug.assert_any_call("Found source folders: []")


@patch("ai_unit_test.cli.logger.debug")
def test_load_pyproject_config_file_not_found_logging(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the debug log is correctly generated when the pyproject.toml file is not found.
    """
    mock_path = Path("non_existent_file.toml")
    config = load_pyproject_config(mock_path)
    assert config == {}
    mock_logger_debug.assert_called_with("pyproject.toml not found.")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_test_patterns_from_pyproject_default(mock_logger_debug: MagicMock) -> None:
    """
    Tests that the default test patterns are returned when no test
    patterns are specified in the pyproject.toml.
    """
    data = {"tool": {"ai-unit-test": {}}}  # type: ignore[var-annotated]
    patterns = extract_test_patterns_from_pyproject(data)
    assert patterns == ["test_*.py", "*_test.py"]
    mock_logger_debug.assert_called_with("Found test patterns: ['test_*.py', '*_test.py']")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_test_patterns_from_pyproject_single_pattern(mock_logger_debug: MagicMock) -> None:
    """
    Tests that a single test pattern is returned correctly from the pyproject.toml.
    """
    data = {"tool": {"ai-unit-test": {"test-patterns": ["test_single.py"]}}}
    patterns = extract_test_patterns_from_pyproject(data)
    assert patterns == ["test_single.py"]
    mock_logger_debug.assert_called_with("Found test patterns: ['test_single.py']")


@patch("ai_unit_test.cli.logger.debug")
def test_extract_test_patterns_from_pyproject_multiple_patterns(mock_logger_debug: MagicMock) -> None:
    """
    Tests that multiple test patterns are returned correctly from the pyproject.toml.
    """
    data = {"tool": {"ai-unit-test": {"test-patterns": ["test_*.py", "check_*.py"]}}}
    patterns = extract_test_patterns_from_pyproject(data)
    assert patterns == ["test_*.py", "check_*.py"]
    mock_logger_debug.assert_called_with("Found test patterns: ['test_*.py', 'check_*.py']")


@patch("ai_unit_test.cli.logger.error")
@patch("ai_unit_test.cli.extract_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
def test_resolve_paths_from_config_no_folders(
    mock_load_pyproject_config: MagicMock, mock_extract_from_pyproject: MagicMock, mock_logger_error: MagicMock
) -> None:
    """
    Tests the _resolve_paths_from_config function when no folders are provided.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = ([], "tests", ".coverage")

    with pytest.raises(SystemExit) as excinfo:
        _resolve_paths_from_config(None, "tests", ".coverage", False)
    assert excinfo.value.code == 1
    mock_logger_error.assert_called_once_with(
        "Source code folders not defined (--folders) and not found in pyproject.toml."
    )


@patch("ai_unit_test.cli.logger.error")
@patch("ai_unit_test.cli.extract_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
def test_resolve_paths_from_config_no_tests_folder(
    mock_load_pyproject_config: MagicMock, mock_extract_from_pyproject: MagicMock, mock_logger_error: MagicMock
) -> None:
    """
    Tests the _resolve_paths_from_config function when no tests_folder is provided.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = (["src"], None, ".coverage")

    with pytest.raises(SystemExit) as excinfo:
        _resolve_paths_from_config(["src"], None, ".coverage", False)
    assert excinfo.value.code == 1
    mock_logger_error.assert_called_once_with(
        "Tests folder not defined (--tests-folder) and not found in pyproject.toml."
    )


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_multiple_methods(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with multiple methods.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase):\n"
        "    def test_one(self):\n"
        "        pass\n"
        "    def test_two(self):\n"
        "        pass\n"
        "    def test_three(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_multiple_methods.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_multiple_asserts(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with multiple assert statements.
    """
    mock_read_file_content.return_value = "def test_example():\n" "    assert 1 == 1\n" "    assert 2 == 2\n"
    result = _detect_test_style(Path("tests/test_example_pytest_multiple_asserts.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class_with_class_docstring(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with a class docstring.
    """
    mock_read_file_content.return_value = (
        "import unittest\n"
        "class TestExample(unittest.TestCase):\n"
        '    """This is a test case."""\n'
        "    def test_example(self):\n"
        "        pass"
    )
    result = _detect_test_style(Path("tests/test_example_class_docstring.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_class_docstring(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with a function docstring.
    """
    mock_read_file_content.return_value = (
        "def test_example():\n" '    """This is a test function."""\n' "    assert True"
    )
    result = _detect_test_style(Path("tests/test_example_pytest_function_docstring.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.logger.info")
@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.chunk_test_file")
@patch("ai_unit_test.cli.read_file_content")
@pytest.mark.asyncio
async def test_process_missing_info_no_uncovered_lines(
    mock_read_file_content: MagicMock,
    mock_chunk_test_file: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
    mock_logger_info: MagicMock,
) -> None:
    """
    Tests the _process_missing_info function when there are no uncovered lines in the chunk.
    """
    func_path_str = "src/example.py"
    test_path_str = "tests/example.py"
    func_name = "example_function"
    missing_info = {Path(func_path_str): [140, 141, 142]}
    mock_find_test_file.return_value = Path(test_path_str)
    mock_chunk_test_file.return_value = [
        Chunk(
            name=func_name,
            type="function",
            source_code="def example_function(): pass",
            start_line=150,
            end_line=155,
            chunk_id="",
            content_hash="",
            text_preview="",
        )
    ]
    mock_read_file_content.return_value = "existing_test_code"

    await _process_missing_info(missing_info, "tests")

    mock_logger_info.assert_called_once_with(f"Processing source file: {func_path_str}")
    mock_write_file_content.assert_not_called()


@patch("ai_unit_test.cli.logger.info")
@patch("ai_unit_test.cli.save_faiss_index")
@patch("ai_unit_test.cli.generate_embeddings")
@patch("ai_unit_test.cli.find_all_test_files")
@patch("ai_unit_test.cli.chunk_test_file")
@patch("ai_unit_test.cli.extract_test_patterns_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
def test_index_with_found_test_files(
    mock_load_pyproject_config: MagicMock,
    mock_extract_test_patterns_from_pyproject: MagicMock,
    mock_chunk_test_file: MagicMock,
    mock_find_all_test_files: MagicMock,
    mock_generate_embeddings: MagicMock,
    mock_save_faiss_index: MagicMock,
    mock_logger_info: MagicMock,
) -> None:
    """
    Tests the index function when test files are found and indexed successfully.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_test_patterns_from_pyproject.return_value = ["test_*.py"]
    mock_find_all_test_files.return_value = [Path("tests/test_file.py")]
    mock_chunk_test_file.return_value = [
        Chunk(
            name="test_chunk",
            type="function",
            source_code="def test_chunk(): pass",
            start_line=1,
            end_line=1,
            chunk_id="",
            content_hash="",
            text_preview="",
        )
    ]
    mock_generate_embeddings.return_value = ["embedding1"]

    index("tests_folder")

    mock_logger_info.assert_any_call("Found 1 test files to index.")
    mock_save_faiss_index.assert_called_once()


@patch("ai_unit_test.cli.semantic_search")
@patch("ai_unit_test.cli.logger.info")
def test_search_results_found_multiple(mock_logger_info: MagicMock, mock_semantic_search: MagicMock) -> None:
    """
    Tests the search function when multiple results are found for the query.
    """
    query = "existing_query"
    index_dir = ".ai_unit_test_cache/faiss_index"
    k = 5
    threshold = 0.7

    # Mocking the semantic_search function to return multiple results
    mock_semantic_search.return_value = [
        ({"source_filepath": "file1.py", "start_line": 1, "end_line": 2, "text_preview": "Preview 1."}, 0.9),
        ({"source_filepath": "file2.py", "start_line": 3, "end_line": 4, "text_preview": "Preview 2."}, 0.85),
    ]

    search(query, index_dir, k, threshold)

    mock_semantic_search.assert_called_once_with(query, index_dir, k, threshold)
    mock_logger_info.assert_any_call("Found 2 results:")
    mock_logger_info.assert_any_call("  1. Similarity: 0.9000 | file1.py:1-2")
    mock_logger_info.assert_any_call("      Preview: Preview 1.")
    mock_logger_info.assert_any_call("  2. Similarity: 0.8500 | file2.py:3-4")
    mock_logger_info.assert_any_call("      Preview: Preview 2.")
