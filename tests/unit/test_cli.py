import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from ai_unit_test.cli import (
    _detect_test_style,
    _main,
    _resolve_paths_from_config,
    app,
    extract_from_pyproject,
    load_pyproject_config,
)
from ai_unit_test.file_helper import Chunk


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
@patch("ai_unit_test.cli.get_source_code_chunks")
@patch("ai_unit_test.cli.extract_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
def test_main_auto_discovery(
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_extract_from_pyproject: MagicMock,
    mock_get_source_code_chunks: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
    mock_read_file_content: MagicMock,
) -> None:
    """
    Tests the _main function with auto-discovery enabled.
    """
    mock_extract_from_pyproject.return_value = (["src"], "tests", ".coverage")
    mock_collect_missing_lines.return_value = {Path("src/main.py"): [1]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.return_value = "test_code"
    mock_get_source_code_chunks.return_value = [
        Chunk(
            name="main",
            type="function",
            source_code="def main(): pass",
            start_line=1,
            end_line=1,
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
@patch("ai_unit_test.cli.get_source_code_chunks")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
def test_main_explicit_args(
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_get_source_code_chunks: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
    mock_read_file_content: MagicMock,
) -> None:
    """
    Tests the _main function with explicit arguments.
    """
    mock_collect_missing_lines.return_value = {Path("src/main.py"): [1]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.return_value = "test_code"
    mock_get_source_code_chunks.return_value = [
        Chunk(
            name="main",
            type="function",
            source_code="def main(): pass",
            start_line=1,
            end_line=1,
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


@patch("ai_unit_test.cli.logger.error")
@patch("sys.exit")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("ai_unit_test.cli.extract_from_pyproject")
def test_resolve_paths_from_config_no_tests_folder(
    mock_extract_from_pyproject: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_exit: MagicMock,
    mock_logger_error: MagicMock,
) -> None:
    """
    Tests the _resolve_paths_from_config function when no tests_folder is provided.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = (["src"], None, ".coverage")

    folders, tests_folder, coverage_file = _resolve_paths_from_config(["src"], None, ".coverage", True)

    mock_logger_error.assert_called_with("Tests folder not defined (--tests-folder) and not found in pyproject.toml.")

    assert folders == ["src"]
    assert tests_folder is None
    assert coverage_file == ".coverage"


@patch("ai_unit_test.cli.logger.debug")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("ai_unit_test.cli.extract_from_pyproject")
def test_resolve_paths_from_config_with_auto(
    mock_extract_from_pyproject: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_logger_debug: MagicMock,
) -> None:
    """
    Tests the _resolve_paths_from_config function when auto discovery is enabled.
    """
    mock_load_pyproject_config.return_value = {}
    mock_extract_from_pyproject.return_value = (["src"], "tests", ".coverage")

    folders, tests_folder, coverage_file = _resolve_paths_from_config(None, None, ".coverage", True)

    mock_logger_debug.assert_any_call("Using source folders from pyproject.toml: ['src']")
    mock_logger_debug.assert_any_call("Using tests folder from pyproject.toml: tests")
    assert folders == ["src"]
    assert tests_folder == "tests"
    assert coverage_file == ".coverage"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_unittest_class(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes.
    """
    mock_read_file_content.return_value = (
        "import unittest\nclass TestExample(unittest.TestCase):\n    def test_example(self):\n        pass"
    )
    result = _detect_test_style(Path("tests/test_example.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style.
    """
    mock_read_file_content.return_value = "def test_example():\n    assert True"
    result = _detect_test_style(Path("tests/test_example.py"))
    assert result == "pytest_function"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_empty_file(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function when the file is empty.
    """
    mock_read_file_content.return_value = ""
    result = _detect_test_style(Path("tests/test_empty.py"))
    assert result == "unknown"


@patch("ai_unit_test.cli.read_file_content")
@patch("ai_unit_test.cli.logger.warning")
def test_detect_test_style_syntax_error(mock_logger_warning: MagicMock, mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function when there is a syntax error in the file.
    """
    mock_read_file_content.return_value = "def test_example():\n    assert True\n    invalid_syntax("
    result = _detect_test_style(Path("tests/test_syntax_error.py"))
    mock_logger_warning.assert_called_once()
    assert result == "unknown"


@patch("ai_unit_test.cli.logger.error")
def test_main_no_folders_or_tests_folder(mock_logger_error: MagicMock) -> None:
    """
    Tests the main command when no folders or tests_folder are provided.
    """
    runner = CliRunner()
    result = runner.invoke(app, ["main"])
    assert result.exit_code == 1
    mock_logger_error.assert_called_with("Coverage file not found: .coverage")


def test_detect_test_style_unittest_class_inheritance(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for unittest.TestCase classes with inheritance.
    """
    mock_read_file_content.return_value = (
        "import unittest\nclass BaseTest(unittest.TestCase):\n    pass\nclass"
        " TestExample(BaseTest):\n    def test_example(self):\n        pass"
    )
    result = _detect_test_style(Path("tests/test_example.py"))
    assert result == "unittest_class"


@patch("ai_unit_test.cli.read_file_content")
def test_detect_test_style_pytest_function_with_decorator(mock_read_file_content: MagicMock) -> None:
    """
    Tests the _detect_test_style function for pytest function style with a decorator.
    """
    mock_read_file_content.return_value = (
        "@pytest.mark.parametrize('input,expected', [(1, 2)])\ndef"
        " test_example(input, expected):\n    assert input + 1 == expected"
    )
    result = _detect_test_style(Path("tests/test_example.py"))
    assert result == "pytest_function"


@patch("sys.exit")
@patch("ai_unit_test.cli.extract_function_source")
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.logger.warning")
def test_func_command_no_tests_folder(
    mock_logger_warning: MagicMock,
    mock_find_test_file: MagicMock,
    mock_extract_function_source: MagicMock,
    mock_exit: MagicMock,
) -> None:
    """
    Tests the func command when no tests folder is provided.
    """
    runner = CliRunner()
    mock_extract_function_source.return_value = "source_code"
    mock_find_test_file.return_value = None

    result = runner.invoke(
        app,
        [
            "func",
            "src/simple_math.py",
            "add",
        ],
    )

    assert result.exit_code == 0
    mock_logger_warning.assert_called_once_with("Test file not found for src/simple_math.py, skipping.")
    mock_exit.assert_called_once_with(0)
