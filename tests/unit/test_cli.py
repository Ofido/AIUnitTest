import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from ai_unit_test.cli import _main, app, extract_from_pyproject, load_pyproject_config


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
@patch("ai_unit_test.cli.read_file_content")
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("ai_unit_test.cli.extract_from_pyproject")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
def test_main_auto_discovery(
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_extract_from_pyproject: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_read_file_content: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
) -> None:
    """Tests the _main function with auto-discovery enabled."""
    mock_extract_from_pyproject.return_value = (["src"], "tests", ".coverage")
    mock_collect_missing_lines.return_value = {"src/main.py": ["1-10"]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.side_effect = lambda path: {
        "src/main.py": "source_code",
        Path("tests/test_main.py"): "test_code",
    }.get(path)
    mock_update_test_with_llm.return_value = "updated_test_code"

    asyncio.run(_main(auto=True, folders=["src"], tests_folder="tests", coverage_file=".coverage"))

    mock_load_pyproject_config.assert_called_once()
    mock_collect_missing_lines.assert_called_once_with(".coverage")
    mock_find_test_file.assert_called_once_with("src/main.py", "tests")
    mock_read_file_content.assert_any_call("src/main.py")
    mock_read_file_content.assert_any_call(Path("tests/test_main.py"))
    mock_update_test_with_llm.assert_called_once()
    mock_write_file_content.assert_called_once_with(Path("tests/test_main.py"), "updated_test_code")


@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.read_file_content")
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.collect_missing_lines")
@patch("pathlib.Path.exists", return_value=True)
@patch("pathlib.Path.rglob", return_value=[])
def test_main_explicit_args(
    mock_rglob: MagicMock,
    mock_path_exists: MagicMock,
    mock_collect_missing_lines: MagicMock,
    mock_find_test_file: MagicMock,
    mock_read_file_content: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
) -> None:
    """Tests the _main function with explicit arguments."""
    mock_collect_missing_lines.return_value = {"src/main.py": ["1-10"]}
    mock_find_test_file.return_value = Path("tests/test_main.py")
    mock_read_file_content.side_effect = lambda path: {
        "src/main.py": "source_code",
        Path("tests/test_main.py"): "test_code",
    }.get(path)
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
    mock_find_test_file.assert_called_once_with("src/main.py", "tests")
    mock_read_file_content.assert_any_call("src/main.py")
    mock_read_file_content.assert_any_call(Path("tests/test_main.py"))
    mock_update_test_with_llm.assert_called_once()
    mock_write_file_content.assert_called_once_with(Path("tests/test_main.py"), "updated_test_code")


@patch("ai_unit_test.cli.write_file_content")
@patch("ai_unit_test.cli.update_test_with_llm", new_callable=AsyncMock)
@patch("ai_unit_test.cli.read_file_content")
@patch("ai_unit_test.cli.find_test_file")
@patch("ai_unit_test.cli.extract_function_source")
@patch("ai_unit_test.cli.load_pyproject_config")
@patch("pathlib.Path.rglob", return_value=[])
def test_func_command(
    mock_rglob: MagicMock,
    mock_load_pyproject_config: MagicMock,
    mock_extract_function_source: MagicMock,
    mock_find_test_file: MagicMock,
    mock_read_file_content: MagicMock,
    mock_update_test_with_llm: AsyncMock,
    mock_write_file_content: MagicMock,
) -> None:
    """Tests the func command."""
    runner = CliRunner()
    mock_extract_function_source.return_value = "source_code"
    mock_find_test_file.return_value = Path("tests/test_simple_math.py")
    mock_read_file_content.return_value = "test_code"
    mock_update_test_with_llm.return_value = "updated_test_code"

    result = runner.invoke(
        app,
        [
            "func",
            "src/simple_math.py",
            "add",
            "--tests-folder",
            "tests",
        ],
    )

    assert result.exit_code == 0
    mock_extract_function_source.assert_called_once_with("src/simple_math.py", "add")
    mock_find_test_file.assert_called_once_with("src/simple_math.py", "tests")
    mock_read_file_content.assert_called_once_with(Path("tests/test_simple_math.py"))
    mock_update_test_with_llm.assert_called_once()
    mock_write_file_content.assert_called_once_with(Path("tests/test_simple_math.py"), "updated_test_code")
