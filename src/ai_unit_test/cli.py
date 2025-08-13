import ast
import asyncio
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any

import typer

from ai_unit_test.coverage_helper import collect_missing_lines
from ai_unit_test.file_helper import (
    extract_function_source,
    find_relevant_tests,
    find_test_file,
    get_source_code_chunks,
    insert_new_test,
    read_file_content,
    write_file_content,
)
from ai_unit_test.llm import update_test_with_llm

logger = logging.getLogger(__name__)

app = typer.Typer()


PYPROJECT_TOML_PATH = Path("pyproject.toml")


def load_pyproject_config(pyproject_path: Path = PYPROJECT_TOML_PATH) -> dict[str, Any]:
    """Loads the pyproject.toml file, if it exists, otherwise returns an empty dictionary."""
    logger.debug(f"Attempting to load pyproject config from: {pyproject_path}")
    if not pyproject_path.exists():
        logger.debug("pyproject.toml not found.")
        return {}
    with pyproject_path.open("rb") as fp:
        config: dict[str, Any] = tomllib.load(fp)
        logger.debug("pyproject.toml loaded successfully.")
        return config


def extract_from_pyproject(
    data: dict[str, Any],
) -> tuple[list[str], str | None, str | None]:
    """
    Extracts (source_folders, tests_folder, coverage_file)
    from the standard pyproject.toml structure.
    """
    logger.debug("Extracting configuration from pyproject.toml data.")
    folders: list[str] = []
    tests_folder: str | None = None
    coverage_path: str | None = None

    folders = data.get("tool", {}).get("coverage", {}).get("run", {}).get("source", [])
    logger.debug(f"Found source folders: {folders}")

    tests_folder = data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths", [None])[0]
    logger.debug(f"Found tests folder: {tests_folder}")

    if not tests_folder and Path("tests").is_dir():
        logger.debug("No tests folder in pyproject.toml, falling back to 'tests' directory.")
        tests_folder = "tests"

    coverage_path = data.get("tool", {}).get("coverage", {}).get("run", {}).get("data_file")
    logger.debug(f"Found coverage file path: {coverage_path}")

    return folders, tests_folder, coverage_path


def _resolve_paths_from_config(
    folders: list[str] | None,
    tests_folder: str | None,
    coverage_file: str,
    auto: bool,
) -> tuple[list[str], str, str]:
    if auto or not (folders and tests_folder):
        logger.info("Auto-discovery enabled or folders/tests_folder not provided. Loading from pyproject.toml.")
        cfg: dict[str, Any] = load_pyproject_config()
        folders_from_cfg: list[str]
        tests_dir_from_cfg: str | None
        cov_file_from_cfg: str | None
        folders_from_cfg, tests_dir_from_cfg, cov_file_from_cfg = extract_from_pyproject(cfg)

        if not folders:
            folders = folders_from_cfg
            logger.debug(f"Using source folders from pyproject.toml: {folders}")
        if not tests_folder:
            tests_folder = tests_dir_from_cfg
            logger.debug(f"Using tests folder from pyproject.toml: {tests_folder}")
        if coverage_file == ".coverage" and cov_file_from_cfg:
            coverage_file = cov_file_from_cfg
            logger.debug(f"Using coverage file from pyproject.toml: {coverage_file}")

    if not folders:
        logger.error("Source code folders not defined (--folders) and not found in pyproject.toml.")
        sys.exit(1)
    if not tests_folder:
        logger.error("Tests folder not defined (--tests-folder) and not found in pyproject.toml.")
        sys.exit(1)

    return folders, tests_folder, coverage_file


def _detect_test_style(test_file_path: Path) -> str:
    """Detects if the test file uses unittest.TestCase classes or pytest functions."""
    test_content = read_file_content(test_file_path)
    if not test_content:
        return "unknown"

    try:
        tree = ast.parse(test_content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "TestCase":
                        return "unittest_class"
                    elif isinstance(base, ast.Name) and base.id == "TestCase":
                        return "unittest_class"
        # If no TestCase classes are found, assume pytest function style
        return "pytest_function"
    except SyntaxError:
        logger.warning(f"Could not parse test file {test_file_path} for style detection.")
        return "unknown"


async def _process_missing_info(missing_info: dict[Path, list[int]], tests_folder: str) -> None:
    for source_file_path, uncovered_lines_list in missing_info.items():
        logger.info(f"Processing source file: {source_file_path}")
        test_file: Path | None = find_test_file(str(source_file_path), tests_folder)
        if not test_file:
            logger.warning(f"Test file not found for {source_file_path}, skipping.")
            continue

        # Detect test style
        test_style = _detect_test_style(test_file)
        logger.debug(f"Detected test style for {test_file}: {test_style}")

        # Get all logical chunks (classes and functions) from the source file
        code_chunks = get_source_code_chunks(source_file_path)

        other_tests_content = read_file_content(test_file)

        for chunk in code_chunks:
            chunk_uncovered_lines = []
            for line_num in uncovered_lines_list:
                if chunk.start_line <= line_num <= chunk.end_line:
                    chunk_uncovered_lines.append(line_num)

            if not chunk_uncovered_lines:
                continue  # No uncovered lines in this chunk, skip

            logger.info(
                f"Updating {test_file} for chunk '{chunk.name}' "
                f"(lines {chunk.start_line}-{chunk.end_line}) with uncovered lines: {chunk_uncovered_lines}"
            )
            try:
                existing_content = read_file_content(test_file)
                updated_test: str = await update_test_with_llm(
                    chunk.source_code,  # Pass chunk source code
                    existing_content or "",  # Still pass the whole test file for context
                    str(source_file_path),
                    chunk_uncovered_lines,  # Pass chunk-specific uncovered lines
                    other_tests_content,
                    test_style,  # Pass the detected test style
                )
                new_content = insert_new_test(existing_content, updated_test)
                write_file_content(test_file, new_content)  # Overwrite the file with the new content
                logger.info(f"✅ Test file updated successfully: {test_file}")
            except Exception as exc:  # pragma: no cover
                logger.error(f"Error updating {test_file}: {exc}")


async def _main(
    folders: list[str] | None = None,
    tests_folder: str | None = None,
    coverage_file: str = ".coverage",
    auto: bool = False,
) -> None:
    logger.info("Starting AI Unit Test generation process.")
    logger.debug(
        f"Initial parameters: folders={folders}, "
        f"tests_folder={tests_folder}, "
        f"coverage_file={coverage_file}, "
        f"auto={auto}"
    )

    folders, tests_folder, coverage_file = _resolve_paths_from_config(folders, tests_folder, coverage_file, auto)

    logger.info(f"Using source folders: {folders}")
    logger.info(f"Using tests folder: {tests_folder}")
    logger.info(f"Using coverage file: {coverage_file}")

    if not Path(coverage_file).exists():
        logger.error(f"Coverage file not found: {coverage_file}")
        sys.exit(1)

    missing_info = collect_missing_lines(coverage_file)
    if not missing_info:
        logger.info("No files with missing coverage 🎉")
        return
    logger.info(f"👉 Found {len(missing_info)} files with missing coverage.")

    await _process_missing_info(missing_info, tests_folder)


DEFAULT_FOLDERS_OPTION = typer.Option(None, "--folders", help="Source code folders to analyze.")
DEFAULT_TESTS_FOLDER_OPTION = typer.Option(None, "--tests-folder", help="Folder where the tests are located.")
DEFAULT_COVERAGE_FILE_OPTION = typer.Option(".coverage", "--coverage-file", help=".coverage file.")
DEFAULT_AUTO_OPTION = typer.Option(False, "--auto", help="Try to discover folders/tests from pyproject.toml.")
DEFAULT_FILE_PATH_ARGUMENT = typer.Argument(..., help="Path to the source file.")
DEFAULT_FUNCTION_NAME_ARGUMENT = typer.Argument(..., help="Name of the function to test.")


@app.command()
def func(
    file_path: str = DEFAULT_FILE_PATH_ARGUMENT,
    function_name: str = DEFAULT_FUNCTION_NAME_ARGUMENT,
    tests_folder: str | None = DEFAULT_TESTS_FOLDER_OPTION,
    auto: bool = DEFAULT_AUTO_OPTION,
) -> None:
    """
    Generates a test for a specific function in a file.
    """
    logger.info(f"Generating test for function '{function_name}' in file '{file_path}'.")

    if auto or not tests_folder:
        logger.info("Auto-discovery enabled or tests_folder not provided. Loading from pyproject.toml.")
        cfg: dict[str, Any] = load_pyproject_config()
        _, tests_dir_from_cfg, _ = extract_from_pyproject(cfg)

        if not tests_folder:
            tests_folder = tests_dir_from_cfg
            logger.debug(f"Using tests folder from pyproject.toml: {tests_folder}")

    if not tests_folder:
        logger.error("Tests folder not defined (--tests-folder) and not found in pyproject.toml.")
        sys.exit(1)

    source_code = extract_function_source(file_path, function_name)
    if not source_code:
        logger.error(f"Function '{function_name}' not found in '{file_path}'.")
        sys.exit(1)

    test_file: Path | None = find_test_file(file_path, tests_folder)
    if not test_file:
        logger.warning(f"Test file not found for {file_path}, skipping.")
        return

    # Detect test style
    test_style = _detect_test_style(test_file)
    logger.debug(f"Detected test style for {test_file}: {test_style}")

    # Read all other test files for context
    other_tests_content = find_relevant_tests(file_path, tests_folder)

    logger.info(f"Updating {test_file} for function '{function_name}'.")
    try:
        existing_content = read_file_content(test_file)
        updated_test: str = asyncio.run(
            update_test_with_llm(source_code, existing_content, str(file_path), [], other_tests_content, test_style)
        )
        new_content = insert_new_test(existing_content, updated_test)
        write_file_content(test_file, new_content)
        logger.info(f"✅ Test file updated successfully: {test_file}")
    except Exception as exc:  # pragma: no cover
        logger.error(f"Error updating {test_file}: {exc}")


@app.command()
def main(
    folders: list[str] | None = DEFAULT_FOLDERS_OPTION,
    tests_folder: str | None = DEFAULT_TESTS_FOLDER_OPTION,
    coverage_file: str = DEFAULT_COVERAGE_FILE_OPTION,
    auto: bool = DEFAULT_AUTO_OPTION,
) -> None:
    """
    Automatically updates unit tests using the .coverage file and
    the settings declared in pyproject.toml
    """
    logger.debug("CLI 'main' command invoked.")
    asyncio.run(
        _main(
            folders=folders,
            tests_folder=tests_folder,
            coverage_file=coverage_file,
            auto=auto,
        )
    )
