import asyncio
import logging
import sys
import tomllib
from pathlib import Path
from typing import Any, Optional  # noqa: F401

import typer

from ai_unit_test.coverage_helper import collect_missing_lines
from ai_unit_test.file_helper import find_test_file, read_file_content, write_file_content
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


DEFAULT_FOLDERS_OPTION = typer.Option(None, "--folders", help="Source code folders to analyze.")
DEFAULT_TESTS_FOLDER_OPTION = typer.Option(None, "--tests-folder", help="Folder where the tests are located.")
DEFAULT_COVERAGE_FILE_OPTION = typer.Option(".coverage", "--coverage-file", help=".coverage file.")
DEFAULT_AUTO_OPTION = typer.Option(False, "--auto", help="Try to discover folders/tests from pyproject.toml.")


async def _main(  # noqa: C901
    folders: list[str] | None = DEFAULT_FOLDERS_OPTION,
    tests_folder: str | None = DEFAULT_TESTS_FOLDER_OPTION,
    coverage_file: str = DEFAULT_COVERAGE_FILE_OPTION,
    auto: bool = DEFAULT_AUTO_OPTION,
) -> None:
    logger.info("Starting AI Unit Test generation process.")
    logger.debug(
        f"Initial parameters: folders={folders}, "
        f"tests_folder={tests_folder}, "
        f"coverage_file={coverage_file}, "
        f"auto={auto}"
    )

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

    for source_file_path, uncovered_lines_list in missing_info.items():
        logger.info(f"Processing source file: {source_file_path}")
        test_file: Path | None = find_test_file(source_file_path, tests_folder)
        if not test_file:
            logger.warning(f"Test file not found for {source_file_path}, skipping.")
            continue

        source_code: str | None = read_file_content(source_file_path)
        test_code: str = read_file_content(test_file) or ""
        if source_code is None:
            logger.warning(f"Could not read source file {source_file_path}, skipping.")
            continue

        logger.info(f"Updating {test_file} for uncovered lines: {uncovered_lines_list}")
        try:
            updated_test: str = await update_test_with_llm(
                source_code, test_code, str(source_file_path), uncovered_lines_list
            )
            write_file_content(test_file, updated_test)
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
    the settings declared in pyproject.toml.
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
