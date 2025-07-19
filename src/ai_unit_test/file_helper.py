import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def find_test_file(source_file: Path, tests_folder: str | Path) -> Path | None:
    """Tries to locate the test file corresponding to a source module."""
    logger.debug(f"Finding test file for {source_file} in {tests_folder}")
    name: str = source_file.stem
    candidates: list[str] = [
        f"test_{name}.py",
        f"{name}_test.py",
        f"test{name}.py",
    ]
    logger.debug(f"Test file candidates: {candidates}")
    tests_folder = Path(tests_folder)
    for root, _, files in os.walk(tests_folder):
        for f in files:
            if f in candidates:
                logger.debug(f"Found test file: {Path(root) / f}")
                return Path(root) / f
    logger.debug(f"Test file not found for {source_file}")
    return None


def read_file_content(path: Path) -> str | None:
    logger.debug(f"Reading file content from {path}")
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    return path.read_text(encoding="utf-8")


def write_file_content(path: Path, content: str) -> None:
    logger.debug(f"Writing file content to {path}")
    path.write_text(content, encoding="utf-8")
