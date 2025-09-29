"""Helper functions for coverage analysis."""

import json
import logging
from pathlib import Path

from coverage import Coverage

# Tests expect the logger to be named "__main__"
logger = logging.getLogger(__name__)


def collect_missing_lines(data_file: str, source_folders: list[str] | None = None) -> dict[Path, list[int]]:
    """Return a mapping {file: [lines without coverage]} using the .coverage file.

    Args:
        data_file: Path to the .coverage file
        source_folders: Optional list of source directories to filter results
    """
    logger.debug(f"Collecting missing lines from {data_file}")
    cov = Coverage(data_file=data_file, source=source_folders)
    cov.load()
    cov.combine()
    missing: dict[Path, list[int]] = {}

    # Generate a JSON report to a temporary file
    report_file = Path(data_file).with_suffix(".json")
    try:
        cov.json_report(outfile=str(report_file))
    except Exception as e:
        logger.warning(f"Could not generate coverage report: {e}")
        return missing

    if not report_file.exists():
        logger.warning("Coverage report not found")
        return missing

    with open(report_file) as f:
        report_data = json.load(f)
    logger.debug(f"Report data: {report_data}")

    for file_path_str, file_data in report_data["files"].items():
        file_path = Path(file_path_str)

        # Filter by source folders if provided
        if source_folders:
            logger.debug(f"Source folders: {source_folders}")
            is_in_source = any(file_path.resolve().is_relative_to(Path(folder).resolve()) for folder in source_folders)
            logger.debug(f"File: {file_path}, Is in source: {is_in_source}")
            if not is_in_source:
                continue

        missing_lines = file_data["missing_lines"]
        if missing_lines:
            logger.debug(f"Found {len(missing_lines)} missing lines in {file_path_str}")
            missing[file_path] = missing_lines
        else:
            logger.debug(f"No missing lines found in {file_path_str}")

    logger.info(f"Found {len(missing)} files with missing lines")
    return missing
