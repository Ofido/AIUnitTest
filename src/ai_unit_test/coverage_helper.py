import logging
from pathlib import Path

from coverage import Coverage

logger = logging.getLogger(__name__)


def collect_missing_lines(data_file: str) -> dict[Path, list[int]]:
    """Returns a mapping {file: [lines without coverage]} using the .coverage file."""
    logger.debug(f"Collecting missing lines from {data_file}")
    cov = Coverage(data_file=data_file)
    cov.load()
    missing: dict[Path, list[int]] = {}
    measured_files = cov.get_data().measured_files()
    logger.debug(f"Measured files by coverage: {measured_files}")
    for file_path_str in measured_files:
        logger.debug(f"Processing measured file: {file_path_str}")
        # The method analysis2 is marked as private by the library, but it is the best way to get the missing lines.
        # The official API does not provide a direct way to get the missing lines for a specific file.
        # The analysis method returns (statements, excluded, missing, annotate_html)
        _, _, missing_lines, _ = cov.analysis(file_path_str)
        logger.debug(f"Analysis for {file_path_str}: missing_lines={missing_lines}")
        if missing_lines:
            logger.debug(f"Found {len(missing_lines)} missing lines in {file_path_str}")
            missing[Path(file_path_str)] = missing_lines
        else:
            logger.debug(f"No missing lines found in {file_path_str}")
    logger.info(f"Found {len(missing)} files with missing lines")
    return missing
