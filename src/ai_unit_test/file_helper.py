"""Helper functions for file operations."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_test_file(source_file_path: str, tests_folder: str) -> Path | None:
    """Find the corresponding test file for a given source file."""
    source_file = Path(source_file_path)
    test_file_name = f"test_{source_file.name}"
    # Look for the test file in the tests_folder and its subdirectories
    test_files = list(Path(tests_folder).rglob(test_file_name))
    if test_files:
        return test_files[0]
    return None


def find_relevant_tests(source_file_path: str, tests_folder: str) -> str:
    """Find the most relevant test file for a given source file and return its content.

    The primary strategy is to find a test file with a similar name.
    """
    test_file_path = find_test_file(source_file_path, tests_folder)
    if test_file_path:
        return read_file_content(test_file_path)
    return ""


def read_file_content(file_path: Path | str) -> str:
    """Read the content of a file."""
    try:
        with open(file_path) as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return ""


def write_file_content(file_path: Path, content: str, mode: str = "w") -> None:
    """Write content to a file."""
    if mode not in ["w", "a", "w+", "a+"]:
        raise ValueError(f"Invalid mode: {mode}")
    with open(file_path, mode) as f:
        f.write(content)


def insert_new_test(existing_content: str, new_test: str) -> str:
    """Insert a new test into the existing content.

    The new test is inserted before the `if __name__ == "__main__":` block if it exists.
    """
    # Handle both single and double-quoted main guards
    candidates = [
        "if __name__ == '__main__':",
        'if __name__ == "__main__":',
    ]
    guard_idx = -1
    guard_text = None
    for cand in candidates:
        idx = existing_content.find(cand)
        if idx != -1 and (guard_idx == -1 or idx < guard_idx):
            guard_idx = idx
            guard_text = cand

    new_test_clean = new_test.strip()

    if guard_idx != -1 and guard_text is not None:
        before = existing_content[:guard_idx]
        after = existing_content[guard_idx + len(guard_text) :]
        # Keep existing whitespace before the guard and insert the new test with spacing
        result = before + "\n\n" + new_test_clean + "\n\n" + "\n" + guard_text + after
        return result

    # No main guard - append with proper spacing
    if existing_content.strip():
        return existing_content.rstrip() + "\n\n" + new_test_clean
    else:
        return "\n" + new_test_clean
