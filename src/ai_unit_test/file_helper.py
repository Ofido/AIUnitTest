import ast
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def find_test_file(source_file_path: str, tests_folder: str) -> Path | None:
    """Finds the corresponding test file for a given source file."""
    source_file = Path(source_file_path)
    test_file_name = f"test_{source_file.name}"
    # Look for the test file in the tests_folder and its subdirectories
    test_files = list(Path(tests_folder).rglob(test_file_name))
    if test_files:
        return test_files[0]
    return None


def find_relevant_tests(source_file_path: str, tests_folder: str) -> str:
    """
    Finds the most relevant test file for a given source file and returns its content.
    The primary strategy is to find a test file with a similar name.
    """
    test_file_path = find_test_file(source_file_path, tests_folder)
    if test_file_path:
        return read_file_content(test_file_path)
    return ""


def read_file_content(file_path: Path | str) -> str:
    """Reads the content of a file."""
    try:
        with open(file_path) as f:
            return f.read()
    except (FileNotFoundError, PermissionError) as e:
        logger.warning(f"Error reading file {file_path}: {e}")
        return ""


def write_file_content(file_path: Path, content: str, mode: str = "w") -> None:
    """Writes content to a file."""
    if mode not in ["w", "a", "w+", "a+"]:
        raise ValueError(f"Invalid mode: {mode}")
    with open(file_path, mode) as f:
        f.write(content)


def insert_new_test(existing_content: str, new_test: str) -> str:
    """
    Inserts a new test into the existing content, before the `if __name__ == "__main__":` block if it exists.
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
        result = before + "\n\n" + new_test_clean + "\n\n" + guard_text + after
        return result

    # No main guard - append with proper spacing
    if existing_content.strip():
        return existing_content.rstrip() + "\n\n" + new_test_clean
    else:
        return "\n" + new_test_clean


def extract_function_source(file_path: str, function_name: str) -> str | None:
    """Extracts the source code of a specific function from a file."""
    try:
        with open(file_path) as f:
            file_content = f.read()
            tree = ast.parse(file_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return ast.get_source_segment(file_content, node)
    except (FileNotFoundError, SyntaxError) as e:
        logger.error(f"Error reading or parsing {file_path}: {e}")
    return None


def find_all_test_files(tests_folder: str, patterns: list[str]) -> list[Path]:
    """Finds all test files in a given directory, based on a list of glob patterns."""
    test_files: list[Path] = []
    for pattern in patterns:
        test_files.extend(list(Path(tests_folder).rglob(pattern)))
    return test_files
