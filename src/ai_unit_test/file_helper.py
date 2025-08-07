import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    name: str
    type: Literal["class", "function"]
    source_code: str
    start_line: int
    end_line: int


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
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return ""


def write_file_content(file_path: Path, content: str, mode: str = "w") -> None:
    """Writes content to a file."""
    with open(file_path, mode) as f:
        f.write(content)


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


def get_source_code_chunks(file_path: Path) -> list[Chunk]:
    """
    Extracts top-level classes and functions from a Python file as code chunks.
    """
    chunks: list[Chunk] = []
    try:
        file_content = read_file_content(file_path)
        tree = ast.parse(file_content)

        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                source_segment = ast.get_source_segment(file_content, node)
                if source_segment is not None:
                    chunks.append(
                        Chunk(
                            name=node.name,
                            type="class" if isinstance(node, ast.ClassDef) else "function",
                            source_code=source_segment,
                            start_line=node.lineno,
                            end_line=(
                                node.end_lineno
                                if hasattr(node, "end_lineno") and node.end_lineno is not None
                                else node.lineno
                            ),
                        )
                    )
    except (FileNotFoundError, SyntaxError) as e:
        logger.error(f"Error reading or parsing {file_path}: {e}")
    return chunks
