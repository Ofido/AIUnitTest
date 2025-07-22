from pathlib import Path
from unittest.mock import mock_open, patch

from ai_unit_test.file_helper import (
    extract_function_source,
    find_test_file,
    read_file_content,
    write_file_content,
)


def test_find_test_file_found() -> None:
    """
    Tests that find_test_file correctly finds an existing test file.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/test_dummy_source.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/test_dummy_source.py")


def test_find_test_file_not_found() -> None:
    """
    Tests that find_test_file returns None when no test file is found.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/non_existent_source.py", "tests/unit")
        assert test_file is None


def test_read_file_content_exists() -> None:
    """
    Tests that read_file_content correctly reads the content of an existing file.
    """
    with patch("builtins.open", mock_open(read_data="file content")) as mock_file:
        content = read_file_content("dummy.txt")
        mock_file.assert_called_once_with("dummy.txt", "r")
        assert content == "file content"


def test_read_file_content_not_exists() -> None:
    """
    Tests that read_file_content returns an empty string when the file does not exist.
    """
    with patch("builtins.open", side_effect=FileNotFoundError) as mock_file:
        content = read_file_content("non_existent.txt")
        mock_file.assert_called_once_with("non_existent.txt", "r")
        assert content == ""


def test_write_file_content() -> None:
    """
    Tests that write_file_content correctly writes content to a file.
    """
    with patch("builtins.open", mock_open()) as mock_file:
        write_file_content(Path("dummy.txt"), "new content")
        mock_file.assert_called_once_with(Path("dummy.txt"), "w")
        mock_file().write.assert_called_once_with("new content")


def test_extract_function_source_found() -> None:
    """
    Tests that extract_function_source correctly extracts the source of a function.
    """
    file_content = """
def func_a():
    pass

def func_b():
    return 1
"""
    with patch("builtins.open", mock_open(read_data=file_content)):
        source = extract_function_source("dummy.py", "func_b")
        assert source == "def func_b():\n    return 1"


def test_extract_function_source_not_found() -> None:
    """
    Tests that extract_function_source returns None when the function is not found.
    """
    file_content = """
def func_a():
    pass
"""
    with patch("builtins.open", mock_open(read_data=file_content)):
        source = extract_function_source("dummy.py", "non_existent_func")
        assert source is None


def test_extract_function_source_file_not_found() -> None:
    """
    Tests that extract_function_source returns None when the file is not found.
    """
    with patch("builtins.open", side_effect=FileNotFoundError):
        source = extract_function_source("non_existent.py", "func")
        assert source is None


def test_extract_function_source_syntax_error() -> None:
    """
    Tests that extract_function_source returns None when there is a syntax error in the file.
    """
    file_content = """
def func_a(
    pass
"""
    with patch("builtins.open", mock_open(read_data=file_content)):
        source = extract_function_source("dummy.py", "func_a")
        assert source is None
