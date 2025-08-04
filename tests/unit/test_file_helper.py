from pathlib import Path
from unittest.mock import mock_open, patch

from ai_unit_test.file_helper import (
    Chunk,
    extract_function_source,
    find_relevant_tests,
    find_test_file,
    get_source_code_chunks,
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
        mock_file.assert_called_once_with("dummy.txt")
        assert content == "file content"


def test_read_file_content_not_exists() -> None:
    """
    Tests that read_file_content returns an empty string when the file does not exist.
    """
    with patch("builtins.open", side_effect=FileNotFoundError):
        content = read_file_content("non_existent.txt")
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


def test_find_relevant_tests_found() -> None:
    """
    Tests that find_relevant_tests correctly finds relevant tests for a given source file.
    """
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"
    test_file_content = "def test_dummy_source():\n    assert True"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = Path("tests/unit/test_dummy_source.py")
        with patch("ai_unit_test.file_helper.read_file_content", return_value=test_file_content):
            relevant_content = find_relevant_tests(source_file_path, tests_folder)
            assert relevant_content == test_file_content


def test_find_relevant_tests_not_found() -> None:
    """
    Tests that find_relevant_tests returns an empty string when no relevant tests are found.
    """
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = None
        relevant_content = find_relevant_tests(source_file_path, tests_folder)
        assert relevant_content == ""


def test_get_source_code_chunks_valid_file() -> None:
    """
    Tests that get_source_code_chunks correctly extracts classes and functions from a valid Python file.
    """
    file_content = """class MyClass:
    def method_a(self):
        pass

def my_function():
    return 42
"""
    with patch("ai_unit_test.file_helper.read_file_content", return_value=file_content):
        chunks = get_source_code_chunks(Path("dummy.py"))
        assert len(chunks) == 2
        assert chunks[0] == Chunk(
            name="MyClass",
            type="class",
            source_code="class MyClass:\n    def method_a(self):\n        pass",
            start_line=1,
            end_line=3,
        )
        assert chunks[1] == Chunk(
            name="my_function",
            type="function",
            source_code="def my_function():\n    return 42",
            start_line=5,
            end_line=6,
        )


def test_get_source_code_chunks_file_not_found() -> None:
    """
    Tests that get_source_code_chunks handles a FileNotFoundError gracefully.
    """
    with patch("ai_unit_test.file_helper.read_file_content", side_effect=FileNotFoundError):
        chunks = get_source_code_chunks(Path("non_existent.py"))
        assert chunks == []


def test_get_source_code_chunks_syntax_error() -> None:
    """
    Tests that get_source_code_chunks handles a SyntaxError gracefully.
    """
    file_content = """def valid_function():
    return 1

def invalid_function(
    return 2
"""
    with patch("ai_unit_test.file_helper.read_file_content", return_value=file_content):
        chunks = get_source_code_chunks(Path("dummy.py"))
        assert len(chunks) == 0


def test_find_test_file_multiple_found() -> None:
    """
    Tests that find_test_file returns the first found test file when multiple exist.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/test_dummy_source.py"), Path("tests/unit/test_another_source.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/test_dummy_source.py")


def test_find_test_file_empty_path() -> None:
    """
    Tests that find_test_file returns None when the source file path is empty.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("", "tests/unit")
        assert test_file is None


def test_find_test_file_invalid_folder() -> None:
    """
    Tests that find_test_file returns None when the tests folder does not exist.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/dummy_source.py", "invalid_folder")
        assert test_file is None


def test_find_test_file_source_file_not_found() -> None:
    """
    Tests that find_test_file returns None when the source file does not exist.
    """
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/non_existent_source.py", "tests/unit")
        assert test_file is None


def test_find_test_file_test_file_name_format() -> None:
    """
    Tests that find_test_file constructs the correct test file name from the source file name.
    """
    source_file_path = "src/my_script.py"
    expected_test_file_name = "test_my_script.py"
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path(f"tests/unit/{expected_test_file_name}")]
        test_file = find_test_file(source_file_path, "tests/unit")
        assert test_file is not None
        assert test_file.name == expected_test_file_name
