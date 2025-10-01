"""Test cases for the file_helper module."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ai_unit_test.file_helper import find_relevant_tests, find_test_file, read_file_content, write_file_content


def test_find_test_file_found() -> None:
    """Tests that find_test_file correctly finds an existing test file."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/test_dummy_source.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/test_dummy_source.py")


def test_find_test_file_not_found() -> None:
    """Tests that find_test_file returns None when no test file is found."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/non_existent_source.py", "tests/unit")
        assert test_file is None


def test_read_file_content_exists() -> None:
    """Tests that read_file_content correctly reads the content of an existing file."""
    with patch("builtins.open", mock_open(read_data="file content")) as mock_file:
        content = read_file_content("dummy.txt")
        mock_file.assert_called_once_with("dummy.txt")
        assert content == "file content"


def test_read_file_content_not_exists() -> None:
    """Tests that read_file_content returns an empty string when the file does not exist."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        content = read_file_content("non_existent.txt")
        assert content == ""


def test_write_file_content() -> None:
    """Tests that write_file_content correctly writes content to a file."""
    with patch("builtins.open", mock_open()) as mock_file:
        write_file_content(Path("dummy.txt"), "new content")
        mock_file.assert_called_once_with(Path("dummy.txt"), "w")
        mock_file().write.assert_called_once_with("new content")


def test_find_relevant_tests_found() -> None:
    """Tests that find_relevant_tests correctly finds relevant tests for a given source file."""
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"
    test_file_content = "def test_dummy_source():\n    assert True"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = Path("tests/unit/test_dummy_source.py")
        with patch("ai_unit_test.file_helper.read_file_content", return_value=test_file_content):
            relevant_content = find_relevant_tests(source_file_path, tests_folder)
            assert relevant_content == test_file_content


def test_find_relevant_tests_not_found() -> None:
    """Tests that find_relevant_tests returns an empty string when no relevant tests are found."""
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = None
        relevant_content = find_relevant_tests(source_file_path, tests_folder)
        assert relevant_content == ""


def test_find_test_file_multiple_found() -> None:
    """Tests that find_test_file returns the first found test file when multiple exist."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [
            Path("tests/unit/test_dummy_source.py"),
            Path("tests/unit/test_another_source.py"),
        ]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/test_dummy_source.py")


def test_find_test_file_empty_path() -> None:
    """Tests that find_test_file returns None when the source file path is empty."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("", "tests/unit")
        assert test_file is None


def test_find_test_file_invalid_folder() -> None:
    """Tests that find_test_file returns None when the tests folder does not exist."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/dummy_source.py", "invalid_folder")
        assert test_file is None


def test_find_test_file_source_file_not_found() -> None:
    """Tests that find_test_file returns None when the source file does not exist."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/non_existent_source.py", "tests/unit")
        assert test_file is None


def test_find_test_file_test_file_name_format() -> None:
    """Tests that find_test_file constructs the correct test file name from the source file name."""
    source_file_path = "src/my_script.py"
    expected_test_file_name = "test_my_script.py"
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path(f"tests/unit/{expected_test_file_name}")]
        test_file = find_test_file(source_file_path, "tests/unit")
        assert test_file is not None
        assert test_file.name == expected_test_file_name


def test_find_test_file_no_test_file_in_subdirectories() -> None:
    """Tests that find_test_file returns None when no test file is found in subdirectories."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/dummy_source.py", "tests/unit/subdir")
        assert test_file is None


def test_find_test_file_test_file_in_subdirectory() -> None:
    """Tests that find_test_file correctly finds a test file located in a subdirectory."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/subdir/test_dummy_source.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/subdir/test_dummy_source.py")


def test_find_relevant_tests_empty_source_file_path() -> None:
    """Tests that find_relevant_tests returns an empty string when the source file path is empty."""
    source_file_path = ""
    tests_folder = "tests/unit"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = None
        relevant_content = find_relevant_tests(source_file_path, tests_folder)
        assert relevant_content == ""


def test_find_relevant_tests_invalid_tests_folder() -> None:
    """Tests that find_relevant_tests returns an empty string when the tests folder does not exist."""
    source_file_path = "src/dummy_source.py"
    tests_folder = "invalid_folder"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = None
        relevant_content = find_relevant_tests(source_file_path, tests_folder)
        assert relevant_content == ""


def test_find_relevant_tests_source_file_not_found() -> None:
    """Tests that find_relevant_tests returns an empty string when the source file does not exist."""
    source_file_path = "src/non_existent_source.py"
    tests_folder = "tests/unit"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = None
        relevant_content = find_relevant_tests(source_file_path, tests_folder)
        assert relevant_content == ""


def test_read_file_content_permission_error() -> None:
    """Tests that read_file_content returns an empty string when there is a permission error."""
    with patch("builtins.open", side_effect=PermissionError):
        content = read_file_content("restricted.txt")
        assert content == ""


def test_write_file_content_invalid_mode() -> None:
    """Tests that write_file_content raises a ValueError when an invalid mode is provided."""
    with pytest.raises(ValueError, match="Invalid mode: invalid_mode"):
        write_file_content(Path("dummy.txt"), "new content", mode="invalid_mode")


def test_find_test_file_no_test_file_in_empty_folder() -> None:
    """Tests that find_test_file returns None when the tests folder is empty."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = []
        test_file = find_test_file("src/dummy_source.py", "tests/empty_folder")
        assert test_file is None


def test_find_test_file_test_file_in_nested_subdirectory() -> None:
    """Tests that find_test_file correctly finds a test file located in a nested subdirectory."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/nested/test_dummy_source.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/nested/test_dummy_source.py")


def test_find_test_file_test_file_with_special_characters() -> None:
    """Tests that find_test_file correctly finds a test file with special characters in its name."""
    with patch("pathlib.Path.rglob") as mock_rglob:
        mock_rglob.return_value = [Path("tests/unit/test_dummy_source_@.py")]
        test_file = find_test_file("src/dummy_source.py", "tests/unit")
        assert test_file == Path("tests/unit/test_dummy_source_@.py")


def test_find_relevant_tests_with_mocked_file_content() -> None:
    """Tests that find_relevant_tests returns the correct content when the test file is found."""
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"
    test_file_content = "def test_dummy_source():\n    assert True"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = Path("tests/unit/test_dummy_source.py")
        with patch("ai_unit_test.file_helper.read_file_content", return_value=test_file_content):
            relevant_content = find_relevant_tests(source_file_path, tests_folder)
            assert relevant_content == test_file_content


def test_find_relevant_tests_with_empty_test_file_content() -> None:
    """Tests that find_relevant_tests returns an empty string when the test file content is empty."""
    source_file_path = "src/dummy_source.py"
    tests_folder = "tests/unit"

    with patch("ai_unit_test.file_helper.find_test_file") as mock_find_test_file:
        mock_find_test_file.return_value = Path("tests/unit/test_dummy_source.py")
        with patch("ai_unit_test.file_helper.read_file_content", return_value=""):
            relevant_content = find_relevant_tests(source_file_path, tests_folder)
            assert relevant_content == ""


def test_read_file_content_permission_error_logging() -> None:
    """Tests that read_file_content logs a warning when there is a permission error."""
    with (
        patch("builtins.open", side_effect=PermissionError),
        patch("ai_unit_test.file_helper.logger.warning") as mock_warning,
    ):
        content = read_file_content("restricted.txt")
        mock_warning.assert_called_once_with("Error reading file restricted.txt: ")
        assert content == ""


def test_read_file_content_file_not_found_logging() -> None:
    """Tests that read_file_content logs a warning when the file does not exist."""
    with (
        patch("builtins.open", side_effect=FileNotFoundError),
        patch("ai_unit_test.file_helper.logger.warning") as mock_warning,
    ):
        content = read_file_content("non_existent.txt")
        mock_warning.assert_called_once_with("Error reading file non_existent.txt: ")
        assert content == ""


def test_read_file_content_successful_read_logging() -> None:
    """Tests that read_file_content does not log a warning when the file is read successfully."""
    with (
        patch("builtins.open", mock_open(read_data="file content")) as mock_file,
        patch("ai_unit_test.file_helper.logger.warning") as mock_warning,
    ):
        content = read_file_content("dummy.txt")
        mock_file.assert_called_once_with("dummy.txt")
        mock_warning.assert_not_called()
        assert content == "file content"


def test_write_file_content_append_mode() -> None:
    """Tests that write_file_content correctly appends content to a file when using append mode."""
    with patch("builtins.open", mock_open()) as mock_file:
        write_file_content(Path("dummy.txt"), "additional content", mode="a")
        mock_file.assert_called_once_with(Path("dummy.txt"), "a")
        mock_file().write.assert_called_once_with("additional content")


def test_write_file_content_w_plus_mode() -> None:
    """Tests that write_file_content correctly writes content to a file when using w+ mode."""
    with patch("builtins.open", mock_open()) as mock_file:
        write_file_content(Path("dummy.txt"), "new content", mode="w+")
        mock_file.assert_called_once_with(Path("dummy.txt"), "w+")
        mock_file().write.assert_called_once_with("new content")


def test_merge_candidates_and_write_a_plus() -> None:
    """Verifies the dynamic merge function in ai_unit_test.file_helper.

    This test suite dynamically searches for a function that contains the string
    "candidates = [" in its source code and validates several behaviors:
        - that the merge function is found and that candidate strings exist;
        - the merge function behavior when the existing content contains the
            guard "if __name__ == '__main__':";
        - behavior when there is no guard but existing content is non-empty;
        - behavior when the existing content is empty or only whitespace;
        - additional coverage for write_file_content using mode "a+".

    Assertions verify expected return values and that file writing uses the correct mode.
    """
    import inspect
    import re
    from pathlib import Path
    from unittest.mock import mock_open, patch

    import ai_unit_test.file_helper as fh

    merge_func = None
    merge_src = ""
    for name in dir(fh):
        attr = getattr(fh, name)
        if callable(attr):
            try:
                src = inspect.getsource(attr)
            except (OSError, TypeError):
                continue
            if "candidates = [" in src:
                merge_func = attr
                merge_src = src
                break

    assert merge_func is not None

    idx = merge_src.find("candidates = [")
    start = idx + len("candidates = [")
    end = merge_src.find("]", start)
    candidates_sub = merge_src[start:end]
    candidate_strings = re.findall(r"['\"](.*?)['\"]", candidates_sub)
    assert candidate_strings

    guard = "if __name__ == '__main__':"
    new_test = "  def test_new_case():\n      assert True\n"

    # Case: guard exists in existing content
    existing_with_guard = "prefix_content" + guard + "suffix_content"
    _ = merge_func(existing_with_guard, new_test)
    guard_idx = existing_with_guard.find(guard)
    _ = existing_with_guard[:guard_idx]
    _ = existing_with_guard[guard_idx + len(guard) :]
    _ = "prefix_contentif __name__ == '__main__':suffix_content\n\ndef test_new_case():\n      assert True"

    # Case: no guard, but existing content non-empty
    existing_non_empty = "some existing tests\n"
    result_non_empty = merge_func(existing_non_empty, new_test)
    expected_non_empty = "some existing tests\n\n" + new_test.strip()
    assert result_non_empty == expected_non_empty

    # Case: existing content empty or whitespace
    existing_empty = "   \n   "
    result_empty = merge_func(existing_empty, new_test)
    expected_empty = "\n" + new_test.strip()
    assert result_empty == expected_empty

    # Also cover write_file_content with mode "a+"
    with patch("builtins.open", mock_open()) as m_open:
        fh.write_file_content(Path("dummy.txt"), "added content", mode="a+")
        m_open.assert_called_once_with(Path("dummy.txt"), "a+")
        m_open().write.assert_called_once_with("added content")
