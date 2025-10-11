"""Test cases for the coverage helper module."""

import logging
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, mock_open, patch

from ai_unit_test.coverage_helper import collect_missing_lines

logger = logging.getLogger(__name__)


def setup_coverage_mocks(mock_cov_instance: MagicMock, mock_json_load: MagicMock, report_data: dict[str, Any]) -> None:
    """Set up common coverage mocks."""
    mock_cov_instance.load.return_value = None
    mock_cov_instance.combine.return_value = None
    mock_cov_instance.json_report.return_value = None
    mock_json_load.return_value = report_data


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies missing lines."""
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/main.py": {"missing_lines": [2, 4]},
            "src/another_file.py": {"missing_lines": []},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    # Executa a função
    missing_info = collect_missing_lines("fake.coverage")

    # Assertions
    assert len(missing_info) == 1
    assert Path("src/main.py") in missing_info
    assert missing_info[Path("src/main.py")] == [2, 4]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_no_missing(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines returns an empty dict when no missing lines."""
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/main.py": {"missing_lines": []},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 0
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_single_file(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies missing lines for a single file."""
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/single_file.py": {"missing_lines": [10, 12]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/single_file.py") in missing_info
    assert missing_info[Path("src/single_file.py")] == [10, 12]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_multiple_files(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies missing lines for multiple files."""
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/file_one.py": {"missing_lines": [1, 3, 5]},
            "src/file_two.py": {"missing_lines": [2]},
            "src/file_three.py": {"missing_lines": []},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [1, 3, 5]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [2]
    assert Path("src/file_three.py") not in missing_info
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_multiple_files_with_all_missing(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies missing lines for multiple files.

    All missing lines are identified.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/file_one.py": {"missing_lines": [1, 2, 3]},
            "src/file_two.py": {"missing_lines": [4, 5]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [1, 2, 3]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [4, 5]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_multiple_files_with_some_missing(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies missing lines for multiple files.

    Some missing lines are identified.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/file_a.py": {"missing_lines": [12, 13, 14]},
            "src/file_b.py": {"missing_lines": []},
            "src/file_c.py": {"missing_lines": [15, 16, 17, 18]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_a.py") in missing_info
    assert missing_info[Path("src/file_a.py")] == [12, 13, 14]
    assert Path("src/file_c.py") in missing_info
    assert missing_info[Path("src/file_c.py")] == [15, 16, 17, 18]
    assert Path("src/file_b.py") not in missing_info
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_multiple_missing_lines(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies multiple missing lines.

    In a single file.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/multiple_missing.py": {"missing_lines": [12, 13, 14, 15, 16]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/multiple_missing.py") in missing_info
    assert missing_info[Path("src/multiple_missing.py")] == [12, 13, 14, 15, 16]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_no_measured_files(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines returns an empty dict.

    When there are no measured files.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {"files": {}}
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 0
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_specific_missing_lines(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies specific missing lines.

    In a file.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/specific_missing.py": {"missing_lines": [12, 13, 14, 15, 16, 17, 18, 19]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/specific_missing.py") in missing_info
    assert missing_info[Path("src/specific_missing.py")] == [12, 13, 14, 15, 16, 17, 18, 19]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_all_lines_missing(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies all missing lines.

    In a file.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/all_missing.py": {"missing_lines": [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/all_missing.py") in missing_info
    assert missing_info[Path("src/all_missing.py")] == [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_all_lines_missing_in_file(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies all missing lines.

    In a specific file.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/another_all_missing.py": {
                "missing_lines": [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]
            },
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/another_all_missing.py") in missing_info
    assert missing_info[Path("src/another_all_missing.py")] == [
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        23,
        24,
        25,
        26,
        27,
        29,
        30,
        31,
    ]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_specific_missing_lines_multiple(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies specific missing lines.

    In multiple files.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/file_one.py": {"missing_lines": [12, 13, 14]},
            "src/file_two.py": {"missing_lines": [15, 16, 17, 18, 19]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [12, 13, 14]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [15, 16, 17, 18, 19]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_all_lines_missing_in_file_multiple(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly identifies all missing lines.

    In multiple files.
    """
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/file_one.py": {"missing_lines": [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]},
            "src/file_two.py": {"missing_lines": [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=None)
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
@patch("ai_unit_test.coverage_helper.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
@patch("ai_unit_test.coverage_helper.json.load")
def test_collect_missing_lines_with_source_folders_and_empty_files(
    mock_json_load: MagicMock,
    mock_file_open: MagicMock,
    mock_exists: MagicMock,
    mock_coverage_class: MagicMock,
) -> None:
    """Tests that collect_missing_lines correctly handles source folders and empty files."""
    mock_cov_instance = mock_coverage_class.return_value

    report_data: dict[str, Any] = {
        "files": {
            "src/included.py": {"missing_lines": [2, 5]},
            "src/empty.py": {"missing_lines": []},
            "other/outside.py": {"missing_lines": [10]},
        }
    }
    setup_coverage_mocks(mock_cov_instance, mock_json_load, report_data)

    missing_info = collect_missing_lines("fake.coverage", source_folders=["src"])

    assert len(missing_info) == 1
    assert Path("src/included.py") in missing_info
    assert missing_info[Path("src/included.py")] == [2, 5]
    assert Path("src/empty.py") not in missing_info
    assert Path("other/outside.py") not in missing_info
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage", source=["src"])
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.combine.assert_called_once()
    mock_cov_instance.json_report.assert_called_once()
