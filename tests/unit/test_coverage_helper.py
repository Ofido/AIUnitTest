import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_unit_test.coverage_helper import collect_missing_lines

logger = logging.getLogger(__name__)


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines(mock_coverage_class: MagicMock) -> None:
    """
    Tests that collect_missing_lines correctly identifies missing lines.
    """
    # Mock the Coverage object and its methods
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/main.py",
        "src/another_file.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        ("", "", [2, 4], ""),  # Missing lines for src/main.py
        ("", "", [], ""),  # No missing lines for src/another_file.py
    ]

    # Call the function under test
    missing_info = collect_missing_lines("fake.coverage")

    # Assertions
    assert len(missing_info) == 1
    assert Path("src/main.py") in missing_info
    assert missing_info[Path("src/main.py")] == [2, 4]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 2
    mock_cov_instance.analysis.assert_any_call("src/main.py")
    mock_cov_instance.analysis.assert_any_call("src/another_file.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_no_missing(mock_coverage_class: MagicMock) -> None:
    """
    Tests that collect_missing_lines returns an empty dict when no missing lines.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/main.py"]
    mock_cov_instance.analysis.return_value = ("", "", [], "")

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 0
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/main.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_single_file(mock_coverage_class: MagicMock) -> None:
    """
    Tests that collect_missing_lines correctly identifies missing lines for a single file.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/single_file.py"]
    mock_cov_instance.analysis.return_value = ("", "", [5, 10], "")

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/single_file.py") in missing_info
    assert missing_info[Path("src/single_file.py")] == [5, 10]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/single_file.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_multiple_files(mock_coverage_class: MagicMock) -> None:
    """
    Tests that collect_missing_lines correctly identifies missing lines for multiple files.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/file_one.py",
        "src/file_two.py",
        "src/file_three.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        ("", "", [1, 3, 5], ""),  # Missing lines for src/file_one.py
        ("", "", [2], ""),  # Missing lines for src/file_two.py
        ("", "", [], ""),  # No missing lines for src/file_three.py
    ]

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [1, 3, 5]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [2]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 3
    mock_cov_instance.analysis.assert_any_call("src/file_one.py")
    mock_cov_instance.analysis.assert_any_call("src/file_two.py")
    mock_cov_instance.analysis.assert_any_call("src/file_three.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_multiple_files_with_all_missing(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies missing lines for multiple files with all missing lines.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/file_one.py",
        "src/file_two.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        ("", "", [1, 2, 3], ""),  # Missing lines for src/file_one.py
        ("", "", [4, 5], ""),  # Missing lines for src/file_two.py
    ]

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [1, 2, 3]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [4, 5]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 2
    mock_cov_instance.analysis.assert_any_call("src/file_one.py")
    mock_cov_instance.analysis.assert_any_call("src/file_two.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_multiple_files_with_some_missing(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies missing lines for multiple files with some missing lines.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/file_a.py",
        "src/file_b.py",
        "src/file_c.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        ("", "", [12, 13, 14], ""),  # Missing lines for src/file_a.py
        ("", "", [], ""),  # No missing lines for src/file_b.py
        ("", "", [15, 16, 17, 18], ""),  # Missing lines for src/file_c.py
    ]

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_a.py") in missing_info
    assert missing_info[Path("src/file_a.py")] == [12, 13, 14]
    assert Path("src/file_c.py") in missing_info
    assert missing_info[Path("src/file_c.py")] == [15, 16, 17, 18]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 3
    mock_cov_instance.analysis.assert_any_call("src/file_a.py")
    mock_cov_instance.analysis.assert_any_call("src/file_b.py")
    mock_cov_instance.analysis.assert_any_call("src/file_c.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_multiple_missing_lines(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies multiple missing lines in a single file.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/multiple_missing.py"]
    mock_cov_instance.analysis.return_value = ("", "", [12, 13, 14, 15, 16], "")

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/multiple_missing.py") in missing_info
    assert missing_info[Path("src/multiple_missing.py")] == [12, 13, 14, 15, 16]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/multiple_missing.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_no_measured_files(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines returns an empty dict when there are no measured files.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = []

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 0
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_specific_missing_lines(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies specific missing lines in a file.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/specific_missing.py"]
    mock_cov_instance.analysis.return_value = (
        "",
        "",
        [12, 13, 14, 15, 16, 17, 18, 19],
        "",
    )

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/specific_missing.py") in missing_info
    assert missing_info[Path("src/specific_missing.py")] == [
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
    ]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/specific_missing.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_all_lines_missing(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies all missing lines in a file.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/all_missing.py"]
    mock_cov_instance.analysis.return_value = (
        "",
        "",
        [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31],
        "",
    )

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 1
    assert Path("src/all_missing.py") in missing_info
    assert missing_info[Path("src/all_missing.py")] == [
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
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/all_missing.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_all_lines_missing_in_file(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies all missing lines in a specific file.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = ["src/another_all_missing.py"]
    mock_cov_instance.analysis.return_value = (
        "",
        "",
        [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31],
        "",
    )

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
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    mock_cov_instance.analysis.assert_called_once_with("src/another_all_missing.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_specific_missing_lines_multiple(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies specific missing lines in multiple files.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/file_one.py",
        "src/file_two.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        ("", "", [12, 13, 14], ""),  # Missing lines for src/file_one.py
        ("", "", [15, 16, 17, 18, 19], ""),  # Missing lines for src/file_two.py
    ]

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [12, 13, 14]
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [15, 16, 17, 18, 19]
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 2
    mock_cov_instance.analysis.assert_any_call("src/file_one.py")
    mock_cov_instance.analysis.assert_any_call("src/file_two.py")


@patch("ai_unit_test.coverage_helper.Coverage")
def test_collect_missing_lines_with_all_lines_missing_in_file_multiple(
    mock_coverage_class: MagicMock,
) -> None:
    """
    Tests that collect_missing_lines correctly identifies all missing lines in multiple files.
    """
    mock_cov_instance = mock_coverage_class.return_value
    mock_cov_instance.get_data.return_value.measured_files.return_value = [
        "src/file_one.py",
        "src/file_two.py",
    ]
    mock_cov_instance.analysis.side_effect = [
        (
            "",
            "",
            [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31],
            "",
        ),  # Missing lines for src/file_one.py
        (
            "",
            "",
            [12, 13, 14, 15, 16, 17, 18, 19, 23, 24, 25, 26, 27, 29, 30, 31],
            "",
        ),  # Missing lines for src/file_two.py
    ]

    missing_info = collect_missing_lines("fake.coverage")

    assert len(missing_info) == 2
    assert Path("src/file_one.py") in missing_info
    assert missing_info[Path("src/file_one.py")] == [
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
    assert Path("src/file_two.py") in missing_info
    assert missing_info[Path("src/file_two.py")] == [
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
    mock_coverage_class.assert_called_once_with(data_file="fake.coverage")
    mock_cov_instance.load.assert_called_once()
    mock_cov_instance.get_data.assert_called_once()
    assert mock_cov_instance.analysis.call_count == 2
    mock_cov_instance.analysis.assert_any_call("src/file_one.py")
    mock_cov_instance.analysis.assert_any_call("src/file_two.py")
