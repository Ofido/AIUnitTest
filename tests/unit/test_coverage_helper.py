from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_unit_test.coverage_helper import collect_missing_lines


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
