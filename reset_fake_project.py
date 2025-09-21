"""This module provides a function to reset the fake project's test file."""

from pathlib import Path


def reset_test_file() -> None:
    """Reset the content of 'tests/fake_project/tests/test_simple_math.py' to its original state."""
    original_content = """from src.simple_math import add

def test_add() -> None:
    assert add(1, 2) == 3  # nosec
    assert add(-1, 1) == 0  # nosec
"""
    file_path = Path("tests/fake_project/tests/test_simple_math.py")
    file_path.write_text(original_content)
    print(f"File {file_path} reset to original content.")


if __name__ == "__main__":
    reset_test_file()
