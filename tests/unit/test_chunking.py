from pathlib import Path

from ai_unit_test.chunking import chunk_test_file


def test_chunk_test_file() -> None:
    # Create a dummy test file
    dummy_test_file = Path("dummy_test.py")
    dummy_test_file.write_text(
        """def test_addition():
    assert 1 + 1 == 2

class TestClass:
    def test_subtraction(self):
        assert 2 - 1 == 1
"""
    )

    chunks = chunk_test_file(str(dummy_test_file))

    assert len(chunks) == 2
    assert chunks[0].name == "def test_addition():"
    assert chunks[1].name == "class TestClass:"

    dummy_test_file.unlink()
