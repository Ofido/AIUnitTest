from src.simple_math import add


def test_add() -> None:
    assert add(1, 2) == 3  # nosec
    assert add(-1, 1) == 0  # nosec
