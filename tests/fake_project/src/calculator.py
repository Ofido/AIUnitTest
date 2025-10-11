"""Calculator module for testing."""


class Calculator:
    """A simple calculator class."""

    def __init__(self) -> None:
        """Initialize calculator."""
        self.history: list[str] = []

    def add(self, a: int | float, b: int | float) -> int | float:
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: int | float, b: int | float) -> int | float:
        """Subtract second number from first."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: int | float, b: int | float) -> int | float:
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: int | float, b: int | float) -> int | float:
        """Divide first number by second."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def power(self, base: int | float, exponent: int | float) -> int | float:
        """Raise base to the power of exponent."""
        result = base**exponent
        self.history.append(f"{base} ** {exponent} = {result}")
        return result

    def clear_history(self) -> None:
        """Clear calculation history."""
        self.history.clear()

    def get_history(self) -> list[str]:
        """Get calculation history."""
        return self.history.copy()
