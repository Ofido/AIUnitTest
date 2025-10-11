"""Test CLI performance."""

import subprocess  # nosec B404
import sys
import time
from pathlib import Path


class TestCLIPerformance:
    """Test CLI performance."""

    def test_cli_startup_time(self, performance_config: dict[str, int]) -> None:
        """Test that the CLI starts up within a reasonable time."""
        # Arrange
        max_startup_time = 10.0

        # Act
        start_time = time.perf_counter()
        result = subprocess.run(  # nosec B603
            [sys.executable, "-m", "ai_unit_test", "--help"],
            capture_output=True,
            text=True,
        )
        end_time = time.perf_counter()

        # Assert
        assert result.returncode == 0
        startup_time = end_time - start_time
        assert startup_time < max_startup_time

    def test_test_generation_performance(self, performance_config: dict[str, int], temp_dir: Path) -> None:
        """Test that the test generation command performs within a reasonable time."""
        # Arrange
        max_generation_time = 60.0  # 1 minute
        fake_project_path = temp_dir / "fake_project"
        fake_project_path.mkdir()
        (fake_project_path / "src").mkdir()
        (fake_project_path / "tests").mkdir()

        source_file = fake_project_path / "src" / "simple_math.py"
        source_file.write_text(
            """
def add(a, b):
    return a + b
"""
        )

        pyproject_toml = fake_project_path / "pyproject.toml"
        pyproject_toml.write_text(
            """
[tool.ai-unit-test]
llm.provider = \"mock\"
"""
        )

        # Act
        start_time = time.perf_counter()
        result = subprocess.run(  # nosec B603
            [sys.executable, "-m", "ai_unit_test", "generate"],
            capture_output=True,
            text=True,
            cwd=fake_project_path,
        )
        end_time = time.perf_counter()

        # Assert
        assert result.returncode == 0, f"CLI command failed with output: {result.stdout}\n{result.stderr}"
        generation_time = end_time - start_time
        assert generation_time < max_generation_time
