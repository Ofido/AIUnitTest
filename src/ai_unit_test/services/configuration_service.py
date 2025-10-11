"""Service for handling configuration management."""

import logging
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ai_unit_test.core.exceptions import ConfigurationError
from ai_unit_test.services.base_service import BaseService

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentStatus:
    """Represents the status of the environment."""

    python_version: str
    working_directory: str
    pyproject_exists: bool
    tests_directory_exists: bool
    coverage_file_exists: bool
    environment_variables: dict[str, bool]


class ConfigurationService(BaseService):
    """Service for managing application configuration."""

    def __repr__(self) -> str:
        """Return a string representation of the service."""
        return "ConfigurationService"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the ConfigurationService.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self._pyproject_cache: dict[str, Any] | None = None

    def get_service_name(self) -> str:
        """Return the name of this service for logging purposes."""
        return "Configuration"

    def load_pyproject_config(self, pyproject_path: Path | None = None) -> dict[str, Any]:
        """Load and cache pyproject.toml configuration."""
        if pyproject_path is None:
            pyproject_path = Path("pyproject.toml")

        if self._pyproject_cache is not None:
            return self._pyproject_cache

        self.logger.debug(f"Loading pyproject config from: {pyproject_path}")

        if not pyproject_path.exists():
            self.logger.debug("pyproject.toml not found, using empty config")
            self._pyproject_cache = {}
            return self._pyproject_cache

        try:
            with pyproject_path.open("rb") as fp:
                self._pyproject_cache = tomllib.load(fp)
                self.logger.debug("pyproject.toml loaded successfully")
                return self._pyproject_cache
        except Exception as e:
            raise ConfigurationError(f"Failed to load pyproject.toml: {e}")

    def extract_source_configuration(
        self, pyproject_data: dict[str, Any] | None = None
    ) -> tuple[list[str], str | None, str | None]:
        """Extract source folders, tests folder, and coverage file from config."""
        if pyproject_data is None:
            pyproject_data = self.load_pyproject_config()

        self.logger.debug("Extracting source configuration from pyproject.toml")

        # Extract source folders
        folders: list[str] = pyproject_data.get("tool", {}).get("coverage", {}).get("run", {}).get("source", [])
        self.logger.debug(f"Found source folders: {folders}")

        # Extract tests folder
        tests_folder = (
            pyproject_data.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("testpaths", [None])[0]
        )

        # Fallback to 'tests' directory if not specified
        if not tests_folder and Path("tests").is_dir():
            self.logger.debug("No tests folder in config, using 'tests' directory")
            tests_folder = "tests"

        self.logger.debug(f"Found tests folder: {tests_folder}")

        # Extract coverage file path
        coverage_path = pyproject_data.get("tool", {}).get("coverage", {}).get("run", {}).get("data_file")
        self.logger.debug(f"Found coverage file path: {coverage_path}")

        return folders, tests_folder, coverage_path

    def extract_test_patterns(self, pyproject_data: dict[str, Any] | None = None) -> list[str]:
        """Extract test file patterns from configuration."""
        if pyproject_data is None:
            pyproject_data = self.load_pyproject_config()

        patterns: list[str] = (
            pyproject_data.get("tool", {}).get("ai-unit-test", {}).get("test-patterns", ["test_*.py", "*_test.py"])
        )

        self.logger.debug(f"Found test patterns: {patterns}")
        return patterns

    def resolve_paths_from_config(
        self,
        folders: list[str] | None = None,
        tests_folder: str | None = None,
        coverage_file: str = ".coverage",
        auto_discovery: bool = False,
    ) -> tuple[list[str], str, str]:
        """Resolve and validate all path configurations."""
        if auto_discovery or not (folders and tests_folder):
            self.logger.info("Auto-discovery enabled or paths not provided")
            cfg = self.load_pyproject_config()
            cfg_folders, cfg_tests, cfg_coverage = self.extract_source_configuration(cfg)

            if not folders:
                folders = cfg_folders
                self.logger.debug(f"Using source folders from config: {folders}")

            if not tests_folder and cfg_tests:
                tests_folder = cfg_tests
                self.logger.debug(f"Using tests folder from config: {tests_folder}")

            if coverage_file == ".coverage" and cfg_coverage:
                coverage_file = cfg_coverage
                self.logger.debug(f"Using coverage file from config: {coverage_file}")

        # Validate required paths
        if not folders:
            raise ConfigurationError(
                "Source folders not defined (--folders) and not found in pyproject.toml. "
                "Please specify source code directories to analyze."
            )

        if not tests_folder:
            raise ConfigurationError(
                "Tests folder not defined (--tests-folder) and not found in pyproject.toml. "
                "Please specify the directory containing test files."
            )

        # Validate that paths exist
        for folder in folders:
            if not Path(folder).exists():
                self.logger.warning(f"Source folder does not exist: {folder}")

        if not Path(tests_folder).exists():
            self.logger.warning(f"Tests folder does not exist: {tests_folder}")

        return folders, tests_folder, coverage_file

    def get_llm_config(self, pyproject_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get LLM configuration from pyproject.toml."""
        if pyproject_data is None:
            pyproject_data = self.load_pyproject_config()

        ai_unit_test_config = pyproject_data.get("tool", {}).get("ai-unit-test", {})
        result: dict[str, Any] = ai_unit_test_config.get("llm", {})
        if "llm_provider" in ai_unit_test_config:
            result["provider"] = ai_unit_test_config.pop("llm_provider")
        return result

    def get_indexing_config(self, pyproject_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Get indexing configuration from pyproject.toml."""
        if pyproject_data is None:
            pyproject_data = self.load_pyproject_config()

        result: dict[str, Any] = pyproject_data.get("tool", {}).get("ai-unit-test", {}).get("indexing", {})
        return result

    def validate_environment(self) -> EnvironmentStatus:
        """Validate environment and return status information."""
        status = EnvironmentStatus(
            python_version=sys.version,
            working_directory=str(Path.cwd()),
            pyproject_exists=Path("pyproject.toml").exists(),
            tests_directory_exists=Path("tests").exists(),
            coverage_file_exists=Path(".coverage").exists(),
            environment_variables={},
        )

        # Check important environment variables
        import os

        env_vars = ["OPENAI_API_KEY", "HF_API_KEY", "OPENAI_API_URL"]
        env_status: dict[str, bool] = {}
        for var in env_vars:
            env_status[var] = var in os.environ

        status.environment_variables = env_status

        return status
