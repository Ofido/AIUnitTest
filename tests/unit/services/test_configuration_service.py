"""Test configuration service."""

from typing import Any
from unittest.mock import mock_open, patch

import pytest

from ai_unit_test.core.exceptions import ConfigurationError
from ai_unit_test.services.configuration_service import ConfigurationService


class TestConfigurationService:
    """Test configuration service functionality."""

    def test_init(self) -> None:
        """Test service initialization."""
        service = ConfigurationService()
        assert service.config == {}
        assert service._pyproject_cache is None

        config = {"test": "value"}
        service = ConfigurationService(config)
        assert service.config == config

    def test_load_pyproject_config_missing_file(self) -> None:
        """Test loading config when file doesn't exist."""
        service = ConfigurationService()

        with patch("pathlib.Path.exists", return_value=False):
            config = service.load_pyproject_config()

        assert config == {}
        assert service._pyproject_cache == {}

    def test_load_pyproject_config_success(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test successful config loading."""
        service = ConfigurationService()

        # Mock file reading
        toml_content = """
[tool.coverage.run]
source = ["src"]
data_file = ".coverage"

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ai-unit-test.llm]
provider = "mock"
model = "mock-model"
"""

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.open", mock_open(read_data=toml_content.encode())):
                config = service.load_pyproject_config()

        assert "tool" in config
        assert config["tool"]["coverage"]["run"]["source"] == ["src"]

        # Test caching
        config2 = service.load_pyproject_config()
        assert config2 is config  # Same object due to caching

    def test_load_pyproject_config_invalid_toml(self) -> None:
        """Test handling of invalid TOML file."""
        service = ConfigurationService()

        invalid_toml = "invalid [ toml content"

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.open", mock_open(read_data=invalid_toml.encode())):
                with pytest.raises(ConfigurationError) as exc_info:
                    service.load_pyproject_config()

        assert "Failed to load pyproject.toml" in str(exc_info.value)

    def test_extract_source_configuration(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test extracting source configuration."""
        service = ConfigurationService()

        folders, tests_folder, coverage_path = service.extract_source_configuration(sample_pyproject_config)

        assert folders == ["src"]
        assert tests_folder == "tests"
        assert coverage_path == ".coverage"

    def test_extract_source_configuration_missing_values(self) -> None:
        """Test extracting config with missing values."""
        service = ConfigurationService()

        with patch("pathlib.Path.is_dir", return_value=True):
            folders, tests_folder, coverage_path = service.extract_source_configuration({})

        assert folders == []
        assert tests_folder == "tests"  # Fallback value
        assert coverage_path is None

    def test_resolve_paths_from_config_auto_discovery(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test path resolution with auto-discovery."""
        service = ConfigurationService()
        service._pyproject_cache = sample_pyproject_config

        folders, tests_folder, coverage_file = service.resolve_paths_from_config(auto_discovery=True)

        assert folders == ["src"]
        assert tests_folder == "tests"
        assert coverage_file == ".coverage"

    def test_resolve_paths_validation_errors(self) -> None:
        """Test path resolution validation errors."""
        service = ConfigurationService()
        service._pyproject_cache = {}

        with patch.object(service, "load_pyproject_config", return_value={}):
            with patch("pathlib.Path.is_dir", return_value=False):
                # Test missing folders
                with pytest.raises(ConfigurationError) as exc_info:
                    service.resolve_paths_from_config(folders=None, tests_folder=None, auto_discovery=False)
                assert "Source folders not defined" in str(exc_info.value)

                # Test missing tests folder
                with pytest.raises(ConfigurationError) as exc_info:
                    service.resolve_paths_from_config(folders=["src"], tests_folder=None, auto_discovery=False)
                assert "Tests folder not defined" in str(exc_info.value)

    def test_get_llm_config(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test getting LLM configuration."""
        service = ConfigurationService()

        llm_config = service.get_llm_config(sample_pyproject_config)

        assert llm_config["provider"] == "mock"
        assert llm_config["model"] == "mock-model"
        assert llm_config["temperature"] == 0.1

    def test_get_indexing_config(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test getting indexing configuration."""
        service = ConfigurationService()

        indexing_config = service.get_indexing_config(sample_pyproject_config)

        assert indexing_config["backend"] == "memory"
        assert indexing_config["index_directory"] == "data/test_index"

    def test_validate_environment(self) -> None:
        """Test environment validation."""
        service = ConfigurationService()

        def exists_side_effect(path: str) -> bool:
            return str(path) == "pyproject.toml"

        with patch("pathlib.Path.exists", new=exists_side_effect):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
                status = service.validate_environment()
                assert hasattr(status, "python_version")
                assert hasattr(status, "working_directory")
                assert status.pyproject_exists is True
                assert status.environment_variables["OPENAI_API_KEY"] is True
                assert status.environment_variables["HF_API_KEY"] is False

    def test_cover_misc_branches(self, sample_pyproject_config: dict[str, Any]) -> None:
        """Test miscellaneous branches."""
        service = ConfigurationService()

        # __repr__
        assert service.__repr__() == "ConfigurationService"

        # extract_source_configuration with pyproject_data None uses load_pyproject_config
        with patch.object(service, "load_pyproject_config", return_value=sample_pyproject_config):
            folders, tests_folder, coverage_path = service.extract_source_configuration(None)
        assert folders == ["src"]
        assert tests_folder == "tests"
        assert coverage_path == ".coverage"

        # get_llm_config with llm_provider present
        pyproject_llm = {"tool": {"ai-unit-test": {"llm_provider": "openai", "llm": {"model": "gpt"}}}}
        llm = service.get_llm_config(pyproject_llm)
        assert llm["provider"] == "openai"
        assert llm["model"] == "gpt"

        # get_indexing_config when pyproject_data is None (uses load)
        with patch.object(
            service, "load_pyproject_config", return_value={"tool": {"ai-unit-test": {"indexing": {"backend": "disk"}}}}
        ):
            idx = service.get_indexing_config(None)
        assert idx["backend"] == "disk"

        # resolve_paths_from_config should log warnings for missing folders/tests and respect coverage replacement
        service._pyproject_cache = sample_pyproject_config
        test_folders = ["nonexistent_src"]
        test_tests_folder = "nonexistent_tests"
        test_coverage = ".coverage"
        with patch("pathlib.Path.exists", return_value=False):
            with patch.object(service.logger, "warning") as mock_warn:
                res_folders, res_tests, res_cov = service.resolve_paths_from_config(
                    folders=test_folders,
                    tests_folder=test_tests_folder,
                    coverage_file=test_coverage,
                    auto_discovery=False,
                )
        assert res_folders == test_folders
        assert res_tests == test_tests_folder
        assert res_cov == ".coverage"
        assert mock_warn.call_count == 2
