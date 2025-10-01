"""Test cases for TestProcessingService class."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_unit_test.core.exceptions import TestGenerationError
from ai_unit_test.services.processing_service import CoverageProcessingResult, FileTestResult, TestProcessingService


class TestFileTestResult:
    """Test cases for FileTestResult dataclass."""

    def test_file_test_result_creation(self) -> None:
        """Test FileTestResult creation with all fields."""
        result = FileTestResult(
            status="success",
            test_generated=True,
            reason=None,
            test_file="/path/to/test.py",
            test_style="pytest_function",
            uncovered_lines=[1, 2, 3],
            new_test_length=150,
        )

        assert result.status == "success"
        assert result.test_generated is True
        assert result.reason is None
        assert result.test_file == "/path/to/test.py"
        assert result.test_style == "pytest_function"
        assert result.uncovered_lines == [1, 2, 3]
        assert result.new_test_length == 150

    def test_file_test_result_minimal(self) -> None:
        """Test FileTestResult creation with minimal fields."""
        result = FileTestResult(status="error", test_generated=False)

        assert result.status == "error"
        assert result.test_generated is False
        assert result.reason is None
        assert result.test_file is None
        assert result.test_style is None
        assert result.uncovered_lines is None
        assert result.new_test_length is None


class TestCoverageProcessingResult:
    """Test cases for CoverageProcessingResult dataclass."""

    def test_coverage_processing_result_creation(self) -> None:
        """Test CoverageProcessingResult creation."""
        result = CoverageProcessingResult(
            status="success",
            files_processed=5,
            tests_generated=3,
            message="All good",
            errors=["error1", "error2"],
            file_results={"file1": FileTestResult("success", True)},
        )

        assert result.status == "success"
        assert result.files_processed == 5
        assert result.tests_generated == 3
        assert result.message == "All good"
        assert result.errors == ["error1", "error2"]
        assert len(result.file_results) == 1

    def test_coverage_processing_result_defaults(self) -> None:
        """Test CoverageProcessingResult with default values."""
        result = CoverageProcessingResult(status="success", files_processed=0, tests_generated=0)

        assert result.status == "success"
        assert result.files_processed == 0
        assert result.tests_generated == 0
        assert result.message is None
        assert result.errors == []
        assert result.file_results == {}


class TestTestProcessingService:
    """Test cases for TestProcessingService class."""

    @pytest.fixture
    def service(self) -> TestProcessingService:
        """Create a TestProcessingService instance."""
        return TestProcessingService({})

    @pytest.fixture
    def service_with_config(self, tmp_path: Path) -> TestProcessingService:
        """Create a TestProcessingService instance with config."""
        index_directory = tmp_path / "index"
        index_directory.mkdir()
        config = {
            "llm": {"model": "gpt-4", "temperature": 0.2},
            "indexing": {"index_directory": str(index_directory)},
        }
        return TestProcessingService(config)

    def test_get_service_name(self, service: TestProcessingService) -> None:
        """Test service name identification."""
        assert service.get_service_name() == "TestProcessing"

    @patch("ai_unit_test.services.processing_service.LLMConnectorFactory")
    @patch("ai_unit_test.services.processing_service.IndexOrganizerFactory")
    async def test_initialize_dependencies_success(
        self, mock_index_factory: MagicMock, mock_llm_factory: MagicMock, service_with_config: TestProcessingService
    ) -> None:
        """Test successful dependency initialization."""
        mock_llm = AsyncMock()
        mock_llm_factory.create_from_config_file.return_value = mock_llm

        mock_index = AsyncMock()
        mock_index_factory.create_from_config_file.return_value = mock_index

        with patch("pathlib.Path.exists", return_value=True):
            await service_with_config.initialize_dependencies()

        assert service_with_config.llm_connector == mock_llm
        assert service_with_config.index_organizer == mock_index
        mock_llm.initialize.assert_called_once()
        mock_index.load_index.assert_called_once()

    @patch("ai_unit_test.services.processing_service.LLMConnectorFactory")
    @patch("ai_unit_test.services.processing_service.IndexOrganizerFactory")
    async def test_initialize_dependencies_no_index_dir(
        self, mock_index_factory: MagicMock, mock_llm_factory: MagicMock, service: TestProcessingService
    ) -> None:
        """Test dependency initialization without index directory."""
        mock_llm = AsyncMock()
        mock_llm_factory.create_from_config_file.return_value = mock_llm

        mock_index = AsyncMock()
        mock_index_factory.create_from_config_file.return_value = mock_index

        await service.initialize_dependencies()

        assert service.llm_connector == mock_llm
        assert service.index_organizer == mock_index
        mock_llm.initialize.assert_called_once()
        mock_index.load_index.assert_not_called()

    @patch("ai_unit_test.services.processing_service.LLMConnectorFactory")
    async def test_initialize_dependencies_failure(
        self, mock_llm_factory: MagicMock, service: TestProcessingService
    ) -> None:
        """Test dependency initialization failure."""
        mock_llm_factory.create_from_config_file.side_effect = Exception("LLM error")

        with pytest.raises(TestGenerationError, match="Failed to initialize dependencies"):
            await service.initialize_dependencies()

    @patch("ai_unit_test.services.processing_service.collect_missing_lines")
    async def test_process_missing_coverage_no_missing(
        self, mock_collect: MagicMock, service: TestProcessingService
    ) -> None:
        """Test processing when no missing coverage found."""
        mock_collect.return_value = {}
        service.llm_connector = AsyncMock()

        result = await service.process_missing_coverage(["src"], "tests", "coverage.xml")

        assert result.status == "success"
        assert result.message == "No missing coverage found"
        assert result.files_processed == 0
        assert result.tests_generated == 0

    @patch("ai_unit_test.services.processing_service.collect_missing_lines")
    async def test_process_missing_coverage_with_files(
        self, mock_collect: MagicMock, service: TestProcessingService
    ) -> None:
        """Test processing with missing coverage files."""
        mock_collect.return_value = {Path("src/module.py"): [10, 15, 20]}
        service.llm_connector = AsyncMock()

        with patch.object(service, "_process_single_file") as mock_process:
            mock_process.return_value = FileTestResult("success", True)

            result = await service.process_missing_coverage(["src"], "tests", "coverage.xml")

            assert result.status == "success"
            assert result.files_processed == 1
            assert result.tests_generated == 1
            mock_process.assert_called_once_with(Path("src/module.py"), [10, 15, 20], "tests")

    @patch("ai_unit_test.services.processing_service.collect_missing_lines")
    async def test_process_missing_coverage_with_errors(
        self, mock_collect: MagicMock, service: TestProcessingService
    ) -> None:
        """Test processing with some file errors."""
        mock_collect.return_value = {Path("src/module1.py"): [10, 15], Path("src/module2.py"): [20, 25]}
        service.llm_connector = AsyncMock()

        with patch.object(service, "_process_single_file") as mock_process:
            mock_process.side_effect = [FileTestResult("success", True), Exception("Processing error")]

            result = await service.process_missing_coverage(["src"], "tests", "coverage.xml")

            assert result.status == "partial_success"
            assert result.files_processed == 1
            assert result.tests_generated == 1
            assert len(result.errors) == 1
            assert "Processing error" in result.errors[0]

    @patch("ai_unit_test.services.processing_service.collect_missing_lines")
    async def test_process_missing_coverage_initialization_needed(
        self, mock_collect: MagicMock, service: TestProcessingService
    ) -> None:
        """Test processing when LLM connector needs initialization."""
        mock_collect.return_value = {}

        with patch.object(service, "initialize_dependencies") as mock_init:
            _ = await service.process_missing_coverage(["src"], "tests", "coverage.xml")

            mock_init.assert_called_once()

    @patch("ai_unit_test.services.processing_service.collect_missing_lines")
    async def test_process_missing_coverage_collection_error(
        self, mock_collect: MagicMock, service: TestProcessingService
    ) -> None:
        """Test processing when coverage collection fails."""
        mock_collect.side_effect = Exception("Coverage error")
        service.llm_connector = AsyncMock()

        with pytest.raises(TestGenerationError, match="Coverage processing failed"):
            await service.process_missing_coverage(["src"], "tests", "coverage.xml")

    @patch("ai_unit_test.services.processing_service.find_test_file")
    @patch("ai_unit_test.services.processing_service.write_file_content")
    async def test_process_single_file_creates_test_file(
        self, mock_write: MagicMock, mock_find_test: MagicMock, service: TestProcessingService
    ) -> None:
        """Test single file processing when test file doesn't exist."""
        mock_find_test.return_value = None
        service.llm_connector = AsyncMock()

        with (
            patch.object(service, "_detect_test_style", return_value="pytest_function"),
            patch.object(service, "_extract_source_code_for_lines", return_value="def func(): pass"),
            patch("ai_unit_test.services.processing_service.read_file_content", return_value=""),
            patch.object(service, "_create_empty_test_file_content", return_value="# empty test"),
            patch.object(service, "_find_relevant_test_context", return_value=""),
            patch.object(service, "_generate_test_with_llm", return_value="def test_func(): pass"),
            patch("ai_unit_test.services.processing_service.insert_new_test", return_value="updated content"),
        ):

            result = await service._process_single_file(Path("src/module.py"), [10, 15], "tests")

            mock_write.assert_called()
            assert result.status == "success"
            assert result.test_generated is True

    async def test_process_single_file_no_extractable_code(self, service: TestProcessingService) -> None:
        """Test single file processing when no code can be extracted."""
        with (
            patch("ai_unit_test.services.processing_service.find_test_file", return_value=Path("test.py")),
            patch.object(service, "_detect_test_style", return_value="pytest_function"),
            patch.object(service, "_extract_source_code_for_lines", return_value=""),
        ):

            result = await service._process_single_file(Path("src/module.py"), [10, 15], "tests")

            assert result.status == "skipped"
            assert result.reason == "no_extractable_code"
            assert result.test_generated is False

    async def test_process_single_file_llm_error(self, service: TestProcessingService) -> None:
        """Test single file processing when LLM generation fails."""
        service.llm_connector = AsyncMock()

        with (
            patch("ai_unit_test.services.processing_service.find_test_file", return_value=Path("test.py")),
            patch.object(service, "_detect_test_style", return_value="pytest_function"),
            patch.object(service, "_extract_source_code_for_lines", return_value="def func(): pass"),
            patch("ai_unit_test.services.processing_service.read_file_content", return_value="# test"),
            patch.object(service, "_find_relevant_test_context", return_value=""),
            patch.object(service, "_generate_test_with_llm", side_effect=Exception("LLM error")),
        ):

            result = await service._process_single_file(Path("src/module.py"), [10, 15], "tests")

            assert result.status == "error"
            assert "LLM error" in result.reason  # type: ignore
            assert result.test_generated is False

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_detect_test_style_unittest(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test test style detection for unittest."""
        mock_read.return_value = """
import unittest

class MyTest(unittest.TestCase):
    def test_something(self) -> None:
        pass
"""

        style = service._detect_test_style(Path("test.py"))
        assert style == "unittest_class"

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_detect_test_style_pytest(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test test style detection for pytest."""
        mock_read.return_value = """
def test_something():
    assert True

def test_another():
    pass
"""

        style = service._detect_test_style(Path("test.py"))
        assert style == "pytest_function"

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_detect_test_style_empty_file(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test test style detection for empty file."""
        mock_read.return_value = ""

        style = service._detect_test_style(Path("test.py"))
        assert style == "pytest_function"

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_detect_test_style_syntax_error(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test test style detection with syntax error."""
        mock_read.return_value = "invalid python code {"

        style = service._detect_test_style(Path("test.py"))
        assert style == "pytest_function"

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_extract_source_code_for_lines(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test source code extraction for specific lines."""
        mock_read.return_value = """line 1
line 2
line 3
line 4
line 5"""

        result = service._extract_source_code_for_lines(Path("src.py"), [2, 3, 5])

        assert "line 2" in result
        assert "line 3" in result
        assert "line 5" in result
        assert "Lines 2-3:" in result
        assert "Lines 5-5:" in result

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_extract_source_code_empty_file(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test source code extraction from empty file."""
        mock_read.return_value = ""

        result = service._extract_source_code_for_lines(Path("src.py"), [1, 2])
        assert result == ""

    @patch("ai_unit_test.services.processing_service.read_file_content")
    def test_extract_source_code_read_error(self, mock_read: MagicMock, service: TestProcessingService) -> None:
        """Test source code extraction when file read fails."""
        mock_read.side_effect = Exception("Read error")

        result = service._extract_source_code_for_lines(Path("src.py"), [1, 2])
        assert result == ""

    def test_group_consecutive_lines_empty(self, service: TestProcessingService) -> None:
        """Test grouping empty line list."""
        result = service._group_consecutive_lines([])
        assert result == []

    def test_group_consecutive_lines_single(self, service: TestProcessingService) -> None:
        """Test grouping single line."""
        result = service._group_consecutive_lines([5])
        assert result == [[5]]

    def test_group_consecutive_lines_consecutive(self, service: TestProcessingService) -> None:
        """Test grouping consecutive lines."""
        result = service._group_consecutive_lines([1, 2, 3, 5, 6, 9])
        assert result == [[1, 2, 3], [5, 6], [9]]

    def test_group_consecutive_lines_unordered(self, service: TestProcessingService) -> None:
        """Test grouping unordered lines."""
        result = service._group_consecutive_lines([9, 1, 3, 2])
        assert result == [[1, 2, 3], [9]]

    async def test_find_relevant_test_context_no_organizer(self, service: TestProcessingService) -> None:
        """Test finding test context when no index organizer."""
        service.index_organizer = None

        result = await service._find_relevant_test_context(Path("src.py"), "tests")
        assert result == ""

    @patch("ai_unit_test.services.processing_service.find_relevant_tests")
    async def test_find_relevant_test_context_success(
        self, mock_find: MagicMock, service: TestProcessingService
    ) -> None:
        """Test finding test context successfully."""
        service.index_organizer = AsyncMock()
        mock_find.return_value = ["test1 content", "test2 content", "test3 content", "test4 content"]

        result = await service._find_relevant_test_context(Path("src.py"), "tests")
        assert result == "test1 content\n\ntest2 content\n\ntest3 content"

    @patch("ai_unit_test.services.processing_service.find_relevant_tests")
    async def test_find_relevant_test_context_error(self, mock_find: MagicMock, service: TestProcessingService) -> None:
        """Test finding test context with error."""
        service.index_organizer = AsyncMock()
        mock_find.side_effect = Exception("Find error")

        result = await service._find_relevant_test_context(Path("src.py"), "tests")
        assert result == ""

    async def test_generate_test_with_llm_no_connector(self, service: TestProcessingService) -> None:
        """Test LLM generation without connector."""
        service.llm_connector = None

        with pytest.raises(TestGenerationError, match="LLM connector not initialized"):
            await service._generate_test_with_llm("code", "test", "file.py", [1], "", "pytest")

    async def test_generate_test_with_llm_success(self, service: TestProcessingService) -> None:
        """Test successful LLM generation."""
        mock_response = MagicMock()
        mock_response.content = "def test_generated(): pass"

        service.llm_connector = AsyncMock()
        service.llm_connector.generate_response.return_value = mock_response

        result = await service._generate_test_with_llm(
            "def func(): pass", "# test", "file.py", [1], "", "pytest_function"
        )

        assert result == "def test_generated(): pass"
        service.llm_connector.generate_response.assert_called_once()

    def test_build_system_message_unittest(self, service: TestProcessingService) -> None:
        """Test building system message for unittest style."""
        message = service._build_system_message("unittest_class")

        assert "expert Python test developer" in message
        assert "unittest.TestCase" in message
        assert "self" in message

    def test_build_system_message_pytest(self, service: TestProcessingService) -> None:
        """Test building system message for pytest style."""
        message = service._build_system_message("pytest_function")

        assert "expert Python test developer" in message
        assert "test function" in message

    def test_build_system_message_unknown(self, service: TestProcessingService) -> None:
        """Test building system message for unknown style."""
        message = service._build_system_message("unknown")

        assert "expert Python test developer" in message
        assert "adapting to the existing" in message

    def test_build_user_message(self, service: TestProcessingService) -> None:
        """Test building user message for LLM."""
        message = service._build_user_message(
            "file.py",
            [1, 2],
            "def func(): pass",
            "# test",
            "# reference",
        )

        assert "file.py" in message
        assert "[1, 2]" in message
        assert "def func(): pass" in message
        assert "# test" in message
        assert "# reference" in message

    def test_create_empty_test_file_content_unittest(self, service: TestProcessingService) -> None:
        """Test creating empty unittest test file."""
        content = service._create_empty_test_file_content("unittest_class", Path("my_module.py"))

        assert "import unittest" in content
        assert "class MyModuleTest(unittest.TestCase)" in content
        assert "unittest.main()" in content

    def test_create_empty_test_file_content_pytest(self, service: TestProcessingService) -> None:
        """Test creating empty pytest test file."""
        content = service._create_empty_test_file_content("pytest_function", Path("my_module.py"))

        assert "Tests for my_module.py" in content
        assert "from my_module import *" in content

    async def test_context_manager_enter(self, service: TestProcessingService) -> None:
        """Test async context manager entry."""
        with patch.object(service, "initialize_dependencies") as mock_init:
            result = await service.__aenter__()

            assert result == service
            mock_init.assert_called_once()

    async def test_context_manager_exit(self, service: TestProcessingService) -> None:
        """Test async context manager exit."""
        mock_llm = AsyncMock()
        mock_index = AsyncMock()
        service.llm_connector = mock_llm
        service.index_organizer = mock_index

        await service.__aexit__(None, None, None)

        mock_llm.__aexit__.assert_called_once_with(None, None, None)
        mock_index.__aexit__.assert_called_once_with(None, None, None)

    async def test_context_manager_exit_no_dependencies(self, service: TestProcessingService) -> None:
        """Test async context manager exit without dependencies."""
        service.llm_connector = None
        service.index_organizer = None

        # Should not raise any errors
        await service.__aexit__(None, None, None)
