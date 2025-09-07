"""Service for test processing and generation."""

import ast
import logging
from pathlib import Path
from typing import Any

from ai_unit_test.core.exceptions import TestGenerationError
from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.coverage_helper import collect_missing_lines
from ai_unit_test.file_helper import (
    extract_function_source,
    find_relevant_tests,
    find_test_file,
    insert_new_test,
    read_file_content,
    write_file_content,
)
from ai_unit_test.services.base_service import BaseService

logger = logging.getLogger(__name__)


class TestProcessingService(BaseService):
    """Service for processing test generation requests."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.llm_connector = None
        self.index_organizer = None

    async def initialize_dependencies(self) -> None:
        """Initialize LLM and indexing dependencies."""
        try:
            # Initialize LLM connector
            llm_config = self.config.get("llm", {})
            self.llm_connector = LLMConnectorFactory.create_from_config_file(
                {"tool": {"ai-unit-test": {"llm": llm_config}}}
            )
            await self.llm_connector.initialize()

            # Initialize index organizer
            indexing_config = self.config.get("indexing", {})
            self.index_organizer = IndexOrganizerFactory.create_from_config_file(
                {"tool": {"ai-unit-test": {"indexing": indexing_config}}}
            )

            self.logger.info("Test processing dependencies initialized")

        except Exception as e:
            raise TestGenerationError(f"Failed to initialize dependencies: {e}")

    async def process_missing_coverage(
        self, source_folders: list[str], tests_folder: str, coverage_file: str
    ) -> dict[str, Any]:
        """Process missing coverage and generate tests."""

        if not self.llm_connector:
            await self.initialize_dependencies()

        try:
            # Collect missing coverage information
            self.logger.info("Collecting coverage information...")
            missing_info = collect_missing_lines(source_folders, coverage_file)

            if not missing_info:
                return {
                    "status": "success",
                    "message": "No missing coverage found",
                    "files_processed": 0,
                    "tests_generated": 0,
                }

            # Process each file with missing coverage
            results = {
                "status": "success",
                "files_processed": 0,
                "tests_generated": 0,
                "errors": [],
                "file_results": {},
            }

            for source_file_path, uncovered_lines in missing_info.items():
                try:
                    file_result = await self._process_single_file(source_file_path, uncovered_lines, tests_folder)

                    results["file_results"][str(source_file_path)] = file_result
                    results["files_processed"] += 1

                    if file_result.get("test_generated"):
                        results["tests_generated"] += 1

                except Exception as e:
                    error_msg = f"Error processing {source_file_path}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)

            if results["errors"]:
                results["status"] = "partial_success"

            return results

        except Exception as e:
            self.logger.error(f"Coverage processing failed: {e}")
            raise TestGenerationError(f"Coverage processing failed: {e}")

    async def _process_single_file(
        self, source_file_path: Path, uncovered_lines: list[int], tests_folder: str
    ) -> dict[str, Any]:
        """Process a single source file for test generation."""

        self.logger.info(f"Processing source file: {source_file_path}")

        # Find corresponding test file
        test_file = find_test_file(str(source_file_path), tests_folder)
        if not test_file:
            self.logger.warning(f"Test file not found for {source_file_path}")
            return {"status": "skipped", "reason": "test_file_not_found", "test_generated": False}

        # Detect test style
        test_style = self._detect_test_style(test_file)
        self.logger.debug(f"Detected test style: {test_style}")

        # Extract source code for uncovered lines
        source_code = self._extract_source_code_for_lines(source_file_path, uncovered_lines)
        if not source_code:
            return {"status": "skipped", "reason": "no_extractable_code", "test_generated": False}

        # Get existing test content
        test_content = read_file_content(test_file)
        if not test_content:
            test_content = self._create_empty_test_file_content(test_style, source_file_path)

        # Find relevant tests for context
        relevant_tests = await self._find_relevant_test_context(source_file_path, tests_folder)

        # Generate new test using LLM
        try:
            new_test = await self._generate_test_with_llm(
                source_code=source_code,
                test_code=test_content,
                file_name=str(source_file_path),
                coverage_lines=uncovered_lines,
                other_tests_content=relevant_tests,
                test_style=test_style,
            )

            # Insert new test into file
            updated_content = insert_new_test(test_content, new_test, test_style)
            write_file_content(test_file, updated_content)

            return {
                "status": "success",
                "test_file": str(test_file),
                "test_style": test_style,
                "uncovered_lines": uncovered_lines,
                "test_generated": True,
                "new_test_length": len(new_test),
            }

        except Exception as e:
            self.logger.error(f"Test generation failed for {source_file_path}: {e}")
            return {"status": "error", "reason": str(e), "test_generated": False}

    def _detect_test_style(self, test_file_path: Path) -> str:
        """Detect test file style (unittest/pytest)."""
        test_content = read_file_content(test_file_path)
        if not test_content:
            return "pytest_function"  # Default

        try:
            tree = ast.parse(test_content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        if isinstance(base, ast.Attribute) and base.attr == "TestCase":
                            return "unittest_class"
                        elif isinstance(base, ast.Name) and base.id == "TestCase":
                            return "unittest_class"
            return "pytest_function"
        except SyntaxError:
            self.logger.warning(f"Could not parse {test_file_path} for style detection")
            return "pytest_function"

    def _extract_source_code_for_lines(self, source_file_path: Path, uncovered_lines: list[int]) -> str:
        """Extract source code chunks for uncovered lines."""
        try:
            # Group consecutive lines into chunks
            chunks = self._group_consecutive_lines(uncovered_lines)

            source_code_parts = []
            for chunk in chunks:
                chunk_code = extract_function_source(str(source_file_path), chunk)
                if chunk_code:
                    source_code_parts.append(chunk_code)

            return "\n\n".join(source_code_parts)

        except Exception as e:
            self.logger.error(f"Failed to extract source code: {e}")
            return ""

    def _group_consecutive_lines(self, lines: list[int]) -> list[list[int]]:
        """Group consecutive line numbers into chunks."""
        if not lines:
            return []

        sorted_lines = sorted(lines)
        chunks = []
        current_chunk = [sorted_lines[0]]

        for line in sorted_lines[1:]:
            if line == current_chunk[-1] + 1:
                current_chunk.append(line)
            else:
                chunks.append(current_chunk)
                current_chunk = [line]

        chunks.append(current_chunk)
        return chunks

    async def _find_relevant_test_context(self, source_file_path: Path, tests_folder: str) -> str:
        """Find relevant tests for context using semantic search."""
        try:
            if not self.index_organizer:
                return ""

            # Use semantic search to find similar tests
            relevant_tests = find_relevant_tests(str(source_file_path), tests_folder)
            return "\n\n".join(relevant_tests[:3])  # Limit context

        except Exception as e:
            self.logger.warning(f"Failed to find relevant test context: {e}")
            return ""

    async def _generate_test_with_llm(
        self,
        source_code: str,
        test_code: str,
        file_name: str,
        coverage_lines: list[int],
        other_tests_content: str,
        test_style: str,
    ) -> str:
        """Generate test using LLM connector."""

        system_msg = self._build_system_message(test_style)
        user_msg = self._build_user_message(file_name, coverage_lines, source_code, test_code, other_tests_content)

        from ai_unit_test.core.interfaces.llm_connector import LLMRequest

        request = LLMRequest(
            system_message=system_msg,
            user_message=user_msg,
            model=self.config.get("llm", {}).get("model", "gpt-4o-mini"),
            temperature=self.config.get("llm", {}).get("temperature", 0.1),
        )

        response = await self.llm_connector.generate_response(request)
        return response.content

    def _build_system_message(self, test_style: str) -> str:
        """Build system message for LLM based on test style."""
        base_msg = (
            "You are an expert Python test developer. Your task is to write new unit tests "
            "to cover missing lines in a given code chunk. You must follow the style of existing "
            "tests provided as reference. Your response must be only the new test code, without "
            "any explanations, comments, or markdown formatting. Your response must be only valid Python code."
        )

        if test_style == "unittest_class":
            style_msg = (
                "Generate a new test method to be added inside a `unittest.TestCase` class. "
                "The method name should start with `test_` and it should accept `self` as its first argument."
            )
        elif test_style == "pytest_function":
            style_msg = "Generate a new test function. The function name should start with `test_`."
        else:
            style_msg = "Generate a new test function or method, adapting to the existing test file's style."

        return f"{base_msg} {style_msg}"

    def _build_user_message(
        self, file_name: str, coverage_lines: list[int], source_code: str, test_code: str, other_tests_content: str
    ) -> str:
        """Build user message for LLM with all context."""
        return f"""Here is the information for the test generation:

<file_to_be_tested>
{file_name}
</file_to_be_tested>

<uncovered_lines>
{coverage_lines}
</uncovered_lines>

<source_code_chunk>
{source_code}
</source_code_chunk>

<existing_tests>
{test_code}
</existing_tests>

<style_reference_tests>
{other_tests_content}
</style_reference_tests>
"""

    def _create_empty_test_file_content(self, test_style: str, source_file_path: Path) -> str:
        """Create empty test file content based on style."""
        if test_style == "unittest_class":
            class_name = source_file_path.stem.replace("_", " ").title().replace(" ", "") + "Test"
            return f"""import unittest
from {source_file_path.stem} import *

class {class_name}(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()
"""
        else:
            return f"""# Tests for {source_file_path.name}
from {source_file_path.stem} import *

# Test functions will be added here
"""

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize_dependencies()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.llm_connector:
            await self.llm_connector.__aexit__(exc_type, exc_val, exc_tb)
        if self.index_organizer:
            await self.index_organizer.__aexit__(exc_type, exc_val, exc_tb)
