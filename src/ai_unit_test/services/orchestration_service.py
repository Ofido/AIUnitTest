"""Orchestration service for coordinating complex workflows."""

import ast
import asyncio
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory
from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory
from ai_unit_test.core.interfaces.llm_connector import EmbeddingRequest
from ai_unit_test.services.base_service import BaseService
from ai_unit_test.services.configuration_service import ConfigurationService, EnvironmentStatus
from ai_unit_test.services.processing_service import TestProcessingService

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of source code."""

    file_path: str
    start_line: int
    end_line: int
    content: str


@dataclass
class ConfigHealth:
    """Represents the health status of the configuration."""

    healthy: bool
    pyproject_loaded: bool = False
    environment: EnvironmentStatus | None = None
    error: str | None = None
    timestamp: float | None = None


@dataclass
class LlmHealth:
    """Represents the health status of the LLM connector."""

    healthy: bool
    timestamp: float | None = None
    connector_info: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class IndexHealth:
    """Represents the health status of the indexing service."""

    healthy: bool
    available_backends: list[str] | None = None
    error: str | None = None


@dataclass
class HealthStatusChecks:
    """Container for various health check statuses."""

    config: ConfigHealth | None = None
    llm: LlmHealth | None = None
    indexing: IndexHealth | None = None


@dataclass
class HealthStatus:
    """Overall health status of the system."""

    status: str
    timestamp: float
    checks: HealthStatusChecks = field(default_factory=HealthStatusChecks)
    error: str | None = None
    failed_checks: list[str] = field(default_factory=list)


class OrchestrationService(BaseService):
    """Service for orchestrating complex AI Unit Test workflows."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the OrchestrationService.

        Args:
            config: Optional configuration dictionary.
        """
        super().__init__(config)
        self.config_service = ConfigurationService(config)
        self.test_service = None

    def get_service_name(self) -> str:
        """Return the name of this service for logging purposes."""
        return "Orchestration"

    async def run_test_generation_workflow(
        self,
        folders: list[str] | None = None,
        tests_folder: str | None = None,
        coverage_file: str = ".coverage",
        auto_discovery: bool = False,
        index_directory: str | None = None,  # Add this line
    ) -> dict[str, Any]:
        """Run complete test generation workflow."""
        workflow_start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Resolve configuration
            self.logger.info("Starting test generation workflow")
            resolved_folders, resolved_tests_folder, resolved_coverage = self.config_service.resolve_paths_from_config(
                folders, tests_folder, coverage_file, auto_discovery
            )

            # Step 2: Validate environment
            env_status = self.config_service.validate_environment()
            self.logger.debug(f"Environment validation: {env_status}")

            test_config = {
                "llm": self.config_service.get_llm_config(),
                "indexing": self.config_service.get_indexing_config(),
            }
            if index_directory:
                test_config["indexing"]["index_directory"] = index_directory

            async with TestProcessingService(test_config) as test_service:
                # Step 4: Process missing coverage
                coverage_result = await test_service.process_missing_coverage(
                    resolved_folders, resolved_tests_folder, resolved_coverage
                )

            # Step 5: Add workflow metadata
            workflow_end_time = asyncio.get_event_loop().time()

            results = asdict(coverage_result)
            results.update(
                {
                    "workflow_duration_seconds": workflow_end_time - workflow_start_time,
                    "configuration": {
                        "source_folders": resolved_folders,
                        "tests_folder": resolved_tests_folder,
                        "coverage_file": resolved_coverage,
                        "auto_discovery": auto_discovery,
                    },
                    "environment": env_status,
                }
            )

            self.logger.info(f"Workflow completed in {results['workflow_duration_seconds']:.2f}s")
            return results

        except Exception as e:
            self.logger.error(f"Workflow failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "workflow_duration_seconds": asyncio.get_event_loop().time() - workflow_start_time,
            }

    async def run_index_creation_workflow(
        self,
        source_folders: list[str],
        index_directory: str,
        force_rebuild: bool = False,
    ) -> dict[str, Any]:
        """Run index creation workflow."""
        self.logger.info("Starting index creation workflow")
        workflow_start_time = asyncio.get_event_loop().time()

        try:
            index_path = Path(index_directory)
            if index_path.exists() and not force_rebuild:
                return {
                    "status": "skipped",
                    "message": f"Index already exists at {index_directory}. Use --force to rebuild.",
                    "workflow_duration_seconds": asyncio.get_event_loop().time() - workflow_start_time,
                }

            # Initialize LLM and Indexing services
            llm_config = self.config_service.get_llm_config()
            llm_connector = LLMConnectorFactory.create_from_config_file({"tool": {"ai-unit-test": {"llm": llm_config}}})
            await llm_connector.initialize()

            indexing_config = self.config_service.get_indexing_config()
            index_organizer = IndexOrganizerFactory.create_from_config_file(
                {"tool": {"ai-unit-test": {"indexing": indexing_config}}}
            )

            # 1. Find all python files
            self.logger.info(f"Searching for Python files in {source_folders}")
            source_files = []
            for folder in source_folders:
                source_files.extend(list(Path(folder).rglob("*.py")))
            self.logger.info(f"Found {len(source_files)} Python files.")

            # 2. Chunk files
            self.logger.info("Chunking files...")
            all_chunks: list[Chunk] = []
            for file_path in source_files:
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                        all_chunks.extend(self._chunk_source_code(content, str(file_path)))
                except Exception as e:
                    self.logger.warning(f"Could not process file {file_path}: {e}")
            self.logger.info(f"Created {len(all_chunks)} chunks.")

            if not all_chunks:
                return {
                    "status": "success",
                    "message": "No source code found to index.",
                    "workflow_duration_seconds": asyncio.get_event_loop().time() - workflow_start_time,
                }

            # 3. Generate embeddings
            self.logger.info("Generating embeddings...")
            chunk_contents = [chunk.content for chunk in all_chunks]
            embedding_model = llm_config.get("embedding_model", "text-embedding-ada-002")
            embedding_request = EmbeddingRequest(texts=chunk_contents, model=embedding_model)
            embedding_response = await llm_connector.generate_embeddings(embedding_request)
            embeddings = embedding_response.embeddings

            # 4. Create metadata
            self.logger.info("Creating metadata...")
            metadata = [
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "content_preview": (chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content),
                }
                for chunk in all_chunks
            ]

            # 5. Create index
            self.logger.info(f"Creating index at {index_directory}...")
            await index_organizer.create_index(
                embeddings=embeddings,
                metadata=metadata,
                index_path=index_path,
                model_name=embedding_model,
            )

            workflow_end_time = asyncio.get_event_loop().time()
            return {
                "status": "success",
                "message": f"Index created successfully with {len(all_chunks)} documents.",
                "files_processed": len(source_files),
                "chunks_created": len(all_chunks),
                "workflow_duration_seconds": workflow_end_time - workflow_start_time,
            }

        except Exception as e:
            self.logger.error(f"Index creation workflow failed: {e}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "workflow_duration_seconds": asyncio.get_event_loop().time() - workflow_start_time,
            }

    def _chunk_source_code(self, source_code: str, file_path: str) -> list[Chunk]:
        """Chunk source code by classes and functions."""
        chunks = []
        try:
            tree = ast.parse(source_code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno
                    end_line = node.end_lineno
                    if end_line is None:
                        # Estimate end line for nodes without it
                        source_segment = ast.get_source_segment(source_code, node)
                        if source_segment is None:
                            continue
                        end_line = start_line + len(source_segment.splitlines())

                    chunks.append(
                        Chunk(
                            file_path=file_path,
                            start_line=start_line,
                            end_line=end_line,
                            content=ast.get_source_segment(source_code, node) or "",
                        )
                    )
        except SyntaxError as e:
            self.logger.warning(f"Could not parse {file_path} for chunking: {e}")
            # Fallback to chunking the whole file
            chunks.append(
                Chunk(
                    file_path=file_path,
                    start_line=1,
                    end_line=len(source_code.splitlines()),
                    content=source_code,
                )
            )
        return chunks

    async def run_health_check_workflow(self) -> HealthStatus:
        """Run comprehensive system health check."""
        self.logger.info("Running health check workflow")

        health_status: HealthStatus = HealthStatus(
            status="healthy",
            timestamp=asyncio.get_event_loop().time(),
        )

        try:
            # Check configuration
            health_status.checks.config = await self._check_configuration_health()

            # Check LLM connectivity
            health_status.checks.llm = await self._check_llm_health()

            # Check index availability
            health_status.checks.indexing = await self._check_indexing_health()

            # Determine overall status
            failed_checks = [
                name
                for name, check in vars(health_status.checks).items()
                if check is not None and hasattr(check, "healthy") and not check.healthy
            ]

            if failed_checks:
                health_status.status = "unhealthy"
                health_status.failed_checks = failed_checks

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            health_status.status = "error"
            health_status.error = str(e)
            return health_status

    async def _check_configuration_health(self) -> ConfigHealth:
        """Check configuration health."""
        try:
            config = self.config_service.load_pyproject_config()
            env_status = self.config_service.validate_environment()

            return ConfigHealth(healthy=True, pyproject_loaded=bool(config), environment=env_status)
        except Exception as e:
            return ConfigHealth(healthy=False, error=str(e), timestamp=asyncio.get_event_loop().time())

    async def _check_llm_health(self) -> LlmHealth:
        """Check LLM connector health."""
        try:
            from ai_unit_test.core.factories.llm_factory import LLMConnectorFactory

            llm_config = self.config_service.get_llm_config()
            connector = LLMConnectorFactory.create_from_config_file({"tool": {"ai-unit-test": {"llm": llm_config}}})

            async with connector:
                healthy = await connector.health_check()
                info = connector.get_connector_info()

                return LlmHealth(healthy=healthy, connector_info=info)

        except Exception as e:
            return LlmHealth(healthy=False, error=str(e))

    async def _check_indexing_health(self) -> IndexHealth:
        """Check index organizer health."""
        try:
            from ai_unit_test.core.factories.index_factory import IndexOrganizerFactory

            available_backends = IndexOrganizerFactory.get_available_organizers()

            return IndexHealth(
                healthy=len(available_backends) > 0,
                available_backends=available_backends,
            )

        except Exception as e:
            return IndexHealth(healthy=False, error=str(e))

    async def run_coverage_analysis_workflow(self, folders: list[str], tests_folder: str) -> dict[str, Any]:
        """Run coverage analysis workflow."""
        self.logger.info("Starting coverage analysis workflow")
        workflow_start_time = asyncio.get_event_loop().time()

        try:
            # Step 1: Initialize test processing service
            test_config = {
                "llm": self.config_service.get_llm_config(),
                "indexing": self.config_service.get_indexing_config(),
            }

            async with TestProcessingService(test_config) as test_service:
                # Step 2: Process missing coverage
                coverage_result = await test_service.process_missing_coverage(folders, tests_folder, ".coverage")

            workflow_end_time = asyncio.get_event_loop().time()
            self.logger.info(f"Coverage analysis workflow completed in {workflow_end_time - workflow_start_time:.2f}s")
            return asdict(coverage_result)

        except Exception as e:
            self.logger.error(f"Coverage analysis workflow failed: {e}", exc_info=True)
            return {}

    async def load_index(self, index_path: Path) -> None:
        """Load the index."""
        self.logger.info(f"Loading index from {index_path}")
        indexing_config = self.config_service.get_indexing_config()
        self.index_organizer = IndexOrganizerFactory.create_from_config_file(
            {"tool": {"ai-unit-test": {"indexing": indexing_config}}}
        )
        await self.index_organizer.load_index(index_path)
