"""Orchestration service for coordinating complex workflows."""

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import Any

from ai_unit_test.services.base_service import BaseService
from ai_unit_test.services.configuration_service import ConfigurationService, EnvironmentStatus
from ai_unit_test.services.processing_service import TestProcessingService

logger = logging.getLogger(__name__)


@dataclass
class ConfigHealth:
    healthy: bool
    pyproject_loaded: bool = False
    environment: EnvironmentStatus | None = None
    error: str | None = None
    timestamp: float | None = None


@dataclass
class LlmHealth:
    healthy: bool
    timestamp: float | None = None
    connector_info: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class IndexHealth:
    healthy: bool
    available_backends: list[str] | None = None
    error: str | None = None


@dataclass
class HealthStatusChecks:
    config: ConfigHealth | None = None
    llm: LlmHealth | None = None
    indexing: IndexHealth | None = None


@dataclass
class HealthStatus:
    status: str
    timestamp: float
    checks: HealthStatusChecks = HealthStatusChecks()
    error: str | None = None
    failed_checks: list[str] = []


class OrchestrationService(BaseService):
    """Service for orchestrating complex AI Unit Test workflows."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
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

            # Step 3: Initialize test processing service
            test_config = {
                "llm": self.config_service.get_llm_config(),
                "indexing": self.config_service.get_indexing_config(),
            }

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
            self.logger.error(f"Workflow failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "workflow_duration_seconds": asyncio.get_event_loop().time() - workflow_start_time,
            }

    async def run_index_creation_workflow(
        self, source_folders: list[str], index_directory: str, force_rebuild: bool = False
    ) -> dict[str, Any]:
        """Run index creation workflow."""

        self.logger.info("Starting index creation workflow")

        try:
            # Implementation for index creation workflow
            # This would coordinate between file processing, embedding generation,
            # and index creation services

            return {"status": "success", "message": "Index creation workflow not yet implemented"}

        except Exception as e:
            self.logger.error(f"Index creation workflow failed: {e}")
            return {"status": "error", "error": str(e)}

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
            self.logger.error(f"Health check failed: {e}")
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

            return IndexHealth(healthy=len(available_backends) > 0, available_backends=available_backends)

        except Exception as e:
            return IndexHealth(healthy=False, error=str(e))
