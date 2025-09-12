"""CLI interface for AI Unit Test - Pure presentation layer."""

import asyncio
import logging
from typing import Any

import typer

from ai_unit_test.services.orchestration_service import (
    HealthStatus,
    OrchestrationService,
)

logger = logging.getLogger(__name__)
app = typer.Typer()

FOLDERS_OPTION = typer.Option(None, "--folders", "-f", help="Source code folders to analyze for coverage")
TESTS_FOLDER_OPTION = typer.Option(None, "--tests-folder", "-t", help="Directory containing test files")
COVERAGE_FILE_OPTION = typer.Option(".coverage", "--coverage-file", "-c", help="Path to coverage data file")
AUTO_OPTION = typer.Option(False, "--auto", "-a", help="Auto-discover configuration from pyproject.toml")
INDEX_DIR_OPTION = typer.Option(None, "--index-dir", help="Directory containing semantic search index")
FOLDERS_REQUIRED_OPTION = typer.Option(..., "--folders", "-f", help="Source code folders to index")
INDEX_DIR_DEFAULT_OPTION = typer.Option("data/faiss_index", "--index-dir", help="Directory to save the index")
FORCE_OPTION = typer.Option(False, "--force", help="Force rebuild even if index exists")


def handle_cli_exception(e: Exception) -> None:
    """Handle CLI exceptions with appropriate logging and user feedback."""
    logger.error(f"CLI command failed: {e}")
    typer.echo(f"❌ Error: {e}", err=True)


@app.command()
def generate_tests(
    folders: list[str] | None = FOLDERS_OPTION,
    tests_folder: str | None = TESTS_FOLDER_OPTION,
    coverage_file: str = COVERAGE_FILE_OPTION,
    auto: bool = AUTO_OPTION,
    index_dir: str | None = INDEX_DIR_OPTION,
) -> None:
    """Generate unit tests for uncovered code using AI."""
    try:
        # Create orchestration service
        config: dict[str, Any] = {"indexing": {"index_directory": index_dir}} if index_dir else {}
        orchestration_service = OrchestrationService(config)

        # Run workflow
        typer.echo("🚀 Starting test generation...")

        results = asyncio.run(
            orchestration_service.run_test_generation_workflow(
                folders=folders,
                tests_folder=tests_folder,
                coverage_file=coverage_file,
                auto_discovery=auto,
            )
        )

        # Display results
        _display_test_generation_results(results)

        # Exit with appropriate code
        if results["status"] == "error":
            raise typer.Exit(1)
        elif results["status"] == "partial_success":
            typer.echo("⚠️  Some files had issues, but tests were generated for others.")
            raise typer.Exit(2)
        else:
            typer.echo("✅ Test generation completed successfully!")

    except (SystemExit, typer.Exit):
        # Normal CLI exit, don't handle as error
        raise
    except Exception as e:
        handle_cli_exception(e)
        raise typer.Exit(1)


@app.command()
def create_index(
    folders: list[str] = FOLDERS_REQUIRED_OPTION,
    index_dir: str = INDEX_DIR_DEFAULT_OPTION,
    force: bool = FORCE_OPTION,
) -> None:
    """Create semantic search index from source code."""
    try:
        orchestration_service = OrchestrationService()

        typer.echo("🏗️  Creating semantic search index...")

        results = asyncio.run(
            orchestration_service.run_index_creation_workflow(
                source_folders=folders, index_directory=index_dir, force_rebuild=force
            )
        )

        _display_index_creation_results(results)

        if results["status"] == "error":
            raise typer.Exit(1)
        else:
            typer.echo("✅ Index creation completed!")

    except (SystemExit, typer.Exit):
        # Normal CLI exit, don't handle as error
        raise
    except Exception as e:
        handle_cli_exception(e)
        raise typer.Exit(1)


@app.command()
def health_check() -> None:
    """Check system health and configuration."""
    try:
        orchestration_service = OrchestrationService()

        typer.echo("🏥 Running health check...")

        results = asyncio.run(orchestration_service.run_health_check_workflow())

        _display_health_check_results(results)

        if results.status == "unhealthy":
            raise typer.Exit(1)
        elif results.status == "error":
            raise typer.Exit(2)
        else:
            typer.echo("✅ System is healthy!")

    except (SystemExit, typer.Exit):
        # Normal CLI exit, don't handle as error
        raise
    except Exception as e:
        handle_cli_exception(e)
        raise typer.Exit(1)


def _display_test_generation_results(results: dict[str, Any]) -> None:
    """Display test generation results in user-friendly format."""
    typer.echo("\n📊 Test Generation Results:")
    typer.echo(f"  Status: {results['status']}")
    typer.echo(f"  Files processed: {results.get('files_processed', 0)}")
    typer.echo(f"  Tests generated: {results.get('tests_generated', 0)}")

    if results.get("workflow_duration_seconds"):
        typer.echo(f"  Duration: {results['workflow_duration_seconds']:.2f}s")

    # Display file-specific results
    file_results = results.get("file_results", {})
    if file_results:
        typer.echo("\n📁 File Results:")
        for file_path, file_result in file_results.items():
            status_icon = "✅" if file_result.get("test_generated") else "⚠️"
            typer.echo(f"  {status_icon} {file_path}: {file_result.get('status', 'unknown')}")

    # Display errors
    errors = results.get("errors", [])
    if errors:
        typer.echo("\n❌ Errors:")
        for error in errors:
            typer.echo(f"  • {error}")


def _display_index_creation_results(results: dict[str, Any]) -> None:
    """Display index creation results."""
    typer.echo("\n📚 Index Creation Results:")
    typer.echo(f"  Status: {results['status']}")

    if results["status"] == "error":
        typer.echo(f"  Error: {results.get('error', 'Unknown error')}")


def _display_health_check_results(results: HealthStatus) -> None:
    """Display health check results."""
    typer.echo("\n🏥 Health Check Results:")
    typer.echo(f"  Overall Status: {results.status}")

    for check_name, check_result in vars(results.checks).items():
        status_icon = "✅" if check_result.healthy else "❌"
        status_text = "Healthy" if check_result.healthy else "Unhealthy"
        typer.echo(f"  {status_icon} {check_name.title()}: {status_text}")

        if not check_result.healthy and hasattr(check_result, "error"):
            typer.echo(f"      Error: {check_result.error}")


if __name__ == "__main__":
    app()
