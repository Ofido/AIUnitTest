"""CLI interface for AI Unit Test - Pure presentation layer."""

import asyncio
import logging
from typing import List, Optional

import typer

from ai_unit_test.services.orchestration_service import OrchestrationService

logger = logging.getLogger(__name__)
app = typer.Typer()


@app.command()
def generate_tests(
    folders: list[str] | None = typer.Option(
        None, "--folders", "-f", help="Source code folders to analyze for coverage"
    ),
    tests_folder: str | None = typer.Option(None, "--tests-folder", "-t", help="Directory containing test files"),
    coverage_file: str = typer.Option(".coverage", "--coverage-file", "-c", help="Path to coverage data file"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-discover configuration from pyproject.toml"),
    index_dir: str | None = typer.Option(None, "--index-dir", help="Directory containing semantic search index"),
) -> None:
    """Generate unit tests for uncovered code using AI."""

    # Create orchestration service
    config = {"indexing": {"index_directory": index_dir}} if index_dir else {}
    orchestration_service = OrchestrationService(config)

    # Run workflow
    typer.echo("🚀 Starting test generation...")

    results = asyncio.run(
        orchestration_service.run_test_generation_workflow(
            folders=folders, tests_folder=tests_folder, coverage_file=coverage_file, auto_discovery=auto
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


@app.command()
def create_index(
    folders: list[str] = typer.Option(..., "--folders", "-f", help="Source code folders to index"),
    index_dir: str = typer.Option("data/faiss_index", "--index-dir", help="Directory to save the index"),
    force: bool = typer.Option(False, "--force", help="Force rebuild even if index exists"),
) -> None:
    """Create semantic search index from source code."""

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


@app.command()
def health_check() -> None:
    """Check system health and configuration."""

    orchestration_service = OrchestrationService()

    typer.echo("🏥 Running health check...")

    results = asyncio.run(orchestration_service.run_health_check_workflow())

    _display_health_check_results(results)

    if results["status"] == "unhealthy":
        raise typer.Exit(1)
    elif results["status"] == "error":
        raise typer.Exit(2)
    else:
        typer.echo("✅ System is healthy!")


@app.command()
def generate_tests(
    folders: list[str] | None = typer.Option(
        None, "--folders", "-f", help="Source code folders to analyze for coverage"
    ),
    tests_folder: str | None = typer.Option(None, "--tests-folder", "-t", help="Directory containing test files"),
    coverage_file: str = typer.Option(".coverage", "--coverage-file", "-c", help="Path to coverage data file"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-discover configuration from pyproject.toml"),
    index_dir: str | None = typer.Option(None, "--index-dir", help="Directory containing semantic search index"),
) -> None:
    """Generate unit tests for uncovered code using AI."""

    try:
        # Create orchestration service
        config = {"indexing": {"index_directory": index_dir}} if index_dir else {}
        orchestration_service = OrchestrationService(config)

        # Run workflow
        typer.echo("🚀 Starting test generation...")

        results = asyncio.run(
            orchestration_service.run_test_generation_workflow(
                folders=folders, tests_folder=tests_folder, coverage_file=coverage_file, auto_discovery=auto
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
    folders: list[str] = typer.Option(..., "--folders", "-f", help="Source code folders to index"),
    index_dir: str = typer.Option("data/faiss_index", "--index-dir", help="Directory to save the index"),
    force: bool = typer.Option(False, "--force", help="Force rebuild even if index exists"),
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

        if results["status"] == "unhealthy":
            raise typer.Exit(1)
        elif results["status"] == "error":
            raise typer.Exit(2)
        else:
            typer.echo("✅ System is healthy!")

    except (SystemExit, typer.Exit):
        # Normal CLI exit, don't handle as error
        raise
    except Exception as e:
        handle_cli_exception(e)
        raise typer.Exit(1)


def _display_test_generation_results(results: dict) -> None:
    """Display test generation results in user-friendly format."""

    typer.echo(f"\n📊 Test Generation Results:")
    typer.echo(f"  Status: {results['status']}")
    typer.echo(f"  Files processed: {results.get('files_processed', 0)}")
    typer.echo(f"  Tests generated: {results.get('tests_generated', 0)}")

    if results.get("workflow_duration_seconds"):
        typer.echo(f"  Duration: {results['workflow_duration_seconds']:.2f}s")

    # Display file-specific results
    file_results = results.get("file_results", {})
    if file_results:
        typer.echo(f"\n📁 File Results:")
        for file_path, file_result in file_results.items():
            status_icon = "✅" if file_result.get("test_generated") else "⚠️"
            typer.echo(f"  {status_icon} {file_path}: {file_result.get('status', 'unknown')}")

    # Display errors
    errors = results.get("errors", [])
    if errors:
        typer.echo(f"\n❌ Errors:")
        for error in errors:
            typer.echo(f"  • {error}")


def _display_index_creation_results(results: dict) -> None:
    """Display index creation results."""

    typer.echo(f"\n📚 Index Creation Results:")
    typer.echo(f"  Status: {results['status']}")

    if results["status"] == "error":
        typer.echo(f"  Error: {results.get('error', 'Unknown error')}")


def _display_health_check_results(results: dict) -> None:
    """Display health check results."""

    typer.echo(f"\n🏥 Health Check Results:")
    typer.echo(f"  Overall Status: {results['status']}")

    checks = results.get("checks", {})
    for check_name, check_result in checks.items():
        status_icon = "✅" if check_result.get("healthy") else "❌"
        typer.echo(f"  {status_icon} {check_name.title()}: {'Healthy' if check_result.get('healthy') else 'Unhealthy'}")

        if not check_result.get("healthy") and "error" in check_result:
            typer.echo(f"      Error: {check_result['error']}")


# Legacy command aliases for backward compatibility
@app.command()
def main(
    folders: list[str] | None = typer.Option(
        None, "--folders", "-f", help="Source code folders to analyze for coverage"
    ),
    tests_folder: str | None = typer.Option(None, "--tests-folder", "-t", help="Directory containing test files"),
    coverage_file: str = typer.Option(".coverage", "--coverage-file", "-c", help="Path to coverage data file"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-discover configuration from pyproject.toml"),
) -> None:
    """Legacy alias for generate_tests command."""
    typer.echo("⚠️  'main' command is deprecated. Use 'generate-tests' instead.")

    # Call the new command with same parameters
    generate_tests(folders=folders, tests_folder=tests_folder, coverage_file=coverage_file, auto=auto, index_dir=None)


@app.command()
def index(
    tests_folder: str | None = typer.Option(None, "--tests-folder", "-t", help="Directory containing test files"),
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-discover configuration from pyproject.toml"),
    index_dir: str = typer.Option("data/faiss_index", "--index-dir", help="Directory to save the index"),
) -> None:
    """Legacy alias for create_index command."""
    typer.echo("⚠️  'index' command is deprecated. Use 'create-index' instead.")

    # For backward compatibility, derive folders from tests_folder
    if tests_folder:
        folders = [tests_folder]
    else:
        # Use auto-discovery
        try:
            from ai_unit_test.services.configuration_service import ConfigurationService

            config_service = ConfigurationService()
            config_data = config_service.load_pyproject_config()
            source_folders, _, _ = config_service.extract_source_configuration(config_data)
            folders = source_folders or ["src"]
        except Exception:
            folders = ["src"]

    create_index(folders=folders, index_dir=index_dir, force=False)


@app.command()
def search(
    query: str,
    index_dir: str = typer.Option("data/faiss_index", "--index-dir", help="Directory containing the search index"),
    k: int = typer.Option(5, "--k", help="Number of results to return"),
    threshold: float = typer.Option(0.7, "--threshold", help="Similarity threshold"),
) -> None:
    """Search for similar code in the index."""

    try:
        from ai_unit_test.semantic_search import search as semantic_search

        typer.echo(f"🔍 Searching for: '{query}'")
        results = semantic_search(query, index_dir, k, threshold)

        if not results:
            typer.echo("No results found.")
            return

        typer.echo(f"Found {len(results)} results:")
        for i, (result_meta, distance) in enumerate(results):
            typer.echo(
                f"  {i+1}. Similarity: {distance:.4f} | "
                f"{result_meta['source_filepath']}:{result_meta['start_line']}-{result_meta['end_line']}"
            )
            typer.echo(f"      Preview: {result_meta['text_preview'].strip()}")

    except ImportError:
        typer.echo("❌ Search functionality not available. Make sure FAISS is installed.")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Search failed: {e}")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
