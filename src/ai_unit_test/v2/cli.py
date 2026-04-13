"""CLI commands for AIUnitTest v2 (incubation namespace)."""

import asyncio
import logging

import typer

from ai_unit_test.v2.backends.base import BackendRegistry
from ai_unit_test.v2.backends.copilot_cli import CopilotCliBackend
from ai_unit_test.v2.backends.gemini_cli import GeminiCliBackend
from ai_unit_test.v2.context.builder import FileContextBuilder
from ai_unit_test.v2.models import RunRequest
from ai_unit_test.v2.orchestrator import V2Orchestrator
from ai_unit_test.v2.patching.workspace import PatchApplier
from ai_unit_test.v2.reporting.renderer import JsonRenderer, TerminalRenderer
from ai_unit_test.v2.reporting.store import RunStore
from ai_unit_test.v2.targeting.selectors import ExplicitFileSelector
from ai_unit_test.v2.validation.feedback import FeedbackSummarizer
from ai_unit_test.v2.validation.runners import PytestValidator, SyntaxValidator

logger = logging.getLogger(__name__)

v2_app = typer.Typer(name="v2", help="AIUnitTest v2 — tool-first test execution layer (incubation).")

FILE_OPTION = typer.Option(..., "--file", "-f", help="Target source file to generate tests for.")
BACKEND_OPTION = typer.Option("copilot-cli", "--backend", "-b", help="Reasoning backend to use.")
MAX_ATTEMPTS_OPTION = typer.Option(3, "--max-attempts", help="Maximum retry attempts.")
DRY_RUN_OPTION = typer.Option(False, "--dry-run", help="Inspect proposal without writing files.")
ALLOW_SOURCE_EDITS_OPTION = typer.Option(False, "--allow-source-edits", help="Allow edits to non-test files.")
JSON_OUTPUT_OPTION = typer.Option(False, "--json", help="Output report as JSON.")


def _build_registry() -> BackendRegistry:
    """Build and populate the backend registry."""
    registry = BackendRegistry()
    registry.register(CopilotCliBackend())
    registry.register(GeminiCliBackend())
    return registry


@v2_app.command()
def run(
    file: str = FILE_OPTION,
    backend: str = BACKEND_OPTION,
    max_attempts: int = MAX_ATTEMPTS_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    allow_source_edits: bool = ALLOW_SOURCE_EDITS_OPTION,
    json_output: bool = JSON_OUTPUT_OPTION,
) -> None:
    """Run the v2 test generation workflow on a target file."""
    try:
        registry = _build_registry()

        try:
            backend_instance = registry.get(backend)
        except KeyError:
            typer.echo(f"❌ Unknown backend: {backend}. Available: {', '.join(registry.list_names())}", err=True)
            raise typer.Exit(1)

        request = RunRequest(
            file_path=file,
            backend_name=backend,
            max_attempts=max_attempts,
            dry_run=dry_run,
            allow_source_edits=allow_source_edits,
        )

        orchestrator = V2Orchestrator(
            backend=backend_instance,
            target_selector=ExplicitFileSelector(),
            context_builder=FileContextBuilder(),
            patch_applier=PatchApplier(test_patterns=request.test_patterns),
            validators=[SyntaxValidator(), PytestValidator()],
            feedback_summarizer=FeedbackSummarizer(),
            run_store=RunStore(),
        )

        typer.echo("🚀 Starting v2 run...")
        report = asyncio.run(orchestrator.run(request))

        if json_output:
            typer.echo(JsonRenderer().render(report))
        else:
            typer.echo(TerminalRenderer().render(report))

        if not report.success:
            raise typer.Exit(1)

    except typer.Exit:
        raise
    except Exception as e:
        logger.error("v2 run failed: %s", e)
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(1)


REPORT_JSON_OPTION = typer.Option(False, "--json", help="Output report as JSON.")


@v2_app.command()
def report(
    json_output: bool = REPORT_JSON_OPTION,
) -> None:
    """Display the last run report."""
    store = RunStore()
    last_report = store.load_last_report()

    if last_report is None:
        typer.echo("No previous runs found.")
        raise typer.Exit(1)

    if json_output:
        typer.echo(JsonRenderer().render(last_report))
    else:
        typer.echo(TerminalRenderer().render(last_report))


@v2_app.command("backends")
def list_backends() -> None:
    """List available reasoning backends."""
    registry = _build_registry()
    typer.echo("Available backends:")
    for name in registry.list_names():
        typer.echo(f"  • {name}")


@v2_app.command()
def doctor() -> None:
    """Check v2 backend availability and system readiness."""
    typer.echo("🏥 AIUnitTest v2 Doctor\n")

    checks = [
        ("copilot-cli", CopilotCliBackend.is_available),
        ("gemini-cli", GeminiCliBackend.is_available),
    ]

    all_ok = True
    for name, check_fn in checks:
        available = asyncio.run(check_fn())
        icon = "✅" if available else "❌"
        status = "available" if available else "not found"
        typer.echo(f"  {icon} {name}: {status}")
        if not available:
            all_ok = False

    typer.echo("")
    if all_ok:
        typer.echo("All backends available.")
    else:
        typer.echo("Some backends are missing. Install them to use all features.")
