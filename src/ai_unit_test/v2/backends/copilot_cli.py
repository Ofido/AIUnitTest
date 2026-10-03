"""GitHub Copilot reasoning backend for AIUnitTest v2.

This backend invokes `gh models run <model>`, which is the non-interactive,
automation-friendly entry point for GitHub-hosted models (provided by the
`gh-models` extension). The previous implementation used
`gh copilot suggest`, but that command only generates shell/git/gh commands
and is interactive, so it cannot produce Python tests.

Requirements at runtime:
  - GitHub CLI authenticated: `gh auth login`
  - Models extension installed: `gh extension install github/gh-models`
"""

import asyncio
import logging
import os

from ai_unit_test.v2.backends.base import BaseCliBackend

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"


class CopilotCliBackend(BaseCliBackend):
    """Invoke `gh models run` as a reasoning backend."""

    name: str = "copilot-cli"

    def __init__(self, model: str | None = None) -> None:
        """Configure the model.

        Resolution order: explicit `model` arg > `AI_UNIT_TEST_COPILOT_MODEL`
        env var > `DEFAULT_MODEL`.
        """
        self.model = model or os.environ.get("AI_UNIT_TEST_COPILOT_MODEL", DEFAULT_MODEL)

    def _subprocess_argv(self, prompt: str) -> list[str]:
        """Return argv for `gh models run <model>`. Prompt is fed via stdin."""
        return ["gh", "models", "run", self.model]

    def _not_found_message(self) -> str:
        return (
            "gh CLI or the gh-models extension is not available. Install GitHub CLI, "
            "authenticate with 'gh auth login', and run "
            "'gh extension install github/gh-models'."
        )

    @staticmethod
    async def is_available() -> bool:
        """Check that gh, the gh-models extension, and auth are all set up."""
        if not await _gh_command_ok(["gh", "--version"]):
            return False

        ext_stdout = await _gh_command_capture(["gh", "extension", "list"])
        if ext_stdout is None or b"gh-models" not in ext_stdout:
            return False

        return await _gh_command_ok(["gh", "auth", "status"])


async def _gh_command_ok(argv: list[str]) -> bool:
    """Return True if argv exits with code 0."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        return proc.returncode == 0
    except FileNotFoundError:
        return False


async def _gh_command_capture(argv: list[str]) -> bytes | None:
    """Run argv and return stdout bytes on success, None on failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            return None
        return stdout
    except FileNotFoundError:
        return None
