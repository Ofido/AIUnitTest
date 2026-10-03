"""Gemini CLI reasoning backend for AIUnitTest v2.

Invokes the `gemini` CLI in non-interactive mode. The prompt is fed via
stdin to avoid blowing past `ARG_MAX` for large context bundles, and
`-o json` is requested so the wire format is structured output rather
than mixed terminal markup.

Requirements at runtime:
  - `gemini` CLI installed and authenticated.
"""

import asyncio
import json
import logging
import os

from ai_unit_test.v2.backends.base import BaseCliBackend, _ParseFailure
from ai_unit_test.v2.models import PatchCandidate

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiCliBackend(BaseCliBackend):
    """Invoke Gemini CLI as a reasoning backend."""

    name: str = "gemini-cli"

    def __init__(self, model: str | None = None) -> None:
        """Configure the model.

        Resolution order: explicit `model` arg > `AI_UNIT_TEST_GEMINI_MODEL`
        env var > `DEFAULT_MODEL`.
        """
        self.model = model or os.environ.get("AI_UNIT_TEST_GEMINI_MODEL", DEFAULT_MODEL)

    def _subprocess_argv(self, prompt: str) -> list[str]:
        """Return argv. Prompt body is supplied via stdin (see `_subprocess_stdin`)."""
        return [
            "gemini",
            "--yolo",
            "--model",
            self.model,
            "--output-format",
            "json",
        ]

    def _not_found_message(self) -> str:
        return (
            "gemini CLI not found. Install with `npm install -g @google/gemini-cli` "
            "and authenticate before retrying."
        )

    def _parse_response(self, raw_output: str) -> PatchCandidate:
        """Parse Gemini-specific structured output before falling back to base parsing.

        With `--output-format json`, Gemini wraps the model's response in a
        top-level envelope. We try to unwrap it first, then delegate to the
        base parser to extract our own JSON contract.
        """
        envelope: dict[str, object] | None = None
        try:
            data = json.loads(raw_output)
            if isinstance(data, dict):
                envelope = data
        except json.JSONDecodeError:
            envelope = None

        if envelope is not None:
            inner = _extract_response_text(envelope)
            if inner:
                try:
                    return super()._parse_response(inner)
                except _ParseFailure:
                    pass

        return super()._parse_response(raw_output)

    @staticmethod
    async def is_available() -> bool:
        """Check if the Gemini CLI binary is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "gemini",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False


def _extract_response_text(envelope: dict[str, object]) -> str | None:
    """Best-effort extraction of the textual response from Gemini's JSON envelope.

    The envelope shape has shifted across Gemini CLI releases; we probe a few
    known keys and return the first non-empty string found.
    """
    for key in ("response", "text", "output", "content"):
        value = envelope.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None
