"""Base contracts for AIUnitTest v2 reasoning backends."""

import abc
import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Protocol

from ai_unit_test.v2.models import ContextBundle, PatchCandidate

logger = logging.getLogger(__name__)

# ANSI escape sequences (colors, cursor moves, OSC, etc.) emitted by
# interactive CLI tools that we want to clean before parsing.
_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


class ReasoningBackend(Protocol):
    """Contract for external reasoning backends used by AIUnitTest v2."""

    name: str

    async def propose_patch(self, context: ContextBundle) -> PatchCandidate:
        """Return a plan and patch candidate for the provided context."""
        ...


class BackendRegistry:
    """In-memory registry for v2 reasoning backends."""

    def __init__(self) -> None:
        """Initialize an empty backend registry."""
        self._backends: dict[str, ReasoningBackend] = {}

    def register(self, backend: ReasoningBackend) -> None:
        """Register a backend instance by its name."""
        self._backends[backend.name] = backend

    def get(self, name: str) -> ReasoningBackend:
        """Retrieve a registered backend by name."""
        return self._backends[name]

    def list_names(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends.keys())


class BaseCliBackend(abc.ABC):
    """Shared scaffold for CLI-based reasoning backends.

    Subclasses must implement:
      - `name` (class attribute)
      - `_subprocess_argv()` returning the argv to invoke the CLI
      - `_subprocess_stdin(prompt)` returning bytes to feed via stdin
      - `is_available()` static method

    The base class handles prompt construction, subprocess invocation with
    timeout, output sanitization, JSON extraction, and artifact persistence
    on parse failures.
    """

    name: str = "base-cli"
    timeout_seconds: float = 300.0
    artifacts_dir: Path = Path("logs/v2-backends")

    async def propose_patch(self, context: ContextBundle) -> PatchCandidate:
        """Send context to the underlying CLI and parse a patch candidate."""
        prompt = self._build_prompt(context)
        argv = self._subprocess_argv(prompt)
        stdin_payload = self._subprocess_stdin(prompt)

        logger.debug("[%s] invoking: %s", self.name, " ".join(argv))
        logger.debug("[%s] prompt size: %d bytes", self.name, len(prompt))

        raw_output = await self._run_subprocess(argv, stdin_payload)
        cleaned = _ANSI_RE.sub("", raw_output)

        try:
            return self._parse_response(cleaned)
        except _ParseFailure as exc:
            artifact_path = self._dump_artifact(raw_output, prompt)
            logger.warning(
                "[%s] could not parse structured output (%s); raw saved to %s",
                self.name,
                exc,
                artifact_path,
            )
            return PatchCandidate(
                backend_name=self.name,
                plan_summary=f"Could not parse structured output. Raw saved to {artifact_path}",
                patch_text=cleaned,
                touched_files=[],
            )

    async def _run_subprocess(self, argv: list[str], stdin_payload: bytes | None) -> str:
        """Run the subprocess with timeout and return decoded stdout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin_payload is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(self._not_found_message()) from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=stdin_payload),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            with _suppress_exception():
                proc.kill()
                await proc.communicate()
            raise RuntimeError(
                f"{self.name} timed out after {self.timeout_seconds:.0f} seconds."
            ) from exc

        raw_output = stdout.decode("utf-8", errors="replace")
        err_text = stderr.decode("utf-8", errors="replace").strip()

        if proc.returncode != 0:
            raise RuntimeError(
                f"{self.name} exited with code {proc.returncode}: "
                f"{err_text or raw_output[:200] or '<no output>'}"
            )

        if not raw_output.strip():
            raise RuntimeError(f"{self.name} returned empty output.")

        return raw_output

    def _build_prompt(self, context: ContextBundle) -> str:
        """Build a structured prompt that instructs the model to emit JSON."""
        parts: list[str] = [
            "You are AIUnitTest, a tool that generates Python unit tests for the pytest framework.",
            "",
            "RULES:",
            "  - Generate ONLY pytest-style unit tests (no `unittest.TestCase` unless requested).",
            "  - Prefer fixtures, parametrize, and isolated assertions.",
            "  - Cover normal cases, edge cases, and error paths.",
            "  - File paths in your response MUST be repository-relative (e.g. tests/test_foo.py),",
            "    NEVER absolute and NEVER prefixed with ./.",
            "  - By default, only files matching test_*.py or *_test.py may be created or modified.",
            "  - Do not import the module under test from a path that does not exist in the source.",
            "  - Do not invent helpers, fixtures, or fixtures files that are not provided.",
            "",
            "RESPONSE FORMAT (strict):",
            "  Return a single JSON object, no prose, no markdown fences:",
            '    {"plan_summary": "<one paragraph>",',
            '     "files": {"<repo-relative-path>": "<full file content>"}}',
            "  The `files` map must contain the FULL desired content of each file (not a diff).",
            "",
        ]

        if context.target.files:
            parts.append(f"TARGET FILES: {', '.join(context.target.files)}")
            parts.append("")

        for path, content in context.source_snippets.items():
            parts.append(f"--- Source: {path} ---")
            parts.append(content)
            parts.append("")

        for path, content in context.test_snippets.items():
            parts.append(f"--- Existing tests: {path} ---")
            parts.append(content)
            parts.append("")

        if context.project_config:
            for path, content in context.project_config.items():
                parts.append(f"--- Project config: {path} ---")
                parts.append(content)
                parts.append("")

        if context.validator_feedback:
            parts.append("--- Previous attempt feedback ---")
            parts.extend(context.validator_feedback)
            parts.append("")

        if context.previous_patch:
            parts.append("--- Previous patch (failed) ---")
            parts.append(context.previous_patch)
            parts.append("")

        return "\n".join(parts)

    def _parse_response(self, raw_output: str) -> PatchCandidate:
        """Parse backend output into a PatchCandidate.

        Strategy: try direct JSON parse, then fenced ```json``` block, then
        balanced-brace extraction. Falls back to fenced ```python``` blocks
        as a partial result. Raises `_ParseFailure` when nothing usable is
        found so the caller can persist the raw output for debugging.
        """
        # 1. Direct JSON
        candidate = _try_json(raw_output)
        if candidate is not None:
            return self._from_json(candidate)

        # 2. Fenced ```json block
        json_match = re.search(r"```json\s*\n(.*?)```", raw_output, re.DOTALL)
        if json_match:
            candidate = _try_json(json_match.group(1))
            if candidate is not None:
                return self._from_json(candidate)

        # 3. Balanced brace extraction
        extracted = _extract_json_object(raw_output)
        if extracted is not None:
            candidate = _try_json(extracted)
            if candidate is not None:
                return self._from_json(candidate)

        # 4. Partial fallback: fenced python blocks
        python_blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", raw_output, re.DOTALL)
        if python_blocks:
            patch_text = "\n\n".join(python_blocks)
            return PatchCandidate(
                backend_name=self.name,
                plan_summary="Extracted from fenced Python blocks (no path info).",
                patch_text=patch_text,
                touched_files=[],
            )

        raise _ParseFailure("no JSON object or python fenced block found")

    def _from_json(self, data: dict[str, object]) -> PatchCandidate:
        """Build PatchCandidate from a JSON response."""
        plan_summary = str(data.get("plan_summary", ""))
        raw_files = data.get("files", {})
        if not isinstance(raw_files, dict):
            raise _ParseFailure(f"`files` must be an object, got {type(raw_files).__name__}")

        patch_lines: list[str] = []
        touched: list[str] = []
        for path, content in raw_files.items():
            patch_lines.append(f"--- file: {path}")
            patch_lines.append(str(content))
            touched.append(str(path))

        return PatchCandidate(
            backend_name=self.name,
            plan_summary=plan_summary,
            patch_text="\n".join(patch_lines),
            touched_files=touched,
        )

    def _dump_artifact(self, raw_output: str, prompt: str) -> Path:
        """Persist raw output and prompt for offline debugging."""
        try:
            self.artifacts_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S")
            path = self.artifacts_dir / f"{self.name}-{stamp}.txt"
            path.write_text(
                f"=== prompt ({len(prompt)} bytes) ===\n{prompt}\n\n"
                f"=== raw output ({len(raw_output)} bytes) ===\n{raw_output}\n",
                encoding="utf-8",
            )
            return path
        except OSError as exc:
            logger.warning("Failed to dump backend artifact: %s", exc)
            return Path("<artifact-write-failed>")

    @abc.abstractmethod
    def _subprocess_argv(self, prompt: str) -> list[str]:
        """Return the argv used to invoke the underlying CLI."""

    def _subprocess_stdin(self, prompt: str) -> bytes | None:
        """Return bytes to feed via stdin. Default: feed the prompt."""
        return prompt.encode("utf-8")

    def _not_found_message(self) -> str:
        """Return the user-facing message when the CLI binary is missing."""
        return f"{self.name} CLI not found."


class _ParseFailure(Exception):
    """Raised internally when a backend response cannot be parsed."""


def _try_json(text: str) -> dict[str, object] | None:
    """Attempt to JSON-decode text, returning None on failure."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced top-level JSON object in `text`.

    Handles strings with escaped quotes. Returns the JSON substring or None
    if no balanced object is found.
    """
    start = text.find("{")
    while start != -1:
        end = _find_balanced_close(text, start)
        if end is not None:
            return text[start : end + 1]
        start = text.find("{", start + 1)
    return None


def _find_balanced_close(text: str, start: int) -> int | None:
    """Return the index of the `}` that closes the `{` at `text[start]`."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            in_str, escape = _advance_string(ch, escape)
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _advance_string(ch: str, escape: bool) -> tuple[bool, bool]:
    """Advance one character inside a JSON string literal.

    Returns (still_in_string, next_escape_state).
    """
    if escape:
        return True, False
    if ch == "\\":
        return True, True
    if ch == '"':
        return False, False
    return True, False


class _suppress_exception:
    """Context manager that swallows any exception (used for best-effort cleanup)."""

    def __enter__(self) -> "_suppress_exception":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return True
