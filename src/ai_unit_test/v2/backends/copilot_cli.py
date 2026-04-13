"""Copilot CLI reasoning backend for AIUnitTest v2."""

import asyncio
import json
import logging
import re

from ai_unit_test.v2.models import ContextBundle, PatchCandidate

logger = logging.getLogger(__name__)


class CopilotCliBackend:
    """Invoke GitHub Copilot CLI as a reasoning backend."""

    name: str = "copilot-cli"

    async def propose_patch(self, context: ContextBundle) -> PatchCandidate:
        """Send context to Copilot CLI and parse a patch candidate."""
        prompt = self._build_prompt(context)

        try:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "copilot",
                "suggest",
                "-t",
                "shell",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(input=prompt.encode()), timeout=300)
            raw_output = stdout.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Copilot CLI exited with code {proc.returncode}: {err_text or raw_output[:200]}")

            if not raw_output.strip():
                raise RuntimeError("Copilot CLI returned empty output.")
        except FileNotFoundError:
            raise RuntimeError("gh CLI not found. Install GitHub CLI and authenticate with 'gh auth login'.")
        except TimeoutError:
            raise RuntimeError("Copilot CLI timed out after 300 seconds.")

        return self._parse_response(raw_output)

    def _build_prompt(self, context: ContextBundle) -> str:
        """Build a structured prompt from a ContextBundle."""
        parts: list[str] = []
        parts.append("Generate Python unit tests for the following source code.")
        parts.append('Respond with JSON: {"plan_summary": "...", "files": {"path": "content"}}')
        parts.append("")

        for path, content in context.source_snippets.items():
            parts.append(f"--- Source: {path} ---")
            parts.append(content)
            parts.append("")

        for path, content in context.test_snippets.items():
            parts.append(f"--- Existing tests: {path} ---")
            parts.append(content)
            parts.append("")

        if context.validator_feedback:
            parts.append("--- Previous attempt feedback ---")
            for line in context.validator_feedback:
                parts.append(line)
            parts.append("")

        if context.previous_patch:
            parts.append("--- Previous patch (failed) ---")
            parts.append(context.previous_patch)
            parts.append("")

        return "\n".join(parts)

    def _parse_response(self, raw_output: str) -> PatchCandidate:
        """Parse backend output into a PatchCandidate."""
        # Try JSON parsing first
        try:
            data = json.loads(raw_output)
            return self._from_json(data)
        except (json.JSONDecodeError, KeyError):
            pass

        # Try extracting JSON block from fenced code
        json_match = re.search(r"```json\s*\n(.*?)```", raw_output, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return self._from_json(data)
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: extract fenced python blocks
        python_blocks = re.findall(r"```python\s*\n(.*?)```", raw_output, re.DOTALL)
        if python_blocks:
            patch_text = "\n\n".join(python_blocks)
            return PatchCandidate(
                backend_name=self.name,
                plan_summary="Extracted from fenced Python blocks.",
                patch_text=patch_text,
                touched_files=[],
            )

        return PatchCandidate(
            backend_name=self.name,
            plan_summary="Could not parse structured output.",
            patch_text=raw_output,
            touched_files=[],
        )

    def _from_json(self, data: dict[str, object]) -> PatchCandidate:
        """Build PatchCandidate from a JSON response."""
        plan_summary = str(data.get("plan_summary", ""))
        files: dict[str, str] = data.get("files", {})  # type: ignore[assignment]

        patch_lines: list[str] = []
        touched: list[str] = []
        for path, content in files.items():
            patch_lines.append(f"--- file: {path}")
            patch_lines.append(str(content))
            touched.append(path)

        return PatchCandidate(
            backend_name=self.name,
            plan_summary=plan_summary,
            patch_text="\n".join(patch_lines),
            touched_files=touched,
        )

    @staticmethod
    async def is_available() -> bool:
        """Check if the Copilot CLI backend is available."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "gh",
                "copilot",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except FileNotFoundError:
            return False
