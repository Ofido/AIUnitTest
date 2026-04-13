# AIUnitTest v2

AIUnitTest v2 is a redesign around a different product category.

v1 is a direct AI test generator driven by missing coverage.
v2 is a tool-first test execution layer for coding agents.

It should become the specialized testing layer that works with coding agents,
instead of competing with them as a general-purpose agent.

The goal is not to make the current generator slightly better.
The goal is to turn a testing target into a validated patch with explicit control
over target selection, context assembly, validator execution, retry policy,
and reporting.

## Official positioning

AIUnitTest v2 should be understood as:

- a test-focused orchestration engine
- a local CLI client built on top of that engine
- a future MCP-friendly tool surface for external agents
- a backend-agnostic system for reasoning
- a validation-first workflow that aims to produce a patch, not just generated text

AIUnitTest v2 is not:

- a generic coding agent
- a thin wrapper around a single model or provider
- a VS Code extension pretending to be the product
- a PR review bot for every possible concern
- a replacement for Copilot CLI, Gemini CLI, or similar agentic runtimes

## Delivery principle

The architecture is hybrid.
The delivery should be sequential.

That means:

- build the core engine first
- ship the first usable workflow through the CLI
- add CI and PR delivery after the local loop is reliable
- expose MCP only after the core contracts are stable

This avoids a common failure mode where "hybrid" is interpreted as
"build every surface at the same time."

## Product promise

Given one of these targets:

- uncovered code
- a git diff
- an explicit file or symbol
- a failing test

AIUnitTest v2 should:

1. decide what to fix first
2. assemble only the context needed for that target
3. request a patch proposal from a reasoning backend or external agent
4. apply the patch under explicit guardrails
5. run validators and collect structured feedback
6. retry when the first attempt fails
7. emit a report that can be reviewed locally or in CI

## Why this shape exists

Modern coding agents are getting better at reasoning, but that does not remove
the testing problem. It shifts the product boundary.

The scarce part is no longer raw code generation.
The scarce part is test-specific orchestration:

- selecting the right target
- choosing the minimum useful context
- enforcing validation
- coordinating retries
- producing audit-friendly diffs and reports

That is the part AIUnitTest v2 should own.

## Product surfaces

### Core engine

The core engine owns:

- target selection
- context building
- patch application
- validation
- retry policy
- report generation

This is the real product.

### CLI client

The CLI is the first shipping surface because it is the fastest way to prove value in:

- local development
- benchmark runs
- CI and headless execution
- demos

The CLI should call the core engine. It should not contain the core logic.

### Future MCP surface

The future MCP surface should expose stable, test-specific capabilities from the same core.
That lets Copilot, Gemini, Claude, or custom agents use AIUnitTest as a testing tool
instead of forcing AIUnitTest to compete as a general agent.

### CI and PR reporting

CI and PR integrations should reuse the same run artifacts and reporters produced by the core.
They are delivery surfaces, not separate products.

## Reasoning strategy

AIUnitTest v2 keeps reasoning backend-agnostic.

Initial backend targets:

- Copilot CLI
- Gemini CLI

Possible future backends:

- OpenAI API
- local models
- MCP-mediated or SDK-backed runtimes

This means AIUnitTest does not try to out-think the strongest general agent.
It delegates general reasoning and keeps ownership of test-focused execution.

## Risks and guardrails

The v2 strategy only works if the product enforces guardrails instead of trusting raw model output.

The first cut should treat these as non-negotiable:

- bounded retry limits
- test-file-first writes by default
- explicit opt-in for source edits
- persisted diffs and validator artifacts for every run
- no automatic commit, push, or merge behavior in the core workflow
- validation summaries that make failures reviewable instead of opaque

## First useful scope

The first useful scope for v2 is intentionally narrow:

- explicit file targeting
- one backend adapter
- one patch application path
- targeted pytest validation
- bounded retry loop
- terminal and JSON reporting

That is enough to prove the new thesis without widening into PR bots,
editor plugins, or generic review automation.

## Docs in this folder

- `architecture.md` defines the concrete layer split and interfaces
- `implementation-plan.md` defines the bounded MVP and expansion path

## Current status

The v2 MVP is implemented and tested. The following components are functional:

- **Core models:** `RunRequest`, `TargetSpec`, `ContextBundle`, `PatchCandidate`, `PatchApplication`, `ValidationResult`, `RunReport`
- **Target selection:** `ExplicitFileSelector` for explicit file targeting
- **Context building:** `FileContextBuilder` reads source, related tests, and project config
- **Backend adapters:** `CopilotCliBackend` and `GeminiCliBackend` (subprocess-based, JSON-first parsing)
- **Patch application:** `PatchApplier` with test-file-first guardrails, rollback between retries
- **Validation:** `SyntaxValidator` (py_compile) and `PytestValidator` (subprocess, overrides global addopts)
- **Feedback:** `FeedbackSummarizer` for structured retry context
- **Reporting:** `RunStore` (persists report.json, summary.md, patch.diff), `TerminalRenderer`, `JsonRenderer`
- **Orchestrator:** `V2Orchestrator` wires the full loop with bounded retries
- **CLI:** `v2 run`, `v2 report`, `v2 backends`, `v2 doctor` registered under the `v2` namespace

Work stays isolated from v1. The v1 CLI remains fully functional.
