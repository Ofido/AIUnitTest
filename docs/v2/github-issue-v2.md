# AIUnitTest v2 Issue Draft

## Is your feature request related to a problem?

Yes.

The current AIUnitTest flow is built around direct test generation from uncovered lines.
That worked as an initial thesis, but it is now too weak for the actual problem the project is trying to solve.

Modern agentic tools are better because they do more than generate text:

- inspect the target
- build context
- propose a patch
- run validation tools
- repair failures
- report the final result

AIUnitTest currently stops too early in that loop.
It can produce tests, but it does not yet reliably produce validated test patches.

As a result, the current architecture is increasingly misaligned with the problem space.

## Describe the solution you'd like

Redesign AIUnitTest as **AIUnitTest v2**, a **tool-first test execution layer for coding agents**.

The v2 product should:

1. Select a target from coverage, diff, or explicit user input.
2. Build a focused context package from source code, tests, config, and recent failures.
3. Call an external reasoning backend such as Copilot CLI or Gemini CLI.
4. Receive a structured plan and patch candidate.
5. Apply the patch in a controlled workspace.
6. Run validators such as pytest, syntax checks, and optional coverage comparison.
7. Retry with feedback when validation fails.
8. Return a final patch and execution report.

This makes AIUnitTest responsible for orchestration, targeting, validation, guardrails,
and reporting, while delegating heavy reasoning to the best available backend.

The product direction is to become the specialized testing layer that works with coding agents,
instead of competing with them as a general-purpose agent.

## Describe alternatives you've considered

### 1. Keep improving the current direct generation flow

This would likely produce incremental gains, but it would not solve the structural gap.
The problem is not only prompt quality or provider quality. The product boundary itself is too narrow.

### 2. Turn the project into a generic PR code reviewer

This is broader, noisier, and less differentiated.
It would also move the project away from the original testing problem.

PR review should exist later as a surface of the same engine, not as the primary product thesis.

### 3. Build a VS Code extension first

That would increase implementation cost and ecosystem coupling too early.
The v2 engine should be CLI-first and reusable before any editor integration is attempted.

## Additional context

This issue proposes a product-level pivot, not just a refactor.

The intended v2 definition is:

- **What it is:** a hybrid architecture with a tool-first core, a CLI-first delivery path, and future MCP exposure
- **What it is not:** a thin LLM wrapper, a generic reviewer, or an editor-specific plugin

Delivery rule:

- architecture is hybrid
- delivery is sequential
- core and CLI come first
- CI and PR delivery come after the local loop is reliable
- MCP comes after the core contracts are stable

Initial non-negotiable guardrails:

- bounded retries
- test-file-first writes by default
- explicit opt-in for source edits
- persisted diffs and validator artifacts for every run
- no automatic commit, push, or merge in the core workflow

Planned implementation order:

1. Define contracts and v2 module boundaries
2. Build one local backend adapter
3. Build one validation loop
4. Prove the flow on a narrow real benchmark
5. Add PR mode via GitHub Actions

Initial benchmark recommendation:

- use a narrow Financas domain slice as a real corpus
- do not use the whole Financas app as the first target

Supporting docs for this issue:

- `docs/v2/README.md`
- `docs/v2/architecture.md`
- `docs/v2/implementation-plan.md`
