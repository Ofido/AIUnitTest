# AIUnitTest v2 Implementation Plan

## Product objective

AIUnitTest v2 must prove one thing first:
given a concrete testing target, it can produce a validated patch and a reviewable report
without making the user manually stitch together context, commands, and retries.

## Delivery rule

The architecture is hybrid, but delivery is sequential.

Order matters:

- core engine first
- CLI client second
- CI and PR delivery after the local loop works
- MCP only after the core contracts and artifacts are stable

This keeps the MVP from collapsing under too many surfaces at once.

## MVP boundary

The first cut must stay intentionally narrow.

Included in the MVP:

- explicit file targeting
- one backend adapter
- one controlled patch application path
- targeted pytest validation
- bounded retry loop
- persisted terminal and JSON reporting

Explicitly out of the MVP:

- IDE integration
- generic PR review across all concerns
- multiple backends in the same run
- MCP server delivery
- full coverage discovery automation as the primary entry point

## MVP interfaces

The MVP should standardize these core models:

- `TargetSpec`
- `ContextBundle`
- `PatchCandidate`
- `ValidationResult`
- `RunReport`

The MVP should introduce these implementation interfaces around them:

- `TargetSelector`
- `ContextBuilder`
- `PatchApplier`
- `Validator`
- `RunStore`

## MVP command surface

The first public command surface for v2 should stay namespaced so v1 remains stable.

```text
ai-unit-test v2 run --file path/to/module.py --backend copilot-cli
ai-unit-test v2 run --file path/to/module.py --backend gemini-cli --max-attempts 2
ai-unit-test v2 run --file path/to/module.py --backend copilot-cli --dry-run
ai-unit-test v2 report --last-run
ai-unit-test v2 backends list
ai-unit-test v2 doctor
```

Important MVP flags:

- `--file` to define the initial target
- `--backend` to choose the reasoning adapter
- `--max-attempts` to bound retries
- `--dry-run` to inspect the proposal without writing files
- `--json` to emit machine-readable output
- `--allow-source-edits` kept off by default

## MVP execution contract

One full MVP run should do the following:

1. accept an explicit file target
2. build context from source, nearby tests, project config, and optional prior feedback
3. request a patch candidate from the selected backend
4. apply the patch with test-first write guardrails
5. run targeted validation
6. retry on failure up to the configured bound
7. persist artifacts and emit a summary

## MVP validation criteria

The MVP is usable only if all of these conditions are met:

- a successful run can create or update tests for one explicit file target
- the run emits a reviewable diff and a machine-readable report
- failed validation is summarized in structured form, not only as raw terminal noise
- at least one retry path is exercised in a controlled benchmark
- the tool exits clearly on success, recoverable failure, and hard failure
- v1 commands remain unaffected

## MVP safety guardrails

The MVP should enforce these constraints from day one:

- retries are bounded by configuration
- writes default to test files only
- source edits require explicit opt-in
- every run saves artifacts for later inspection
- no commit or push behavior is part of the core run command

Suggested mandatory artifacts per run:

- `report.json`
- `summary.md`
- `patch.diff`

## Implementation sequence

### 1. Definition and contracts

Deliverables:

- v2 docs and issue
- stable core models
- isolated v2 module namespace

Exit criteria:

- the product shape is clear enough to implement without re-litigating scope

### 2. Local single-backend workflow

Deliverables:

- one backend adapter
- one context builder
- one patch applier path
- one pytest validator
- one persisted run report

Exit criteria:

- v2 can run locally on an explicit file target and produce artifacts

### 3. Bounded repair loop

Deliverables:

- validation feedback summarization
- retry controller
- stable failed-run reporting

Exit criteria:

- v2 can recover from at least one failed attempt in a controlled benchmark

### 4. Diff-aware expansion

Deliverables:

- diff target selector
- changed-file context builder
- scoped test patch reporting

Exit criteria:

- v2 can focus on changed code without falling back to whole-project context

### 5. CI and PR delivery

Deliverables:

- GitHub Actions path or equivalent CI runner
- markdown reporter for PR surfaces

Exit criteria:

- v2 can publish a useful report outside the local terminal

## First benchmark recommendation

Use a narrow Financas domain slice, not the whole application.

Good candidates are modules with:

- clear business rules
- deterministic outcomes
- high value from regression protection

Avoid starting with:

- OCR pipelines
- sync infrastructure
- UI-heavy flows

## What not to build now

- a VS Code-first experience
- a generic code reviewer persona
- a custom orchestration platform for unrelated workflows
- a broad MCP surface before the core contracts are stable
- a large benchmark corpus before the first explicit-file path works
