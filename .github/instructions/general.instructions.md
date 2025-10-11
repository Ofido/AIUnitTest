---
applyTo: "**"
---

# AIUnitTest – Copilot & Developer Instructions

## Running Tests

1. **Activate the virtual environment:**

   ```bash
   source env/bin/activate
   ```

2. **Install dependencies (if needed):**

   ```bash
   make install-dev
   ```

3. **Run tests:**

   - All unit tests (default):

     ```bash
     make test
     ```

   - Only unit tests:

     ```bash
     make test-unit
     ```

   - Unit tests with coverage report:

     ```bash
     make test-unit-cov
     ```

     - Coverage HTML report: `htmlcov/index.html`

   - Integration tests:

     ```bash
     make test-integration
     ```

   - All tests (including slow):

     ```bash
     make test-all
     ```

   - Fast tests (excluding slow):

     ```bash
     make test-fast
     ```

   - Watch mode (auto re-run on file change):

     ```bash
     make test-watch
     ```

   - Using global pytest (not recommended):

     ```bash
     make test-simple
     ```

> **Note:** All test commands require the virtual environment to be active and dependencies installed.

## Project Structure

- **Source code:** `src/ai_unit_test/`
- **Tests:** `tests/`
- **Configs:** `pyproject.toml`, `requirements.txt`
- **Docs:** `README.md`, `docs/`

## Coding Conventions

- Use descriptive names for functions and variables.
- Cover normal and edge cases in tests.
- Follow mypy standards (see: mypy-lang.org/) and configured `flake8`/`bandit` rules.
- Never skip pre-commit hooks.
- Always run commands inside the project environment (`source env/bin/activate`).
- Follow clean code best practices.

## Quick Onboarding

- Clone the repo and enter the project folder.

- Create and activate the virtual environment:

  ```bash
  python3 -m venv env
  source env/bin/activate
  ```

- Install all dependencies:

  ```bash
  make install-dev
  ```

- Run tests and check code quality before committing.

---

If you have questions, check the documentation or ask your team!
