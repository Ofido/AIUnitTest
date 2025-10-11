
# Gemini CLI Instructions

## How to run the tests

- **Always activate the project virtual environment first:**

    ```bash
    source env/bin/activate
    ```

- **To run all unit tests (default):**

    ```bash
    make test
    ```

- **To run only unit tests:**

    ```bash
    make test-unit
    ```

- **To run unit tests with coverage report:**

    ```bash
    make test-unit-cov
    ```

- The HTML coverage report will be generated at `htmlcov/index.html`.

- **To run integration tests:**

    ```bash
    make test-integration
    ```

- **To run all tests (including slow ones):**

    ```bash
    make test-all
    ```

- **To run only fast tests (excluding slow ones):**

    ```bash
    make test-fast
    ```

- **To run tests in watch mode (auto re-run on file change):**

    ```bash
    make test-watch
    ```

- **If you want to run tests using the global pytest (not recommended):**

    ```bash
    make test-simple
    ```

- **Tip:** All test commands require the virtual environment to be active and dependencies installed.
    If you see errors about missing dependencies, run:

    ```bash
    make install-dev
    ```

- **Test files are located in the `tests/` folder.**

## Project Structure

- Source code: `src/ai_unit_test/`
- Tests: `tests/`
- Configurations: `pyproject.toml`, `requirements.txt`
- Documentation: `README.md`, `docs/`

## Conventions

- Follow mypy standards for Python according to the configured flake8 and bandit rules.
- Use descriptive names for functions and variables.
- Tests must cover normal and edge cases.
- Never skip pre-commit.
- Always run commands inside the project env (`source env/bin/activate`).
- Follow clean code best practices.
- Always think as much as necessary.
