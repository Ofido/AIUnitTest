# AIUnitTest

AIUnitTest is a command-line tool that reads your `pyproject.toml` and
test coverage data (`.coverage`) to generate and update missing Python
unit tests using AI.

## Features

- **Coverage Analysis**: Uses Coverage.py API to identify untested lines.
- **AI-Powered Test Generation**: Calls OpenAI GPT to create or enhance test cases.
- **Config-Driven**: Automatically picks up `coverage.run.source` and `pytest.ini_options.testpaths` from `pyproject.toml`.
- **Auto Mode**: `--auto` flag sets source and tests directories without manual arguments.
- **Async & Parallel**: Speeds up OpenAI requests for large codebases.

## How to Run

1. **Install the project:**

   ```bash
   pip install .
   ```

2. **Run the script:**

   ```bash
   ai-unit-test --auto
   ```
