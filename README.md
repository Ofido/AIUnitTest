# AIUnitTest

[![PyPI version](https://badge.fury.io/py/AIUnitTest.svg)](https://badge.fury.io/py/AIUnitTest)
[![Python versions](https://img.shields.io/pypi/pyversions/AIUnitTest.svg)](https://pypi.org/project/AIUnitTest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Linter: flake8](https://img.shields.io/badge/linter-flake8-blue.svg)](https://flake8.pycqa.org/en/latest/)
[![Static typing: mypy](https://img.shields.io/badge/static%20typing-mypy-blue.svg)](https://mypy-lang.org/)
[![Testing: pytest](https://img.shields.io/badge/testing-pytest-blue.svg)](https://pytest.org)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://coverage.readthedocs.io/)
[![CI/CD](https://github.com/ofido/AIUnitTest/actions/workflows/ci.yml/badge.svg)](https://github.com/ofido/AIUnitTest/actions/workflows/ci.yml)
[![GitHub last commit](https://img.shields.io/github/last-commit/ofido/AIUnitTest)](https://github.com/ofido/AIUnitTest/commits/main)
[![GitHub repo size](https://img.shields.io/github/repo-size/ofido/AIUnitTest)](https://github.com/ofido/AIUnitTest)
[![GitHub issues](https://img.shields.io/github/issues/ofido/AIUnitTest)](https://github.com/ofido/AIUnitTest/issues)

AIUnitTest is a command-line tool that reads your `pyproject.toml` and test coverage
data (`.coverage`) to generate and update missing Python unit tests using AI.

## How it Works

1. **Coverage Analysis**: The tool uses `coverage.py` to identify lines of code
    that are not covered by your existing test suite.
2. **Source Code Chunking**: It breaks down the source code into logical chunks
    (functions and classes).
3. **AI-Powered Test Generation**: For each chunk with uncovered lines,
    it sends the source code and the uncovered line numbers to an AI model
    (like OpenAI's GPT) to generate new test cases.
4. **Test File Updates**: The newly generated tests are appended to the
    corresponding test file.

## Features

- **Coverage Analysis**: Uses the Coverage.py API to identify untested lines.
- **AI-Powered Test Generation**: Calls OpenAI GPT to create or enhance test cases.
- **Config-Driven**: Automatically picks up `coverage.run.source` and
  `pytest.ini_options.testpaths` from `pyproject.toml`.
- **Auto Mode**: The `--auto` flag sets source and tests directories without
  manual arguments.
- **Async & Parallel**: Speeds up OpenAI requests for large codebases.

## Installation

There are two ways to install AIUnitTest:

### From PyPI

You can install the latest stable version from PyPI:

```bash
pip install AIUnitTest
```

### From GitHub (for the latest development version)

1. **Clone the repository:**

    ```bash
    git clone https://github.com/ofido/AIUnitTest.git
    cd AIUnitTest
    ```

2. **Install the project in editable mode:**

    ```bash
    pip install -e .
    ```

## Usage

### Automatic Mode

The easiest way to run the tool is in automatic mode.
It will automatically discover your source and test folders
from your `pyproject.toml` file.

```bash
ai-unit-test --auto
```

### Manual Mode

You can also specify the source and test folders manually:

```bash
ai-unit-test --folders src --tests-folder tests
```

### Generating a Test for a Specific Function

You can also generate a test for a single function:

```bash
ai-unit-test func my_module/my_file.py my_function
```

### Command-Line Options

- `--folders`: The source code folders to analyze.
- `--tests-folder`: The folder where the tests are located.
- `--coverage-file`: The path to the `.coverage` file.
- `--auto`: Try to discover folders/tests from `pyproject.toml`.

## Configuration

AIUnitTest uses the standard `pyproject.toml` file for configuration.
Here are the relevant sections:

- **`[tool.coverage.run]`**:
  - `source`: A list of source code folders.
- **`[tool.pytest.ini_options]`**:
  - `testpaths`: A list of test folders.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License
see the [LICENSE](LICENSE) file for details.
