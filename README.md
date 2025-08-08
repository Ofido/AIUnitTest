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
