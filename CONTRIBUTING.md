# Contributing to AIUnitTest

First off, thank you for considering contributing to AIUnitTest! It's people like you that make open source such a
great community.

We welcome any type of contribution, not only code. You can help with a, documentation, reporting bugs, or suggesting
new features.

## Getting Started

### 1. Fork the Repository

If you don't have write access, start by forking the repository and then clone your fork to your local machine:

```bash
git clone https://github.com/YOUR_USERNAME/AIUnitTest.git
cd AIUnitTest
```

### 2. Set Up the Development Environment

We use `pip` and a virtual environment to manage dependencies. To get your environment set up, follow these steps:

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

2. **Install the project in editable mode with development dependencies:**

    This will install the project itself, plus all the tools needed for testing and linting
    (`pytest`, `black`, `flake8`, etc.).

    ```bash
    pip install -e .[dev]
    ```

### 3. Install Pre-Commit Hooks

We use `pre-commit` to automatically enforce code style and quality before you even commit. After installing the dependencies,
set up the git hooks:

```bash
pre-commit install
```

Now, `pre-commit` will run automatically on every `git commit`.

## Development Workflow

1. **Create a new branch:**

    Create a new branch from `main` for your feature or bug fix:

    ```bash
    git checkout -b your-feature-name
    ```

2. **Make your changes:**

    Write your code, add your tests, and make sure everything works as expected.

3. **Run tests:**

    Before submitting, ensure that all tests pass:

    ```bash
    pytest
    ```

4. **Commit your changes:**

    When you commit, the `pre-commit` hooks will run. If they fail, you'll need to fix the issues and `git add` the files
    again before successfully committing.

5. **Push to your fork:**

    ```bash
    git push origin your-feature-name
    ```

6. **Open a Pull Request:**

    Go to the original AIUnitTest repository on GitHub and open a new Pull Request. The PR template will be automatically
    populated. Please fill it out as completely as possible.

## Code Style

- **Code formatting:** We use `black` for code formatting.
- **Import sorting:** We use `isort`.
- **Linting:** We use `flake8`.
- **Static Typing:** We use `mypy`.

All of these are enforced automatically by the `pre-commit` hooks, so you don't have to worry about running them manually.

## Questions?

If you have any questions, feel free to open an issue or start a discussion.
