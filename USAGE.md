# How to Run

## Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/AIUnitTest.git
   cd AIUnitTest
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv env
   source env/bin/activate
   ```

3. **Install the project:**

   ```bash
   pip install .
   ```

   Or with optional dependencies:

   ```bash
   # Install with HuggingFace support
   pip install .[huggingface]

   # Install with FAISS indexing support
   pip install .[faiss]

   # Install all optional features
   pip install .[all]

   # Install development dependencies
   pip install .[dev]
   ```

## Optional Features

AIUnitTest includes several optional features:

- **`huggingface`**: Enables HuggingFace model support for test generation
- **`faiss`**: Enables semantic search and indexing of tests using FAISS
- **`all`**: Installs all optional features
- **`dev`**: Installs development tools (linters, formatters, type checkers)

## Running the Tool

To run the AI Unit Test tool, use the following command:

```bash
   ai-unit-test --help
```

This will display the available options and commands.

## Indexing Tests

To index your existing tests for semantic search, run the following command:

```bash
ai-unit-test index
```

This will create a FAISS index of your tests in the `.ai_unit_test_cache/` directory.

## Searching Tests

Once you have indexed your tests, you can search for tests related to a specific query:

```bash
ai-unit-test search "my search query"
```
