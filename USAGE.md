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

This will create a FAISS index of your tests in the `data/` directory.

## Searching Tests

Once you have indexed your tests, you can search for tests related to a specific query:

```bash
ai-unit-test search "my search query"
```
