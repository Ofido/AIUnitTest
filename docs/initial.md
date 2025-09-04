# Initial Project Documentation

This section describes the initial idea, objectives, and architectural overview of the AIUnitTest project.

1. Objective
  The Auto Test Updater aims to automate the generation and updating of unit tests in Python projects,
  reducing manual work and continuously improving coverage.

2. Context
  Many teams struggle to keep test coverage high. Missing or outdated tests
  can lead to regressions and hinder refactoring. By integrating an LLM, we can:
    - Quickly detect untested lines.
    - Generate test cases consistent with the project's style.
    - Update existing tests when code logic changes.

3. Main Components
   - CLI (cli.py)
     - Argument parsing and configuration loading from pyproject.toml.
     - Manual mode vs. --auto.
   - Coverage Helper (coverage_helper.py)
     - Use Coverage.py API to collect uncovered lines.
     - Returns a mapping file → missing lines.
   - File Helper (file_helper.py)
     - Locate source modules and test files.
     - Read and write files.
   - LLM Integration (llm.py)
     - Asynchronous connection with OpenAI GPT to generate or complete tests.
     - Prompt assembly (system + user).
   - Orchestration (main.py)
     - Main asyncio flow: collect coverage, find tests, call LLM, and write results.
     - Logging and error handling.

4. Execution Flow

    ```mermaid
      flowchart TD
      A[Start CLI] --> B{--auto ?}
      B -- yes --> C[Load config from pyproject.toml]
      B -- no --> D[Use provided args]
      D --> E[Validate paths]
          C --> E
          E --> F[Collect Missing Lines]
      F --> G{Missing files?}
      G -- no --> H[End]
      G -- yes --> I[For each file]
          I --> J[Find Test File]
          J --> K[Read source & test]
      K --> L[Call LLM Async]
      L --> M[Write test file]
          M --> I
          M --> H
    ```

5. Success Criteria
   - Minimum coverage of X% after execution.
   - Generated tests pass without failures.
   - Acceptable execution time (< Y seconds for N files).

6. Future Roadmap
   - Phase 1: Context Improvement via Simple Search
     - Enrich the LLM prompt with examples of existing tests found by simple text search.
     - This grounds the model in the project's style and conventions, improving test quality.

   - Phase 2: RAG with a Simplified Embedding Module
     - Implement a semantic search system to find the most relevant test examples.
     - Create a command to index existing tests, generating embeddings and saving them locally.
     - Use embedding similarity search to find the best examples for the prompt.
     - Support for multiple LLM providers.
     - Automated generation of mocks and fixtures.
     - Integration with external CI pipelines (GitHub Apps, GitLab CI).
     - Web dashboard to visualize coverage progress.

   - Phase 3: Refactoring and Simplification of Existing Tests
     - The LLM can analyze existing tests and suggest refactoring to improve readability and conciseness,
       applying effective testing patterns (e.g., smart use of `pytest.fixture`, simplifying complex
       assertions, removing duplication).
     - Benefit: Improved maintainability of tests.

   - Phase 4: Intelligent Test Data Generation
     - The LLM can analyze function signatures, types, docstrings, and source code to suggest or generate
       test data (parameter values, mocked objects) that cover edge cases and different execution paths.
     - Benefit: Increased effectiveness of tests by covering a wider range of scenarios.
