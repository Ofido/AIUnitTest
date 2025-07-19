from pathlib import Path

import typer

from ai_unit_test.coverage_helper import process_coverage_data
from ai_unit_test.file_helper import read_file_content
from ai_unit_test.llm import get_llm_response

app = typer.Typer()


@app.command()
def analyze(file_path: Path) -> None:
    """Analyzes a file to generate unit tests."""
    file_content = read_file_content(file_path)
    prompt = f"Generate unit tests for the following code:\n\n{file_content}"
    llm_response = get_llm_response(prompt)
    print(f"LLM Response: {llm_response}")

    # This is a placeholder for coverage analysis
    coverage_data = "..."
    processed_data = process_coverage_data(coverage_data)
    print(f"Coverage Data: {processed_data}")
