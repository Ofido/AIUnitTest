from pathlib import Path


def read_file_content(file_path: Path) -> str:
    """Reads the content of a file."""
    with open(file_path) as f:
        return f.read()
