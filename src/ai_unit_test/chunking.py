from langchain.text_splitter import PythonCodeTextSplitter

from ai_unit_test.file_helper import Chunk, read_file_content


def chunk_test_file(file_path: str) -> list[Chunk]:
    """Chunks a test file into a list of logical blocks (functions or classes)."""
    code = read_file_content(file_path)
    if not code:
        return []

    python_splitter = PythonCodeTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = python_splitter.create_documents([code])

    chunks: list[Chunk] = []
    for doc in docs:
        # For simplicity, we'll use the first line of the chunk as the name
        # and assume the chunk type is a function.
        # A more sophisticated approach would be to parse the chunk content.
        chunk_name = doc.page_content.split("\n")[0]
        chunks.append(
            Chunk(
                name=chunk_name,
                type="function",
                source_code=doc.page_content,
                start_line=0,  # Line numbers are not easily available from langchain splitter
                end_line=0,
            )
        )

    return chunks
