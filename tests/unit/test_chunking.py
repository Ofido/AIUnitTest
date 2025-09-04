import ast
from pathlib import Path

from ai_unit_test.chunking import ASTChunker, chunk_test_file


def test_chunk_test_file() -> None:
    # Create a dummy test file
    dummy_test_file = Path("dummy_test.py")
    dummy_test_file.write_text(
        """def test_addition():
    assert 1 + 1 == 2

class TestClass:
    def test_subtraction(self):
        assert 2 - 1 == 1
"""
    )

    chunks = chunk_test_file(str(dummy_test_file))

    assert len(chunks) == 2
    assert chunks[0].name == "test_addition"
    assert chunks[1].name == "TestClass"

    dummy_test_file.unlink()


def test_chunk_file_empty_content() -> None:
    chunker = ASTChunker()
    result = chunker.chunk_file("non_existent_file.py")
    assert result == []


def test_chunk_file_syntax_error() -> None:
    dummy_test_file = Path("dummy_syntax_error.py")
    dummy_test_file.write_text("def test_function(:\n    pass\n")

    chunker = ASTChunker()
    result = chunker.chunk_file(str(dummy_test_file))
    assert result == []

    dummy_test_file.unlink()


def test_create_chunk() -> None:
    chunker = ASTChunker()
    chunk = chunker._create_chunk(
        name="test_function",
        type="function",
        source_code="def test_function(): pass",
        start_line=1,
        end_line=1,
        file_path="dummy_file.py",
    )
    assert chunk.name == "test_function"
    assert chunk.type == "function"
    assert chunk.start_line == 1
    assert chunk.end_line == 1
    assert chunk.file_path == "dummy_file.py"


def test_chunk_test_file_with_non_python_extension() -> None:
    dummy_test_file = Path("dummy_test.txt")
    dummy_test_file.write_text("This is not a Python test file.")

    chunks = chunk_test_file(str(dummy_test_file))

    assert len(chunks) == 0

    dummy_test_file.unlink()


def test_create_chunk_with_part_index() -> None:
    chunker = ASTChunker()
    chunk = chunker._create_chunk(
        name="test_function",
        type="function",
        source_code="def test_function(): pass",
        start_line=1,
        end_line=1,
        file_path="dummy_file.py",
        part_index=1,
    )
    assert chunk.name == "test_function"


def test_chunk_file_with_valid_python_content() -> None:
    dummy_test_file = Path("dummy_valid.py")
    dummy_test_file.write_text(
        """def test_valid_function():
    assert True
"""
    )

    chunker = ASTChunker()
    result = chunker.chunk_file(str(dummy_test_file))
    assert len(result) > 0
    assert result[0].name == "test_valid_function"

    dummy_test_file.unlink()


def test_chunk_test_file_with_invalid_file_path() -> None:
    invalid_file_path = "invalid_path.py"
    chunks = chunk_test_file(invalid_file_path)
    assert chunks == []


def test_split_large_chunk_function() -> None:
    # Função com 10 linhas (simula uma função grande)
    source_code = (
        "def big_function():\n"
        "    a = 1\n"
        "    b = 2\n"
        "    c = 3\n"
        "    d = 4\n"
        "    e = 5\n"
        "    f = 6\n"
        "    g = 7\n"
        "    h = 8\n"
        "    i = 9\n"
    )
    file_path = "dummy_big.py"
    # Parse AST
    tree = ast.parse(source_code)
    func_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef))

    # Chunker com chunk pequeno para testar divisão
    chunker = ASTChunker(max_chunk_lines=4, overlap_lines=2)
    sub_chunks = chunker._split_large_chunk(func_node, file_path, source_code)

    # Deve dividir em partes com sobreposição
    assert len(sub_chunks) > 1
    # Verifica se o nome está correto
    assert sub_chunks[0].name.startswith("big_function-part")
    # Verifica se a assinatura está presente nos sub-chunks
    for i, chunk in enumerate(sub_chunks):
        if i > 0:
            assert "def big_function()" in chunk.source_code
            assert "..." in chunk.source_code

    # Verifica se as linhas estão corretas e não ultrapassam o total
    for chunk in sub_chunks:
        assert chunk.start_line >= func_node.lineno
        assert chunk.end_line <= func_node.lineno + len(source_code.splitlines()) - 1
