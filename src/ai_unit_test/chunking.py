import ast
import hashlib
import logging
from dataclasses import dataclass
from typing import Literal

from ai_unit_test.file_helper import read_file_content

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    name: str
    type: Literal["class", "function", "docstring", "module"]
    source_code: str
    start_line: int
    end_line: int
    file_path: str | None = None  # Make file_path optional
    chunk_id: str | None = None  # Make chunk_id optional
    content_hash: str | None = None  # Make content_hash optional
    text_preview: str | None = None  # Make text_preview optional


class ASTChunker:
    def __init__(self: "ASTChunker", max_chunk_lines: int = 150, overlap_lines: int = 20) -> None:
        self.max_chunk_lines = max_chunk_lines
        self.overlap_lines = overlap_lines

    def chunk_file(self: "ASTChunker", file_path: str) -> list[Chunk]:
        """Chunks a Python file using AST parsing."""
        file_content = read_file_content(file_path)
        if not file_content:
            return []

        try:
            tree = ast.parse(file_content)
            return self._chunk_node(tree, file_path, file_content)
        except SyntaxError as e:
            logger.error(f"Error parsing {file_path}: {e}")
            return []

    def _chunk_node(self: "ASTChunker", node: ast.Module, file_path: str, file_content: str) -> list[Chunk]:
        chunks: list[Chunk] = []
        last_end: int = 0
        for child_node in node.body:
            if isinstance(child_node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                # Handle code before the current node as a module-level chunk
                if child_node.lineno - 1 > last_end:
                    module_chunk_code = "\n".join(
                        file_content.splitlines()[last_end : child_node.lineno - 1],
                    )
                    if module_chunk_code.strip():
                        chunks.append(
                            self._create_chunk(
                                name="module-level",
                                type="module",
                                source_code=module_chunk_code,
                                start_line=last_end + 1,
                                end_line=child_node.lineno - 1,
                                file_path=file_path,
                            )
                        )

                chunks.extend(self._process_structured_node(child_node, file_path, file_content))
                if child_node.end_lineno:
                    last_end = child_node.end_lineno

        # Handle any remaining code after the last structured node
        if len(file_content.splitlines()) > last_end:
            module_chunk_code = "\n".join(file_content.splitlines()[last_end:])
            if module_chunk_code.strip():
                chunks.append(
                    self._create_chunk(
                        name="module-level",
                        type="module",
                        source_code=module_chunk_code,
                        start_line=last_end + 1,
                        end_line=len(file_content.splitlines()),
                        file_path=file_path,
                    )
                )

        return chunks

    def _process_structured_node(
        self: "ASTChunker",
        node: ast.AST,
        file_path: str,
        file_content: str,
        *,
        signature: str | None = None,
    ) -> list[Chunk]:
        # Gracefully handle unsupported node types
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            return []
        chunks: list[Chunk] = []
        source_segment = ast.get_source_segment(file_content, node) or ""

        # Handle docstring as a separate chunk
        docstring = ast.get_docstring(node)
        if docstring:
            chunks.append(
                self._create_chunk(
                    name=f"{node.name} docstring",
                    type="docstring",
                    source_code=docstring,
                    start_line=node.lineno,
                    end_line=node.lineno,  # Approximate line number
                    file_path=file_path,
                )
            )

        # If a signature override is provided, ensure the source includes it at the top
        if signature and not source_segment.startswith(signature):
            source_segment = signature + source_segment

        # Handle the node itself
        if source_segment:
            chunks.append(
                self._create_chunk(
                    name=node.name,
                    type="class" if isinstance(node, ast.ClassDef) else "function",
                    source_code=source_segment,
                    start_line=node.lineno,
                    end_line=node.end_lineno or node.lineno,
                    file_path=file_path,
                )
            )
        return chunks

    def _split_large_chunk(
        self: "ASTChunker",
        node: ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef,
        file_path: str,
        file_content: str,
    ) -> list[Chunk]:
        """Splits a large chunk into smaller overlapping sub-chunks."""
        source_segment = ast.get_source_segment(file_content, node)
        if not source_segment:
            return []

        lines = source_segment.splitlines(True)
        if not lines:
            return []

        # Preserve the signature of the function/class in each sub-chunk
        signature = ""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Extract just the signature line(s)
            signature_end_line = node.body[0].lineno - 1 if node.body else node.lineno
            signature_lines = file_content.splitlines(True)[node.lineno - 1 : signature_end_line]
            signature = "".join(signature_lines)
        elif isinstance(node, ast.ClassDef):
            # Extract class definition line
            signature_end_line = node.body[0].lineno - 1 if node.body else node.lineno
            signature_lines = file_content.splitlines(True)[node.lineno - 1 : signature_end_line]
            signature = "".join(signature_lines)

        sub_chunks: list[Chunk] = []
        start_line_offset = 0
        part_index = 1

        while start_line_offset < len(lines):
            end_line_offset = start_line_offset + self.max_chunk_lines
            chunk_lines = lines[start_line_offset:end_line_offset]

            # Prepend signature to sub-chunks, avoiding duplication if it's already there
            chunk_content = "".join(chunk_lines)
            if start_line_offset > 0 and signature:
                chunk_content = signature + "    ...\n" + chunk_content

            sub_chunks.append(
                self._create_chunk(
                    name=f"{node.name}-part{part_index}",
                    type="function" if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "class",
                    source_code=chunk_content,
                    start_line=node.lineno + start_line_offset,
                    end_line=node.lineno + start_line_offset + len(chunk_lines) - 1,
                    file_path=file_path,
                    part_index=part_index,
                )
            )
            start_line_offset += self.max_chunk_lines - self.overlap_lines
            part_index += 1
        return sub_chunks

    def _create_chunk(
        self: "ASTChunker",
        name: str,
        type: Literal["class", "function", "docstring", "module"],
        source_code: str,
        start_line: int,
        end_line: int,
        file_path: str,
        part_index: int | None = None,
    ) -> Chunk:
        # Build a human-readable chunk_id that starts with the file location (tests assert startswith)
        chunk_id_str = f"{file_path}:{start_line}:{end_line}"
        if part_index:
            chunk_id_str += f":part_{part_index}"

        return Chunk(
            name=name,
            type=type,
            source_code=source_code,
            start_line=start_line,
            end_line=end_line,
            file_path=file_path,
            chunk_id=chunk_id_str,
            content_hash=hashlib.sha256(source_code.encode()).hexdigest(),
            text_preview=source_code[:250] + "..." if len(source_code) > 250 else source_code,
        )


def chunk_test_file(file_path: str) -> list[Chunk]:
    """Chunks a test file into a list of logical blocks (functions or classes)."""
    chunker = ASTChunker()
    return chunker.chunk_file(file_path)
