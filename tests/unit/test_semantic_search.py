"""
Testes corrigidos para semantic_search.py - versão final
"""

from builtins import isinstance as original_isinstance
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_unit_test.semantic_search import search


def test_search_with_faiss_index() -> None:
    """Testa busca com índice FAISS mockado"""

    # Mock do índice FAISS
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.9, 0.5, 0.1]]),  # scores (já convertidos, não distâncias)
        np.array([[0, 1, 2]]),  # índices
    )
    mock_index.d = 4  # dimensão do índice

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata (chunks)
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance para retornar True apenas para FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN001
            # Para o índice mock e FAISS, retornar True  # noqa: ANN201
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            # Para outros casos, usar isinstance original
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Executar busca com threshold baixo para pegar todos os resultados
        results = search("test query", "dummy_dir", k=3, threshold=0.0)

    # Checks - function returns a list of tuples (metadata, score)
    assert len(results) == 3
    assert results[0] == ("chunk0", 0.9)  # direct score from FAISS
    assert results[1] == ("chunk1", 0.5)  # direct score from FAISS
    assert results[2] == ("chunk2", 0.1)  # direct score from FAISS


def test_search_with_sklearn_index() -> None:
    """Testa busca com índice sklearn mockado"""
    # Mock do índice sklearn (sem método search, com kneighbors)
    mock_index = MagicMock()
    del mock_index.search  # Remove search para forçar uso do sklearn
    mock_index.kneighbors.return_value = (np.array([[0.1, 0.5, 0.9]]), np.array([[0, 1, 2]]))  # distâncias  # índices

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Executar busca com threshold baixo para pegar todos os resultados
        results = search("test query", "dummy_dir", k=3, threshold=0.0)

        # Verificações - sklearn converte distâncias para scores com 1 - distance
        assert len(results) == 3
        assert results[0][0] == "chunk0"
        assert abs(results[0][1] - 0.9) < 1e-6  # 1 - 0.1
        assert results[1][0] == "chunk1"
        assert abs(results[1][1] - 0.5) < 1e-6  # 1 - 0.5
        assert results[2][0] == "chunk2"
        assert abs(results[2][1] - 0.1) < 1e-6  # 1 - 0.9


def test_search_with_threshold_filtering() -> None:
    """Testa filtragem por threshold"""

    # Mock do índice FAISS
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.9, 0.5, 0.1]]), np.array([[0, 1, 2]]))  # scores  # índices
    mock_index.d = 4  # dimensão do índice

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance para retornar True apenas para FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Executar busca com threshold alto (só vai pegar scores > 0.8)
        results = search("test query", "dummy_dir", k=3, threshold=0.8)

        # Verificações - apenas o primeiro resultado deve passar no threshold
        assert len(results) == 1
        assert results[0] == ("chunk0", 0.9)


def test_search_empty_results() -> None:
    """Testa busca que retorna resultados vazios devido ao threshold"""

    # Mock do índice FAISS
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.1, 0.05, 0.01]]), np.array([[0, 1, 2]]))  # scores baixos  # índices
    mock_index.d = 4  # dimensão do índice

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance para retornar True apenas para FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Executar busca com threshold alto
        results = search("test query", "dummy_dir", k=3, threshold=0.5)

        # Verificações - nenhum resultado deve passar no threshold
        assert len(results) == 0


def test_search_invalid_index_dir() -> None:
    """Testa erro quando diretório do índice é inválido"""
    with patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load:
        mock_load.side_effect = FileNotFoundError("Index directory not found")

        # Executar busca deve lançar exceção
        with pytest.raises(FileNotFoundError, match="Index directory not found"):
            search("test query", "invalid_dir", k=3)


def test_search_dimension_mismatch() -> None:
    """Testa erro quando dimensões não batem com FAISS"""

    # Mock do índice FAISS
    mock_index = MagicMock()
    mock_index.d = 10  # índice espera 10 dimensões

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo que retorna dimensão errada (4D em vez de 10D)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance para retornar True para FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Executar busca deve lançar exceção por dimensão incompatível
        with pytest.raises(ValueError, match="query embedding dimension .* incompatible"):
            search("test query", "dummy_dir", k=3)


def test_search_unsupported_index_type() -> None:
    """Testa erro quando tipo de índice não é suportado"""
    # Mock de um índice que não é nem FAISS nem sklearn
    mock_index = MagicMock()
    del mock_index.search  # Remove search
    del mock_index.kneighbors  # Remove kneighbors

    # Mock do manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock dos metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock do modelo
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Executar busca deve lançar exceção de tipo não suportado
        with pytest.raises(TypeError, match="Unsupported index type"):
            search("test query", "dummy_dir", k=3)
