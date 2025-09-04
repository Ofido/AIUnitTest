"""
Final version of fixed tests for semantic_search.py
"""

from builtins import isinstance as original_isinstance
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_unit_test.semantic_search import search


def test_search_with_faiss_index() -> None:
    """Tests search with a mocked FAISS index"""

    # Mock FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (
        np.array([[0.9, 0.5, 0.1]]),  # scores (already similarity, not distances)
        np.array([[0, 1, 2]]),  # indices
    )
    mock_index.d = 4  # index dimension

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata (chunks)
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance to return True only for FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN001
            # For the mock index and FAISS, return True  # noqa: ANN201
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            # For other cases, use original isinstance
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Run search with a low threshold to include all results
        results = search("test query", "dummy_dir", k=3, threshold=0.0)

    # Checks - function returns a list of tuples (metadata, score)
    assert len(results) == 3
    assert results[0] == ("chunk0", 0.9)  # type: ignore[comparison-overlap] # direct score from FAISS
    assert results[1] == ("chunk1", 0.5)  # type: ignore[comparison-overlap] # direct score from FAISS
    assert results[2] == ("chunk2", 0.1)  # type: ignore[comparison-overlap] # direct score from FAISS


def test_search_with_sklearn_index() -> None:
    """Tests search with a mocked sklearn index"""
    # Mock sklearn index (no search method, using kneighbors)
    mock_index = MagicMock()
    del mock_index.search  # Remove search to force sklearn path
    mock_index.kneighbors.return_value = (np.array([[0.1, 0.5, 0.9]]), np.array([[0, 1, 2]]))  # distances  # indices

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Run search with a low threshold to include all results
        results = search("test query", "dummy_dir", k=3, threshold=0.0)

        # Checks - sklearn converts distances to scores with 1 - distance
        assert len(results) == 3
        assert results[0][0] == "chunk0"  # type: ignore[comparison-overlap]
        assert abs(results[0][1] - 0.9) < 1e-6  # 1 - 0.1
        assert results[1][0] == "chunk1"  # type: ignore[comparison-overlap]
        assert abs(results[1][1] - 0.5) < 1e-6  # 1 - 0.5
        assert results[2][0] == "chunk2"  # type: ignore[comparison-overlap]
        assert abs(results[2][1] - 0.1) < 1e-6  # 1 - 0.9


def test_search_with_threshold_filtering() -> None:
    """Tests filtering by threshold"""

    # Mock FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.9, 0.5, 0.1]]), np.array([[0, 1, 2]]))  # scores  # indices
    mock_index.d = 4  # index dimension

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance to return True only for FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Run search with a high threshold (only scores > 0.8)
        results = search("test query", "dummy_dir", k=3, threshold=0.8)

        # Checks - only the first result should pass the threshold
        assert len(results) == 1
        assert results[0] == ("chunk0", 0.9)  # type: ignore[comparison-overlap]


def test_search_empty_results() -> None:
    """Tests search that returns empty results due to threshold"""

    # Mock FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.1, 0.05, 0.01]]), np.array([[0, 1, 2]]))  # low scores  # indices
    mock_index.d = 4  # index dimension

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance to return True only for FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore  # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Run search with a high threshold
        results = search("test query", "dummy_dir", k=3, threshold=0.5)

        # Checks - no result should pass the threshold
        assert len(results) == 0


def test_search_invalid_index_dir() -> None:
    """Tests error when the index directory is invalid"""
    with patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load:
        mock_load.side_effect = FileNotFoundError("Index directory not found")

        # Running search should raise an exception
        with pytest.raises(FileNotFoundError, match="Index directory not found"):
            search("test query", "invalid_dir", k=3)


def test_search_dimension_mismatch() -> None:
    """Tests error when dimensions don't match for FAISS"""

    # Mock FAISS index
    mock_index = MagicMock()
    mock_index.d = 10  # index expects 10 dimensions

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
        patch("ai_unit_test.semantic_search.FAISS_AVAILABLE", True),
        patch("ai_unit_test.semantic_search.isinstance") as mock_isinstance,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model that returns wrong dimension (4D instead of 10D)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Mock isinstance to return True for FAISS
        def mock_isinstance_func(obj, cls):  # type: ignore # noqa: ANN201, ANN001
            if obj is mock_index and hasattr(cls, "__name__") and "Index" in str(cls):
                return True
            return original_isinstance(obj, cls)

        mock_isinstance.side_effect = mock_isinstance_func

        # Running search should raise an exception due to incompatible dimension
        with pytest.raises(ValueError, match="query embedding dimension .* incompatible"):
            search("test query", "dummy_dir", k=3)


def test_search_unsupported_index_type() -> None:
    """Tests error when index type is not supported"""
    # Mock an index that is neither FAISS nor sklearn
    mock_index = MagicMock()
    del mock_index.search  # Remove search
    del mock_index.kneighbors  # Remove kneighbors

    # Mock manifest
    mock_manifest = {"embedding_model": "all-MiniLM-L6-v2"}

    # Mock metadata
    mock_metadata = ["chunk0", "chunk1", "chunk2"]

    with (
        patch("ai_unit_test.semantic_search.load_faiss_index") as mock_load,
        patch("ai_unit_test.semantic_search.SentenceTransformer") as mock_transformer,
    ):

        mock_load.return_value = (mock_index, mock_metadata, mock_manifest)

        # Mock model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
        mock_transformer.return_value = mock_model

        # Running search should raise unsupported type exception
        with pytest.raises(TypeError, match="Unsupported index type"):
            search("test query", "dummy_dir", k=3)
