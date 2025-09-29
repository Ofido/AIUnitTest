"""Test cases for the sklearn index organizer."""

import asyncio
import tempfile
from importlib import import_module
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from ai_unit_test.core.exceptions import IndexError


class FakeNN:
    """Fake NearestNeighbors for testing."""

    def __init__(self, algorithm: str | None = None, metric: str | None = None, n_jobs: int | None = None) -> None:
        """Initialize FakeNN."""
        self.algorithm = algorithm
        self.metric = metric
        self.n_jobs = n_jobs
        self.fitted = False
        self.X: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Fit the model."""
        self.fitted = True
        self.X = np.asarray(X)

    def kneighbors(self, query: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
        """Find the k-nearest neighbors."""
        X = self.X if self.X is not None else np.zeros((1, query.shape[1]))
        k = min(n_neighbors, X.shape[0])
        indices = np.arange(k).reshape(1, k)
        # distances: smaller is closer. For cosine metric use zeros, else some positive distances
        if self.metric == "cosine":
            distances = np.zeros((1, k))
        else:
            distances = np.ones((1, k)) * 0.2
        return distances, indices


# stub cosine_similarity to predictable values
def fake_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fake cosine similarity."""
    # return high similarity for all
    return np.ones((a.shape[0], b.shape[0])) * 0.9


class FakeJoblib:
    """Fake joblib for testing."""

    @staticmethod
    def dump(obj: Any, path: Path) -> None:  # noqa
        """Dump an object to a file."""
        joblib.dump(obj, path)

    @staticmethod
    def load(path: Path) -> Any:  # noqa
        """Load an object from a file."""
        return joblib.load(path)


def test_sklearn_organizer_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the flow of the sklearn organizer."""
    mod = import_module("ai_unit_test.core.implementations.indexing.sklearn_organizer")

    # Monkeypatch module attributes
    monkeypatch.setattr(mod, "SKLEARN_AVAILABLE", True)
    monkeypatch.setattr(mod, "NearestNeighbors", FakeNN)
    monkeypatch.setattr(mod, "cosine_similarity", fake_cosine_similarity)
    monkeypatch.setattr(mod, "joblib", FakeJoblib)

    async def run_flow(tmp_dir: str) -> None:
        index_path = Path(tmp_dir) / "sklearn_index"
        index_path.mkdir(parents=True, exist_ok=True)

        # Create organizer with non-cosine metric to go through cosine_similarity branch
        organizer = mod.SklearnIndexOrganizer({"metric": "euclidean", "algorithm": "ball_tree", "n_jobs": 1})

        # 1) Attempt to create index with empty embeddings => should raise IndexError due to validation
        with pytest.raises(IndexError):
            await organizer.create_index(np.array([]), [], index_path=index_path, model_name="test_model")

        # 2) Create proper index
        embeddings = np.random.random((3, 4))
        metadata = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        index_info = await organizer.create_index(embeddings, metadata, index_path=index_path, model_name="test_model")
        assert index_info.total_documents == 3
        assert organizer._index_loaded is True
        assert organizer._index_path == index_path

        # 3) Load index into new organizer
        new_org = mod.SklearnIndexOrganizer({"metric": "euclidean"})
        loaded_info = await new_org.load_index(index_path)
        assert loaded_info.total_documents == 3
        assert new_org._index_loaded is True
        assert new_org.embeddings.shape == (3, 4)

        # 4) Search with wrong-dimension query => IndexError
        wrong_query = np.random.random(2)  # 1D wrong size
        with pytest.raises(IndexError):
            await new_org.search(wrong_query, k=1)

        # 5) Successful search
        query = np.random.random((1, 4))
        results = await new_org.search(query, k=2, threshold=0.0)
        assert isinstance(results, list)
        assert all(hasattr(r, "score") and hasattr(r, "document_id") for r in results)

        # 6) Add documents
        add_embeddings = np.random.random((1, 4))
        add_metadata = [{"id": "d"}]
        await new_org.add_documents(add_embeddings, add_metadata)
        assert new_org.index_info.total_documents == 4

        # 7) Remove documents (remove the newly added one which is at index 3)
        await new_org.remove_documents(["3"])
        assert new_org.index_info.total_documents == 3
        assert len(new_org.metadata) == 3

        # 8) Update a document
        await new_org.update_document("1", np.random.random((4,)), {"id": "b_updated"})
        assert new_org.metadata[1]["id"] == "b_updated"

        # 9) get_index_info
        info = await new_org.get_index_info()
        assert info.total_documents == 3

        # 10) validate_index returns True for valid index
        is_valid = await new_org.validate_index(index_path)
        assert is_valid is True

        # 11) get_stats returns IndexStats with some fields
        stats = await new_org.get_stats()
        assert hasattr(stats, "total_documents")
        assert stats.total_documents == 3

        # 12) optimize retrains model (no error)
        await new_org.optimize_index()
        assert new_org.model.fitted is True

    with tempfile.TemporaryDirectory() as td:
        asyncio.run(run_flow(td))
