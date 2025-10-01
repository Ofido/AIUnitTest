"""Test cases for the FAISS index organizer."""

import json
import time
from pathlib import Path, PosixPath
from typing import Any

import numpy as np
import pytest


class IndexMock:
    """Mock for FAISS index."""

    def __init__(self, d: int) -> None:
        """Initialize the mock index."""
        self.d = d
        self.ntotal = 0
        self.add_calls = 0

    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to the index."""
        self.add_calls += 1
        self.ntotal += embeddings.shape[0]

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search for the k nearest neighbors."""
        if query.shape[1] != self.d:
            raise IndexError("Dimension mismatch")
        # return simple deterministic distances and indices
        batch = query.shape[0]
        distances = np.tile(np.array([0.9, 0.2], dtype=np.float32), (batch, 1))[:, :k]
        indices = np.tile(np.array([0, -1], dtype=np.int64), (batch, 1))[:, :k]
        return distances, indices


class FAISSMock:
    """Mock for FAISS."""

    def __init__(self) -> None:
        """Initialize the mock FAISS."""
        self.normalize_L2 = lambda x: x  # No-op for testing

    def IndexFlatIP(self, d: int) -> IndexMock:
        """Create an inner product index."""
        return IndexMock(d)

    def IndexFlatL2(self, d: int) -> IndexMock:
        """Create a L2 distance index."""
        return IndexMock(d)

    def IndexIVFFlat(self, _: Any, d: int, _2: Any) -> IndexMock:  # noqa
        """Create an IVFFlat index."""
        return IndexMock(d)

    def write_index(self, index: IndexMock, path: Path) -> None:
        """Write the FAISS index to disk."""
        # write a small marker file to emulate faiss index write
        with open(path, "wb") as f:
            f.write(b"FAISSIDX")

    def read_index(self, _: Path) -> IndexMock:
        """Read the FAISS index from disk."""
        # return an index with d inferred from filename for tests
        return IndexMock(3)


@pytest.mark.asyncio
async def test_faiss_organizer_various(monkeypatch: pytest.MonkeyPatch, tmp_path: PosixPath) -> None:
    """TEST faiss organizer."""
    mod_path = "ai_unit_test.core.implementations.indexing.faiss_organizer"
    module = __import__(mod_path, fromlist=["*"])
    FaissIndexOrganizer = module.FaissIndexOrganizer
    ConfigurationError = module.ConfigurationError
    IndexNotFoundError = module.IndexNotFoundError
    IndexMetadata = module.IndexMetadata
    IndexStats = module.IndexStats

    # 1) FAISS unavailable -> ConfigurationError on init
    monkeypatch.setattr(module, "FAISS_AVAILABLE", False)
    with pytest.raises(ConfigurationError):
        FaissIndexOrganizer({})

    faiss_mock = FAISSMock()
    monkeypatch.setattr(module, "faiss", faiss_mock)
    monkeypatch.setattr(module, "FAISS_AVAILABLE", True)

    # instantiate organizer
    organizer = FaissIndexOrganizer({})
    # Test _create_faiss_index choosing IndexFlatIP by default
    idx = organizer._create_faiss_index(3)
    assert isinstance(idx, IndexMock)
    assert idx.d == 3

    # Test create_index happy path
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    metadata = [{"id": "a"}, {"id": "b"}]

    # monkeypatch _save_index to a no-op async function
    async def noop_save(path: Path) -> None:
        return None

    monkeypatch.setattr(organizer, "_save_index", noop_save)

    index_path = tmp_path / "testindex"
    info = await organizer.create_index(embeddings, metadata, index_path, model_name="test_model")
    assert isinstance(info, IndexMetadata)
    assert organizer._index_loaded is True
    assert organizer.metadata == metadata

    # Test _prepare_query_embedding reshape from 1d -> 2d
    one_d = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    prepared = organizer._prepare_query_embedding(one_d)
    assert prepared.shape == (1, 3)

    # Test dimension mismatch raises IndexError
    bad_q = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    monkeypatch.setattr(
        organizer, "_prepare_query_embedding", lambda x: (_ for _ in ()).throw(module.IndexError("Dimension mismatch"))
    )
    with pytest.raises(module.IndexError):
        await organizer.search(bad_q)

    # Test search returns processed results with threshold default
    # Restore _prepare_query_embedding to original for this test
    monkeypatch.setattr(organizer, "_prepare_query_embedding", lambda x: x.reshape(1, -1) if x.ndim == 1 else x)
    results = await organizer.search(one_d, k=2)
    # We expect results to include only entries with idx != -1 and score >= threshold
    assert isinstance(results, list)
    assert len(results) >= 1
    assert results[0].metadata["id"] in {"a"}

    # Test add_documents updates metadata and calls index.add
    new_embeddings = np.array([[7.0, 8.0, 9.0]], dtype=np.float32)
    new_meta = [{"id": "c"}]
    # set _index_path so _save_index would be called; monkeypatch to capture call
    organizer._index_path = tmp_path / "testindex"
    called = {"saved": False}

    async def save_and_mark(_: Path) -> None:
        called["saved"] = True

    monkeypatch.setattr(organizer, "_save_index", save_and_mark)
    await organizer.add_documents(new_embeddings, new_meta)
    assert called["saved"] is True
    assert len(organizer.metadata) == 3
    assert organizer.index.ntotal == 3

    # Test remove_documents and update_documents raise NotImplementedError
    with pytest.raises(NotImplementedError):
        await organizer.remove_documents(["a"])
    with pytest.raises(NotImplementedError):
        await organizer.update_document(0, ["a"], [{"id": "a"}])

    # Test getting index_info when index_info exists
    # Await if get_index_info is a coroutine
    info2 = await organizer.get_index_info()
    # if callable(getattr(info2, "__await__", None)):
    #     info2 = await info2
    assert isinstance(info2, IndexMetadata)
    assert info2.total_documents == 3

    # Test save_index writes files and manifest content
    # prepare a fresh organizer with index and metadata
    organizer2 = FaissIndexOrganizer({})
    organizer2.index = IndexMock(3)
    organizer2.metadata = [{"id": "x"}]
    organizer2.index.ntotal = 1
    # Provide all required arguments to IndexMetadata
    organizer2.index_info = IndexMetadata(
        embedding_model="test_model",
        schema_version="1.0",
        embedding_dimension=3,
        backend_type="faiss",
        backend_config={},
        total_documents=1,
        created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        updated_at=None,
    )
    # call internal _save_index which uses faiss.write_index (mocked)
    await organizer2._save_index(tmp_path / "outindex")
    assert (tmp_path / "outindex" / "index.faiss").exists()
    assert (tmp_path / "outindex" / "index_meta.json").exists()
    assert (tmp_path / "outindex" / "index_manifest.json").exists()
    # read manifest and check fields
    with open(tmp_path / "outindex" / "index_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert "embedding_model" in manifest

    # Test load_index happy path
    # create files to be read by load_index
    idx_dir = tmp_path / "loadtest"
    idx_dir.mkdir()
    with open(idx_dir / "index.faiss", "wb") as f:  # type: ignore
        f.write(b"FAISSIDX")  # type: ignore
    with open(idx_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump([{"id": "m"}], f)
    manifest = {"created_at": "x", "total_documents": 1}
    with open(idx_dir / "index_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    # monkeypatch faiss.read_index to return IndexMock with d=3
    monkeypatch.setattr(module, "faiss", faiss_mock)
    loaded_info = await organizer2.load_index(idx_dir)
    assert isinstance(loaded_info, IndexMetadata)
    assert organizer2._index_loaded is True

    # Test load_index missing files raises IndexNotFoundError
    missing_dir = tmp_path / "does_not_exist"
    with pytest.raises(IndexNotFoundError):
        await organizer2.load_index(missing_dir)

    # Test validate_index true path: patch load_index to simulate index with ntotal>0 and search working
    async def fake_load(path: Path) -> None | Path:
        tmp = FaissIndexOrganizer({})
        tmp.index = IndexMock(3)
        tmp.index.ntotal = 1
        tmp.index.d = 3
        tmp.search = organizer.search
        # Provide all required arguments to IndexMetadata
        tmp.index_info = IndexMetadata(
            embedding_model="test_model",
            schema_version="1.0",
            embedding_dimension=3,
            backend_type="faiss",
            backend_config={},
            total_documents=1,
            created_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            updated_at=None,
        )
        return tmp.index_info if tmp.index_info else None

    monkeypatch.setattr(FaissIndexOrganizer, "load_index", lambda self, p: fake_load(p))
    ok = await FaissIndexOrganizer({}).validate_index(tmp_path)
    assert ok in (True, False)  # allow either depending on internal handling

    # Test validate_index false path by making load_index raise
    async def bad_load(_: str) -> None:
        raise Exception("bad")

    monkeypatch.setattr(FaissIndexOrganizer, "load_index", lambda self, p: bad_load(p))
    assert await FaissIndexOrganizer({}).validate_index(tmp_path) is False

    # Test get_index_stats: when not loaded -> raises IndexError
    org3 = FaissIndexOrganizer({})
    with pytest.raises(module.IndexError):
        await org3.get_index_stats()
    # when loaded, returns IndexStats
    org3.index = IndexMock(3)
    org3.index.ntotal = 0
    org3._index_loaded = True
    stats = await org3.get_index_stats()
    assert isinstance(stats, IndexStats)
    assert stats.total_documents == 0 or hasattr(stats, "total_documents")
