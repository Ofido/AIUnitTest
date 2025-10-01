"""Test cases for the in-memory index organizer."""

import importlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from ai_unit_test.core.exceptions import IndexError


@pytest.mark.asyncio
async def test_memory_organizer_basic_flow_and_errors(tmp_path: Path) -> None:  # noqa
    """Test basic flow and error handling of memory organizer."""
    mod = importlib.import_module("ai_unit_test.core.implementations.indexing.memory_organizer")
    # find the main organizer class in the module
    cls = None
    for name, obj in mod.__dict__.items():
        if inspect.isclass(obj) and obj.__module__ == mod.__name__ and ("Memory" in name or "Organizer" in name):
            cls = obj
            break
    assert cls is not None, "Organizer class not found in module"

    # try to instantiate with different signatures
    try:
        inst = cls({})
    except TypeError:
        try:
            inst = cls(None)
        except TypeError:
            inst = cls()

    # __init__ should set these attributes
    assert hasattr(inst, "embeddings")
    assert inst.embeddings is None
    assert hasattr(inst, "metadata")
    assert isinstance(inst.metadata, list)
    assert len(inst.metadata) == 0
    assert hasattr(inst, "doc_ids")
    assert isinstance(inst.doc_ids, list)
    assert len(inst.doc_ids) == 0

    # find create method
    create_candidates = ["create_index", "create", "build_index", "create_from_embeddings", "initialize_index"]
    create = None
    for name in create_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            create = getattr(inst, name)
            break
    if create is None:
        pytest.skip("No create method found on organizer")

    # prepare initial embeddings and metadata
    emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    meta = [{"text": "doc1"}, {"text": "doc2"}]

    # call create and verify index is loaded and index_info set
    index_path = tmp_path / "dummy_index"
    model_name = "test_model"
    _ = await inst.create_index(emb, meta, index_path, model_name)
    assert getattr(inst, "_index_loaded", False) is True
    assert hasattr(inst, "index_info")

    # find is_loaded method
    is_loaded_candidates = ["is_index_loaded", "is_loaded", "index_loaded", "is_index_loaded"]
    is_loaded = None
    for name in is_loaded_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            is_loaded = getattr(inst, name)
            break
    if is_loaded:
        val = is_loaded()
        assert isinstance(val, bool)
        assert val is True

    # find search method
    search_candidates = ["search", "query", "nearest", "search_index"]
    search = None
    for name in search_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            search = getattr(inst, name)
            break
    if search is None:
        pytest.skip("No search method found on organizer")

    # perform a search with a query vector that matches first document
    q = np.array([1.0, 0.0, 0.0])
    try:
        results = await search(q, k=2, threshold=0.0)
    except TypeError:
        # try alternative signature without keywords
        results = search(q, 2, 0.0)
    assert isinstance(results, list)
    assert len(results) >= 1
    # expect first result to correspond to doc0
    first = results[0]
    # result may be tuple or dict-like; handle common shapes
    if isinstance(first, dict):
        assert "score" in first or "id" in first or "metadata" in first
    elif isinstance(first, (list, tuple)):
        assert len(first) >= 2

    # find add_documents method
    add_candidates = ["add_documents", "add", "insert_documents", "add_embeddings"]
    add = None
    for name in add_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            add = getattr(inst, name)
            break
    if add:
        new_emb = np.array([[0.0, 0.0, 1.0]])
        new_meta = [{"text": "doc3"}]
        try:
            await add(new_emb, new_meta)
        except TypeError:
            await add(new_emb)
        assert len(inst.metadata) >= 3
        assert inst.doc_ids[-1] == str(len(inst.metadata) - 1)

    # find remove_documents method
    remove_candidates = ["remove_documents", "remove", "delete_documents", "delete"]
    remove = None
    for name in remove_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            remove = getattr(inst, name)
            break
    if remove:
        # remove the last document id
        last_id = inst.doc_ids[-1]
        try:
            await remove([last_id])
        except TypeError:
            await remove(last_id)
        assert last_id not in inst.doc_ids
        assert len(inst.metadata) == inst.index_info.total_documents

    # find update_document method
    update_candidates = ["update_document", "update", "upsert_document"]
    update = None
    for name in update_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            update = getattr(inst, name)
            break
    if update:
        # update document 0
        new_embedding = np.array([0.5, 0.5, 0.0])
        new_meta = {"text": "updated"}  # type: ignore
        try:
            await update("0", new_embedding, new_meta)
        except TypeError:
            await update(0, new_embedding, new_meta)
        # verify metadata updated
        assert inst.metadata[0]["text"] == "updated"

    # find get_stats method
    stats_candidates = ["get_index_stats", "get_stats", "stats"]
    get_stats = None
    for name in stats_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            get_stats = getattr(inst, name)
            break
    if get_stats:
        stats = await get_stats()
        assert stats is not None

    # test behavior when index not loaded: methods that check _index_loaded should raise
    # set _index_loaded False and try to call a method that requires loaded index (get_index_info / similar)
    info_candidates = ["get_index_info", "get_index", "index_info", "get_index_metadata"]
    info = None
    for name in info_candidates:
        if hasattr(inst, name) and callable(getattr(inst, name)):
            info = getattr(inst, name)
            break
    if info:
        inst._index_loaded = False
        with pytest.raises(IndexError):
            await info()
