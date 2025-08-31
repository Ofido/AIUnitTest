import json
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import numpy as np

try:
    # Suprimir warnings do SWIG/FAISS temporariamente
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")
        import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = MagicMock()

from ai_unit_test.llm import EMBEDDING_MODEL

SCHEMA_VERSION = "1.1.0"  # Incremented version for the new structure
FAISS_INDEX_FILENAME = "index.faiss"
SKLEARN_INDEX_FILENAME = "index.joblib"
METADATA_FILENAME = "index_meta.json"
MANIFEST_FILENAME = "index_manifest.json"

logger = logging.getLogger(__name__)


def _validate_inputs(embeddings: list[list[float]], metadatas: list[dict[str, Any]]) -> None:
    """Validates the inputs for the save_faiss_index function."""
    if not embeddings:
        raise ValueError("Validation Error: The embeddings list is empty. Cannot create an empty index.")

    if len(embeddings) != len(metadatas):
        msg = (
            f"Validation Error: The number of embeddings ({len(embeddings)}) does not "
            f"match the number of metadata entries ({len(metadatas)}). "
            "The index cannot be saved."
        )
        raise ValueError(msg)

    if not metadatas:
        raise ValueError("The metadata list is empty")

    first_dim = len(embeddings[0])
    if not all(len(e) == first_dim for e in embeddings):
        raise ValueError(
            "Validation Error: Embeddings have inconsistent dimensions. " "All vectors must have the same size."
        )

    if not all(isinstance(item, dict) for item in metadatas):
        raise ValueError("Validation Error: Not all items in the metadata list are valid dictionaries.")


def save_faiss_index(
    embeddings: list[list[float]],
    metadata: list[dict[str, Any]],
    index_dir: str,
    model_name: str = EMBEDDING_MODEL,
) -> None:
    """
    Saves the FAISS index, metadata, and a manifest file to a directory.

    Args:
        embeddings (list[list[float]]): List of embeddings.
        metadata (list[dict[str, Any]]): List of metadata dictionaries for each chunk.
        index_dir (str): Directory to save the index and associated files.
        model_name (str): Name of the embedding model used.
    """
    _validate_inputs(embeddings, metadata)

    os.makedirs(index_dir, exist_ok=True)

    embeddings_np = np.array(embeddings, dtype=np.float32)
    if FAISS_AVAILABLE:
        # Embeddings are expected to be normalized before this function is called.
        dimension = embeddings_np.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings_np)  # pyright: ignore[reportCallIssue]
        index_filename = FAISS_INDEX_FILENAME
        faiss.write_index(index, os.path.join(index_dir, index_filename))
    else:
        import joblib
        from sklearn.neighbors import NearestNeighbors

        index = NearestNeighbors(n_neighbors=5, metric="cosine")
        index.fit(embeddings_np)
        index_filename = SKLEARN_INDEX_FILENAME
        joblib.dump(index, os.path.join(index_dir, index_filename))

    logger.info(f"Index saved to {os.path.join(index_dir, index_filename)}")

    metadata_path = os.path.join(index_dir, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": model_name,
        "vector_dimension": embeddings_np.shape[1],
        "index_file": index_filename,
        "metadata_file": METADATA_FILENAME,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunk_count": len(metadata),
        "index_type": "faiss" if FAISS_AVAILABLE else "sklearn",
    }
    manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest saved to {manifest_path}")


def load_faiss_index(index_dir: str) -> tuple[object, list[dict[str, Any]], dict[str, Any]]:
    """
    Loads the FAISS index, metadata, and manifest from a directory.

    Args:
        index_dir (str): Directory where the index and associated files are stored.

    Returns:
        tuple[object, list[dict[str, Any]], dict[str, Any]]: A tuple containing the
        index (FAISS or scikit-learn), the metadata, and the manifest.
    """
    manifest_path = os.path.join(index_dir, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest file not found at {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Schema version mismatch. Expected {SCHEMA_VERSION}, found {manifest['schema_version']}")

    # Validate index type
    index_type = manifest.get("index_type", "faiss")
    if index_type not in ["faiss", "sklearn"]:
        raise ValueError(f"Invalid index type: {index_type}. Supported types are 'faiss' and 'sklearn'.")

    index_path = os.path.join(index_dir, manifest["index_file"])
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index file not found at {index_path}")

    if manifest.get("index_type", "faiss") == "faiss":
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS not installed, but index was created with FAISS.")
        index = faiss.read_index(index_path)
    else:
        import joblib

        index = joblib.load(index_path)

    logger.info(f"Index loaded from {index_path}")

    metadata_path = os.path.join(index_dir, manifest["metadata_file"])
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    logger.info(f"Metadata loaded from {metadata_path}")

    if FAISS_AVAILABLE and isinstance(index, faiss.Index) and index.ntotal != len(metadata):
        raise ValueError(
            f"Inconsistency detected: FAISS index has {index.ntotal} vectors, "
            f"but metadata file has {len(metadata)} entries."
        )

    logger.info("Index and metadata validation successful.")

    return index, metadata, manifest
