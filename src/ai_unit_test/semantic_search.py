import logging
import warnings
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from ai_unit_test.indexing import FAISS_AVAILABLE, load_faiss_index

if FAISS_AVAILABLE:
    # Suppress SWIG/FAISS warnings temporarily
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="<frozen importlib._bootstrap>")
        import faiss

logger = logging.getLogger(__name__)


def search(
    query: str,
    index_dir: str,
    k: int = 5,
    threshold: float = 0.7,
) -> list[tuple[dict[str, Any], float]]:
    """Searches the index for the most similar chunks to a given query."""
    index, metadata, manifest = load_faiss_index(index_dir)

    # Load the model used for indexing from the manifest
    model_name = manifest.get("embedding_model", "all-MiniLM-L6-v2")
    model = SentenceTransformer(model_name)

    # Generate and validate query embedding
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding_np = np.array(query_embedding, dtype=np.float32)

    if FAISS_AVAILABLE and isinstance(index, faiss.Index):  # pyright: ignore[reportPossiblyUnboundVariable]
        # Validation
        query_dim = query_embedding_np.shape[1]
        index_dim = index.d
        if query_dim != index_dim:
            error_message = (
                f"Critical Search Error: The query embedding dimension ({query_dim}) "
                f"is incompatible with the index dimension ({index_dim})."
            )
            instruction = (
                "This usually occurs when the embedding model has been changed. "
                "Please rebuild the index with the current model."
            )
            logger.error(f"{error_message}\n{instruction}")
            raise ValueError(f"{error_message}\n{instruction}")

        scores, indices = index.search(query_embedding_np, k)  # pyright: ignore[reportCallIssue]
        scores = scores[0]
        indices = indices[0]
    elif hasattr(index, "kneighbors"):
        distances, indices = index.kneighbors(  # pyright: ignore[reportAttributeAccessIssue]
            query_embedding_np, n_neighbors=k
        )
        scores = 1 - distances[0]  # Convert cosine distance to similarity
        indices = indices[0]
    else:
        raise TypeError("Unsupported index type")

    results = []
    for i, score in zip(indices, scores):
        if i != -1 and score >= threshold:
            results.append((metadata[i], float(score)))

    return results
