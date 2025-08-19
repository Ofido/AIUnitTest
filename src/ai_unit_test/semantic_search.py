import numpy as np
from sentence_transformers import SentenceTransformer

from ai_unit_test.indexing import load_faiss_index

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def search(
    query: str,
    index_path: str,
    k: int = 5,
    threshold: float = 1.0,
) -> list[tuple[int, float]]:
    """Searches the FAISS index for the most similar chunks to a given query."""
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query])
    index = load_faiss_index(index_path)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

    results = []
    for i, distance in zip(indices[0], distances[0]):
        if distance <= threshold:
            results.append((i, distance))

    return results
