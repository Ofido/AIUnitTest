import os
from pathlib import Path

import numpy as np

from ai_unit_test.indexing import save_faiss_index
from ai_unit_test.semantic_search import search


def test_search() -> None:
    # Create dummy embeddings
    embeddings = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32).tolist()

    # Define index path
    index_path = Path("test_search_index.faiss")

    # Save the index
    save_faiss_index(embeddings, str(index_path))

    # Search for a query
    results = search("test", str(index_path), k=1, threshold=100.0)

    # Check if the search returns any results
    assert len(results) > 0

    # Clean up the index file
    os.remove(index_path)
