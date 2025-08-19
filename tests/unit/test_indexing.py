import os
from pathlib import Path

import numpy as np

from ai_unit_test.indexing import load_faiss_index, save_faiss_index


def test_save_and_load_faiss_index() -> None:
    # Create dummy embeddings
    embeddings = np.random.rand(10, 128).tolist()

    # Define index path
    index_path = Path("test_index.faiss")

    # Save the index
    save_faiss_index(embeddings, str(index_path))

    # Check if the index file was created
    assert index_path.exists()

    # Load the index
    loaded_index = load_faiss_index(str(index_path))

    # Check if the loaded index has the correct number of embeddings
    assert loaded_index.ntotal == 10

    # Clean up the index file
    os.remove(index_path)
