import faiss
import numpy as np


def save_faiss_index(embeddings: list[list[float]], index_path: str) -> None:
    """Saves the FAISS index to a file."""
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, index_path)


def load_faiss_index(index_path: str) -> faiss.Index:
    """Loads the FAISS index from a file."""
    return faiss.read_index(index_path)
