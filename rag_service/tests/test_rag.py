import numpy as np
from src.rag.vector_store import VectorStore


def test_vector_store_add_and_search() -> None:
    store = VectorStore(dimension=4)
    texts = ["hello world", "goodbye world", "hello there"]
    embeddings = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
    ])
    store.add(texts, embeddings)
    assert len(store) == 3

    query = np.array([[1.0, 0.0, 0.0, 0.0]])
    results = store.search(query, k=2)
    assert len(results) == 2
    assert results[0][0] == "hello world"
