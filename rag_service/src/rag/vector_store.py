import pickle
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int) -> None:
        self.index = faiss.IndexFlatIP(dimension)
        self.texts: list[str] = []

    def add(self, texts: list[str], embeddings: np.ndarray) -> None:
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        self.index.add(embeddings.astype(np.float32))
        self.texts.extend(texts)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.texts) and idx >= 0:
                results.append((self.texts[idx], float(dist)))
        return results

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        texts_path = path.with_suffix(".pkl")
        with open(texts_path, "wb") as f:
            pickle.dump(self.texts, f)

    def load(self, path: Path) -> None:
        self.index = faiss.read_index(str(path))
        texts_path = path.with_suffix(".pkl")
        with open(texts_path, "rb") as f:
            self.texts = pickle.load(f)

    def __len__(self) -> int:
        return len(self.texts)
