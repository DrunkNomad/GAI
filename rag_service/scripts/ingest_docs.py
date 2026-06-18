"""
Ingest documents into the vector store.

Usage:
    python scripts/ingest_docs.py path/to/document.txt
    python scripts/ingest_docs.py data/documents/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def chunk_text(text: str, chunk_size: int = 256) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < chunk_size:
            buffer += "\n\n" + para if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks if chunks else [text]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest_docs.py <path>")
        sys.exit(1)

    source = Path(sys.argv[1])
    if source.is_file():
        texts = [load_text(source)]
    elif source.is_dir():
        texts = [load_text(f) for f in source.glob("*.txt")]
    else:
        print(f"Path not found: {source}")
        sys.exit(1)

    chunk_size = settings.chunk_size
    all_chunks: list[str] = []
    for text in texts:
        all_chunks.extend(chunk_text(text, chunk_size))

    print(f"Chunks: {len(all_chunks)}")
    embedder = EmbeddingModel(settings.embedding_model)
    store = VectorStore(embedder.dim)

    vec_path = Path(settings.vector_db_path)
    if vec_path.exists():
        store.load(vec_path)
        print(f"Loaded existing index with {len(store)} docs")

    print("Embedding chunks...")
    embeddings = embedder.encode(all_chunks)
    store.add(all_chunks, embeddings)
    store.save(vec_path)
    print(f"Saved index with {len(store)} docs total")


if __name__ == "__main__":
    main()
