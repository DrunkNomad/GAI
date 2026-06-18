from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.generator import CustomGPTGenerator
from src.core.config import settings


class RAGPipeline:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        generator: CustomGPTGenerator,
    ) -> None:
        self.embedder = embedding_model
        self.store = vector_store
        self.generator = generator

    def retrieve(self, question: str, k: int | None = None) -> list[tuple[str, float]]:
        if k is None:
            k = settings.rag_top_k
        q_emb = self.embedder.encode([question])
        return self.store.search(q_emb, k=k)

    def answer(self, question: str, use_rag: bool = True) -> tuple[str, list[tuple[str, float]]]:
        if use_rag and len(self.store) > 0:
            results = self.retrieve(question)
            context = "\n\n".join([text for text, _ in results])
            prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        else:
            results = []
            prompt = f"User: {question}\nAssistant:"

        answer = self.generator.generate(prompt, max_new_tokens=settings.max_tokens)
        return answer, results
