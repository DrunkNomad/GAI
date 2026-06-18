from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.core.config import settings
from src.rag.embeddings import EmbeddingModel
from src.rag.vector_store import VectorStore
from src.rag.generator import CustomGPTGenerator
from src.rag.retriever import RAGPipeline
from src.api import health, chat, documents


def create_app(
    pipeline: RAGPipeline | None = None,
    model_path: Path | None = None,
) -> FastAPI:
    app = FastAPI(
        title="GAI RAG Service",
        version="0.1.0",
        description="Production RAG service powered by a custom GPT model",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if pipeline is None:
        if model_path is None:
            model_path = settings.model_path

        embedder = EmbeddingModel(settings.embedding_model)
        vector_store = VectorStore(embedder.dim)

        vector_db_path = Path(settings.vector_db_path)
        if vector_db_path.exists():
            vector_store.load(vector_db_path)

        if settings.use_local_model and model_path.exists():
            generator = CustomGPTGenerator(
                model_path,
                temperature=settings.temperature,
                top_k=settings.top_k,
            )
        else:
            raise RuntimeError(f"Model not found at {model_path}")

        pipeline = RAGPipeline(embedder, vector_store, generator)

    app.state.pipeline = pipeline

    app.include_router(health.router)
    app.include_router(chat.router, prefix="/api/v1")
    app.include_router(documents.router, prefix="/api/v1")

    return app


app: FastAPI | None = None


def get_app() -> FastAPI:
    global app
    if app is None:
        app = create_app()
    return app


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:get_app", host=settings.host, port=settings.port, reload=True, factory=True)
