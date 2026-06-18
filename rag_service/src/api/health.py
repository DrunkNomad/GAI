from fastapi import APIRouter, Depends

from src.models.schemas import HealthResponse
from src.rag.retriever import RAGPipeline
from src.api.dependencies import get_rag_pipeline

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health(pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> HealthResponse:
    return HealthResponse(
        status="ok" if pipeline.generator.model is not None else "no_model",
        model_loaded=pipeline.generator.model is not None,
        documents_indexed=len(pipeline.store),
        embedding_model=pipeline.embedder.model.model_name,
    )
