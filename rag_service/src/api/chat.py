from fastapi import APIRouter, Depends

from src.models.schemas import ChatRequest, ChatResponse, Source
from src.rag.retriever import RAGPipeline
from src.api.dependencies import get_rag_pipeline

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, pipeline: RAGPipeline = Depends(get_rag_pipeline)) -> ChatResponse:
    answer, sources = pipeline.answer(req.question, use_rag=req.use_rag)
    return ChatResponse(
        answer=answer,
        sources=[Source(text=t, score=s) for t, s in sources],
        model=pipeline.generator.model_name,
    )
