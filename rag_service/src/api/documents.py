from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from src.models.schemas import DocumentIngestResponse
from src.rag.retriever import RAGPipeline
from src.api.dependencies import get_rag_pipeline
from src.core.config import settings

router = APIRouter(tags=["Documents"])


@router.post("/documents/upload", response_model=DocumentIngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
) -> DocumentIngestResponse:
    if not file.filename:
        raise HTTPException(400, "No file provided")

    content = (await file.read()).decode("utf-8")
    chunks = _chunk_text(content)
    embeddings = pipeline.embedder.encode(chunks)
    pipeline.store.add(chunks, embeddings)

    return DocumentIngestResponse(
        status="ok",
        chunks_added=len(chunks),
        total_documents=len(pipeline.store),
    )


def _chunk_text(text: str) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for para in paragraphs:
        if len(buffer) + len(para) < settings.chunk_size:
            buffer += "\n\n" + para if buffer else para
        else:
            if buffer:
                chunks.append(buffer)
            buffer = para
    if buffer:
        chunks.append(buffer)
    return chunks if chunks else [text]
