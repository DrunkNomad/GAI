from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str
    use_rag: bool = True


class Source(BaseModel):
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    model: str = "custom-gpt"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    documents_indexed: int
    embedding_model: str


class DocumentIngestResponse(BaseModel):
    status: str
    chunks_added: int
    total_documents: int
