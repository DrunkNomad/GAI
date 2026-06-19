FROM python:3.11-slim

WORKDIR /app

# Copy gai package (the custom neural network framework)
COPY gai/ /app/gai/
COPY gai_model_v5_max.pkl /app/gai_model_v5_max.pkl

# Copy and install RAG service
COPY rag_service/pyproject.toml /app/rag_service/
RUN pip install --no-cache-dir -e /app/rag_service[dev] && \
    pip install --no-cache-dir sentence-transformers faiss-cpu torch

COPY rag_service/ /app/rag_service/

WORKDIR /app/rag_service

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
