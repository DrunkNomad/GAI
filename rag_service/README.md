# GAI RAG Service

Production-grade RAG (Retrieval-Augmented Generation) service powered by a custom GPT model trained from scratch.

## Architecture

```
User → FastAPI
         │
    ┌────┴────┐
    │         │
  FAISS    Custom GPT
  (semantic  (own implementation
   search)   in numpy + autograd)
    │         │
    └────┬────┘
         │
    Answer + Sources
```

- **Custom GPT** — a decoder-only transformer with multi-head attention, pre-LN, GELU activation, trained from scratch on conversational data
- **FAISS** — efficient vector similarity search for document retrieval
- **Embeddings** — `intfloat/multilingual-e5-small` for multilingual semantic search
- **FastAPI** — async REST API with automatic OpenAPI docs

## Quick Start

```bash
# Install dependencies
pip install -e .

# Start the service
uvicorn src.main:app --reload

# Open the API docs
open http://localhost:8000/docs
```

### With Docker

```bash
docker compose up --build
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service health + model status |
| POST | `/api/v1/chat` | Ask a question (with or without RAG) |
| POST | `/api/v1/documents/upload` | Upload a document for indexing |

### Chat Example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a transformer?"}'
```

Response:
```json
{
  "answer": "A transformer uses attention mechanism...",
  "sources": [
    {"text": "Transformers are...", "score": 0.85}
  ],
  "model": "custom-gpt-256dim-6l"
}
```

## Ingesting Documents

```bash
# Single file
python scripts/ingest_docs.py data/documents/sample.txt

# Directory
python scripts/ingest_docs.py data/documents/
```

## Model Training

The GPT model is trained from scratch using only NumPy (no PyTorch/TensorFlow).

```bash
cd ..
python scripts/train_model_final.py \
  --embed 256 --heads 8 --layers 6 --seq 256 --steps 10000
```

## Tech Stack

- **Python 3.11+**
- **FastAPI** — API framework
- **FAISS** — vector search
- **sentence-transformers** — embeddings
- **NumPy** — tensor operations
- **Docker** — containerization
- **Gradio** — web UI

## Web UI

```bash
make ui
# or
python -m src.ui
```

Opens at http://localhost:7860
