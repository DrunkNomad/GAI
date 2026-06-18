# GAI — Custom GPT + Production RAG Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/ganin-dev/gai/actions/workflows/ci.yml/badge.svg)](https://github.com/ganin-dev/gai/actions/workflows/ci.yml)

**GAI** (Ganin Artificial Intelligence) is a two-part project:

1. **Custom GPT from scratch** — a decoder-only transformer built entirely on NumPy with a custom autograd engine
2. **Production RAG service** — FastAPI + FAISS retrieval-augmented generation wrapping the custom GPT model

---

## Project Structure

```
gai/                  Custom transformer (NumPy, no frameworks)
gai_legacy/           Clean NumPy-only copy (original version)
rag_service/          Production RAG service (FastAPI, FAISS, Docker)
scripts/              Training scripts
```

### gai/ — Custom Transformer

Built from scratch with zero dependencies beyond NumPy:

| Component | Description |
|-----------|-------------|
| `tensor.py` | Tensor with reverse-mode autograd |
| `nn/attention.py` | Multi-head scaled dot-product attention |
| `nn/transformer.py` | Transformer block (Pre-LN, GELU, residual) |
| `nn/normalization.py` | Layer normalization |
| `model/gpt.py` | GPT decoder-only model |
| `tokenizer/bpe.py` | BPE tokenizer (GPT-2-style regex) |
| `optim/adam.py` | Adam optimizer |
| `train/trainer.py` | Training loop |

Architecture: Pre-LN, GELU activation, learned positional embeddings, causal masking.

### rag_service/ — RAG Service

Production-grade document-augmented chat:

```
User → FastAPI API
         │
    ┌────┴────┐
  FAISS     Custom GPT
  (semantic  (trained model)
   search)    │
    │         │
    └────┬────┘
         │
    Answer + Sources
```

| Component | Tech |
|-----------|------|
| API | FastAPI (Pydantic, Swagger docs) |
| Vector search | FAISS (IndexFlatIP) |
| Embeddings | intfloat/multilingual-e5-small |
| LLM | Custom GPT model (NumPy) |
| Containerization | Docker + docker-compose |
| CI | GitHub Actions |

---

## Quick Start

### Legacy version (NumPy only)

```bash
cd gai_legacy
pip install numpy
python scripts/chat.py                 # chat with pre-trained model
python scripts/test_tensor.py          # test autograd engine
python scripts/train_model.py          # train your own model
```

### RAG Service

```bash
cd rag_service
pip install -e .
uvicorn src.main:get_app --factory --reload
```

Then open http://localhost:8000/docs

### Docker

```bash
cd rag_service
docker compose up --build
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Service status, model info, document count |
| POST | `/api/v1/chat` | Ask a question (with RAG context) |
| POST | `/api/v1/documents/upload` | Upload a text document for indexing |

### Chat Example

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a transformer?", "use_rag": true}'
```

Response:
```json
{
  "answer": "Transformers use attention mechanism...",
  "sources": [{"text": "document content...", "score": 0.85}],
  "model": "custom-gpt-128dim-6l"
}
```

---

## Model Training

```bash
# Train a custom model (NumPy, no GPU needed)
python scripts/train_model_final.py \
  --embed 128 --heads 4 --layers 6 --seq 256 --steps 5000 --vocab 1024
```

The model is saved as a pickle file and loaded by the RAG service.

---

## License

MIT
