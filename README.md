# GAI — General Artificial Intelligence

A production-grade Retrieval-Augmented Generation (RAG) service powered by a custom GPT model trained from scratch.

## Project Structure

| Directory | Description |
|-----------|-------------|
| `gai/` | Custom GPT implementation (numpy autograd, transformer from scratch) |
| `rag_service/` | Production RAG service (FastAPI, FAISS, Docker) |
| `scripts/` | Model training scripts |

## RAG Service

```bash
cd rag_service
pip install -e .
uvicorn src.main:app --reload
```

See [rag_service/README.md](rag_service/README.md) for details.

## Model Training

```bash
python scripts/train_model_final.py --embed 256 --heads 8 --layers 6
```
