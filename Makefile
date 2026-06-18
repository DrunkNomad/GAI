.PHONY: legacy-train legacy-chat rag-install rag-run rag-ui

# Legacy (NumPy-only) commands
legacy-install:
	cd gai_legacy && pip install numpy

legacy-chat:
	cd gai_legacy && python scripts/chat.py

legacy-test:
	cd gai_legacy && python scripts/test_tensor.py

legacy-train:
	cd gai_legacy && python scripts/train_model.py

# RAG service commands
rag-install:
	cd rag_service && pip install -e . && pip install gradio

rag-run:
	cd rag_service && uvicorn src.main:get_app --factory --reload

rag-ui:
	cd rag_service && python -m src.ui

rag-docker:
	cd rag_service && docker compose up --build

rag-test:
	cd rag_service && pytest tests/ -v
