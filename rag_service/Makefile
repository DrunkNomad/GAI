.PHONY: install run test docker lint clean

install:
	pip install -e .[dev]
	pip install gradio

run:
	uvicorn src.main:get_app --factory --reload --host 0.0.0.0 --port 8000

ui:
	python -m src.ui

test:
	pytest tests/ -v

docker:
	docker compose up --build

lint:
	ruff check src/ tests/

clean:
	rm -rf __pycache__ .pytest_cache
