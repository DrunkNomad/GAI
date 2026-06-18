from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from src.main import create_app


def _mock_pipeline() -> MagicMock:
    pipeline = MagicMock()
    pipeline.generator.model = MagicMock()
    pipeline.generator.model_name = "test-model"
    pipeline.embedder.model.model_name = "test-embedder"
    pipeline.store.__len__ = MagicMock(return_value=42)
    pipeline.answer = MagicMock(return_value=("test answer", [("source text", 0.95)]))
    return pipeline


def test_health() -> None:
    app = create_app(pipeline=_mock_pipeline())
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_chat() -> None:
    app = create_app(pipeline=_mock_pipeline())
    client = TestClient(app)

    resp = client.post("/api/v1/chat", json={"question": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "test answer"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["text"] == "source text"
