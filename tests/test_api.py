from unittest.mock import patch

import pytest

from app import create_app
from app.config import TestConfig
from app.rag import reset_cache


@pytest.fixture
def client():
    app = create_app(TestConfig)
    reset_cache()
    return app.test_client()


def test_health(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_root_describes_service(client):
    assert client.get("/").status_code == 200


def test_chat_requires_message(client):
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400


def test_chat_unconfigured_returns_503(client):
    # TestConfig has no API keys, so the pipeline cannot be built.
    resp = client.post("/api/chat", json={"message": "How do I make ugali?"})
    assert resp.status_code == 503


@patch("app.api.routes.answer_question")
def test_chat_returns_answer_with_sources(mock_answer, client):
    mock_answer.return_value = {
        "answer": "Boil water, add maize flour, stir into ugali.",
        "sources": [{"source": "Kenya Recipe Book 2018.pdf", "page": 12}],
        "grounded": True,
    }
    resp = client.post("/api/chat", json={"message": "ugali?"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert "ugali" in data["answer"].lower()
    assert data["sources"][0]["page"] == 12
    assert data["grounded"] is True


@patch("app.api.routes.answer_question")
def test_chat_out_of_scope_is_not_grounded(mock_answer, client):
    mock_answer.return_value = {"answer": "I can only help with recipes...", "sources": [], "grounded": False}
    data = client.post("/api/chat", json={"message": "change a tyre?"}).get_json()
    assert data["grounded"] is False
    assert data["sources"] == []
