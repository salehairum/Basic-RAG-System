import os
import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_oauth_callback_missing_code():
    response = client.get("/oauth/callback")
    assert response.status_code == 400
    assert response.json()["detail"] == "Authorization code not found"


def test_query_endpoint_invalid_token():
    headers = {"Authorization": "Bearer invalid_token"}
    response = client.post("/query", json={"query": "test", "top_k": 2}, headers=headers)
    assert response.status_code == 401
    
def test_query_success(monkeypatch):
    monkeypatch.setattr("app.embedder.encode", lambda x: [[0.1]*384])
    monkeypatch.setattr("app.collection.query", lambda **kwargs: {"documents": [["This is a test document"]]})
    monkeypatch.setattr("app.generator.__call__", lambda *args, **kwargs: [{"generated_text": "Test answer"}])
    headers = {"Authorization": "Bearer mock_token"}
    response = client.post("/query", json={"query": "test", "top_k": 1}, headers=headers)
    assert response.status_code == 200
    assert "answer" in response.json()
