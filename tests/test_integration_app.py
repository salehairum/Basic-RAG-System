import pytest
from fastapi.testclient import TestClient
from app import app, client, verify_google_token
from embed_and_store import embed_and_store
import os
import numpy as np

client_api = TestClient(app)

docs = [
    "FastAPI is a modern, fast web framework for Python.",
    "Hugging Face provides powerful pretrained models.",
    "Chroma DB is a vector database optimized for embeddings.",
    "Retrieval-Augmented Generation combines retrieval with generation.",
]

@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    # Setup: create and populate collection
    embed_and_store(docs)
    yield
    # Teardown: delete collection after tests
    try:
        client.delete_collection(name="documents")
    except Exception:
        pass

def test_oauth_callback_missing_code():
    response = client_api.get("/oauth/callback")
    assert response.status_code == 400
    assert response.json()["detail"] == "Authorization code not found"

def test_query_endpoint_invalid_token():
    headers = {"Authorization": "Bearer invalid_token"}
    response = client_api.post("/query", json={"query": "test", "top_k": 2}, headers=headers)
    assert response.status_code == 401

def test_query_success(monkeypatch):
    import numpy as np
    from unittest.mock import MagicMock

    # Mock embedder.encode to return a NumPy array
    monkeypatch.setattr("app.embedder.encode", lambda x: np.array([[0.1]*384]))
    
    # Mock Redis: r.get returns None (simulate cache miss), r.set does nothing
    mock_redis = MagicMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    monkeypatch.setattr("app.r", mock_redis)

    # Mock ChromaDB collection query
    class MockCollection:
        def query(self, query_embeddings, n_results):
            return {"documents": [["This is a test document"]]}
    monkeypatch.setattr("app.get_collection", lambda: MockCollection())

    # Mock text generator
    mock_generator = MagicMock(return_value=[{"generated_text": "Test answer"}])
    monkeypatch.setattr("app.generator", mock_generator)

    # Mock token verification
    app.dependency_overrides[verify_google_token] = lambda: {"aud": os.getenv("GOOGLE_CLIENT_ID")}

    # Make API call
    headers = {"Authorization": "Bearer mock_token"}
    response = client_api.post("/query", json={"query": "test", "top_k": 1}, headers=headers)

    # Assertions
    assert response.status_code == 200
    assert response.json() == {"answer": "Test answer"}
