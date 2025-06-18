import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from app import verify_google_token, query_rag, QueryRequest
import os
from embed_and_store import embed_and_store
import numpy as np 

@pytest.mark.asyncio
async def test_verify_google_token_valid(monkeypatch):
    async def mock_get(*args, **kwargs):
        class MockResponse:
            status_code = 200
            def json(self):
                return {"aud": os.getenv("GOOGLE_CLIENT_ID")}
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="valid_token")
    result = await verify_google_token(credentials)
    assert result["aud"] == os.getenv("GOOGLE_CLIENT_ID")

def test_query_rag_flow(monkeypatch):
    import numpy as np
    from unittest.mock import MagicMock

    # Mock embedder.encode to return a NumPy array
    monkeypatch.setattr("app.embedder.encode", lambda x: np.array([[0.1] * 384]))

    # Mock Redis cache (simulate cache miss and successful set)
    mock_redis = MagicMock()
    mock_redis.get.return_value = None  # Simulate cache miss
    mock_redis.set.return_value = True  # Simulate successful caching
    monkeypatch.setattr("app.r", mock_redis)

    # Mock ChromaDB collection with a fake query result
    class MockCollection:
        def query(self, query_embeddings, n_results):
            return {"documents": [["Test document 1", "Test document 2"]]}
    monkeypatch.setattr("app.get_collection", lambda: MockCollection())

    # Mock text generation pipeline
    mock_generator = MagicMock(return_value=[{"generated_text": "Test answer"}])
    monkeypatch.setattr("app.generator", mock_generator)

    # Create a fake request
    req = QueryRequest(query="What is FastAPI?", top_k=2)

    # Directly call the route function (sync function, no await needed)
    result = query_rag(req, token_info={"aud": os.getenv("GOOGLE_CLIENT_ID")})

    # Validate the output
    assert "answer" in result
    assert result["answer"] == "Test answer"
