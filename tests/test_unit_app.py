import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
import httpx
from app import verify_google_token, query_rag, QueryRequest
import os
from embed_and_store import embed_and_store


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
    # Mock embedding model encode method to return a fixed embedding vector
    monkeypatch.setattr("app.embedder.encode", lambda x: [[0.1] * 384])

    # Create a mock collection class with a query method returning dummy docs
    class MockCollection:
        def query(self, query_embeddings, n_results):
            return {
                "documents": [["Test document 1", "Test document 2"]]
            }

    # Patch app.get_collection to return the mock collection
    monkeypatch.setattr("app.get_collection", lambda: MockCollection())

    # Patch the generator pipeline to return the fixed answer
    from unittest.mock import MagicMock
    mock_generator = MagicMock(return_value=[{"generated_text": "Test answer"}])
    monkeypatch.setattr("app.generator", mock_generator)

    # Create a QueryRequest object
    req = QueryRequest(query="What is FastAPI?", top_k=2)

    # Call the endpoint function directly (note it’s not async)
    result = query_rag(req, token_info={"aud": os.getenv("GOOGLE_CLIENT_ID")})

    # Assertions to verify behavior
    assert "answer" in result
    assert result["answer"] == "Test answer"
