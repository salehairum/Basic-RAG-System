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
    # Mock embedding
    monkeypatch.setattr("app.embedder.encode", lambda x: [[0.1]*384])

    # Mock retrieval
    monkeypatch.setattr("app.collection.query", lambda query_embeddings, n_results: {
        "documents": [["Test document 1", "Test document 2"]]
    })

    # Mock generation
    monkeypatch.setattr("app.generator.__call__", lambda *args, **kwargs: [{"generated_text": "Test answer"}])

    req = QueryRequest(query="What is FastAPI?", top_k=2)
    result = query_rag(req, token_info={"aud": os.getenv("GOOGLE_CLIENT_ID")})

    assert "answer" in result
    assert result["answer"] == "Test answer"