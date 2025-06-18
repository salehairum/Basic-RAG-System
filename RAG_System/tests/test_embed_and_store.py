import pytest
from unittest.mock import patch, MagicMock
import os
import shutil
import chromadb
from embed_and_store import embed_and_store

@pytest.fixture
def sample_docs():
    return [
        "Sample document 1.",
        "Sample document 2.",
    ]
@patch("embed_and_store.SentenceTransformer")
@patch("embed_and_store.chromadb.PersistentClient")
def test_embed_and_store(mock_client_class, mock_embedder_class, sample_docs):
    mock_embedder = MagicMock()
    
    # Fix: make mock_embedder the return value of the mocked SentenceTransformer
    mock_embedder_class.return_value = mock_embedder

    mock_embeddings = MagicMock()
    mock_embeddings.tolist.return_value = [[0.1, 0.2], [0.3, 0.4]]
    mock_embedder.encode.return_value = mock_embeddings

    # Mock DB
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    mock_client_class.return_value = mock_client

    # Call actual function
    embed_and_store(sample_docs)

    # Assertions
    mock_embedder.encode.assert_called_once_with(sample_docs)
    mock_client.get_or_create_collection.assert_called_once_with(name="documents")


    expected_ids = [f"doc{i}" for i in range(len(sample_docs))]
    mock_collection.add.assert_called_once_with(
        ids=expected_ids,
        documents=sample_docs,
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
