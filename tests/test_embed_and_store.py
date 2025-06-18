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

    #Defines what happens when the test calls mock_embedder.encode(...)
    mock_embedder.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]

    #dont use the original sentence transformer class, use the mock instead
    mock_embedder_class.return_value = mock_embedder

    #similarly for the db stuff
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.create_collection.return_value = mock_collection
    mock_client_class.return_value = mock_client

    embed_and_store(sample_docs)

    mock_embedder.encode.assert_called_once_with(sample_docs)
    mock_client.create_collection.assert_called_once_with(name="documents")

    expected_ids = [f"doc{i}" for i in range(len(sample_docs))]
    mock_collection.add.assert_called_once_with(
        ids=expected_ids,
        documents=sample_docs,
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
    )
    
CHROMA_PATH = "./chroma_persist"

def clear_chroma_persist():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

def test_integration_embed_and_store():
    # Setup: clean before
    clear_chroma_persist()

    sample_docs = [
        "Python is a programming language.",
        "Chroma stores embeddings.",
    ]

    # Run the real function
    embed_and_store(sample_docs)

    # Connect to Chroma DB to validate
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name="documents")
    results = collection.get()

    assert len(results["documents"]) == 2
    assert sample_docs[0] in results["documents"]
    assert sample_docs[1] in results["documents"]

    # Teardown: clean after
    clear_chroma_persist()
