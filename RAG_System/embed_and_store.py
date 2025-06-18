from sentence_transformers import SentenceTransformer
import chromadb

def embed_and_store(documents):
    # Initialize embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Initialize Chroma client and collection
    client = chromadb.PersistentClient(path="./chroma_persist")
    collection = client.get_or_create_collection(name="documents")
    
    # Embed documents
    embeddings = embedder.encode(documents).tolist()   #uses BERT contextual embedding.
    
    # Prepare data for Chroma DB
    ids = [f"doc{i}" for i in range(len(documents))]
    
    # Add documents to Chroma DB collection
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
    )
    
    print(f"Added {len(documents)} documents to Chroma DB.")

if __name__ == "__main__":
    docs = [
        "FastAPI is a modern, fast web framework for Python.",
        "Hugging Face provides powerful pretrained models.",
        "Chroma DB is a vector database optimized for embeddings.",
        "Retrieval-Augmented Generation combines retrieval with generation.",
    ]
    embed_and_store(docs)
