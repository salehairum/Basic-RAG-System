# #running fastAPI: uvicorn app:app --reload

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

app = FastAPI()

# Initialize embedding model and Chroma DB client
embedder = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.PersistentClient(path="./chroma_persist")
collection = client.get_collection(name="documents")

generator = pipeline("text2text-generation", model="google/flan-t5-base")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

@app.post("/query")
def query_rag(request: QueryRequest):
    # Embed query
    query_embedding = embedder.encode([request.query])[0].tolist()
    
    # Retrieve relevant docs from Chroma DB
    results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)
    
    if not results['documents'][0]:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    # Combine retrieved documents into context
    context = " ".join(results['documents'][0])
    
    # Generate response using LLM with context + query
    prompt = f"Context: {context}\nQuestion: {request.query}\nAnswer:"
    generated = generator(
    prompt,
    max_length=150,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True
)

    
    return {"answer": generated[0]['generated_text']}
