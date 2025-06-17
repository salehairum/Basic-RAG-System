# #running fastAPI: uvicorn app:app --reload
#basic curl command
#curl -X POST http://localhost:8000/query ^
#   -H "Content-Type: application/json" ^
#   -d "{\"query\": \"What is Retrieval-Augmented Generation?\", \"top_k\": 2}"


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()  # still log to terminal
    ]
)


logger = logging.getLogger(__name__)

app = FastAPI()

Instrumentator().instrument(app).expose(app)

# Initialize embedding model and Chroma DB client
logger.info("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

logger.info("Setting up Chroma DB client...")
client = chromadb.PersistentClient(path="./chroma_persist")
collection = client.get_collection(name="documents")

generator = pipeline("text2text-generation", model="google/flan-t5-base")

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

@app.post("/query")
def query_rag(request: QueryRequest):
    start_time = time.time()
    try:
        logger.info(f"Received query: {request.query} with top_k: {request.top_k}")

        # Embedding
        embed_start = time.time()
        query_embedding = embedder.encode([request.query])[0].tolist()
        logger.info(f"Query embedding computed in {time.time() - embed_start:.2f}s")

        # Retrieval
        retrieval_start = time.time()
        results = collection.query(query_embeddings=[query_embedding], n_results=request.top_k)
        logger.info(f"Document retrieval took {time.time() - retrieval_start:.2f}s")

        if not results['documents'][0]:
            logger.warning("No documents retrieved.")
            raise HTTPException(status_code=404, detail="No relevant documents found.")

        context = " ".join(results['documents'][0])

        # Generation
        generation_start = time.time()
        prompt = f"Context: {context}\nQuestion: {request.query}\nAnswer:"
        generated = generator(
            prompt,
            max_length=150,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        logger.info(f"Text generation took {time.time() - generation_start:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"Total query handling time: {total_time:.2f}s")

        return {"answer": generated[0]['generated_text']}

    except Exception as e:
        logger.exception("Error during RAG query handling")
        raise HTTPException(status_code=500, detail=str(e))