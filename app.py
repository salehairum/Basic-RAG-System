from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import time
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import httpx
from jose import jwt
from jose.exceptions import JWTError
from dotenv import load_dotenv
import os
import redis
import pickle

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

REDIS_HOST = os.getenv("REDIS_HOST", "redis") 
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2


security = HTTPBearer()

load_dotenv()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

@app.get("/oauth/callback")
async def oauth_callback(request: Request):
    
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Authorization code not found")

    # Exchange the code for tokens
    async with httpx.AsyncClient() as client:
        token_response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "redirect_uri": "http://localhost:8000/oauth/callback",
                "grant_type": "authorization_code",
            }
        )
    if token_response.status_code != 200:
        content = await token_response.aread()
        detail = content.decode()
        raise HTTPException(status_code=token_response.status_code, detail=f"Failed to get tokens: {detail}")

    tokens = token_response.json()
    # tokens contains access_token, id_token, refresh_token, expires_in, etc.
    return tokens


async def verify_google_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    try:
        # Verify token by calling Google's tokeninfo endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://oauth2.googleapis.com/tokeninfo",
                params={"id_token": token},
            )
            if response.status_code != 200:
                raise HTTPException(status_code=401, detail="Invalid token")
            token_info = response.json()

            # Verify audience
            if token_info.get("aud") != GOOGLE_CLIENT_ID:
                raise HTTPException(status_code=401, detail="Invalid audience")

            # You can extract user info here if needed
            return token_info

    except httpx.HTTPError:
        raise HTTPException(status_code=401, detail="Failed to validate token")

def get_cached_embedding(text: str):
    cached = r.get(text)
    if cached:
        # Deserialize embedding vector
        return pickle.loads(cached)
    
    # Compute embedding if not cached
    embedding = embedder.encode([text])[0].tolist()
    
    # Serialize and store in Redis with 1 hour expiry
    r.set(text, pickle.dumps(embedding), ex=3600)
    return embedding

@app.post("/query")
def query_rag(request: QueryRequest, token_info: dict = Depends(verify_google_token)):
    start_time = time.time()
    try:
        logger.info(f"Received query: {request.query} with top_k: {request.top_k}")

        # Embedding
        embed_start = time.time()
        query_embedding = get_cached_embedding(request.query)
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