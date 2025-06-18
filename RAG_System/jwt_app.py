# running fastAPI: uvicorn app:app --reload

#basic curl commands for testing (on windows command prompt)

# curl -X POST http://localhost:8000/login ^
#  -H "Content-Type: application/x-www-form-urlencoded" ^
#  -d "username=user1&password=password123"

#curl -X POST http://localhost:8000/query ^
#   -H "Content-Type: application/json" ^
#   -H "Authorization: Bearer JWT_TOKEN" ^
#   -d "{\"query\": \"What is Retrieval-Augmented Generation?\", \"top_k\": 2}"


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
import logging
from prometheus_fastapi_instrumentator import Instrumentator
import time
import jwt
from fastapi import Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timezone, timedelta
from fastapi import Form
from dotenv import load_dotenv
import os

load_dotenv()

#logging
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

JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Simple user "database"
users_db = {
    "user1": pwd_context.hash("password123"),
    "user2": pwd_context.hash("mypassword")
}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 2

security = HTTPBearer()
#utility functions for jwt authentication
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username is None or username not in users_db:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

#endpoints
@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if username not in users_db or not verify_password(password, users_db[username]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    access_token = create_access_token(data={"sub": username}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/query")
def query_rag(request: QueryRequest, current_user: str = Depends(get_current_user)):
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
   