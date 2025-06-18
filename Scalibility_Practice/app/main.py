from fastapi import FastAPI
import socket
import time
import random

app = FastAPI()

@app.get("/query")
def query():
    # Simulate variable processing delay
    time.sleep(random.uniform(0.1, 0.5))
    
    return {
        "response": "Hello from FastAPI!",
        "instance": socket.gethostname()
    }
