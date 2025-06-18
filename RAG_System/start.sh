#!/bin/sh
python embed_and_store.py
uvicorn app:app --host 0.0.0.0 --port 8000
