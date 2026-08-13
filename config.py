"""
Configuration shared by ingest.py, rag_chain.py and app.py
"""

import os
from dotenv import load_dotenv

load_dotenv()

#Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_PATH = os.path.join(DATA_DIR, "sample.pdf")
PERSIST_DIR = "./chroma"
STREAMLIT_SESSION_DIR = os.path.join(BASE_DIR, "chroma_sessions")
COLLECTION_NAME = "cv_collection"


#Chunking 
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

#Embedding model name 
EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5"


#Hybrid retrieval
RETRIEVAL_K = 4
BM25_WEIGHT = 0.5
DENSE_WEIGHT = 0.5

#LLM
LLM_MODEL = "openai/gpt-oss-20b"
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_API_KEY_ENV = "API_KEY"
LLM_TEMPERATURE = 0.2




