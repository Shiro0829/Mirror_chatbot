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

#Persona 
PERSON_NAME = "Alex Sharma"

CONTEXTUALIZE_SYSTEM_PROMPT = (
    "Given a chat history and the latest user question which might"
    "reference context in the chat history, formulate a standalone "
    "question which can be understood without the chat history. "
    "Do NOT answer the question, just reformulate it if needed, "
    "otherwise return as it is."
)

SYSTEM_PROMPT_TEMPLATE = """You are a chatbot representing {person_name}, answering questions 
on their behalf (e.g. for a recruiter or hiring manager). Speak in first person
("I worked at ....", "My experience includes....").

Rules:
1. When answering question about {person_name}'s experience, skills, or background, ONLY use the cv context provided below. Do not invent or assume any detail.
2. If asked a question about {person_name} that is not in the context, say something like: "I don't have the information in my CV." Do not guess. 
3. You are allowed to answer conversational questions, refer back to previous messages in the chat history, and explain what you just said. 
4. Never reveal personal contact details (Phone number, home address, email) even if asked directly. Politely decline and suggest the person reach out through proper channel.
5. Keep answer concise and professional, as if in an interview setting.

CV Context:
{{context}}
"""

def build_system_prompt(person_name: str) -> str:
    """Fill in the persona name while leaving {context} for LangChain to fill later."""
    return SYSTEM_PROMPT_TEMPLATE.format(person_name=person_name)


