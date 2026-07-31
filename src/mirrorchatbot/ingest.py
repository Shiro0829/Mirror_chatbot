"""
Phase 1: ingestion pipeline. 
    
    Step 1: Load the pdf
    Step 2: Extract text from the pdf and clean the text
    Step 3: Split the text into chunks
    Step 4: Create embeddings for each chunk and store them in a vector database
    Step 5: Create a hybrid retriver(BM25 + Vector Search) for retrieval
    Step 6: Test the retriever with a sample query
"""

import os
import re

from langchain_community.document_loaders import PyPDFLoader

## Config Path 
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "sank_resume.pdf") ##absolute path for the data 
PERSIST_DIR = "./chroma" ## Path for storing vector embedding
COLLECTION_NAME = "cv_collection" ## Name of the collection in vector database

##The maximum number of characters or tokens allowed in a single text segment.
CHUNK_SIZE = 600 

##The amount of text copied from the end of one segment and placed at the start of the next one.
CHUNK_OVERLAP = 100 

EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5" ## model for embedding 


##Load the pdf
def load_pdf(pdf_path:str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from the PDF.")
    return docs

def clean_text(text:str)-> str:
    """
    Clean the text by removing unwanted characters and formattings.
    """
    ## split the text into individual lies and strip whitespace
    lines = [line.strip() for line in text.split("\n")]
    ## Remove empty lines
    lines = [line for line in lines if line]
    ## join the lines back into a single string
    return "\n".join(lines)

def clean_documents(docs):
    """
    Clean the text in each document
    """
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    return docs



raw_docs = load_pdf(PDF_PATH)
print(raw_docs[0].page_content)
