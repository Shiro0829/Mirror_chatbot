"""
Phase 1: Ingestion pipeline 

    step1: Load pdf
    step2: Extract text from pdf and clean the text
    step3: Split the data into chunk, 
    step4: Create embedding for each chunk and store them in a vector database
    step5: Create a hybridretriever (BM25 + vector search) for retrieval 
    step6: Test the hybridretriever
"""

import os
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever 
from langchain_classic.retrievers import EnsembleRetriever

## Defining config path
PDF_PATH = os.path.join(os.path.dirname(__file__), "data", "sample.pdf")
PERSIST_DIR = "./chroma" ## Path for storing vector database
COLLECTION_NAME = "cv_collection"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

EMBEDDINGS_MODEL_NAME = "BAAI/bge-small-en-v1.5"

##    Step1: Load pdf
def load_pdf(pdf_path:str):
   ## Load a PDF document from the specified path using PyPDFLoader.
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # print(f"Loaded {len(docs)} pages from the PDF. ")
    return docs


def clean_text(text:str)-> str:
    """
    Cleaning the text by removing unwanted characters and formatting.
    """
    ## Splitting the text into individual lines and removing whitespaces
    lines = [line.strip() for line in text.split("\n")]

    ## Removing empty lines
    lines = [line for line in lines if line]

    ##Joining the lines back into a single string
    return "\n".join(lines)


def clean_documents(docs):
    """
    Clean the text in each document.
    """
    for doc in docs:
        doc.page_content =  clean_text(doc.page_content)
    return docs




def chunk_document(docs):
    """
    Splitting the documents into smaller chunks for processing.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators= ["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(docs)
    return chunks



_embeddings =None
def get_embeddings():
    """
    Get the embedding model. Create it if it doesn't exist.
    """
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name = EMBEDDINGS_MODEL_NAME)
    return _embeddings


def build_vector_store(chunks, persist_dir = PERSIST_DIR, collection_name= COLLECTION_NAME):
    """
    Building a vector store from the document chunks.
    """
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name= COLLECTION_NAME,
        embedding_function= embeddings,
        persist_directory= persist_dir
    )
    existing_count = vector_store._collection.count()
    if existing_count > 0:
        print(f"Collection '{collection_name}'already exists with {existing_count} documents. Skipping insertion.")
        return vector_store
    
    vector_store.add_documents(chunks)
    print(f"Inserted {len(chunks)}documents into the collection '{collection_name}'.")
    return vector_store


def build_hybrid_retriever(vector_store, chunks, top_k: int =4, dense_weight: float= 0.5, bm_25_weight: float = 0.5):
    """
    Build a hybrid retriever using BM25 and vector search.
    """
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = top_k

    dense_retriever =  vector_store.as_retriever(search_kwargs={"k": top_k})
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[bm_25_weight, dense_weight]
    )
    return hybrid_retriever

    

## Test function to check the retriever
def test_hybrid_retriever(retriever, query: str):
    """
    Test the hybrid retriever with sample query
    """
    results = retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f" Content: {doc.page_content[:200]}......")
        print("==================")
    return results




raw_docs = load_pdf(PDF_PATH)
cleaned_docs = clean_documents(raw_docs)
chunks = chunk_document(cleaned_docs)
vector_store = build_vector_store(chunks)
retriever = build_hybrid_retriever(vector_store, chunks)

print("----------------------------------------------")
print(test_hybrid_retriever(retriever, "What CERTIFICATIONS are in document?"))
print("----------------------------------------------")


# print(repr(chunks[0].page_content))
# print("\n ----------------------------------secondchunk---------------------")
# print(repr(chunks[1].page_content))


# print("\n ----------------------------------ThirdChunk---------------------")
# print(repr(chunks[2].page_content))
# if __name__ == "__main__":
#     docs = load_pdf(PDF_PATH)


