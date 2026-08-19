"""
CLI conversational RAG Chain over an already ingested CV (Run ingest.py first)
Reuses config.py for all settings and prompts and ingest.py for the vector-store 
and retriever logic, so there is exactly one embedding model, one prompt template
and one BM25 preprocessing function across the whole project. 
"""

import os 

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import config as config
from ingest import get_embeddings, build_hybrid_retriever_from_store

#Build vector store and return it 
def load_vectorstore(persist_directory: str = config.PERSIST_DIR) -> Chroma:
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )

#Function to build LLM and return it 
def build_llm() -> ChatOpenAI:
    api_key = os.environ.get(config.LLM_API_KEY_ENV) #Retrieving API key 
    if not api_key:
        raise EnvironmentError(f"Set {config.LLM_API_KEY_ENV} in your environment before running")
    return ChatOpenAI(
        model = config.LLM_MODEL,
        base_url = config.LLM_BASE_URL,
        api_key= api_key,
        temperature = config.LLM_TEMPERATURE,
    ) 

def build_contextualize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", config.CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def build_answer_prompt(person_name: str = config.PERSON_NAME) -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", config.build_system_prompt(person_name)),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

def build_history_aware_retriever(llm, vectorstore):
    hybrid_retriever = build_hybrid_retriever_from_store(vectorstore)
    return create_history_aware_retriever(llm, hybrid_retriever, build_contextualize_prompt())

def build_rag_chain(llm, build_history_aware_retriever, person_name: str = config.PERSON_NAME):
    document_chain = create_stuff_documents_chain(llm, build_answer_prompt(person_name))
    return create_retrieval_chain(build_history_aware_retriever, document_chain)

_session_store: dict[str, BaseChatMessageHistory] = {} ## session is used to store the users history 

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
    return _session_store[session_id]

def build_conversational_chain(
    persist_directory: str = config.PERSIST_DIR, 
    person_name: str = config.PERSON_NAME,
):
    vectorstore = load_vectorstore(persist_directory)
    llm = build_llm()
    history_aware_retriever = build_history_aware_retriever(llm, vectorstore)
    rag_chain = build_rag_chain(llm, history_aware_retriever, person_name)

    return RunnableWithMessageHistory(
        rag_chain.
        get_session_history,
        input_messages_key="chat_history",
        output_messages_key= "answer", 

    )



