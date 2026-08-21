"""
Reuses inges.py(load/clean/chunk/embed/hyrbid-retriever) and 
rag_chain.py (LLM, prompts, history-aware retriever, RAG chain ) 
instead of re-implementing  the pipeline for a third time. 
"""

import os
import tempfile
import uuid
import sys

import streamlit as st
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings

import config as config
from ingest import load_pdf, clean_documents, chunk_document, build_vector_store, build_hybrid_retriever
from rag_chain import build_llm, build_contextualize_prompt, build_answer_prompt


def _running_under_streamlit() -> bool:
    return st.runtime.exists()

@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDINGS_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def get_cached_llm():
    return build_llm()

def ingest_uploaded_cv(uploaded_file, session_id: str):
    """
    Write the upload to a temp file, then run it throught the shared
    ingest.py pipeline (load ->clean -> embed).
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        docs = load_pdf(tmp_path)
        docs = clean_documents(docs)
        chunks = chunk_document(docs)

        persist_dir = os.path.join(config.STREAMLIT_SESSION_DIR, session_id)
        vectorstore = build_vector_store(
            chunks, 
            embeddings = get_embeddings(),
            persist_directory = persist_dir,
            collection_name =  config.COLLECTION_NAME,
            force_reembed = True, 
        )
    finally:
        os.unlink(tmp_path)

    return vectorstore, chunks


def build_chain_for_session(vectorstore, chunks, person_name: str, history):
    llm = get_cached_llm()
    hybrid_retriever = build_hybrid_retriever(vectorstore, chunks)

    history_aware_retriever = create_history_aware_retriever(
        llm, hybrid_retriever, build_contextualize_prompt()
    )

    document_chain = create_stuff_documents_chain(llm, build_answer_prompt(person_name))
    rag_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    return RunnableWithMessageHistory(
        rag_chain, 
        lambda _session_id: history, 
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer",
    )

def run_streamlit():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "chat_message_history" not in st.session_state:
        st.session_state.chat_message_history = InMemoryChatMessageHistory()
    if "chain" not in st.session_state:
        st.session_state.chain = None
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
    if "person_name" not in st.session_state:
        st.session_state.person_name = ""

    
    st.set_page_config(page_title="Lets explore about ME", page_icon="🤖")
    st.title("Talk to human Encyclopedia🤖 aka SUMAN")
    st.caption("Upload a cv and chat with 🤖 Suman that answers as you for any of your questions")

    with st.sidebar:
        st.header("1. Upload CV")
        uploaded_file = st.file_uploader("CV(PDF)", type=["pdf"])
        person_name = st.text_input("CV owners name", placeholder= "e.g. Alex Sharma")
        process_clicked = st.button("Process CV ", type = "primary")

        if process_clicked:
            if not uploaded_file:
                st.error("Please upload aPDF first.")
            elif not person_name.strip():
                st.error("Please enter the CV owner's name.")
            else:
                with st.spinner("Reading CV, Chunking, embedding, indexing........... Suman is working.... behold"):
                    try:
                        vectorstore, chunks = ingest_uploaded_cv(uploaded_file, st.session_state.session_id)
                        st.session_state.chain = build_chain_for_session(
                            vectorstore, 
                            chunks, 
                            person_name.strip(),
                            st.session_state.chat_message_history,
                        )
                        st.session_state.person_name = person_name.strip()
                        st.session_state.display_history = []
                        st.session_state.chat_message_history.clear()
                    except Exception as e:
                        st.error(f"Error processing CV: {e}")
                    else:
                        st.success(f"CV loaded. Ask me anything about {person_name.strip()}!")
            if st.button("Reset Session"):
                for key in ["session_id", "chat_message_history", "chain", "display_history", "person_name"]:
                    del st.session_state[key]
                st.rerun()

    for role, text in st.session_state.display_history:
        with st.chat_message(role):
            st.markdown(text)

    user_input = st.chat_input(
        "Ask Suman a question about this CV.......",
        disabled=st.session_state.chain is None,
    )
    
    if user_input:
        st.session_state.display_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking....."):
                response = st.session_state.chain.invoke(
                    { "input": user_input},
                    config = {"configurable": {"session_id": st.session_state.session_id}},
                )
                answer = response["answer"]
                st.markdown(answer)
        
        st.session_state.display_history.append(("assistant", answer))
    
    if st.session_state.chain is None:
        st.info("upload a CV in the sidebar and click 'Process CV' to start chatting.")
    
if __name__ =="__main__":
    if not _running_under_streamlit():
        print("Run this app with: Streamlit run app.py", file=sys.stderr)
        raise SystemExit(1)
    run_streamlit()