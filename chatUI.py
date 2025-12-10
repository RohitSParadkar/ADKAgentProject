import streamlit as st
import tempfile
import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma

from langchain_classic.chains import RetrievalQA

from pypdf import PdfReader

# --- CONFIG ---
MODEL_NAME = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
CHROMA_DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "nomic-embed-text"
# ---------------

st.set_page_config(page_title="Llama 3.2 PDF Chatbot", layout="centered")
st.title("📄 Llama 3.2 Local PDF Chatbot")

# --- SESSION STATE ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR FILE UPLOAD ---
uploaded_file = st.sidebar.file_uploader("📎 Upload PDF file", type=["pdf"])

# --- PROCESS PDF ---
if uploaded_file and not st.session_state.pdf_loaded:
    with st.spinner("Processing PDF..."):

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        reader = PdfReader(temp_path)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() + "\n"

        # Split PDF text into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_text(full_text)

        # Create embeddings and store in Chroma
        embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )

        vectorstore = Chroma.from_texts(
            chunks,
            embedding=embedder,
            persist_directory=CHROMA_DB_DIR
        )
        vectorstore.persist()

        st.session_state.vectorstore = vectorstore
        st.session_state.pdf_loaded = True
        st.sidebar.success("✅ PDF indexed successfully!")

        os.remove(temp_path)

# --- LLM SETUP ---
llm = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)

# --- CHAT UI ---
st.divider()

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("Ask something from the PDF or general question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- PDF MODE (Always RAG) ---
    if st.session_state.pdf_loaded:
        with st.chat_message("assistant"):
            with st.spinner("Searching your PDF..."):
                try:
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=retriever,
                        chain_type="stuff"
                    )
                    answer = qa_chain.run(prompt)
                except Exception as e:
                    answer = f"⚠️ Error while running QA: {e}"

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

    # --- GENERAL CHAT MODE ---
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                payload = {
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False
                }
                try:
                    res = requests.post(
                        f"{OLLAMA_BASE_URL}/api/generate",
                        json=payload,
                        timeout=120
                    )
                    res.raise_for_status()
                    data = res.json()
                    response_text = data.get("response") or data.get("output") or "No response"
                except Exception as e:
                    response_text = f"⚠️ Error: {e}"

                st.markdown(response_text)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )
