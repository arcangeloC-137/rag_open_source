import streamlit as st
import subprocess
import ollama
from rag_pipeline import RAGPipeline
from rag import RAG
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from utils import compute_file_hash
from opensearchpy.exceptions import NotFoundError
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

WEBAPP_TITLE = os.environ.get("WEBAPP_TITLE")
PAGE_NAME = os.environ.get("PAGE_NAME")
FAVICON = os.environ.get("FAVICON")
AVATAR_ICON = os.environ.get("AVATAR_ICON")
USER_ICON = os.environ.get("USER_ICON")
LOGO = os.environ.get("LOGO")

st.set_page_config(
    page_title=f"{PAGE_NAME}",
    page_icon=f"static/{FAVICON}",
    layout="wide"
)

hide_streamlit_style = """
<head>
    <style>
     MainMenu {visibility: hidden;}
     header {visibility: hidden;}
    </style>
</head>
"""

# Carica le variabili d'ambiente dal file .env
load_dotenv()

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL")
OPENSEARCH_USER = os.environ.get("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.environ.get("OPENSEARCH_INITIAL_ADMIN_PASSWORD")
INDEX_NAME = os.environ.get("INDEX_NAME", "rag-index") 

DOCUMENTS_DIR = Path("documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)  
INDEX_REGISTRY = DOCUMENTS_DIR / ".indexed_docs.json"

# Load or initialize registry
if INDEX_REGISTRY.exists():
    with open(INDEX_REGISTRY, "r") as f:
        indexed_docs = json.load(f)
else:
    indexed_docs = {}


# Cache dei modelli disponibili
@st.cache_data
def get_available_models():
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
    return [line.split()[0] for line in result.stdout.splitlines()[1:]]

# Cache delle istanze RAGPipeline e RAG
@st.cache_resource
def get_pipeline():
    return RAGPipeline(OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASSWORD, INDEX_NAME)

@st.cache_resource
def get_rag():
    return RAG(OPENSEARCH_URL, OPENSEARCH_USER, OPENSEARCH_PASSWORD, INDEX_NAME)

# Aggiunto per il debug
def debug_opensearch():
    rag = get_rag()
    return rag.debug_index()

# Sidebar
st.sidebar.image(f"static/{LOGO}")

st.sidebar.header("Model Selection")
available_models = get_available_models()
model_choice = st.sidebar.selectbox("Choose a model", available_models)

# Upload e indicizzazione
st.sidebar.header("📤 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type="pdf")

# Get indexed documents
selected_doc = None
try:
    indexed_docs_list = get_pipeline().get_indexed_documents()
    print(f"Index doc list: {indexed_docs_list}")
    if indexed_docs_list and len(indexed_docs_list) > 0:
        selected_doc = st.sidebar.selectbox("Select document to query", indexed_docs_list)
    else:
        st.sidebar.write("No documents indexed yet!")
except NotFoundError as nfe:
    st.sidebar.write("No documents indexed yet - Index doesn't exist!")
except Exception as e:
    st.sidebar.error(f"Error: {str(e)}")

# Debug button
if st.sidebar.button("Debug OpenSearch"):
    debug_info = debug_opensearch()
    st.sidebar.code(str(debug_info))

# Index documents
if st.sidebar.button("Index Documents") and uploaded_files:
    pipeline = get_pipeline()

    json_indexed_docs = {}
    for file in uploaded_files:
        file_bytes = file.read()
        file_hash = compute_file_hash(file_bytes)

        if file_hash in indexed_docs:
            st.sidebar.info(f"Skipped already indexed: {file.name}")
            continue

        save_path = DOCUMENTS_DIR / file.name
        print(f"Save Path: {save_path}")
        with open(save_path, "wb") as f:
            f.write(file_bytes)

        # Run indexing
        pipeline.run_pipeline(str(save_path))

        # Update registry
        json_indexed_docs[file_hash] = file.name
    
    # Store indexed docs
    with open(INDEX_REGISTRY, "w") as f:
        json.dump(json_indexed_docs, f)

    st.sidebar.success("Indexing complete!")
    st.sidebar.info("Debug OpenSearch to verify index structure")

# Flush all documents
flush_all_docs = st.sidebar.button("Delete all documents")
if flush_all_docs:
    pipeline = get_pipeline()
    pipeline.flush_all()
    # Pulisci anche il registro locale
    if INDEX_REGISTRY.exists():
        os.remove(INDEX_REGISTRY)
        indexed_docs = {}
    st.sidebar.success("All docs deleted successfully!")

# Chat
st.title(f"{WEBAPP_TITLE}")
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    if msg["role"] == "assistant":
        with st.chat_message(msg["role"], avatar=f"static/{AVATAR_ICON}"):
            st.markdown(msg["content"])
    else:
        with st.chat_message(msg["role"], avatar=f"static/{USER_ICON}"):
            st.markdown(msg["content"])

if model_choice and selected_doc:
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user", avatar=f"static/{USER_ICON}"):
            st.markdown(user_input)

        rag = get_rag()
        print(f"Documento selezionato: {selected_doc}")
        chunks = rag.retrieve_chunks(user_input, selected_doc)
        prompt = rag.construct_prompt(user_input, chunks)
        print(f"Prompt: {prompt}")

        try:
            with st.spinner("Thinking..."):
                response = ollama.chat(model=model_choice, messages=[{"role": "user", "content": prompt}])
                reply = response["message"]["content"].split("</think>")[-1]
        except Exception as e:
            reply = f"Error: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        with st.chat_message("assistant", avatar=f"static/{AVATAR_ICON}"):
            st.write(reply)
else:
    if not model_choice:
        st.info("Please select or download a model.")
    elif not selected_doc:
        st.info("Please upload and index a document before asking questions.")
    else:
        st.info("Please select both a model and a document.")