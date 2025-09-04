# app.py
import os
import streamlit as st
from typing import List, Dict
from utils import (
    extract_text, chunk_text, load_embedding_model, embed_texts,
    get_vector_store, build_doc_store, generate_answer
)
import json
import time
import base64

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Retrieval QA (Streamlit + Qdrant)", layout="wide")

# Simple session memory structure:
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts: {"role":"user/assistant/system", "text": "..."}
if "kb_store" not in st.session_state:
    st.session_state.kb_store = None  # actual vector store instance
if "docs_indexed" not in st.session_state:
    st.session_state.docs_indexed = []  # metadata list of docs we added

# Load vector store (prefer Qdrant if env var present)
if st.session_state.kb_store is None:
    try:
        st.session_state.kb_store = get_vector_store(prefer_qdrant=True)
        st.session_state.kb_type = type(st.session_state.kb_store).__name__
    except Exception as e:
        st.session_state.kb_store = get_vector_store(prefer_qdrant=False)
        st.session_state.kb_type = type(st.session_state.kb_store).__name__

st.sidebar.title("Settings")
st.sidebar.write(f"Vector store: **{st.session_state.kb_type}**")
st.sidebar.write("Embedding model: `sentence-transformers/all-MiniLM-L6-v2`")
st.sidebar.write("Generator model: `google/flan-t5-small` (lightweight)")

# -------------------------
# UI - Upload files & options
# -------------------------
st.title("Retrieval QA — Upload PDF/DOCX and Ask Questions")
st.markdown("Upload PDF or DOCX files (≤100 pages). The app chunks documents, stores embeddings, and answers questions using a lightweight open-source model.")

with st.expander("Upload documents"):
    uploaded = st.file_uploader("Upload PDF or DOCX files (multiple)", accept_multiple_files=True, type=["pdf", "docx"])
    if uploaded:
        for file in uploaded:
            name = file.name
            bytestr = file.read()
            try:
                text = extract_text(name, bytestr)
            except Exception as e:
                st.error(f"Failed to extract text from {name}: {e}")
                continue
            # enforce page/size check (best-effort)
            if len(text.splitlines()) < 1:
                st.warning(f"No text found in {name}. Skipping.")
                continue
            # Build knowledge base
            with st.spinner(f"Indexing {name} ..."):
                added = build_doc_store(text, st.session_state.kb_store, chunk_size=1000, overlap=200, source_name=name)
                st.session_state.docs_indexed.append({"name": name, "chunks": len(added)})
                st.success(f"Indexed {name}: {len(added)} chunks")

st.markdown("---")
col1, col2 = st.columns([3,1])
with col1:
    st.subheader("Chat")
    query = st.text_input("Ask a question about the uploaded documents", placeholder="Type something like: 'What are the main conclusions?'")
    ask_button = st.button("Ask")
with col2:
    st.subheader("Controls")
    clear = st.button("Clear conversation")
    export = st.button("Export chat (JSON)")

if clear:
    st.session_state.chat_history = []
    st.success("Cleared conversation")

if export:
    payload = {"chat": st.session_state.chat_history, "indexed_docs": st.session_state.docs_indexed}
    b = json.dumps(payload, indent=2).encode("utf-8")
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="chat_export.json">Download chat JSON</a>'
    st.markdown(href, unsafe_allow_html=True)

# -------------------------
# Query handling
# -------------------------
if ask_button and query:
    st.session_state.chat_history.append({"role":"user", "text": query, "time": time.time()})
    # 1. compute embedding for query
    embed_model = load_embedding_model()
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]

    # 2. retrieve top chunks
    top_k = 5
    hits = st.session_state.kb_store.query(q_emb, top_k=top_k)

    # 3. build RAG prompt: include the retrieved context pieces
    context_texts = [h[2] for h in hits]  # h = (id, score, text, metadata)
    combined_context = "\n\n---\n\n".join(context_texts)
    rag_prompt = f"""You are a helpful assistant. Use the following extracted context from uploaded documents to answer the user's question. If the answer is not in the context, say 'I could not find the answer in the provided documents.' Context:\n\n{combined_context}\n\nQuestion: {query}\n\nAnswer concisely but thoroughly."""
    # 4. generate answer
    with st.spinner("Generating answer..."):
        answer = generate_answer(rag_prompt, max_length=256)
    st.session_state.chat_history.append({"role":"assistant", "text": answer, "time": time.time(), "source_chunks": [{"score": h[1], "metadata": h[3]} for h in hits]})

# -------------------------
# Display chat history
# -------------------------
for msg in st.session_state.chat_history:
    role = msg.get("role", "user")
    if role == "user":
        st.markdown(f"**You:** {msg['text']}")
    else:
        st.markdown(f"**Assistant:** {msg['text']}")
        if "source_chunks" in msg and msg["source_chunks"]:
            with st.expander("Sources / metadata"):
                for s in msg["source_chunks"]:
                    st.write(s)

st.markdown("---")
st.subheader("Indexed Documents")
if st.session_state.docs_indexed:
    for d in st.session_state.docs_indexed:
        st.write(f"- {d['name']} — chunks: {d['chunks']}")
else:
    st.write("No documents indexed yet.")

st.markdown("### Notes & Tips")
st.write("""
- For quick Colab testing you can run this app in a Colab cell using `streamlit run app.py` or run the main functions directly in Python.
- For production or Hugging Face Spaces, set up Qdrant (or Qdrant Cloud) and set `QDRANT_URL` env var to point to it.
- To improve results: increase chunk overlap, use a stronger embedding or generator model, or add chain-of-thought/history summarization.
""")
