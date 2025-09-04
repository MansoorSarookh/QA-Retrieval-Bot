# utils.py
import os
import re
from io import BytesIO
from typing import List, Tuple, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import numpy as np
from pypdf import PdfReader
import docx
from tqdm.auto import tqdm

# Vector store compatibility imports
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import faiss
import uuid
import pickle

# -------------------------
# Document parsing
# -------------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return "\n".join(texts)

def extract_text_from_docx(file_bytes: bytes) -> str:
    f = BytesIO(file_bytes)
    doc = docx.Document(f)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)

def extract_text(filename: str, bytestr: bytes) -> str:
    ext = filename.lower().split('.')[-1]
    if ext == "pdf":
        return extract_text_from_pdf(bytestr)
    elif ext in ("docx", "doc"):
        return extract_text_from_docx(bytestr)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# -------------------------
# Chunking (simple char-based chunks with overlap)
# -------------------------
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    if not text:
        return []
    text = re.sub(r'\n\s*\n', '\n', text)  # collapse multiple blank lines
    start = 0
    chunks = []
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# -------------------------
# Embeddings (SentenceTransformer)
# -------------------------
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embed_model = None

def load_embedding_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model

def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings

# -------------------------
# Generator model (RAG prompt -> generate answer)
# -------------------------
# Use a lightweight seq2seq model that runs reasonably on CPU for small questions.
GEN_MODEL_NAME = os.environ.get("GEN_MODEL", "google/flan-t5-small")
_gen_pipeline = None

def load_generator():
    global _gen_pipeline
    if _gen_pipeline is None:
        # Use Seq2SeqPipeline
        _gen_pipeline = pipeline("text2text-generation", model=GEN_MODEL_NAME, tokenizer=GEN_MODEL_NAME, device=-1)
    return _gen_pipeline

def generate_answer(prompt: str, max_length: int = 256) -> str:
    gen = load_generator()
    out = gen(prompt, max_length=max_length, do_sample=False)
    return out[0]["generated_text"]

# -------------------------
# Vector store wrapper: Qdrant (preferred) or FAISS (fallback)
# -------------------------
class VectorStore:
    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict], texts: List[str]):
        raise NotImplementedError()
    def query(self, embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, str, dict]]:
        """Return list of (id, score, text, metadata)"""
        raise NotImplementedError()
    def persist(self, path: str):
        pass

# Qdrant store
class QdrantStore(VectorStore):
    def __init__(self, collection_name="docs", host=None, port=None, prefer_grpc=False):
        # host expected like "http://localhost:6333" or host + port
        q_host = os.environ.get("QDRANT_URL") or host
        api_key = os.environ.get("QDRANT_API_KEY")
        if q_host:
            # if full url provided, QdrantClient accepts url param
            if q_host.startswith("http"):
                self.client = QdrantClient(url=q_host, api_key=api_key)
            else:
                # assume host & port separated
                self.client = QdrantClient(host=q_host, port=port or 6333, api_key=api_key)
        else:
            raise ValueError("Qdrant URL not provided for QdrantStore")
        self.collection_name = collection_name
        # ensure collection exists
        try:
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 for MiniLM; adjust if using different embed dim
            )
        except Exception:
            # maybe already exists; ignore
            pass

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict], texts: List[str]):
        points = []
        for i, uid in enumerate(ids):
            points.append({"id": uid, "vector": embeddings[i].tolist(), "payload": {"meta": metadatas[i], "text": texts[i]}})
        self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, embedding: np.ndarray, top_k: int = 5):
        hits = self.client.search(collection_name=self.collection_name, query_vector=embedding.tolist(), limit=top_k)
        results = []
        for h in hits:
            metadata = h.payload.get("meta", {})
            text = h.payload.get("text", "")
            results.append((str(h.id), float(h.score), text, metadata))
        return results

# FAISS fallback (in-memory)
class FAISSStore(VectorStore):
    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # inner product (we will normalize)
        self.texts = []
        self.metadatas = []
        self.ids = []

    def add(self, ids: List[str], embeddings: np.ndarray, metadatas: List[dict], texts: List[str]):
        # normalize embeddings for cosine via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms==0] = 1.0
        emb_norm = embeddings / norms
        self.index.add(emb_norm.astype('float32'))
        self.texts.extend(texts)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def query(self, embedding: np.ndarray, top_k: int = 5):
        emb = embedding.reshape(1, -1)
        norm = np.linalg.norm(emb)
        if norm == 0:
            norm = 1.0
        emb = emb / norm
        D, I = self.index.search(emb.astype('float32'), k=top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            results.append((self.ids[idx], float(score), self.texts[idx], self.metadatas[idx]))
        return results

# Utility to create appropriate store
def get_vector_store(prefer_qdrant=True, qdrant_collection="docs", embed_dim=384):
    qdrant_url = os.environ.get("QDRANT_URL")
    if prefer_qdrant and qdrant_url:
        try:
            return QdrantStore(collection_name=qdrant_collection)
        except Exception as e:
            print("Qdrant connection failed; falling back to FAISS. Error:", e)
    # fallback
    return FAISSStore(dim=embed_dim)

# -------------------------
# Building knowledge base: takes document text, chunks, embeds, and stores; returns ids
# -------------------------
def build_doc_store(text: str, store: VectorStore, chunk_size=1000, overlap=200, source_name="uploaded_doc"):
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return []
    embeddings = embed_texts(chunks)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]
    store.add(ids=ids, embeddings=embeddings, metadatas=metadatas, texts=chunks)
    return [{"id": _id, "text": t, "metadata": m} for _id, t, m in zip(ids, chunks, metadatas)]
