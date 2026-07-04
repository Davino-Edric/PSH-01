# test_embeddings_batch.py
import time
import requests
from sentence_transformers import SentenceTransformer

# Simulate ~10 realistic PDF chunks (you'll have way more in practice,
# this is just enough to see batching behavior)
sample_chunks = [
    "Retrieval-Augmented Generation combines a vector database with an LLM to answer questions using external documents.",
    "Qdrant is a vector database that supports metadata filtering alongside similarity search.",
    "Quantization reduces model weight precision, e.g. from 16-bit to 4-bit, to lower RAM usage.",
    "Ollama provides a local REST API for running LLMs and embedding models without manual setup.",
    "Chunking splits long documents into smaller segments so embeddings capture localized meaning.",
    "Cosine similarity measures the angle between two vectors, commonly used in semantic search.",
    "SentenceTransformers is a HuggingFace library built for generating sentence-level embeddings.",
    "A vector database indexes embeddings for fast approximate nearest-neighbor search at scale.",
    "Streamlit allows rapid prototyping of Python web interfaces without frontend code.",
    "Flash attention reduces memory overhead during transformer attention computation.",
]

# --- Ollama: one HTTP call per text (no native batch endpoint) ---
start = time.time()
ollama_vecs = []
for text in sample_chunks:
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    ollama_vecs.append(resp.json()["embedding"])
ollama_batch_time = time.time() - start

print(f"[nomic-embed-text] {len(sample_chunks)} chunks  total={ollama_batch_time:.3f}s  avg={ollama_batch_time/len(sample_chunks):.3f}s/chunk")

# --- HuggingFace: native batch encoding ---
model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # cached after first run now

start = time.time()
hf_vecs = model.encode(sample_chunks, batch_size=10)  # encodes all 10 in one optimized call
hf_batch_time = time.time() - start

print(f"[bge-small-en-v1.5] {len(sample_chunks)} chunks  total={hf_batch_time:.3f}s  avg={hf_batch_time/len(sample_chunks):.3f}s/chunk")