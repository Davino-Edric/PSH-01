from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import requests
import json

COLLECTION = "PSH-01_Documents"  
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"
TOP_K = 5

client = QdrantClient(host="localhost", port=6333)
model  = SentenceTransformer("BAAI/bge-small-en-v1.5")

def retrieve(question: str, top_k: int = TOP_K):
    query_vector = model.encode(question).tolist()

    return client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=top_k
    ).points
    

def build_prompt(question: str, chunks: list) -> list:
    context_blocks = []
    for i, chunk in enumerate(chunks):
        page     = chunk.payload["page"]
        filename = chunk.payload["filename"]
        text     = chunk.payload["chunk_text"]
        context_blocks.append(
            f"[Source {i+1} | {filename} | page {page}]:\n{text}"
        )

    context_str = "\n\n".join(context_blocks)

    return [
        {
            "role": "system",
            "content": (
                "You are a helpful study assistant for a Data Science student. "
                "Answer the user's question using ONLY the context provided below. "
                "Always cite your sources by referencing the page number from the context. "
                "If the answer cannot be found in the provided context, say exactly: "
                "'I cannot find this in the provided material.' "
                "Do not use prior knowledge — only the context."
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context_str}\n\n"
                f"Question: {question}"
            )
        }
    ]


def ask(question: str):
    # Step 1: Retrieve relevant chunks
    chunks = retrieve(question)
    if not chunks:
        print("No relevant chunks found in the collection.")
        return

    # Step 2: Build the prompt
    messages = build_prompt(question, chunks)

    # Step 3: Send to Ollama
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False  # get full response at once, not token by token
    })
    response.raise_for_status()

    answer = response.json()["message"]["content"]

    # Step 4: Print answer + sources used
    sources = [
        {
        "filename": chunk.payload["filename"],
        "page": chunk.payload["page"],
        "score": round(chunk.score, 3)
        }
        for chunk in chunks
    ]
    
    return {
        "answer": answer,
        "sources": sources
    }
    
def ask_stream(question: str, chunks=None, top_k: int = TOP_K):
    if chunks is None:
        chunks = retrieve(question, top_k=top_k)  # fallback if called without pre-fetched chunks

    messages = build_prompt(question, chunks)

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True
    }, stream=True)
    response.raise_for_status()

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            yield chunk["message"]["content"]

if __name__ == "__main__":
    ask("Apa itu Random Forest dan bagaimana cara kerjanya?")