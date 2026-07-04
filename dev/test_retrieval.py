# expanded_test_retrieval_quality.py
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Knowledge base: expanded chunks with deliberate near-duplicates/distractors ---
chunks = [
    "Qdrant is a vector database that supports metadata filtering alongside similarity search.",       # 0
    "Chroma is a lightweight vector database often used for quick prototyping.",                        # 1 (distractor)
    "Quantization reduces model weight precision, e.g. from 16-bit to 4-bit, to lower RAM usage.",       # 2
    "Pruning removes redundant weights from a neural network to reduce its size.",                       # 3 (distractor)
    "Ollama provides a local REST API for running LLMs and embedding models without manual setup.",      # 4
    "LM Studio is another tool for running local LLMs with a graphical interface.",                      # 5 (distractor)
    "Chunking splits long documents into smaller segments so embeddings capture localized meaning.",     # 6
    "Tokenization splits text into smaller units like words or subwords before model processing.",       # 7 (distractor)
    "Cosine similarity measures the angle between two vectors, commonly used in semantic search.",       # 8
    "Euclidean distance measures the straight-line distance between two points in vector space.",        # 9 (distractor)
    # === New chunks ===
    "RAG stands for Retrieval-Augmented Generation, combining vector search with LLM generation.",        # 10
    "FAISS is a library for efficient similarity search on large-scale vector collections.",              # 11 (distractor)
    "LoRA is a technique for efficient fine-tuning of large language models by updating low-rank matrices.", # 12
    "Full fine-tuning updates all parameters of a model, which is computationally expensive.",            # 13 (distractor)
    "Hybrid search combines keyword (BM25) and vector similarity for better retrieval accuracy.",         # 14
    "Dense retrieval uses embeddings to find semantically similar documents.",                           # 15
    "Sparse retrieval relies on exact term matching like TF-IDF or BM25.",                               # 16 (distractor)
    "Embedding models convert text into high-dimensional vectors that capture semantic meaning.",        # 17
    "HNSW is a graph-based indexing algorithm used in many vector databases for fast approximate nearest neighbor search.", # 18
    "IVF is an inverted file index that partitions vectors into clusters for faster search.",             # 19 (distractor)
    "Context window refers to the maximum number of tokens an LLM can process at once.",                  # 20
    "Prompt engineering involves crafting inputs to guide LLMs toward desired outputs.",                  # 21
    "Temperature controls the randomness of LLM outputs, with higher values increasing creativity.",     # 22
    "Top-p sampling, or nucleus sampling, selects tokens from the smallest set whose cumulative probability exceeds a threshold.", # 23
]

# --- 15 Test questions with KNOWN correct chunk index ---
test_cases = [
    {"question": "What database are we using for this project and why?", "correct_idx": 0},
    {"question": "How does quantization help run models on limited hardware?", "correct_idx": 2},
    {"question": "What tool gives us a local API for running LLMs?", "correct_idx": 4},
    {"question": "Why do we split PDFs into smaller pieces before embedding?", "correct_idx": 6},
    {"question": "What similarity metric is typically used in semantic search?", "correct_idx": 8},
    {"question": "What does RAG stand for and how does it work?", "correct_idx": 10},
    {"question": "Which library is known for efficient similarity search on massive vector datasets?", "correct_idx": 11},
    {"question": "What fine-tuning method updates only low-rank matrices for efficiency?", "correct_idx": 12},
    {"question": "How does hybrid search improve upon pure vector search?", "correct_idx": 14},
    {"question": "What is the main difference between dense and sparse retrieval?", "correct_idx": 15},
    {"question": "What do embedding models primarily do?", "correct_idx": 17},
    {"question": "Which indexing algorithm is graph-based and popular in vector DBs for ANN search?", "correct_idx": 18},
    {"question": "What is the context window in the context of LLMs?", "correct_idx": 20},
    {"question": "What technique involves designing effective prompts for large language models?", "correct_idx": 21},
    {"question": "How does temperature affect LLM generation?", "correct_idx": 22},
]

def ollama_embed(text):
    resp = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return resp.json()["embedding"]


def run_test(embed_fn, label):
    print(f"\n=== {label} ===")
    chunk_vecs = np.array([embed_fn(c) for c in chunks])

    correct = 0
    for case in test_cases:
        q_vec = np.array(embed_fn(case["question"])).reshape(1, -1)
        sims = cosine_similarity(q_vec, chunk_vecs)[0]
        ranked_idx = np.argsort(sims)[::-1]  # highest similarity first
        top1 = ranked_idx[0]

        is_correct = (top1 == case["correct_idx"])
        correct += is_correct

        status = "Correct" if is_correct else "Incorrect"
        print(f"{status} Q: {case['question']}")
        print(f"   Top match (idx={top1}, score={sims[top1]:.3f}): \"{chunks[top1][:60]}...\"")
        if not is_correct:
            correct_score = sims[case['correct_idx']]
            print(f"   Correct chunk was idx={case['correct_idx']} (score={correct_score:.3f}) — ranked #{list(ranked_idx).index(case['correct_idx'])+1}")

    print(f"\n{label} accuracy: {correct}/{len(test_cases)}")
    return correct


# Run both
ollama_correct = run_test(ollama_embed, "nomic-embed-text")

hf_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
hf_correct = run_test(lambda t: hf_model.encode(t), "bge-small-en-v1.5")