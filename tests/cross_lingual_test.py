# This script is supposed to be ran on root folder, not inside the tests, folder. 
# Move to Root Folder to run the test.

from query import retrieve

TEST_PAIRS = [
    {"query": "Hutan Acak",              "expect_topic": "Random Forest"},
    {"query": "Klasterisasi",            "expect_topic": "Clustering"},
    
    {"query": "Gradien Turun",           "expect_topic": "Gradient Descent"},
    {"query": "Dimensionality Reduction","expect_topic": "Reduksi Dimensi"},
]

TOP_K = 3

for pair in TEST_PAIRS:
    print(f"\n=== Query: '{pair['query']}'  (expecting content about: {pair['expect_topic']}) ===")
    results = retrieve(pair["query"], top_k=TOP_K)
    
    if not results:
        print("  No results returned at all — treat as a FAIL.")
        continue
    
    for rank, chunk in enumerate(results, start=1):
        filename = chunk.payload.get("filename")
        page = chunk.payload.get("page")
        snippet = chunk.payload.get("chunk_text", "")[:150]
        print(f"  #{rank}  score={chunk.score:.3f}  file={filename}  page={page}")
        print(f"       snippet: {snippet!r}")