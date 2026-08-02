from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

client = QdrantClient(host="localhost", port=6333)

results, _ = client.scroll(
    collection_name="PSH-01_Documents",
    scroll_filter=Filter(
        must=[FieldCondition(key="filename", match=MatchValue(value="Bisindo CNN.pdf"))]
    ),
    limit=50,
    with_payload=True,
    with_vectors=False
)

courses_found = {p.payload.get("course") for p in results}
print(courses_found)
print("Collision test complete, moving to Full tag test. . .")

seen = set()
results, _ = client.scroll(
    collection_name="PSH-01_Documents",
    limit=250,
    with_payload=True,
    with_vectors=False
)
for p in results:
    seen.add((p.payload.get("filename"), p.payload.get("course")))

for filename, course in sorted(seen):
    print(f"{filename:60} -> {course}")
    