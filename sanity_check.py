from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

text = "Qdrant is the vector database used in this project for metadata-filterable retrieval."
vector = model.encode(text).tolist()

client.upsert(
    collection_name="PSH-01_Documents",
    points=[PointStruct(
        id=1,
        vector = vector,
        payload={
                "filename": "sanity_check.txt",
                "page": 1,
                "chunk_text": text,
                "ingested_at": "2026-06-22"
            }
        )
    ]
)

query = "Which vector database supports metadata filtering?"
query_vector = model.encode(query).tolist()

results = client.query_points(
    collection_name="PSH-01_Documents",
    query=query_vector,
    limit=3
).points

for i in results:
    print(f"score={i.score:.3f}  payload={i.payload}")