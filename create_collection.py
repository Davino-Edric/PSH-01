from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

COLLECTION_NAME = "PSH-01_Documents"

client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=384, #small-bge-en-v1.5 embedding size
        distance=Distance.COSINE
    )
)

print(f"Collection '{COLLECTION_NAME}' created.")
