import uuid
from datetime import datetime, timezone
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

PDF_DIR = Path("data/pdfs")
COLLECTION = "PSH-01_Documents"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
EMBED_MODEL = "intfloat/multilingual-e5-base" # Changed from Beijing Academy's bge-small-en to e5-base
BATCH_SIZE = 16

client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer(EMBED_MODEL)
splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

# Making a unique point ID based on filename, page number, and chunk index (So if re-ingested, it will overwrite the previous point)
def make_point_id(filename: str, page: int, chunk_index: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{filename}_{page}_{chunk_index}"))

def is_front_matter(page_label: str) -> bool:
    roman = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}
    return str(page_label).lower() in roman

# In ingest_pdf(), after loading docs:

def ingest_pdf(pdf_path: Path):
    docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    print(f"  Loaded {len(docs)} page(s) before filtering")

    # Filter with visibility
    filtered_docs = []
    for d in docs:
        label = d.metadata.get("page_label", "")
        if is_front_matter(label):
            print(f"  Skipping page: {label}")  # shows exactly what got dropped
        else:
            filtered_docs.append(d)

    docs = filtered_docs
    print(f"  Kept {len(docs)} page(s) after filtering")
    
    nodes = splitter.get_nodes_from_documents(docs)
    print(f"split into {len(nodes)} chunk(s)")
    
    ingested_at = datetime.now(timezone.utc).isoformat()
    
    texts = [node.get_content() for node in nodes]
    prefixed_texts = [f'passage: {t}' for t in texts]
    all_vectors = model.encode(prefixed_texts, batch_size=BATCH_SIZE).tolist()
    
    points = []
    
    for i, (node, text,vectors) in enumerate(zip(nodes,texts,all_vectors)):
        page = node.metadata.get("page_label", "unknown")  # Assuming the metadata has a page label; adjust as necessary
        
        points.append(PointStruct(
            id=make_point_id(pdf_path.name,page,i),
            vector=vectors,
            payload={
                "filename": pdf_path.name,
                "page":page,
                "chunk_index": i,
                "chunk_text": text,
                "ingested_at": ingested_at
            }
        ))
        
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"upserted {len(points)} point(s) to the collection")
    
def ingest_all():
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    
    if not pdfs:
        print(f"No PDFs found in {PDF_DIR}:")
        return
    
    print(f"Found {len(pdfs)} PDF(s) to ingest...")
    for pdf_path in pdfs:
        ingest_pdf(pdf_path)

    # Final count — confirms Qdrant actually received and stored everything
    info = client.get_collection(COLLECTION)
    print(f"\nDone. Collection '{COLLECTION}' now holds {info.points_count} point(s).")

if __name__ == "__main__":
    ingest_all()