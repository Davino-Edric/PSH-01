# PSH-01 Decision Log

## Architecture
- Vector DB: Qdrant via Docker. Chosen for production-realistic metadata
  filtering (matters later for course/date/source tagging in the larger system).
- LLM: qwen2.5:3b-instruct-q4_K_M via Ollama. Benchmarked against 7B and
  Llama3.2:3B — best prompt eval speed (61.72 tok/s) on Ryzen 7 5700H / Vega iGPU / 16GB RAM.
- Embedding: bge-small-en-v1.5 via SentenceTransformers. Tied retrieval quality
  with nomic-embed-text, ~2x faster batch embedding — HF route chosen partly to
  build hands-on HF depth (explicit personal goal), not just for the speed win.
- Chunking: 512 tokens / 50 overlap. Tested 256 — didn't resolve topic bleed,
  so reverted; bleed is inherent to content similarity, not a chunking artifact.
- Point IDs: uuid5(filename + page + chunk_index). Deterministic — re-ingesting
  the same file overwrites cleanly instead of duplicating points.

## Known limitations (accepted, not blockers)
- bge-small treats ID/EN spelling variants as unrelated (e.g. "Linier" vs "Linear")
  — inherent to using an English-trained embedding model on mixed-language docs.
- SLR vs MLR topic bleed at chunk boundaries — content similarity, not a bug.
- is_front_matter() only catches Roman-numeral page labels — won't catch e.g.
  unlabeled cover pages.

## Debugging history
- Qdrant API: query_points() replaces deprecated search(); chained .points to
  avoid Pydantic iteration issues.
- Duplicate points appeared after changing the ID scheme mid-project — resolved
  by the uuid5(filename+page+chunk_index) scheme above.
- TOC pages were contaminating retrieval results before is_front_matter() filter.
- Corrupted/image-only PDFs produce zero extractable text → zero nodes → Qdrant
  rejects the empty upsert with a raw 400 error. Fixed by raising a clear
  ValueError inside ingest_pdf() before the upsert call.

## Streamlit integration notes
- ingest_pdf() and ask() needed no refactoring — both already operated on
  single files/questions internally; only app.py needed to be built around them.
- uploaded_file from st.file_uploader is BytesIO, not a filepath — must write
  to disk before ingest_pdf() (which expects Path) can read it.
- accept_multiple_files=True changes uploaded_file into a list; wrapped each
  file's ingest_pdf() call in its own try/except so one bad file doesn't halt
  the batch.