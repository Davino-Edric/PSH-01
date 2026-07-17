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

# v.1.1 Milestone logs

## Milestone 1: Benchmarking and profiling of bge-small-en-1.5 vs multilingual-e5-base

Context: Tested on a 51 page sample pdf about machine learning terms that are mixed language (AI Generated, still yet to find a proper messy mixed-language journal / paper) seeded from term_pairs.json (both documents are on data/profiling)

- Steady state latency is 1.82-1.85x of bge-small-en's steady state latency
- Idle RAM sits at 337MiB, way smaller than the expected boundary of 1.5GiB
- Peak system RAM usage reaches 99% for all runs of e5-base, reaching 15.34-15.4 out of 15.41 GiB of RAM
- A static term pair test based on term_pairs.json was used as a nother form of accuracy check 
  for the two embedding models. bge-small-en-1.5 got 6/12 term pairs correct, whilst e5-base got 11/12 pairs correct,
  of which the misses on e5-base was a near-miss with only 0.0081 margin

Verdict: Pass with documented caveats, proceeding to Milestone 2:

## Milestone 2: Changing the Qdrant collection to use 768 dims size (previously 384, this is related to bge-small and e5-base dimension size)

Context: To accommodate for e5-base 768 dimension size, the collection must be dropped and recreated with 768 dimension size change.

Verdict: Pass, dropped previous collection along with any points upserted into it.

## Milestone 3: Changing the pipeline scripts to use e5-base

Context: To use e5-base, ingest.py and query.py must be given a prefix of passage and query respectively
and also given a batch size (16 in this case)

- query.py logged changes: Changed embedding models, added prefix to query in retrieve()
- changed the ingestion to a per node-loop that collect the node then batch encode it
- prefix only used for encoding only, no prefix ever gets into the LLM / Citations 
- Result: 13/15, and — worth being specific here rather than just "sometimes bleeds" 
both misses ranked #2 by a narrow margin (~0.01–0.02), not buried
- Both misses are the same failure shape: the question didn't specify the distinguishing detail between two topically-adjacent chunks
- e5-base is tested on only english when it's supposed to be cross language indo-eng, however it's ability on a 50 page profiling proves it works okay with cross language

