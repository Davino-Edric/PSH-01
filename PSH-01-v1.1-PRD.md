# PSH-01 v1.1 — Embedding Layer Migration PRD

**Status:** Locked, pending pre-flight profiling results
**Depends on:** PSH-01 v1.0 (complete)
**Blocks:** PSH-02 v1.0 (topic clustering requires 768-dim vectors)

---

## 1. Decision

Migrate the core embedding model from `bge-small-en-v1.5` (384-dim, SentenceTransformers) to `intfloat/multilingual-e5-base` (768-dim, 278M params, ~1.1GB footprint).

### Why

- **Native cross-lingual alignment.** `bge-small-en-v1.5` treats Indonesian/English spelling variants (e.g. "Regresi Linier" vs "Linear Regression") as unrelated tokens — a documented v1.0 limitation. `multilingual-e5-base` is designed to map these into the same vector neighborhood, closing this gap at the embedding layer instead of papering over it with LLM prompt tricks.
- **Higher-resolution clustering for PSH-02.** 768 dimensions give PSH-02's UMAP+HDBSCAN topic clustering more room to separate closely related sub-topics (e.g. Simple vs. Multivariate Linear Regression) than 384 dimensions reliably allows.
- **Alternatives considered and rejected:**
  - `LazarusNLP/all-indo-e5-small-v4` — zero-overhead transition, but narrower language coverage than needed long-term.
  - `BAAI/bge-m3` — larger context window, but heavier footprint without a corresponding accuracy win for this use case.

### Honest tradeoff

This is **not** a speed upgrade — it's a correctness upgrade traded against latency. `bge-small` won its v1.0 slot on a ~2x speed advantage over `nomic-embed-text`, not on being an undisputed best-in-class model. Moving to `e5-base` is expected to cost roughly a 2.5x latency penalty on batch embedding — effectively landing ingestion speed back around where `nomic-embed-text` would have put it originally. **This number is a hypothesis, not a measured fact, until the pre-flight profiling step runs.**

---

## 2. Pre-Flight Profiling Milestone (required before shipping)

Run a controlled profiling pass on a 50-page mixed-language sample PDF, on the target hardware (Ryzen 7 5700H, Vega iGPU, 16GB RAM), measuring `multilingual-e5-base` exclusively (not `e5-small` — this was a copy-paste artifact in an earlier draft and is explicitly ruled out).

| Metric | Baseline (`bge-small-en-v1.5`) | Target (`multilingual-e5-base`) | Acceptable boundary |
|---|---|---|---|
| Embedding generation latency/chunk | Already measured (v1.0) | TBD via profiling | < 3.0x baseline |
| Idle RAM footprint (Python process) | Negligible | TBD via profiling | < 1.5 GB max |
| Concurrency profile (w/ Ollama loaded) | N/A | TBD via profiling | No OOM, no forced model eviction |
| Cross-lingual topic resolution | Zero (fails on variants, documented v1.0 limitation) | TBD via labeled test pairs | Matches cross-lingual term pairs to same neighborhood |

**Sanity-check framing:** the goal is not to prove `e5-base` matches `bge-small`'s speed — that's already conceded as false. The goal is to confirm batch_size=16 doesn't spike RAM past the 16GB ceiling while Ollama is resident.

---

## 3. Refactoring Checklist

### Ingestion pipeline (`ingest.py`)
- [ ] Update `SentenceTransformer` model init to `intfloat/multilingual-e5-base`
- [ ] Prepend `"passage: "` to every chunk's text before encoding (required by E5's task-prefix convention)
- [ ] Constrain `batch_size` to 16 to preserve CPU responsiveness for interactive tasks during background ingestion

### Vector DB config (`create_collection.py`)
- [ ] Update Qdrant collection vector config from 384 → 768 dimensions
- [ ] Implement a hard-reset migration script that drops the existing 384-dim `PSH-01_Documents` collection

### Query engine (`query.py`)
- [ ] Prepend `"query: "` to the user's raw question before encoding (E5 task-prefix convention)

### Course tagging (new, supports PSH-02)
- [ ] File watcher infers `course_tag` from folder structure: `data/pdfs/[Course_Name]/document.pdf`
- [ ] **Defensive fallback:** files placed directly in `data/pdfs/` root (no course subfolder) receive `course: "uncategorized"` automatically — no forced manual migration for legacy-style placement
- [ ] Legacy Qdrant data note: since the 384→768 migration requires a full collection drop, any previously ingested PDFs are simply re-ingested from scratch under v1.1 — no separate backfill step needed

### SQLite ingestion queue (new, supports PSH-02)
- [ ] Create a lightweight SQLite queue table (e.g. `ingestion_queue`) if it doesn't already exist
- [ ] On successful completion of ingest.py's upsert to Qdrant, write a row: `filename`, `status: "COMPLETED"`, timestamp

---

## 4. Known Limitations Carried Forward / Resolved

| v1.0 limitation | v1.1 status |
|---|---|
| ID/EN spelling variants retrieve unrelated chunks | Expected resolved by E5's cross-lingual alignment — **verify empirically**, don't assume |
| SLR/MLR semantic bleed | Expected improved by 768-dim resolution, not eliminated — inherent content similarity remains a factor |
| Cold-start latency on first query | Unchanged, not in scope for this migration |
| `is_front_matter()` only catches Roman-numeral pages | Unchanged, not in scope for this migration |

---

## 5. Definition of Done

1. Pre-flight profiling completed with real numbers logged in `NOTES.md` (not estimates)
2. Qdrant collection successfully migrated to 768-dim, old collection cleanly dropped
3. `ingest.py` and `query.py` updated with task-prefix conventions, re-tested against `test_retrieval_quality.py`
4. Course-tag folder convention live, defensive fallback verified against a root-level test file
5. Cross-lingual retrieval spot-checked against known ID/EN term pairs — documented pass/fail, not assumed
