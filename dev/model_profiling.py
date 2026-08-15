"""
PSH-01 v1.1 — Pre-Flight Profiling Script (Milestone 1)

Measures intfloat/multilingual-e5-base on the target hardware
(Ryzen 7 5700H / Vega iGPU / 16GB single-channel RAM) before committing
to the migration, per PRD §2. 
"""

import gc
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import requests
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer

# ---- Config (matches PRD-locked decisions — do not change without updating PRD) ----
MODEL_NAME = "intfloat/multilingual-e5-base" # intfloat/multilingual-e5-base // BAAI/bge-small-en-v1.5
BATCH_SIZE = 16
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
USE_PREFIXES = True

SAMPLE_PDF = Path("./data/profiling/ml_journal.pdf")
TERM_PAIRS_FILE = Path("./data/profiling/term_pairs.json")

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_PS_URL = "http://localhost:11434/api/ps"
OLLAMA_MODEL = "qwen2.5:3b-instruct-q4_K_M"

RAM_CEILING_GB = 16.0
IDLE_RAM_TARGET_MB = 1500.0  # PRD acceptable boundary: < 1.5 GB

BASELINE_LATENCY_MS_PER_CHUNK = 129.53


def get_process_ram_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 ** 2)


def check_ollama_models() -> dict | None:
    try:
        resp = requests.get(OLLAMA_PS_URL, timeout=2)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def measure_idle_ram(model_name: str):
    gc.collect()
    ram_before = get_process_ram_mb()
    model = SentenceTransformer(model_name)
    ram_after = get_process_ram_mb()
    footprint = ram_after - ram_before
    return {
        "ram_before_load_mb": round(ram_before, 1),
        "ram_after_load_mb": round(ram_after, 1),
        "idle_footprint_mb": round(footprint, 1),
        "within_1_5gb_target": footprint < IDLE_RAM_TARGET_MB,
    }, model


def load_and_chunk_sample_pdf(pdf_path: Path) -> list[str]:
    docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    return [node.get_content() for node in nodes]


def measure_embedding_latency(model, chunks: list[str]) -> dict:
    prefixed = [f"passage: {c}" for c in chunks] if USE_PREFIXES else list(chunks)

    cold_n = min(BATCH_SIZE, len(prefixed))
    cold_start = time.perf_counter()
    _ = model.encode(prefixed[:cold_n], batch_size=BATCH_SIZE)
    cold_elapsed = time.perf_counter() - cold_start

    remaining = prefixed[cold_n:]
    steady_start = time.perf_counter()
    if remaining:
        _ = model.encode(remaining, batch_size=BATCH_SIZE)
    steady_elapsed = time.perf_counter() - steady_start

    steady_ms_per_chunk = (
        round((steady_elapsed / len(remaining)) * 1000, 2) if remaining else None
    )

    result = {
        "total_chunks": len(prefixed),
        "cold_start_batch_size": cold_n,
        "cold_start_total_s": round(cold_elapsed, 3),
        "cold_start_ms_per_chunk": round((cold_elapsed / cold_n) * 1000, 2) if cold_n else None,
        "steady_state_chunks": len(remaining),
        "steady_state_total_s": round(steady_elapsed, 3),
        "steady_state_ms_per_chunk": steady_ms_per_chunk,
    }

    if BASELINE_LATENCY_MS_PER_CHUNK and steady_ms_per_chunk:
        ratio = steady_ms_per_chunk / BASELINE_LATENCY_MS_PER_CHUNK
        result["ratio_vs_bge_small_baseline"] = round(ratio, 2)
        result["within_3x_boundary"] = ratio < 3.0

    return result


def _ollama_background_chat(marker: dict):
    """Fires a real generation request to simulate concurrent chat usage."""
    try:
        resp = requests.post(
            OLLAMA_CHAT_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": "Explain gradient descent in 3 sentences."}],
                "stream": False,
            },
            timeout=60,
        )
        marker["ollama_status"] = resp.status_code
    except requests.RequestException as e:
        marker["ollama_status"] = f"error: {e}"


def measure_concurrency_with_ollama(model, chunks: list[str]) -> dict:
    models_before = check_ollama_models()
    names_before = {m.get("name") for m in (models_before or {}).get("models", [])}

    peak_used_gb = {"value": 0.0}
    stop_flag = threading.Event()

    def _poll_ram():
        while not stop_flag.is_set():
            vm = psutil.virtual_memory()
            used_gb = (vm.total - vm.available) / (1024 ** 3)
            peak_used_gb["value"] = max(peak_used_gb["value"], used_gb)
            time.sleep(0.2)

    poll_thread = threading.Thread(target=_poll_ram, daemon=True)
    poll_thread.start()

    ollama_marker = {}
    ollama_thread = threading.Thread(target=_ollama_background_chat, args=(ollama_marker,), daemon=True)
    ollama_thread.start()

    prefixed = [f"passage: {c}" for c in chunks[:64]]
    start = time.perf_counter()
    _ = model.encode(prefixed, batch_size=BATCH_SIZE)
    elapsed = time.perf_counter() - start

    ollama_thread.join(timeout=60)
    stop_flag.set()
    poll_thread.join(timeout=1)

    models_after = check_ollama_models()
    names_after = {m.get("name") for m in (models_after or {}).get("models", [])}
    evicted = bool(names_before - names_after)

    return {
        "ollama_reachable_before": models_before is not None,
        "ollama_reachable_after": models_after is not None,
        "ollama_models_before": sorted(names_before),
        "ollama_models_after": sorted(names_after),
        "possible_eviction_detected": evicted,
        "ollama_chat_status": ollama_marker.get("ollama_status"),
        "embedding_batch_elapsed_s": round(elapsed, 3),
        "peak_system_ram_used_gb": round(peak_used_gb["value"], 2),
        "system_total_ram_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "under_16gb_ceiling": peak_used_gb["value"] < RAM_CEILING_GB,
    }


def _cos_sim(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def measure_cross_lingual_resolution(model, term_pairs_path: Path) -> dict:
    if not term_pairs_path.exists():
        return {"error": f"Term pairs file not found: {term_pairs_path}"}

    with open(term_pairs_path, "r", encoding="utf-8") as f:
        pairs = json.load(f)

    all_terms = []
    for pair in pairs:
        all_terms.extend([pair["id"], pair["en"]])

    embeddings = model.encode([f"query: {t}" if USE_PREFIXES else t for t in all_terms])
    term_to_vec = dict(zip(all_terms, embeddings))

    results = []
    for pair in pairs:
        id_vec = term_to_vec[pair["id"]]
        en_vec = term_to_vec[pair["en"]]
        pair_sim = _cos_sim(id_vec, en_vec)

        distractor_sims = [
            _cos_sim(id_vec, term_to_vec[other["en"]])
            for other in pairs
            if other["en"] != pair["en"]
        ]
        max_distractor_sim = max(distractor_sims) if distractor_sims else 0.0

        results.append({
            "id_term": pair["id"],
            "en_term": pair["en"],
            "cosine_similarity": round(pair_sim, 4),
            "max_distractor_similarity": round(max_distractor_sim, 4),
            "correctly_closest": pair_sim > max_distractor_sim,
        })

    passed = sum(1 for r in results if r["correctly_closest"])
    return {
        "total_pairs": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 3) if results else 0.0,
        "detail": results,
    }


def run_profiling():
    print(f"=== PSH-01 v1.1 Pre-Flight Profiling: {MODEL_NAME} ===\n")

    if not SAMPLE_PDF.exists():
        raise FileNotFoundError(
            f"Sample PDF not found at {SAMPLE_PDF}. Drop your 50-page "
            f"mixed-language sample there before running this script."
        )

    print("[1/4] Measuring idle RAM footprint (model load)...")
    idle_ram, model = measure_idle_ram(MODEL_NAME)
    print(f"      idle footprint: {idle_ram['idle_footprint_mb']} MB "
          f"({'OK' if idle_ram['within_1_5gb_target'] else 'OVER TARGET'})")

    print("\nChunking sample PDF...")
    chunks = load_and_chunk_sample_pdf(SAMPLE_PDF)
    print(f"      {len(chunks)} chunk(s) from sample PDF")

    print("\n[2/4] Measuring embedding latency...")
    latency = measure_embedding_latency(model, chunks)
    print(f"      steady-state: {latency['steady_state_ms_per_chunk']} ms/chunk")

    print("\n[3/4] Measuring concurrency behavior with Ollama "
          "(sending a real chat request in the background)...")
    concurrency = measure_concurrency_with_ollama(model, chunks)
    print(f"      Ollama reachable: {concurrency['ollama_reachable_after']}, "
          f"peak system RAM: {concurrency['peak_system_ram_used_gb']} GB "
          f"({'under' if concurrency['under_16gb_ceiling'] else 'OVER'} {RAM_CEILING_GB} GB ceiling), "
          f"eviction detected: {concurrency['possible_eviction_detected']}")

    print("\n[4/4] Measuring cross-lingual topic resolution...")
    cross_lingual = measure_cross_lingual_resolution(model, TERM_PAIRS_FILE)
    if "error" in cross_lingual:
        print(f"      SKIPPED: {cross_lingual['error']}")
    else:
        print(f"      {cross_lingual['passed']}/{cross_lingual['total_pairs']} pairs correctly resolved "
              f"({cross_lingual['pass_rate'] * 100:.1f}%)")

    report = {
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hardware": "Ryzen 7 5700H / Vega iGPU / 16GB single-channel RAM",
        "idle_ram": idle_ram,
        "embedding_latency": latency,
        "concurrency_with_ollama": concurrency,
        "cross_lingual_resolution": cross_lingual,
    }

    out_path = Path("profiling_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFull report written to {out_path} — paste the relevant numbers into NOTES.md")
    return report


if __name__ == "__main__":
    run_profiling()