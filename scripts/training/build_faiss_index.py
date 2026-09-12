"""Build FAISS vector index over historical AppleSupport conversations from train.jsonl.

Strict Zero-Leakage Policy:
Only dialogues from data/splits/train.jsonl are indexed.
Golden set (data/golden/golden_set.jsonl) and test split (data/splits/test.jsonl)
are NEVER indexed, preserving 100% leak-free evaluation purity.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = BASE_DIR / "data" / "splits" / "train.jsonl"
OUTPUT_DIR = BASE_DIR / "models" / "embedding_model"
INDEX_PATH = OUTPUT_DIR / "faiss.index"
METADATA_PATH = OUTPUT_DIR / "case_metadata.json"
SUMMARY_PATH = OUTPUT_DIR / "index_summary.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def load_train_cases(train_path: Path) -> List[Dict[str, Any]]:
    """Load and format historical support cases from train.jsonl."""
    logger.info("Loading training conversations from %s", train_path)
    cases = []
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            customer_text = (item.get("first_inquiry") or "").strip()
            brand_response = (item.get("final_brand_response") or "").strip()
            if not customer_text or not brand_response:
                continue

            cases.append(
                {
                    "case_id": item["conversation_id"],
                    "ticket_id": item.get("ticket_id", f"TICK-{item['conversation_id']}"),
                    "customer_text": customer_text,
                    "brand_response": brand_response,
                    "intent": item.get("intent_code", "GENERAL_INQUIRY"),
                    "resolution_status": "RESOLVED",
                    "has_resolution": item.get("has_resolution", False),
                    "has_dm": item.get("has_dm", False),
                    "has_kb_link": item.get("has_kb_link", False),
                    "turn_count": item.get("turn_count", 2),
                    "created_at": item.get("created_at", ""),
                }
            )

    logger.info("Loaded %d valid historical support cases for indexing.", len(cases))
    return cases


def build_and_save_index(
    cases: List[Dict[str, Any]],
    model_name: str = MODEL_NAME,
    output_dir: Path = OUTPUT_DIR,
) -> None:
    """Encode cases with MiniLM and construct an L2-normalized Inner Product FAISS index."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)

    texts = [case["customer_text"] for case in cases]
    logger.info(
        "Encoding %d customer inquiries into dense %d-d vectors...",
        len(texts),
        EMBEDDING_DIM,
    )
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings.astype(np.float32)

    logger.info("Normalizing embeddings to unit L2 norm for exact cosine similarity...")
    faiss.normalize_L2(embeddings)

    logger.info("Constructing FAISS IndexFlatIP (Inner Product = Cosine on unit vectors)...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    logger.info("Indexed %d vectors in FAISS.", index.ntotal)

    # Save index
    faiss.write_index(index, str(INDEX_PATH))
    logger.info("FAISS index saved to %s", INDEX_PATH)

    # Save metadata mapping vector index position -> case metadata
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    logger.info("Case metadata (%d entries) saved to %s", len(cases), METADATA_PATH)

    # Compute and save summary
    intent_counts: Dict[str, int] = {}
    for case in cases:
        intent = case["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    summary = {
        "model_name": model_name,
        "embedding_dim": EMBEDDING_DIM,
        "total_cases_indexed": len(cases),
        "source_split": str(TRAIN_PATH.relative_to(BASE_DIR)),
        "index_type": "IndexFlatIP",
        "metric": "Cosine Similarity (via L2-normalized Inner Product)",
        "intent_distribution": intent_counts,
        "sample_case_id_range": ([cases[0]["case_id"], cases[-1]["case_id"]] if cases else []),
    }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("Index summary saved to %s", SUMMARY_PATH)


def main() -> None:
    if not TRAIN_PATH.exists():
        logger.error(
            "Training data not found at %s. Run scripts/data/split_dataset.py first.",
            TRAIN_PATH,
        )
        sys.exit(1)

    cases = load_train_cases(TRAIN_PATH)
    if not cases:
        logger.error("No valid historical cases extracted.")
        sys.exit(1)

    build_and_save_index(cases)
    logger.info("Phase 7: FAISS vector indexing complete.")


if __name__ == "__main__":
    main()
