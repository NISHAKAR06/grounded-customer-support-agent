#!/usr/bin/env python3
"""Deterministic Headline Benchmark Reproduction Script.

Demonstrates, measures, and documents whether the Grounded Customer Support Agent's
headline evaluation results can be reproduced in under 15 minutes on a clean environment.

Stages:
  [1/5] Loading evaluation data & verifying required artifacts
  [2/5] Intent classification baseline benchmark (Held-out Test Split: 484 samples)
  [3/5] Dense FAISS vector retrieval benchmark (Golden Set: 200 samples)
  [4/5] Grounded generation 6-barrier safety validation benchmark (Golden Set: 200 samples)
  [5/5] Automated test suite execution (pytest)

Usage:
  python scripts/reproduce_headline.py
"""

import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Safe console output encoding for Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure repository root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Required file paths
TEST_SPLIT_PATH = BASE_DIR / "data" / "splits" / "test.jsonl"
GOLDEN_SET_PATH = BASE_DIR / "data" / "golden" / "golden_set.jsonl"
VECTORIZER_PATH = BASE_DIR / "models" / "intent_classifier" / "tfidf_vectorizer.joblib"
LOGREG_MODEL_PATH = BASE_DIR / "models" / "intent_classifier" / "tfidf_logreg_model.joblib"
FAISS_INDEX_PATH = BASE_DIR / "models" / "embedding_model" / "faiss.index"
CASE_METADATA_PATH = BASE_DIR / "models" / "embedding_model" / "case_metadata.json"

REQUIRED_PATHS = [
    TEST_SPLIT_PATH,
    GOLDEN_SET_PATH,
    VECTORIZER_PATH,
    LOGREG_MODEL_PATH,
    FAISS_INDEX_PATH,
    CASE_METADATA_PATH,
]


def check_prerequisites() -> None:
    """Verify that all required data and model artifacts exist before benchmarking."""
    missing = [str(p.relative_to(BASE_DIR)) for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        print("[ERROR] Missing required artifacts for reproduction:")
        for m in missing:
            print(f"  - {m}")
        print("\nPlease ensure precomputed models and dataset splits are present.")
        sys.exit(1)


def get_cpu_info() -> str:
    """Retrieve readable CPU description across platforms."""
    cpu_name = platform.processor() or platform.machine() or "Unknown CPU"
    if sys.platform == "win32":
        brand = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if brand:
            cpu_name = brand.split(",")[0].strip()
    return cpu_name


def stage_1_load_data() -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """Load test split and golden set evaluation datasets."""
    test_texts: List[str] = []
    test_labels: List[str] = []
    with open(TEST_SPLIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            inquiry = (item.get("first_inquiry") or item.get("customer_text") or "").strip()
            intent = (item.get("intent_code") or "").strip()
            if inquiry and intent:
                test_texts.append(inquiry)
                test_labels.append(intent)

    golden_cases: List[Dict[str, Any]] = []
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            golden_cases.append(json.loads(line))

    return test_texts, test_labels, golden_cases


def stage_2_intent_benchmark(test_texts: List[str], test_labels: List[str]) -> Tuple[float, float]:
    """Evaluate TF-IDF + Logistic Regression baseline on held-out test split."""
    import joblib
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    vectorizer = joblib.load(VECTORIZER_PATH)
    classifier = joblib.load(LOGLOG := LOGREG_MODEL_PATH)  # noqa: F841

    x_test = vectorizer.transform(test_texts)
    predictions = classifier.predict(x_test)

    accuracy = float(accuracy_score(test_labels, predictions))
    _, _, macro_f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average="macro", zero_division=0
    )
    return accuracy, float(macro_f1)


def stage_3_retrieval_benchmark(
    golden_cases: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Evaluate FAISS vector retrieval Recall@3 on the Golden Set."""
    from app.services.retrieval.retriever import Retriever
    from scripts.evaluation.evaluate_retrieval import (
        evaluate_retrieval_on_dataset,
        load_queries_from_jsonl,
    )

    retriever = Retriever()
    golden_queries = load_queries_from_jsonl(GOLDEN_SET_PATH)

    uncond_metrics = evaluate_retrieval_on_dataset(
        retriever,
        golden_queries,
        "Golden Set",
        top_k_max=5,
        use_intent_filter=False,
    )
    cond_metrics = evaluate_retrieval_on_dataset(
        retriever,
        golden_queries,
        "Golden Set",
        top_k_max=5,
        use_intent_filter=True,
    )

    uncond_recall3 = float(uncond_metrics["recall_at_3"])
    cond_recall3 = float(cond_metrics["recall_at_3"])
    return uncond_recall3, cond_recall3


def stage_4_safety_benchmark(
    golden_cases: List[Dict[str, Any]],
) -> Tuple[float, int, int]:
    """Evaluate 6-barrier deterministic validation pass rate across Golden Set."""
    from app.services.agent.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    passed_count = 0
    total_count = len(golden_cases)

    for case in golden_cases:
        query = case.get("customer_message", "")
        res = orchestrator.run(
            customer_message=query,
            brand="AppleSupport",
            provider="grounded_precedent",
        )
        if res.validation.all_passed:
            passed_count += 1

    pass_rate = passed_count / total_count if total_count else 0.0
    return pass_rate, passed_count, total_count


def stage_5_run_tests() -> Tuple[int, bool]:
    """Execute pytest suite and parse number of passing tests."""
    cmd = [sys.executable, "-m", "pytest", "tests/", "-q"]
    proc = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        check=False,
    )
    output = proc.stdout + "\n" + proc.stderr
    match = re.search(r"(\d+)\s+passed", output)
    passed_tests = int(match.group(1)) if match else 0
    success = proc.returncode == 0
    return passed_tests, success


def main() -> None:
    check_prerequisites()

    print("==================================================")
    print("GROUNDED CUSTOMER SUPPORT AGENT")
    print("HEADLINE REPRODUCTION")
    print("==================================================")
    print("\nEnvironment")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CPU:      {get_cpu_info()}")
    print()

    total_start = time.perf_counter()

    # Stage 1: Loading evaluation data
    t0 = time.perf_counter()
    test_texts, test_labels, golden_cases = stage_1_load_data()
    t_stage1 = time.perf_counter() - t0
    print(f"[1/5] Loading evaluation data ............ {t_stage1:>5.2f} s")

    # Stage 2: Intent benchmark
    t0 = time.perf_counter()
    intent_acc, intent_f1 = stage_2_intent_benchmark(test_texts, test_labels)
    t_stage2 = time.perf_counter() - t0
    print(f"[2/5] Intent benchmark .................. {t_stage2:>5.2f} s")

    # Stage 3: Retrieval benchmark
    t0 = time.perf_counter()
    uncond_r3, cond_r3 = stage_3_retrieval_benchmark(golden_cases)
    t_stage3 = time.perf_counter() - t0
    print(f"[3/5] Retrieval benchmark ............... {t_stage3:>5.2f} s")

    # Stage 4: Safety benchmark
    t0 = time.perf_counter()
    safety_rate, safety_passed, safety_total = stage_4_safety_benchmark(golden_cases)
    t_stage4 = time.perf_counter() - t0
    print(f"[4/5] Safety benchmark .................. {t_stage4:>5.2f} s")

    # Stage 5: Tests
    t0 = time.perf_counter()
    tests_passed, tests_ok = stage_5_run_tests()
    t_stage5 = time.perf_counter() - t0
    print(f"[5/5] Tests ............................. {t_stage5:>5.2f} s")

    total_elapsed = time.perf_counter() - total_start
    total_minutes = total_elapsed / 60.0

    print("\nHeadline Results")
    print(f"Intent Accuracy:          {intent_acc * 100:.2f}% (Held-out Test, N={len(test_texts)})")
    print(f"Intent Macro F1:          {intent_f1:.4f}")
    print(
        f"Retrieval Recall@3:       {uncond_r3 * 100:.1f}% unconditioned | {cond_r3 * 100:.1f}% intent-conditioned"
    )
    print(
        f"Safety Validation:        {safety_rate * 100:.1f}% ({safety_passed}/{safety_total} cases)"
    )
    print(
        f"Tests Passed:             {tests_passed} passed (status: {'OK' if tests_ok else 'FAILED'})"
    )

    print(f"\nTOTAL WALL-CLOCK TIME: {total_elapsed:.2f} seconds")
    print(f"TOTAL WALL-CLOCK TIME: {total_minutes:.2f} minutes")

    status = "PASS (<15 minutes)" if total_elapsed < 900.0 and tests_ok else "FAIL (>=15 minutes)"
    print("\nREPRODUCTION STATUS:")
    print(status)
    print("==================================================")


if __name__ == "__main__":
    main()
