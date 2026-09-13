#!/usr/bin/env python3
"""Master Turnkey Pipeline Runner for Grounded Customer Support Agent.

Executes all pipeline stages from a single unified entrypoint:
  1. Train Baseline Intent Classifiers (Majority & TF-IDF LogReg)
  2. Build Dense FAISS Vector Store Index (all-MiniLM-L6-v2)
  3. Run End-to-End CLI Pipeline Smoke Test (4 real customer scenarios)
  4. Run Comprehensive Headless Evaluation Suite (Intent, Retrieval, Generation, Judge)
  5. (Optional) Run Automated Pytest Suite & Linters
  6. (Optional) Run Full Data Preprocessing Pipeline from Raw Data

Usage:
  python scripts/run_all.py              # Execute core pipeline: Train -> Smoke -> Eval
  python scripts/run_all.py --with-tests # Execute core pipeline + full pytest suite
  python scripts/run_all.py --train      # Run only baseline & vector index build
  python scripts/run_all.py --smoke      # Run only live CLI smoke test
  python scripts/run_all.py --eval       # Run only evaluation benchmarks
  python scripts/run_all.py --tests      # Run only pytest test suite
  python scripts/run_all.py --with-data  # Also run raw data exploration & preprocessing
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Project root resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Windows console encoding safeguard
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner(text: str, width: int = 80) -> None:
    """Print high-visibility ASCII section header."""
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def run_stage(
    title: str, script_path: Path, args: Optional[List[str]] = None
) -> Tuple[bool, float, str]:
    """Execute a python script as a managed subprocess and record outcome."""
    print(f"\n>> Running Stage: {title} ...")
    start_time = time.time()

    cmd = [sys.executable, str(script_path)] + (args or [])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            capture_output=False,
            text=True,
            check=False,
        )
        elapsed = time.time() - start_time
        success = (result.returncode == 0)
        return success, elapsed, ""
    except Exception as exc:
        elapsed = time.time() - start_time
        return False, elapsed, str(exc)


def run_command(title: str, cmd: List[str]) -> Tuple[bool, float, str]:
    """Execute an arbitrary CLI command and record outcome."""
    print(f"\n>> Running Command: {title} ...")
    start_time = time.time()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BASE_DIR)

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            capture_output=False,
            text=True,
            check=False,
        )
        elapsed = time.time() - start_time
        success = (result.returncode == 0)
        return success, elapsed, ""
    except Exception as exc:
        elapsed = time.time() - start_time
        return False, elapsed, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Master Turnkey Pipeline Runner for Grounded Customer Support Agent.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Include full pytest test suite execution after evaluation.",
    )
    parser.add_argument(
        "--with-data",
        action="store_true",
        help="Include raw data exploration and thread reconstruction pipeline.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run only baseline models and FAISS index build.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only the 4-scenario end-to-end CLI smoke test.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run only the headless evaluation suite.",
    )
    parser.add_argument(
        "--tests",
        action="store_true",
        help="Run only pytest test suite with coverage report.",
    )

    args = parser.parse_args()

    print_banner("GROUNDED CUSTOMER SUPPORT AGENT — UNIFIED PIPELINE RUNNER")
    print(f"Project Directory: {BASE_DIR}")
    print(f"Python Executable: {sys.executable}")
    print(f"Timestamp:         {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()
    results: List[Tuple[str, bool, float]] = []

    # Optional: Data preprocessing from scratch
    if args.with_data:
        print_banner("PHASE 0: DATA PREPROCESSING PIPELINE")
        data_scripts = [
            ("Dataset Exploration", BASE_DIR / "scripts" / "data" / "explore_dataset.py"),
            ("Brand Profiling (@AppleSupport)", BASE_DIR / "scripts" / "data" / "profile_brand.py"),
            ("Conversation Thread Reconstruction", BASE_DIR / "scripts" / "data" / "preprocess_conversations.py"),
            ("Intent Classification Annotation", BASE_DIR / "scripts" / "data" / "classify_intents.py"),
            ("Dataset Split Generation (Train/Val/Test)", BASE_DIR / "scripts" / "data" / "split_dataset.py"),
            ("Golden Set Curation (200 Samples)", BASE_DIR / "scripts" / "evaluation" / "curate_golden_set.py"),
        ]
        for name, script in data_scripts:
            if script.exists():
                success, elapsed, _ = run_stage(name, script)
                results.append((name, success, elapsed))
                if not success:
                    print(f"[FAILED] {name} encountered an error.")
                    break

    # Phase 1: Model & FAISS Vector Index Training
    if not (args.smoke or args.eval or args.tests):
        print_banner("PHASE 1: TRAINING BASELINE MODELS & VECTOR INDEX")
        train_stages = [
            ("Baseline Classifiers (Majority & TF-IDF)", BASE_DIR / "scripts" / "training" / "train_baselines.py"),
            ("Dense FAISS Vector Index (all-MiniLM-L6-v2)", BASE_DIR / "scripts" / "training" / "build_faiss_index.py"),
        ]
        for name, script in train_stages:
            success, elapsed, _ = run_stage(name, script)
            results.append((name, success, elapsed))
            if not success:
                print(f"[FAILED] {name} encountered an error.")
                break

    # Phase 2: Live End-to-End CLI Smoke Test
    if not (args.train or args.eval or args.tests):
        print_banner("PHASE 2: LIVE END-TO-END CLI SMOKE TEST")
        smoke_script = BASE_DIR / "scripts" / "utilities" / "check_e2e_flow.py"
        success, elapsed, _ = run_stage("CLI Pipeline Smoke Test (4 Scenarios)", smoke_script)
        results.append(("CLI Pipeline Smoke Test", success, elapsed))

    # Phase 3: Comprehensive Evaluation Suite
    if not (args.train or args.smoke or args.tests):
        print_banner("PHASE 3: HEADLESS EVALUATION BENCHMARKS")
        eval_script = BASE_DIR / "scripts" / "run_evaluation.py"
        success, elapsed, _ = run_stage("Unified Evaluation Harness", eval_script, ["--all"])
        results.append(("Evaluation Harness (All 4 Stages)", success, elapsed))

    # Phase 4: Automated Tests (if requested)
    if args.with_tests or args.tests:
        print_banner("PHASE 4: AUTOMATED TEST SUITE & COVERAGE")
        test_cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--cov=app", "--cov-report=term"]
        success, elapsed, _ = run_command("Pytest Test Suite (112 Tests)", test_cmd)
        results.append(("Automated Test Suite (pytest)", success, elapsed))

    total_elapsed = time.time() - total_start

    # Print Final Summary Table
    print_banner("MASTER PIPELINE EXECUTION SUMMARY")
    print(f"{'Stage / Component':<46} {'Status':<12} {'Elapsed Time':<12}")
    print("-" * 72)
    all_passed = True
    for name, success, elapsed in results:
        status_str = "[PASS]" if success else "[FAIL]"
        if not success:
            all_passed = False
        print(f"{name:<46} {status_str:<12} {elapsed:>8.2f}s")
    print("-" * 72)
    overall_status = "ALL STAGES PASSED" if all_passed else "SOME STAGES FAILED"
    print(f"Overall Status: {overall_status} (Total Elapsed: {total_elapsed:.2f}s)\n")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
