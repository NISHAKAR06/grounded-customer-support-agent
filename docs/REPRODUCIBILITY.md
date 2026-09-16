# Reproducibility Report

## Goal

The objective of this report is to empirically demonstrate, measure, and document whether the project's headline evaluation results can be reproduced in under 15 minutes on an evaluation environment without relying on external cloud APIs, synthetic labels, or fabricated data.

---

## Reproduction Scope

### Included in Headline Reproduction
- **Intent Classification Benchmark**: Evaluation of the production champion (TF-IDF + Balanced Logistic Regression) against the held-out test split (`data/splits/test.jsonl`, 484 samples).
- **Historical Retrieval Benchmark**: Evaluation of dense FAISS vector search (`models/embedding_model/faiss.index`, `sentence-transformers/all-MiniLM-L6-v2`) over 2,245 historical cases across the 200 Golden Set cases (`data/golden/golden_set.jsonl`), measuring unconditioned Recall@3 and intent-conditioned Recall@3.
- **Safety Validation Benchmark**: Evaluation of the 6-barrier deterministic validation gate (`ResponseValidator`) across the 200 Golden Set cases, verifying non-empty checks, grounding evidence presence, financial guardrails, URL domain whitelisting, PII interception, and hazardous hardware triggers.
- **Automated Test Suite**: Execution of the entire test suite (`tests/`) containing 122 passing tests.

### Excluded from Automated Runtime Benchmark
- **Human-Labelled Golden Set**: The 200-sample Golden Set is currently heuristically curated from real historical support dialogues. Human intent and routing annotations remain pending human evaluation.
- **Human-vs-LLM Agreement**: Cohen's Kappa ($\kappa$) against independent human raters requires manual human scoring and cannot be computed automatically without human annotators.
- **External Human Ratings**: External qualitative scores are pending actual human study.
- **Live Cloud LLM Generations**: Live API calls to third-party endpoints (e.g., Groq, OpenAI, Google Gemini) are excluded from the default local benchmark to ensure 100% offline determinism, rate-limit resilience, and zero API-key dependencies.

---

## Environment

| Component | Specification |
|:---|:---|
| **Python Version** | 3.13.5 (CI supports Python 3.11 & 3.12) |
| **Operating System** | Microsoft Windows 11 Enterprise (64-bit) |
| **Processor / CPU** | x86_64 Architecture (Multi-core) |
| **scikit-learn** | 1.6.1 |
| **torch** | 2.9.1+cpu |
| **faiss-cpu** | 1.14.2 |
| **sentence-transformers** | >= 2.5.0 |
| **pandas** | 2.2.3 |
| **numpy** | 2.1.3 |

*(All identifying corporate credentials and personal usernames have been redacted.)*

---

## Commands

To reproduce the headline benchmark from a fresh clone:

```bash
# 1. Clone repository
git clone https://github.com/NISHAKART/grounded-customer-support-agent.git
cd grounded-customer-support-agent

# 2. Create and activate virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Run the deterministic headline reproduction script
python scripts/reproduce_headline.py
```

---

## Runtime Evidence

Empirical stage-by-stage wall-clock runtimes measured on the evaluation host using Python standard timing facilities (`time.perf_counter()`):

| Stage | Operation / Component | Runtime |
|:---|:---|---:|
| **[1/5] Data Loading** | Verify artifacts & load test split (484) + Golden Set (200) | 0.04 s |
| **[2/5] Intent Benchmark** | TF-IDF + Logistic Regression inference on held-out test split | 0.26 s |
| **[3/5] Retrieval Benchmark** | FAISS dense retrieval (MiniLM) Recall@3 on Golden Set (200 cases) | 53.66 s |
| **[4/5] Safety Benchmark** | 6-barrier deterministic validation gate over Golden Set (200 cases) | 3.96 s |
| **[5/5] Automated Tests** | Complete pytest test suite execution (122 tests) | 72.62 s |
| **Total Wall-Clock Time** | **Full benchmark execution** | **130.54 s (2.18 min)** |

---

## Repeated Runs

To ensure stability, numerical consistency, and guard against transient timing anomalies, the benchmark was executed in 3 independent, consecutive runs:

| Run | Total Runtime (Seconds) | Total Runtime (Minutes) | Status | Tests Passed |
|:---|---:|---:|:---|:---|
| **Run 1** | 130.54 s | 2.18 min | PASS (<15 min) | 122 / 122 |
| **Run 2** | 191.43 s | 3.19 min | PASS (<15 min) | 122 / 122 |
| **Run 3** | 149.21 s | 2.49 min | PASS (<15 min) | 122 / 122 |

### Summary Statistics
- **Minimum Runtime**: 130.54 seconds (2.18 minutes)
- **Maximum Runtime**: 191.43 seconds (3.19 minutes)
- **Mean Runtime**: 157.06 seconds (2.62 minutes)

---

## Headline Result Verification

Comparison of reproduced values against documented benchmark figures:

| Metric | Documented Target | Reproduced Value | Absolute Difference | Verification Status |
|:---|---:|---:|---:|:---|
| **Intent Accuracy** (Held-Out Test) | 84.92% | 84.92% | 0.00% | **EXACT MATCH** |
| **Intent Macro F1** (Held-Out Test) | 0.7154 | 0.7154 | 0.0000 | **EXACT MATCH** |
| **Retrieval Recall@3** (Golden Unconditioned) | 77.50% | 77.50% | 0.00% | **EXACT MATCH** |
| **Retrieval Recall@3** (Golden Intent-Conditioned) | 96.50% | 96.50% | 0.00% | **EXACT MATCH** |
| **Safety Validation Pass Rate** | 80.5% – 81.0% | 81.0% (162/200) | +0.5% (1 case)* | **VERIFIED** |
| **Automated Tests Passed** | 122 passed | 122 passed | 0 | **EXACT MATCH** |

*\*Note on Safety Validation: `experiments/generation_benchmarks.json` logged 80.5% (161/200) when one generated draft fell below the 20-character threshold. When evaluating offline grounded precedents, all 200 cases produce non-empty candidate responses, yielding 162/200 (81.0%) pass rate on the exact same 6 safety barriers.*

---

## Reproducibility Verdict

### VERIFIED — UNDER 15 MINUTES

**Verdict**: The headline evaluation benchmark is verified to execute in **under 15 minutes** on a standard multi-core machine. Across 3 consecutive timed benchmark runs, execution completed between **2.18 and 3.19 minutes** (mean: **2.62 minutes**).

---

## Limitations & Disclosures

1. **Hardware Differences**: CPU vector indexing and embedding generation vary based on available CPU instructions (AVX-512 / AVX2) and core count. Runtimes on single-core virtual machines will be slightly higher but remain well within the 15-minute ceiling.
2. **First-Time Model Download**: First-time execution without a cached `sentence-transformers/all-MiniLM-L6-v2` model downloads ~90 MB from HuggingFace Hub, requiring standard internet bandwidth (typically ~10–30 seconds).
3. **Dependency Installation**: Installing dependencies from `requirements.txt` into a clean virtual environment takes approximately 1 to 3 minutes depending on pip wheel cache and network speed. Setup time and benchmark execution time are reported separately.
4. **Pre-Built Artifacts**: The benchmark utilizes pre-computed models (`models/intent_classifier/`) and FAISS indexes (`models/embedding_model/`) trained strictly on `data/splits/train.jsonl` (zero leakage into test or golden sets). These artifacts are checked into the repository to guarantee instant, reproducible execution without re-indexing or re-downloading the 2.81M-row raw dataset.
5. **Pending Human Evaluation**: As explicitly disclosed across the repository, the 200 Golden Set cases are heuristically verified rather than human-labelled, and human-vs-LLM agreement ($\kappa$) remains pending manual human evaluation.
