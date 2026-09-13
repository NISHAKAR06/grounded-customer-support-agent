# Evaluation Architecture & Navigation Map

This directory serves as the conceptual root for the evaluation system. To adhere to standard Python package organization, evaluation resources are structured into the following dedicated modules:

## 1. Evaluation Scripts ([`scripts/evaluation/`](../scripts/evaluation/))
All benchmark execution scripts are housed under `scripts/evaluation/`:
- [`evaluate_intent_models.py`](../scripts/evaluation/evaluate_intent_models.py): Benchmarks Intent Classification (Production vs. Majority vs. TF-IDF LogReg baselines).
- [`evaluate_retrieval.py`](../scripts/evaluation/evaluate_retrieval.py): Evaluates FAISS dense vector search (Recall@1, Recall@3, Recall@5, MRR, Intent-Conditioning gain).
- [`evaluate_generation.py`](../scripts/evaluation/evaluate_generation.py): Evaluates multi-barrier safety guardrails, escalation precision/recall, and routing policies.
- [`evaluate_judge.py`](../scripts/evaluation/evaluate_judge.py): Calculates multi-dimensional LLM-as-a-Judge scores and Cohen's Kappa ($\kappa = 0.8118$) inter-annotator agreement.
- [`curate_golden_set.py`](../scripts/evaluation/curate_golden_set.py): Two-stage stratified curation pipeline for the 200 hand-verified evaluation samples.

## 2. Golden Evaluation Benchmark Set ([`data/golden/`](../data/golden/))
- [`golden_set.jsonl`](../data/golden/golden_set.jsonl): 200 hand-verified, leak-free test cases across all 7 domain intents.
- [`golden_set_summary.json`](../data/golden/golden_set_summary.json): Statistical breakdown across categories, turn counts, and risk attributes.
- [`README.md`](../data/golden/README.md): Detailed documentation of the golden set curation methodology.

## 3. Benchmark Output Artifacts ([`experiments/`](../experiments/))
Precomputed and verified evaluation metrics are saved as versioned JSON artifacts:
- [`baseline_benchmarks.json`](../experiments/baseline_benchmarks.json): Intent classification accuracy, macro-F1, and confusion matrices.
- [`retrieval_benchmarks.json`](../experiments/retrieval_benchmarks.json): Top-k retrieval recall and MRR metrics.
- [`generation_benchmarks.json`](../experiments/generation_benchmarks.json): Safety barrier pass rates and escalation accuracy.
- [`judge_benchmarks.json`](../experiments/judge_benchmarks.json): Groundedness, relevance, tone, safety, and Cohen's Kappa calibration.

## 4. Application Evaluation Service & API
- **Service Layer**: [`app/services/evaluation/evaluation_service.py`](../app/services/evaluation/evaluation_service.py)
- **API Routes**: [`app/api/routes/evaluation.py`](../app/api/routes/evaluation.py) (`GET /api/evaluation/metrics`, `GET /api/evaluation/golden-set`)
- **Web UI Dashboard**: Visual evaluation interface rendered at `http://localhost:8000/evaluation`.

## 5. Automated Test Suite ([`tests/evaluation/`](../tests/evaluation/))
- [`test_golden_set.py`](../tests/evaluation/test_golden_set.py): Verifies schema compliance, unique IDs, non-empty fields, and API endpoints.

---

### Running All Evaluations:
```bash
# Master evaluation harness across all 4 stages:
python scripts/run_evaluation.py

# Or unified master pipeline:
python scripts/run_all.py --eval
```
