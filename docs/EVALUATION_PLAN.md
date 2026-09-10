# Evaluation Plan: Grounded Customer Support Agent

## 1. Evaluation Philosophy

The Hiver assignment explicitly states: **"The proof is worth more than the system."**

Our evaluation harness is constructed with complete methodological integrity:
- **No data leakage**: Evaluation is executed exclusively on a held-out test partition and golden set.
- **No synthetic labels**: The golden set contains 100% human-verified labels from real conversations.
- **No metric fabrication**: Every reported number is computed by reproducible evaluation scripts.
- **Comprehensive baselines**: Measured against both a trivial baseline and a simple classical ML baseline.

---

## 2. Intent Classification Benchmark

### 2.1 Evaluated Systems
1. **Baseline 1 (Trivial)**: Majority Class Classifier (predicts the most frequent intent unconditionally).
2. **Baseline 2 (Simple ML)**: TF-IDF feature representation with balanced Logistic Regression.
3. **Final System**: Fine-tuned Dense Classifier / Transformer-based Intent Model.

### 2.2 Metrics & Reporting
- **Accuracy**: Overall proportion of correctly categorized queries.
- **Macro-Averaged F1**: Balances performance across minority and majority intents equally.
- **Precision & Recall per Intent**: Pinpoints vulnerability to specific customer requests.
- **Confusion Matrix**: Visualizes cross-intent misclassifications.

---

## 3. Hand-Labelled Golden Set (150–250 Examples)

- **Dataset Partition**: Fully isolated under `data/golden/golden_set.jsonl`.
- **Target Size**: 150 to 250 real customer messages from the chosen brand.
- **Sampling Methodology**: Stratified random sampling across intent clusters, deliberately including edge cases.
- **Quality Assurance**: Every sample contains `sample_id`, `customer_message`, `gold_intent`, `requires_human`, and `annotator_notes`.

---

## 4. Historical Retrieval Evaluation

- **Recall@1**: Percentage of queries where the top-retrieved case is a relevant resolved precedent.
- **Recall@3**: Percentage of queries where at least one of top-3 retrieved cases is relevant.
- **Recall@5**: Percentage of queries where at least one of top-5 retrieved cases is relevant.
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant historical precedent.

---

## 5. Grounded Generation & LLM-as-a-Judge

Generated responses are evaluated using a multi-dimensional rubric via an independent LLM Judge (with strict JSON-schema output):
- **Groundedness** (1–5)
- **Helpfulness** (1–5)
- **Resolution Fit** (1–5)
- **Unsupported Claims** (Pass/Fail)
- **Tone & Brand Safety** (1–5)

---

## 6. Human-Judge Agreement

A stratified subset of 50 generated replies is independently scored by a human evaluator using the exact same rubric to compute percent exact agreement and Cohen's Kappa ($\kappa$).
