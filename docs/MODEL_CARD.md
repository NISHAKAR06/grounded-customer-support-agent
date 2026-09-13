# Model Card: Grounded Customer Support Agent Models

## 1. Model Details

- **Model Hierarchy**:
  1. **Baseline 1**: Majority Class Classifier.
  2. **Baseline 2**: TF-IDF N-gram Vectorizer + Regularized Logistic Regression.
  3. **Final Intent Model**: Fine-tuned Dense Classifier on Domain Embeddings (`all-MiniLM-L6-v2`).
  4. **Dense Retriever**: `sentence-transformers/all-MiniLM-L6-v2` vector indexing over historical resolved cases via FAISS.
  5. **Grounded Reply Generator**: Google Gemini with offline `LocalLLMProvider` fallback.
- **License**: MIT
- **Primary Frameworks**: PyTorch, Scikit-Learn, Sentence-Transformers, FAISS, FastAPI.

---

## 2. Intended Use

- Customer support triage and intent categorization for enterprise shared inboxes.
- Rapid retrieval of historical resolved support interactions matching incoming customer inquiries.
- Grounded draft generation strictly constrained to brand precedents.
- Deterministic escalation routing to safeguard customer experience.

---

## 3. Training & Evaluation Data

- **Primary Corpus**: Kaggle `thoughtvector/customer-support-on-twitter` filtered to a single selected brand.
- **Split Strategy**: Strict thread-level partitioning (Train: 70%, Validation: 15%, Test: 15%). Zero cross-thread leakage.
- **Evaluation Benchmark**: Hand-labelled Golden Set (150–250 real samples) held completely separate from training.

---

## 4. Performance & Evaluation Metrics

All metrics are verified through reproducible evaluation scripts ([`scripts/run_evaluation.py`](../scripts/run_evaluation.py) and [`scripts/evaluation/evaluate_intent_models.py`](../scripts/evaluation/evaluate_intent_models.py)). Metrics are saved to versioned JSON artifacts in `experiments/` without fabrication.
