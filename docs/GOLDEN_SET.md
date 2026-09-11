# Golden Evaluation Set Specification & Curation Report

## 1. Executive Summary

A core standard of enterprise customer support systems is that **"the proof is worth more than the system."** Evaluating generative agents on automated heuristics alone is insufficient; automated systems must be measured against an uncorrupted, hand-annotated benchmark representing real customer behaviors.

The **Golden Evaluation Set** (`data/golden/golden_set.jsonl`) consists of **200 real `@AppleSupport` conversations** extracted from the Twitter customer support corpus. It establishes an empirical ground truth for evaluating:
1. **Intent Classification Accuracy & Macro-F1** across 7 MECE intent categories.
2. **Precedence Disambiguation** on complex, multi-topic queries.
3. **Automated vs. Escalation Routing Reliability** with verifiable reasons.
4. **Historical Precedent Faithfulness** in drafted agent responses.

---

## 2. Dataset Composition & Stratification

To ensure fair representation across both high-frequency issues (OS updates) and critical low-frequency issues (hardware safety, billing disputes), the 200 samples were curated through stratified sampling:

```
+--------------------------------+---------+------------+-------------+------------------+
| Intent Category                | Samples | Percentage | Auto-Handle | Human Escalation |
+--------------------------------+---------+------------+-------------+------------------+
| OPERATING_SYSTEM_UPDATES       | 40      | 20.0%      | 22          | 18               |
| BATTERY_POWER_HARDWARE         | 35      | 17.5%      | 0           | 35               |
| ACCOUNT_APPLE_ID               | 30      | 15.0%      | 0           | 30               |
| CONNECTIVITY_NETWORKING        | 25      | 12.5%      | 12          | 13               |
| AUDIO_ACCESSORIES              | 20      | 10.0%      | 8           | 12               |
| SUBSCRIPTIONS_BILLING          | 20      | 10.0%      | 0           | 20               |
| GENERAL_INQUIRY                | 30      | 15.0%      | 0           | 30               |
+--------------------------------+---------+------------+-------------+------------------+
| TOTAL                          | 200     | 100.0%     | 42 (21%)    | 158 (79%)        |
+--------------------------------+---------+------------+-------------+------------------+
```

### Complexity Distribution
- **Canonical (Single-Issue)**: 134 samples (67.0%)
- **Multi-Turn Context Required**: 35 samples (17.5%)
- **Boundary Edge Cases**: 28 samples (14.0%)
- **Conflict Queries**: 3 samples (1.5%)

---

## 3. Ground Truth Resolution & Verification

Each sample includes the official resolution provided by `@AppleSupport` agents during the actual customer interaction. This ensures that when the retrieval and generation engines execute in subsequent phases:
1. The drafted response can be compared directly against the historical resolution using semantic similarity and LLM-as-a-Judge rubrics.
2. Hallucinations (such as promising refunds, inventing unverified settings, or guessing warranty eligibility) are immediately detected against the ground truth standard.

---

## 4. Leakage Prevention & Reproducibility

1. **Isolation**: The Golden Set is committed under `data/golden/golden_set.jsonl` and is strictly excluded from all training splits.
2. **Deterministic Sampling**: Generated with fixed random seed (42) via `scripts/evaluation/curate_golden_set.py`.
3. **Integrity Validation**: Automated test suites in `tests/evaluation/test_golden_set.py` enforce non-empty fields, unique conversation IDs, and valid schema properties.
