# Dataset Specification & Methodology

## 1. Primary Dataset Source

The authoritative dataset for the Hiver Grounded Customer Support Agent project is:

- **Dataset**: Customer Support on Twitter
- **Kaggle Slug**: `thoughtvector/customer-support-on-twitter`
- **Volume**: Approximately 2.8 million tweets spanning multi-turn customer support interactions across top global brands (Apple, Amazon, Uber, Delta, Spotify, Nike, British Airways, etc.).

### Note on Secondary Kernel (`fengliplatform/customer-sentiment-analysis`)
During initial research, the Kaggle kernel `fengliplatform/customer-sentiment-analysis` was reviewed.
- **Finding**: This kernel focuses on general sentiment categorization rather than multi-turn conversational support resolution.
- **Decision**: In strict accordance with the Hiver assignment requirements, the primary source remains the original `thoughtvector/customer-support-on-twitter` dataset.

---

## 2. Brand Selection Criteria

The assignment mandates selecting **ONE brand** from the dataset based on empirical evidence:

1. **Total Inbound Message Volume**: Sufficient sample size (>20,000 incoming customer queries).
2. **First-Turn Customer Queries**: Clear opening queries that initiate threads.
3. **Company Response Ratio**: High percentage of inbound tweets that received an official brand response (>70%).
4. **Multi-Turn Resolved Conversations**: Recognizable resolution patterns.
5. **Intent Diversity**: Natural distribution across distinct operational categories.
6. **Data Cleanliness**: Low incidence of malformed text or fragmented interactions.

---

## 3. Data Partitioning & Storage Architecture

Raw datasets are never tracked in Git.

```
data/
├── raw/         # Raw CSV/JSON files (ignored by git)
├── interim/     # Cleaned and thread-linked intermediate tables
├── processed/   # Reconstructed multi-turn conversation records
├── external/    # External references or taxonomies
├── samples/     # Lightweight verified samples for local tests & CI
└── golden/      # 150–250 hand-labelled evaluation set (version controlled)
```

---

## 4. Leakage Prevention & Conversation Splitting

1. **Conversation-Level Splits**: Partitioning is strictly performed at the thread level. All turns of a conversation belong exclusively to one split.
2. **Temporal Integrity**: Older conversations are used for historical retrieval indexing and training; newer conversations populate the evaluation test bed.
3. **Golden Set Isolation**: The 150–250 hand-labelled Golden Set is held out completely and never exposed during model training or prompt tuning.
