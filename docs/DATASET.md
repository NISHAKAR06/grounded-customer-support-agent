# Dataset Specification & Methodology

## 1. Primary Dataset Source

The authoritative dataset for the Grounded Customer Support Agent project is:

- **Dataset**: Customer Support on Twitter
- **Kaggle Slug**: `thoughtvector/customer-support-on-twitter`
- **Volume**: 2,811,774 tweets spanning multi-turn customer support interactions across 108 global brands.
- **File Size**: 492.58 MB uncompressed CSV (`twcs.csv`).
- **Temporal Span**: May 8, 2008 – December 3, 2017 (3,496 days).

### Note on Secondary Kernel (`fengliplatform/customer-sentiment-analysis`)
During initial research, the Kaggle kernel `fengliplatform/customer-sentiment-analysis` was reviewed.
- **Finding**: This kernel focuses on general sentiment categorization rather than multi-turn conversational support resolution.
- **Decision**: In strict accordance with the system design specifications, the primary source remains the original `thoughtvector/customer-support-on-twitter` dataset.

---

## 2. Empirical Dataset Breakdown (Phase 1 EDA Findings)

Streaming analysis of all 2,811,774 records (`experiments/dataset_stats.json`) revealed the following distribution:

| Metric Category | Metric Name | Empirical Value | Context / Significance |
| :--- | :--- | :--- | :--- |
| **Volume Split** | Total Records | 2,811,774 | Full uncompressed Twitter corpus |
| | Inbound (Customer Queries) | 1,537,843 (54.69%) | Incoming customer requests and complaints |
| | Outbound (Brand Responses) | 1,273,931 (45.31%) | Official brand support agent replies |
| | Unique Customers | 702,669 | High author diversity preventing user overfitting |
| | Unique Brands | 108 | Cross-industry customer support presence |
| **Topology** | Root Customer Inquiries | 787,346 (51.2%) | True opening queries initiating support threads |
| | Follow-up Customer Turns | 750,497 (48.8%) | Multi-turn conversational clarifications |
| | Brand In-Reply Turns | 1,266,942 | Reactive brand responses to customer queries |
| | Multi-Reply Parents | 222,426 | Inquiries receiving multi-agent or branched replies |
| **Behavioral Signals**| Brand DM Referrals | 336,310 (26.4%) | Sensitive issues routed to private channels |
| | Brand Link / KB Referrals | 416,589 (32.7%) | Procedural grounding in public documentation |
| | Customer Gratitude Signals | 157,151 | Resolution proxy indicators ("thanks", "resolved") |

### Schema Missingness Audit
- `tweet_id`: 0.0% missing
- `author_id`: 0.0% missing
- `inbound`: 0.0% missing
- `created_at`: 0.0% missing
- `text`: 0.0% missing
- `response_tweet_id`: 37.01% missing (expected for terminal/leaf turns of conversations)
- `in_response_to_tweet_id`: 28.25% missing (expected for root initiating inquiries)

---

## 3. Top Brands by Outbound Response Volume

| Rank | Brand Twitter Handle | Outbound Replies | Industry / Domain | Key Characteristics |
| :---: | :--- | :---: | :--- | :--- |
| 1 | `@AmazonHelp` | 169,840 | E-Commerce / Retail | High order-lookup volume, heavy regional variety |
| 2 | `@AppleSupport` | 106,860 | Consumer Tech / OS | Factual troubleshooting, rich public KB grounding, clean text |
| 3 | `@Uber_Support` | 56,270 | Ride-hailing / Transit | Frequent fare disputes, localized trip inquiries |
| 4 | `@SpotifyCares` | 43,265 | Digital Media / Streaming | Subscription billing, multi-platform app issues |
| 5 | `@Delta` | 42,253 | Airline / Aviation | Baggage claims, flight status, high urgency |

---

## 4. Brand Selection Criteria

The methodology requires selecting **ONE brand** from the dataset based on empirical evidence:

1. **Total Inbound Message Volume**: Sufficient sample size (>20,000 incoming customer queries).
2. **First-Turn Customer Queries**: Clear opening queries that initiate threads.
3. **Company Response Ratio**: High percentage of inbound tweets that received an official brand response (>70%).
4. **Multi-Turn Resolved Conversations**: Recognizable resolution patterns.
5. **Intent Diversity**: Natural distribution across distinct operational categories.
6. **Data Cleanliness**: Low incidence of malformed text or fragmented interactions.

---

## 5. Data Partitioning & Storage Architecture

Raw datasets are never tracked in Git.

```
data/
├── raw/         # Raw CSV/JSON files (ignored by git, resolved via Kagglehub cache)
├── interim/     # Cleaned and thread-linked intermediate tables
├── processed/   # Reconstructed multi-turn conversation records
├── external/    # External references or taxonomies
├── samples/     # Lightweight verified samples for local tests & CI
└── golden/      # 150–250 hand-labelled evaluation set (version controlled)
```

---

## 6. Leakage Prevention & Conversation Splitting

1. **Conversation-Level Splits**: Partitioning is strictly performed at the thread level. All turns of a conversation belong exclusively to one split.
2. **Temporal Integrity**: Older conversations are used for historical retrieval indexing and training; newer conversations populate the evaluation test bed.
3. **Golden Set Isolation**: The 150–250 hand-labelled Golden Set is held out completely and never exposed during model training or prompt tuning.

