# Data Preprocessing & Conversation Reconstruction Pipeline (DATA_PIPELINE.md)

## 1. Pipeline Overview & Objectives

In the raw Kaggle *Customer Support on Twitter* dataset (`twcs.csv`), conversations are flattened across 2.81M individual tweet rows connected only by foreign keys (`in_response_to_tweet_id` and `response_tweet_id`).

To build a production-grade grounded support system, raw tweets must be transformed into clean, chronologically ordered **Multi-Turn Conversation Threads**. 

This pipeline serves three downstream capabilities:
1. **Historical Precedent Retrieval (Phase 7)**: Provides indexed pairs of `(Customer Inquiry, Historical Brand Resolution)` for dense semantic similarity search.
2. **Intent Classification (Phase 4 & 6)**: Extracts clean, unpolluted root customer inquiries mapped to the 7-class taxonomy.
3. **Support Inbox UI**: Powers the interactive dashboard inbox with authentic, multi-turn Apple Support conversations.

---

## 2. Text Normalization & Cleaning Rules

Customer and brand tweets are cleaned through deterministic text transformations implemented in `scripts/data/preprocess_conversations.py`:

```
Raw Tweet Text
  │
  ├─► 1. HTML Entity Decoding       (&amp; → &, &gt; → >, &lt; → <)
  ├─► 2. Leading Handle Stripping   ("^(@\w+\s*)+" removed from tweet head)
  ├─► 3. Technical Mention Preserve (Internal references like "iOS 11" preserved)
  ├─► 4. URL & Link Classification  (apple.co / support.apple.com tagged as KB evidence)
  └─► 5. Whitespace Normalization   (Multiple tabs/spaces collapsed to single space)
  │
Cleaned Canonical Text
```

### Cleaning Specifics:
- **Mention Removal**: Removes greeting handles (e.g. `@AppleSupport `, `@115854 `) at the beginning of turns. Mid-sentence mentions are retained to avoid destroying context (e.g., *"Contacted @AppleSupport yesterday"*).
- **HTML Entities**: Decodes encoded characters commonly found in Twitter feeds (e.g., `Settings &gt; General &gt; About` correctly becomes `Settings > General > About`).
- **Emoji & Encoding Preservation**: UTF-8 characters and unicode symbols are preserved to prevent string corruption.

---

## 3. Conversation Thread Reconstruction Algorithm

The pipeline reconstructs multi-turn dialogue trees via directed graph traversal:

1. **Brand Subset Filtering**:
   - Extracts all outbound tweets where `author_id == "AppleSupport"`.
   - Extracts all inbound tweets mentioning `@AppleSupport`.

2. **Root Identification**:
   - Identifies candidate root inquiries: inbound customer tweets where `in_response_to_tweet_id` is null or references a tweet outside the corpus.

3. **Parent-Child Chain Stitching**:
   - Indexes a reverse parent-to-child map: `parent_tweet_id -> [child_tweet_ids]`.
   - Traverses each chain starting from the root customer inquiry through successive replies in chronological order.
   - Enforces a visited set to guard against circular parent-child references.

4. **Conversation Validity Filter**:
   - A thread is only retained if it contains **at least one official brand response** (`AppleSupport`).
   - Threads with zero brand interactions or standalone customer monologues are discarded.

---

## 4. Reconstructed Conversation Schema (`.jsonl`)

Each line in `data/processed/applesupport_conversations.jsonl` adheres to this schema:

```json
{
  "conversation_id": "conv_698",
  "ticket_id": "TICK-698",
  "customer_id": "115854",
  "brand": "AppleSupport",
  "root_tweet_id": "698",
  "first_inquiry": "why are my I’s changing not showing up correctly on any of my social media platforms?",
  "latest_message": "Lets take a closer look into this issue. Select the following link to join us in a DM...",
  "final_brand_response": "Lets take a closer look into this issue. Select the following link to join us in a DM...",
  "turn_count": 5,
  "turns": [
    {
      "turn_id": 1,
      "tweet_id": "700",
      "author_id": "115854",
      "author_role": "CUSTOMER",
      "text": "why are my I’s changing not showing up correctly on any of my social media platforms?",
      "raw_text": "@AppleSupport why are my I️’s changing not showing up correctly on any of my social media platforms? https://t.co/GyRvpyVnkE",
      "created_at": "Tue Oct 31 22:10:47 +0000 2017"
    },
    {
      "turn_id": 2,
      "tweet_id": "698",
      "author_id": "115854",
      "author_role": "CUSTOMER",
      "text": "https://t.co/NV0yucs0lB",
      "raw_text": "@AppleSupport  https://t.co/NV0yucs0lB",
      "created_at": "Tue Oct 31 22:11:45 +0000 2017"
    },
    {
      "turn_id": 3,
      "tweet_id": "696",
      "author_id": "AppleSupport",
      "author_role": "BRAND",
      "text": "We're here for you. Which version of the iOS are you running? Check from Settings > General > About.",
      "raw_text": "@115854 We're here for you. Which version of the iOS are you running? Check from Settings &gt; General &gt; About.",
      "created_at": "Tue Oct 31 22:24:49 +0000 2017"
    }
  ],
  "has_dm": true,
  "has_kb_link": true,
  "has_resolution": false,
  "status": "Needs Human",
  "created_at": "Tue Oct 31 22:10:47 +0000 2017"
}
```

---

## 5. Operational Heuristics & Triage Tagging

Each thread is tagged with operational markers to drive the support dashboard:

| Tag Field | Detection Rule | Purpose in System |
| :--- | :--- | :--- |
| `has_dm` | Regex matching `\b(dm\|direct message\|join us in a dm)\b` in brand replies. | Identifies issues that escalated off-platform to private agent channels. |
| `has_kb_link` | URL regex matching `apple.co`, `support.apple.com`, or knowledge base guides. | Flags threads grounded in official technical articles for retrieval indexing. |
| `has_resolution` | Customer gratitude keywords (`thank you`, `fixed`, `working now`, `solved`). | Signals confirmed successful in-thread resolution. |
| `status` | Tri-state: `Resolved` (gratitude), `Needs Human` (DM), or `AI Ready` (procedural). | Categorizes conversations for the operator inbox view. |

---

## 6. Pipeline Artifacts & Verification

- **Script**: `scripts/data/preprocess_conversations.py`
- **Output Artifacts**:
  - `data/processed/applesupport_conversations.jsonl`: Full reconstructed conversation corpus.
  - `data/processed/applesupport_sample.jsonl`: Lightweight committed sample for offline development & CI test execution.
  - `experiments/preprocessing_stats.json`: Pipeline metrics and distribution summary.
- **Repository Integration**: `app/repositories/conversation_repository.py` automatically resolves processed conversation files to populate the live inbox with real data.
