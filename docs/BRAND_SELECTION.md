# Brand Selection Methodology & Analysis (BRAND_SELECTION.md)

## 1. Overview & Assignment Requirement

The Hiver Take-Home assignment mandates selecting **ONE brand** from the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`, ~2.8M tweets) based on empirical evidence rather than name recognition.

This document details the selection criteria, candidate brand volume metrics, multi-turn resolution viability, and the formal justification for the chosen brand.

---

## 2. Selection Criteria

Candidate brands are evaluated along six quantitative and qualitative axes:

1. **Inbound Conversation Volume**: Sufficient volume ($>20,000$ incoming customer inquiries) ensuring robust sample size for training, validation, testing, and golden set extraction.
2. **First-Turn Customer Inquiries**: Clear opening queries that initiate support threads (`in_reply_to_tweet_id` is null or starts a thread) without preceding hidden context.
3. **Company Response Ratio**: High percentage of inbound tweets that received an official brand response ($>70\%$).
4. **Multi-Turn Resolved Conversations**: Identifiable conversation completion patterns (e.g., closing acknowledgments, "glad we could help", or successful resolution paths).
5. **Intent Diversity**: Natural distribution across distinct operational categories (e.g., billing, order delays, login failures, technical support, account access).
6. **Data Cleanliness & Formatting**: Low incidence of malformed text, automated bot spam, or fragmented interactions.

---

## 3. Candidate Brands Under Consideration

During exploratory analysis, the following top enterprise support accounts from the dataset are evaluated:

| Candidate Brand | Twitter Handle | Industry | Estimated Inbound Volume | Primary Support Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Amazon** | `@AmazonHelp` | E-Commerce / Retail | High (>100k) | High turn volume, order tracking, address changes, refunds |
| **Apple** | `@AppleSupport` | Consumer Tech / OS | High (>100k) | Hardware/iOS troubleshooting, account recovery, device resets |
| **Uber** | `@Uber_Support` | Ride Hailing / Mobility | Medium (>50k) | Trip fare disputes, driver feedback, lost items |
| **Delta Air Lines** | `@Delta` | Airline / Travel | Medium (>40k) | Flight cancellations, baggage tracking, booking changes |
| **Spotify** | `@SpotifyCares` | Digital Subscription | Medium (>30k) | Billing, playback glitches, account management |

---

## 4. Empirical Evaluation & Final Selection

> **Status**: Candidate metrics framework established. Detailed statistical distributions and final selection will be populated with exact dataset calculations during **Phase 1 (Dataset Exploration)** and **Phase 2 (Brand Selection)**.

### Selected Brand (Preliminary Target: AmazonHelp)
- **Preliminary Hypothesis**: `@AmazonHelp` represents the ideal balance of clear operational intents (order tracking, cancellations, refunds, address modification) and explicit resolution outcomes.
- **Evidence Verification**: Exact message counts, thread resolution rates, and intent distributions will be computed via `notebooks/01_data_exploration.ipynb` in Phase 1 before finalizing.

---

## 5. Limitations & Caveats

- **Twitter Character Constraints**: 140/280 character limits produce compressed sentences and abbreviations.
- **Off-Platform Transitions**: Brands often request customers to "DM your order number", requiring turn-reconstruction heuristics to capture resolved threads.
