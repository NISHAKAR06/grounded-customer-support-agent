---
title: Grounded Customer Support Agent
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Grounded Customer Support Agent

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linter: Ruff](https://img.shields.io/badge/linter-ruff-orange.svg)](https://github.com/astral-sh/ruff)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7.svg)](https://grounded-customer-support-agent.onrender.com/)

> An evidence-grounded AI customer support agent designed for real-world social support traffic. Classifies customer intent into a domain taxonomy, retrieves historically resolved brand precedents via dense vector search, drafts grounded replies, enforces multi-barrier hallucination guardrails, and deterministically routes inquiries between automated resolution (`AUTO_HANDLE`) and human review (`HUMAN_ESCALATION`).

---

## TL;DR / At a Glance

| Area | Implementation & Verified Finding |
| :--- | :--- |
| **Live Deployment** | **[grounded-customer-support-agent.onrender.com](https://grounded-customer-support-agent.onrender.com/)** (Interactive simulator, support inbox, evaluation dashboard, failure analyses) |
| **Problem** | Unconstrained LLM hallucinations, fabricated pricing, and physical safety hazards in automated customer support |
| **Dataset** | Kaggle *Customer Support on Twitter* (`thoughtvector/customer-support-on-twitter`, 2.81M tweets) |
| **Target Brand** | `@AppleSupport` (106,860 brand replies; complex hardware diagnostics, OS updates, and billing disputes) |
| **Intent Classification** | 7-class domain taxonomy with priority disambiguation rules (**88.0% Accuracy**, **0.867 Macro F1** on preprocessed splits; **98.5% Accuracy** audited against hand-labeled Golden Set) |
| **Retrieval Engine** | Dense FAISS `IndexFlatIP` on `all-MiniLM-L6-v2` (2,245 resolved precedents, **Unconditioned Recall@3 = 77.5%**, **Intent-Conditioned Recall@3 = 96.5%** [conditioned on predicted intent partition], MRR = 0.965) |
| **Generation Layer** | Pluggable coordinator: Groq (Llama-3.1-8b), Ollama, OpenAI, Gemini, Claude, with offline precedent fallback |
| **Safety Barriers** | 5 deterministic validators: ungrounded pricing ($), URL allowlisting, public PII, hazard detection, lexical overlap (**predictable, auditable first-line defense**) |
| **Escalation Policy** | Multi-factor routing on intent confidence, evidence similarity, risk phrases, and validation checks (**94.8% Precision**, **96.2% Recall**, **Safety Hazard Recall: 100% on the evaluated explicit acute-hazard subset** vs. historical policy) |
| **Evaluation Harness** | 200 hand-verified Golden Set samples benchmarked against Trivial Majority and TF-IDF LogReg baselines |
| **Human Evaluation Benchmark** | **Response Quality ($N=100$)**: Human Evaluator Mean **3.21 / 5.00** vs. LLM Judge Mean **4.24 / 5.00** (MAE 1.39 pts, Spearman $\rho = 0.0565, p = 0.260$);<br>**Routing Decision ($N=200$)**: Observed Agreement **38.00%**, Cohen's $\kappa = \mathbf{-0.0562}$ (*Very Low / Below Chance-Adjusted Agreement*), with **106 human-escalation cases auto-handled** |
| **Historical Policy Concordance** | Automated system routing rules vs. historical dataset triage policy: **89.5% concordance**, **Cohen's $\kappa = 0.8118$** (distinguished from human evaluation) |
| **Interface** | Full-featured FastAPI web application with Vanilla CSS (Simulate workspace, Support Inbox, Evaluation Dashboard, Failure Modes, Decision Log) |

---

## Table of Contents

- [TL;DR / At a Glance](#tldr--at-a-glance)
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Module Navigation](#module-navigation)
- [Why This Approach](#why-this-approach)
- [Dataset & Corpus Selection](#dataset--corpus-selection)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [System Pipeline Components](#system-pipeline-components)
  - [1. Intent Classification & Disambiguation](#1-intent-classification--disambiguation)
  - [2. Grounded Retrieval (Dense FAISS)](#2-grounded-retrieval-dense-faiss)
  - [3. Grounded Generation](#3-grounded-generation)
  - [4. Deterministic Response Validation](#4-deterministic-response-validation)
  - [5. Escalation Policy Engine](#5-escalation-policy-engine)
- [Evaluation Methodology](#evaluation-methodology)
  - [Golden Evaluation Set Profile ($N=200$)](#golden-evaluation-set-profile-n200)
  - [Human Evaluation Protocol ($N=100$ Quality, $N=200$ Routing)](#human-evaluation-protocol-n100-quality-n200-routing)
  - [Metric Justification](#metric-justification)
  - [Validation Experiments (Probe Tests)](#validation-experiments-probe-tests)
- [Benchmark Results vs. Baselines](#benchmark-results-vs-baselines)
  - [Intent Classification & Routing vs Baselines](#intent-classification--routing-vs-baselines)
  - [Dense Retrieval Performance & Intent-Conditioning Disclosure](#dense-retrieval-performance--intent-conditioning-disclosure)
  - [Human Evaluation & Judge Calibration Results](#human-evaluation--judge-calibration-results)
    - [Human vs. LLM Response Quality ($N=100$)](#human-vs-llm-response-quality-n100)
    - [Human vs. System Routing Agreement ($N=200$)](#human-vs-system-routing-agreement-n200)
    - [Historical System Policy Concordance vs. Human Evaluation](#historical-system-policy-concordance-vs-human-evaluation)
- [Top 5 Empirical Failure Modes](#top-5-empirical-failure-modes)
  - [Primary Empirical Human-Evaluation Failure Modes](#primary-empirical-human-evaluation-failure-modes)
  - [Secondary Engineering & Environmental Failure Modes](#secondary-engineering--environmental-failure-modes)
- [What is Misleading About My Headline Number? / Limitations](#what-is-misleading-about-my-headline-number--limitations)
- [Engineering Trade-Offs](#engineering-trade-offs)
- [How to Run / Reproducibility (< 15 Minutes)](#how-to-run--reproducibility--15-minutes)
- [Real End-to-End Example Walkthrough](#real-end-to-end-example-walkthrough)
- [Testing & CI/CD Pipeline](#testing--cicd-pipeline)
- [Future Work Tied to Actual Limitations](#future-work-tied-to-actual-limitations)
- [AI Tools Used](#ai-tools-used)

---

## Project Overview

The system processes incoming customer inquiries through a deterministic 7-step lifecycle:

```
[Inquiry Ingest] ➔ [Intent Classification] ➔ [Vector Retrieval] ➔ [Prompt Assembly] ➔ [Reply Drafting] ➔ [Safety Barrier Validation] ➔ [Routing Action]
```

1. **Inbound Ingestion**: The raw customer inquiry is sanitized; customer handles (`@user`) are extracted to preserve natural addressing, and formatting artifacts are normalized.
2. **Intent Classification**: A priority-disambiguated 7-class classifier assigns a domain category and confidence score in ~2 ms.
3. **Intent-Conditioned Dense Retrieval**: The classified intent partitions the search space; `sentence-transformers/all-MiniLM-L6-v2` embeds the query to retrieve the top-$k$ most similar resolved historical precedents from a FAISS `IndexFlatIP` index ($\text{sim} \ge 0.55$). *(Note: Conditioned Recall@3 = 96.5% on predicted intent partition; unconditioned Recall@3 = 77.5%).*
4. **Grounded Prompt Synthesis**: Retrieved historical resolutions, customer handle, and anti-hallucination constraints are assembled into an evidence prompt.
5. **Multi-Engine Response Generation**: Generation is dispatched to the active provider (Groq Llama-3.1-8b, local Ollama, OpenAI, Gemini, or Claude). If cloud providers fail, the system falls back to the top grounded historical precedent.
6. **Deterministic Validation Barriers**: The draft reply must pass 5 explicit safety checks: ungrounded pricing detection (`$\d+`), URL allowlisting, public PII solicitation prevention, physical hazard precautions, and lexical overlap auditing.
7. **Escalation Routing Action**: The policy engine evaluates validation results, intent confidence ($\ge 0.80$), retrieval similarity ($\ge 0.60$), and critical risk phrases to produce the final routing action (`AUTO_HANDLE` vs `HUMAN_ESCALATION`) with an audit-logged reason.

---

## System Architecture

```mermaid
flowchart TD
    inbound["Incoming Customer Inquiry<br/>(Tweet / Ticket Payload)"] --> intake["1. Intake & Handle Resolution<br/>(Extract customer handle @user)"]
    intake --> intent["2. Intent Classification<br/>(7-Class Domain Taxonomy + Priority Disambiguation)"]

    subgraph RAG ["Grounded Knowledge Retrieval"]
        intent --> filter["Intent-Conditioned Query Filter"]
        filter --> embed["Dense Vector Embedding<br/>(all-MiniLM-L6-v2)"]
        embed --> faiss[("FAISS IndexFlatIP<br/>(2,245 Resolved Precedents)")]
        faiss --> ranker["Precedent Ranker<br/>(Cosine Sim >= 0.55 + Resolution Status)"]
    end

    ranker --> prompt["3. Grounded Prompt Synthesis<br/>(Inject top-k historical precedents + Strict anti-hallucination rules)"]

    subgraph LLM ["Pluggable Inference Coordinator"]
        prompt --> router{"Active Engine"}
        router -->|Cloud High-Speed| groq["Groq (Llama-3.1-8b)"]
        router -->|Local Self-Hosted| ollama["Ollama (Llama-3.1:8b)"]
        router -->|Commercial Cloud| cloud["OpenAI / Gemini / Claude"]
        router -->|Offline Sandbox / Fallback| fallback["Grounded Precedent Fallback"]
    end

    groq --> draft["Drafted Brand Response<br/>(No markdown asterisks, natural @handle greeting)"]
    ollama --> draft
    cloud --> draft
    fallback --> draft

    subgraph Safety ["Deterministic Guardrails & Policy"]
        draft --> v1["Barrier 1: Ungrounded Pricing ($)"]
        v1 --> v2["Barrier 2: Unauthorized External URLs"]
        v2 --> v3["Barrier 3: Public PII Solicitation (Passwords/SSN)"]
        v3 --> v4["Barrier 4: Dangerous Hardware Hazards (Swollen Battery)"]
        v4 --> v5["Barrier 5: N-Gram Evidence Grounding Overlap"]
        v5 --> valResult{"All Checks Passed?"}
    end

    valResult -->|Passed| policy["Escalation Policy Engine<br/>(Intent Conf >= 0.80 and Sim >= 0.60, Risk-Free)"]
    valResult -->|Failed| humanEscalate["Route: HUMAN_ESCALATION<br/>(Safety / Policy Violation Reason Stated)"]

    policy -->|Eligible| autoHandle["Route: AUTO_HANDLE<br/>(Automated Dispatch Approved)"]
    policy -->|Risk / Low Conf| humanReview["Route: HUMAN_ESCALATION<br/>(Borderline Signals Stated)"]

    autoHandle --> dispatch["Structured Output Payload & UI Workspaces<br/>(JSON API / SSE Stream / Operational Workspaces)"]
    humanReview --> dispatch
    humanEscalate --> dispatch
```

---

## Module Navigation

| Module | Primary File | Responsibility |
| :--- | :--- | :--- |
| **Agent Orchestrator** | [`app/services/agent/agent_orchestrator.py`](app/services/agent/agent_orchestrator.py) | Coordinates the end-to-end 7-stage execution lifecycle and timing telemetry |
| **Intent Classifier** | [`app/services/intent/intent_classifier.py`](app/services/intent/intent_classifier.py) | 7-class taxonomy classifier with hierarchical priority disambiguation rules |
| **Baseline Classifiers** | [`app/services/intent/baseline_classifiers.py`](app/services/intent/baseline_classifiers.py) | Majority class and TF-IDF + Logistic Regression benchmark baselines |
| **Dense Retriever** | [`app/services/retrieval/retriever.py`](app/services/retrieval/retriever.py) | SentenceTransformer query embedding and FAISS inner-product similarity search |
| **Evidence Ranker** | [`app/services/retrieval/evidence_ranker.py`](app/services/retrieval/evidence_ranker.py) | Filters candidates by resolution status and similarity threshold ($\text{sim} \ge 0.55$) |
| **Prompt Builder** | [`app/services/generation/prompt_builder.py`](app/services/generation/prompt_builder.py) | Constructs few-shot prompts with dynamic handle insertion and anti-hallucination directives |
| **LLM Service** | [`app/services/generation/llm_service.py`](app/services/generation/llm_service.py) | Multi-provider manager with automatic circuit breaking and precedent extraction |
| **Provider Factory** | [`app/services/generation/provider_factory.py`](app/services/generation/provider_factory.py) | Dynamic registry for Groq, Ollama, OpenAI, Gemini, Claude, and Grounded Precedent |
| **Response Validator** | [`app/services/validation/response_validator.py`](app/services/validation/response_validator.py) | Enforces 5 deterministic safety checks (pricing, URLs, PII, physical hazards, overlap) |
| **Escalation Policy** | [`app/services/escalation/escalation_policy.py`](app/services/escalation/escalation_policy.py) | Deterministic routing logic evaluating intent, similarity, risk phrases, and validation |
| **LLM-as-a-Judge** | [`app/services/evaluation/judge_service.py`](app/services/evaluation/judge_service.py) | 4-dimension 1–5 rubric evaluator assessed against human ratings via MAE, exact/within-1 agreement, and Spearman rank correlation ($\rho$) |
| **Golden Repository** | [`app/repositories/golden_set_repository.py`](app/repositories/golden_set_repository.py) | Ingests and queries the 200 hand-verified Golden Set evaluation benchmark cases |
| **Web Server & UI** | [`app/main.py`](app/main.py) | FastAPI application mounting API endpoints, SSE streams, and Jinja2 visual workspaces |

---

## Why This Approach

| Design Choice | What We Built | Why We Built It This Way (Rationale) | What We Explicitly Avoided |
| :--- | :--- | :--- | :--- |
| **Target Brand** | `@AppleSupport` | Apple handles technical hardware diagnostics, software update verification loops, billing refunds, and acute hardware hazards. This provides a genuine stress-test for intent disambiguation and safety guardrails. | Generic telecom/airline brands where 90% of replies are identical canned triage (*"Please DM your ticket number"*). |
| **Vector Index** | FAISS `IndexFlatIP` | With 2,245 historical support cases, exact inner product on normalized embeddings executes in **< 1 ms** with 100% recall. Approximate indices (IVF, HNSW) introduce recall degradation for zero practical benefit at this corpus size. | Approximate nearest neighbor indexing (IVF, HNSW) or external cloud vector databases adding unnecessary latency and cost. |
| **Classification** | Priority Rule Taxonomy | Classifies inquiries in **~2 ms** with 0 token cost, deterministic execution, and zero cold-start failure modes. Disambiguation rules enforce safety (swollen batteries always override software updates). | Pure few-shot LLM classification that costs tokens on every request, adds 500ms latency, and exhibits non-deterministic classification drift. |
| **Safety Barriers** | Deterministic Regex & Set Checks | Fact-checking, pricing audit, URL allowlisting, and public PII protection execute via hardcoded deterministic rules *before* message dispatch. | Relying on LLM self-correction or prompt directives ("please do not hallucinate prices"), which routinely fail edge-case jailbreaks. |
| **Multi-Provider Coordinator** | Pluggable Factory with Circuit Breakers | Supports Groq, local Ollama, OpenAI, Gemini, and Claude with automated fallback. If all external APIs fail, the system falls back to the top grounded precedent rather than throwing an HTTP 500. | Single-vendor lock-in that breaks when an API goes down or rate limits are reached. |
| **Frontend UI** | Vanilla CSS & JavaScript | Zero npm build steps, zero node_modules dependencies, and instant browser rendering with Server-Sent Events (SSE) streaming. | Heavy client-side JavaScript frameworks (React, Next.js) that introduce build bloat for an operational dashboard. |

---

## Dataset & Corpus Selection

### Dataset Sources & Repository Locations

| Dataset Role | Dataset Name & Location | Source / Platform | Volume & Scope | Characteristics & Application |
| :--- | :--- | :--- | :--- | :--- |
| **Primary** | **Customer Support on Twitter**<br>• Kaggle: `thoughtvector/customer-support-on-twitter`<br>• Local Path: `data/raw/twcs.csv`<br>• Sample: `data/samples/sample_twcs.csv` | [Kaggle Dataset](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) | **~3M tweets** (2,811,774 rows), 108 brands, multi-turn threads | **Real, noisy, and imperfect.** Used for end-to-end conversation reconstruction, `@AppleSupport` intent classification, dense retrieval indexing, and grounded reply generation. |
| **Secondary** *(Optional)* | **Banking77**<br>• Hugging Face: `PolyAI/banking77`<br>• Local Path: `data/external/banking77` | [Hugging Face PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77) | **13k queries** (13,082 samples), 77 labelled intents | **Intent classification benchmarking only.** Clean single-turn queries used as an intent taxonomy baseline reference; excluded from retrieval/generation because social tech support requires multi-turn hardware and OS troubleshooting. |

> **Model Flexibility**: The system is engineered to work with **any LLM API or open model** via a unified provider interface. Users can seamlessly configure Groq (Llama-3.1/3.2), local Ollama, OpenAI (GPT-4o), Google Gemini, Anthropic Claude, or use the built-in offline `GroundedPrecedentProvider` with zero API keys required.

### Local Repository Data Layout
```
data/
├── raw/                 # Primary raw twcs.csv from Kaggle (~600MB uncompressed, optional)
├── samples/             # Committed offline sample dataset (sample_twcs.csv) for test environments
├── splits/              # Stratified leak-free splits (train.jsonl, val.jsonl, test.jsonl)
├── golden/              # 200 hand-verified Golden Set evaluation benchmark cases (golden_set.jsonl)
├── taxonomy/            # 7-class domain intent taxonomy definitions (intent_taxonomy.json)
└── external/            # Secondary dataset storage (e.g., Banking77 reference cache)
```

### Primary Dataset Profile (`thoughtvector/customer-support-on-twitter`)
The primary Twitter customer support corpus contains real-world multi-turn customer-brand interactions spanning 2008 to 2017:

| Metric | Raw Dataset Volume | Profile & Characteristics |
| :--- | :---: | :--- |
| **Total Tweets** | **2,811,774** | 1,537,843 customer inbound (54.7%) / 1,273,931 brand outbound (45.3%) |
| **Unique Customers** | **702,669** | High diversity of phrasing, typos, and emotional states |
| **Unique Brands** | **108** | Multi-industry distribution (retail, airlines, tech, telecom) |
| **Temporal Span** | **3,496 days** | May 2008 to December 2017 |
| **Target Brand Volume** | **106,860 tweets** | `@AppleSupport` is the **#2 most active brand overall**, offering dense technical dialogue |

### Noise Characteristics in the Raw Primary Data
1. **Thread Fragmentation**: Multi-turn exchanges are stored as flat rows linked by `in_response_to_tweet_id` and `response_tweet_id`, with 28.2% missing parent IDs.
2. **Entity Masking Artifacts**: The raw dataset replaces usernames with numeric IDs (`@115858`) and masks internal ticket numbers, creating synthetic text artifacts.
3. **Dead Link Proliferation**: Historical tweets contain thousands of dead short-links (`apple.co/2xyz`) that no longer resolve.
4. **Canned Non-Resolutions**: A large portion of raw tweets are single-sentence handoffs (*"Send us a DM and we'll take a look"*), requiring resolution filtering before indexing.

---

## Data Preprocessing Pipeline

The preprocessing pipeline ([`scripts/data/preprocess_conversations.py`](scripts/data/preprocess_conversations.py)) converts raw tweets into clean, reconstructed multi-turn conversations:

1. **Brand Ingestion & Filtering**: Isolates `@AppleSupport` tweets from the 2.81M corpus.
2. **Dialogue Graph Reconstruction**: Assembles conversation trees by tracing `in_response_to_tweet_id` chains, grouping root customer inquiries with subsequent turns.
3. **Handle Normalization & Extraction**: Extracts real customer handles from mentions while cleaning numeric Twitter artifacts.
4. **URL Normalization**: Rewrites dead link patterns and normalizes official Apple documentation paths (`support.apple.com`).
5. **Resolution Status Tagging**: Evaluates conversation terminal turns to tag whether the issue reached technical troubleshooting resolution vs private channel escalation (`PRIVATE_CHANNEL_HANDOFF`).
6. **Stratified Splitting (Zero Leakage)**: Isolates the 200 Golden Set cases first using seed 42, then splits the remaining 3,223 conversations into 70% Train, 15% Validation, and 15% Test.

| Data Split | Conversation Count | Percentage | Purpose |
| :--- | :---: | :---: | :--- |
| **Train Split** | 2,256 | 70.0% | Source pool for dense FAISS vector index (2,245 indexed cases) |
| **Validation Split** | 483 | 15.0% | Hyperparameter tuning (similarity thresholds, confidence cutoffs) |
| **Test Split** | 484 | 15.0% | Held-out statistical evaluation for baseline classifiers |
| **Golden Set** | **200** | — | Strictly held-out hand-verified evaluation benchmark (never indexed) |

---

## System Pipeline Components

### 1. Intent Classification & Disambiguation
Incoming inquiries are classified into a 7-class domain taxonomy using keyword and regex pattern matching combined with strict priority disambiguation:

| Intent Code | Description | Example Query | Priority Rules Enforced |
| :--- | :--- | :--- | :--- |
| `OPERATING_SYSTEM_UPDATES` | iOS/macOS update verification loops, install crashes, app freezing | *"My iPhone has been stuck on 'Verifying update' for iOS 11 for 3 hours."* | Post-update battery drain classified under OS Updates |
| `BATTERY_POWER_HARDWARE` | Rapid battery drop, thermal runaway, shutdowns, physical battery swelling | *"My iPhone battery swelled up and popped the screen off."* | **Rule 1 (Safety Preempts All)**: Overrides OS updates on physical hazard |
| `ACCOUNT_APPLE_ID` | 2FA lockout, forgotten passwords, Apple ID activation lock | *"I am locked out of my Apple ID because I changed my phone number."* | Overrides generic OS queries when authentication is blocked |
| `CONNECTIVITY_NETWORKING` | Wi-Fi disconnects, Bluetooth pairing, cellular carrier drops | *"Why does my Wi-Fi keep disconnecting every time my phone locks?"* | General network settings |
| `AUDIO_ACCESSORIES` | AirPods charging, single earbud failure, static sound | *"My right AirPod won't connect or charge in the case."* | **Rule 3 (Accessory Precedence)**: Overrides generic Bluetooth |
| `SUBSCRIPTIONS_BILLING` | Unauthorized iTunes charges, recurring subscription refunds | *"I was charged $9.99 on my bank statement from itunes.com/bill."* | **Rule 2 (Financial Precedence)**: Overrides generic account queries |
| `GENERAL_INQUIRY` | Store appointments, trade-in policies, general compatibility | *"What are the Apple Store Regent Street opening hours on Sunday?"* | Fallback category for non-technical requests |

### 2. Grounded Retrieval (Dense FAISS)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2-normalized).
- **Index**: FAISS `IndexFlatIP` performing exact cosine similarity search over 2,245 historical `@AppleSupport` precedents.
- **Intent Conditioning & Disclosure**: Candidate search is partitioned by the predicted intent, preventing cross-domain noise (e.g. retrieving billing precedents for a battery swelling issue). **Disclosure**: The 96.5% Recall@3 metric is strictly conditioned on the predicted intent partition; when evaluated unconditionally across the entire global index, Recall@3 is **77.5%** (MRR = 0.723).
- **Thresholding**: Filters out candidate precedents with cosine similarity $< 0.55$.

### 3. Grounded Generation
The prompt synthesis layer injects top-$k$ retrieved precedents directly into the system context alongside strict operational constraints:
- **Zero Markdown Asterisks**: Twitter does not render markdown bold (`**word**`), so formatting asterisks are explicitly stripped.
- **Dynamic Customer Handle Addressing**: Responses address the customer naturally by extracted handle (`@username`).
- **Precedent-Grounded Resolution**: Instructions must derive from steps verified in the retrieved precedents.

### 4. Deterministic Response Validation
Before message dispatch, draft replies pass through 5 deterministic barriers in [`ResponseValidator`](app/services/validation/response_validator.py):
1. **Ungrounded Pricing Barrier**: Flags any currency token (`$\d+`) not explicitly attested in the retrieved precedent.
2. **URL Allowlist Barrier**: Restricts links to official Apple domains (`support.apple.com`, `appleid.apple.com`, `locate.apple.com`).
3. **Public PII Barrier**: Blocks any request soliciting passwords, credit cards, or serial numbers publicly on Twitter.
4. **Physical Hazard Barrier**: When inquiry mentions hardware hazards (swelling, smoke, thermal event), enforces safety precautions (unplug device, stop charging, seek authorized service).
5. **N-Gram Lexical Overlap**: Audits word overlap against retrieved evidence, flagging responses below 15% overlap.

### 5. Escalation Policy Engine
The policy engine decides whether the inquiry is eligible for automated resolution (`AUTO_HANDLE`) or requires human review (`HUMAN_ESCALATION`):
- **Triggers for Human Escalation**:
  - Any safety barrier violation (pricing, untrusted URL, PII solicitation, physical hazard).
  - Intent classification confidence $< 0.80$.
  - Vector retrieval similarity $< 0.60$ (insufficient historical evidence).
  - Customer distress keywords (legal action, repeated troubleshooting failure, abusive language).
  - Explicit customer request for human support.

---

## Evaluation Methodology

### Golden Evaluation Set Profile ($N=200$)
Curated under [`data/golden/golden_set.jsonl`](data/golden/golden_set.jsonl), this benchmark represents **200 hand-verified, leak-free customer conversations** with zero synthetic data.

**Sampling Rationale**: A naive random sample of Twitter support data is 80%+ trivial noise. To rigorously test system limits, we used a two-stage stratified sampling approach with targeted edge-case injection:
- **Stratified Intent Coverage**: Balances minority classes (Billing: 20, Accessories: 20) alongside high-volume classes (OS Updates: 40).
- **Dialogue Depth Representation**: 80 multi-turn conversations (40%) and 120 single-turn inquiries (60%).
- **Deliberate Edge-Case Injection**: 66 complex cases (33% of the benchmark):
  - **28 Acute Edge Cases**: Swollen batteries, thermal runaway, shattered glass, stolen Apple IDs.
  - **35 Multi-Turn Escalations**: Threads where customers attempted multiple troubleshooting steps without success.
  - **3 Conflict Queries**: Cross-domain inquiries testing priority disambiguation.

| Intent Code | Golden Samples | Stratification Focus | Edge Cases Tested |
| :--- | :---: | :--- | :--- |
| `OPERATING_SYSTEM_UPDATES` | 40 | iOS 11 update verify loops, autocorrect glitches, app freezing | Post-update battery drain disambiguation |
| `BATTERY_POWER_HARDWARE` | 35 | Rapid battery percentage drop, thermal runaway, shutdowns | Swollen battery physical safety hazards |
| `ACCOUNT_APPLE_ID` | 30 | 2FA lockout, forgotten passwords, activation locks | Identity verification, stolen account recovery |
| `CONNECTIVITY_NETWORKING` | 25 | Wi-Fi drops, carrier cellular failure, Bluetooth pairing | Bluetooth pairing drop vs. audio accessory failures |
| `AUDIO_ACCESSORIES` | 20 | AirPods charging failure, single earbud sound loss | Hardware sound loss vs. generic Bluetooth settings |
| `SUBSCRIPTIONS_BILLING` | 20 | Unauthorized App Store charges, recurring subscriptions | In-app purchase fraud disputes, bank statement charges |
| `GENERAL_INQUIRY` | 30 | Store appointments, trade-in policies, warranty terms | Broad multi-intent queries lacking technical keywords |
| **Total Golden Set** | **200** | **Balanced across 7 MECE categories** | **66 complex / edge / multi-turn cases (33%)** |

### Human Evaluation Protocol ($N=100$ Quality, $N=200$ Routing)

To rigorously audit the system rather than relying solely on automated benchmarks, a human evaluation study was conducted:
1. **Response Quality Study ($N=100$)**: A stratified sample of 100 system-generated responses was scored by a human evaluator alongside the automated LLM Judge (`judge_service.py`). Neither evaluator had access to the other's scores. Both scored responses across four standardized dimensions on a 1–5 scale: Groundedness & Faithfulness, Answer Relevance & Actionability, Brand Voice & Empathy, and Safety & Policy Compliance.
2. **Routing Agreement Study ($N=200$)**: All 200 Golden Set conversations were reviewed by a human evaluator without exposure to system predictions, assigning either `AUTO_HANDLE` or `HUMAN_ESCALATION` based on inquiry complexity, customer distress, and diagnostic risk.
3. **Zero Post-Hoc Tuning**: To preserve evaluation integrity, no prompt rubrics or routing thresholds were re-tuned post-hoc using human ratings.

### Metric Justification

| Metric | Why It Exists (Engineering Purpose) | What It Measures |
| :--- | :--- | :--- |
| **Accuracy** | Standard classification correctness | Overall proportion of correctly predicted intents |
| **Macro F1** | Evaluates minority class performance | Unweighted mean of class F1-scores; prevents high-volume OS updates from masking failures on low-volume Billing queries |
| **Recall@K (K=1,3,5)** | Evaluates retrieval coverage | Proportion of queries where a relevant historical precedent appears in the top-$K$ candidates |
| **MRR (Mean Reciprocal Rank)** | Measures retrieval rank quality | Penalizes retrieval systems that place the relevant precedent lower in the ranked list |
| **Escalation Precision** | Prevents agent queue overflow | Ensures that when the system escalates, human agents receive genuinely complex or risky inquiries |
| **Escalation Recall** | Prevents customer safety incidents | Ensures that high-risk inquiries (hardware swelling, financial fraud) never get auto-handled |
| **Cohen's Kappa ($\kappa$)** | Evaluates inter-rater agreement above chance: $\kappa = \frac{P_o - P_e}{1 - P_e}$ | Distinguishes historical policy heuristic agreement ($\kappa = 0.8118$) from human routing agreement ($\kappa = -0.0562$) |
| **Spearman Rho ($\rho$)** | Evaluates monotonic rank correlation | Measures alignment between continuous automated judge scores and ordinal human evaluator ratings |

### Validation Experiments (Probe Tests)
The test suite ([`tests/unit/test_validation.py`](tests/unit/test_validation.py)) executes explicit probe tests against known failure scenarios:
1. **Pricing Hallucination Probe**: Injects drafted replies containing `$199 repair fee` without precedent support &rarr; Intercepted by Barrier 1, escalated to human.
2. **Untrusted URL Phishing Probe**: Injects third-party domain link `http://apple-fix.com` &rarr; Intercepted by Barrier 2, escalated to human.
3. **Public PII Solicitation Probe**: Injects reply asking customer for password/SSN publicly &rarr; Intercepted by Barrier 3, escalated to human.
4. **Physical Hazard Safety Probe**: Injects query with *"battery is swollen and hot"* &rarr; Intercepted by Barrier 4, forces safety advisory and human escalation.

---

## Benchmark Results vs. Baselines

### Intent Classification & Routing vs Baselines
Benchmarked across the 200 Golden Set evaluation samples:

| Model / Pipeline | Intent Accuracy | Macro F1 | Escalation Precision | Escalation Recall | Escalation F1 | Historical Routing Concordance | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Majority** | 15.0% | 0.037 | 42.0% | 100.0% | 0.592 | 32.0% | < 1 ms |
| **Baseline 2: TF-IDF + LogReg** | 71.0% | 0.674 | 81.2% | 76.5% | 0.788 | 74.5% | ~ 3 ms |
| **Evaluated System (Ours)** | **88.0%** | **0.867** | **94.8%** | **96.2%** | **0.955** | **89.5%** | **~ 2 ms** |

> **Historical Routing Benchmark vs. Human Routing**: The 89.5% routing concordance above measures agreement against the *automated historical system policy heuristics* from the dataset era ($\kappa = 0.8118$). As detailed in the human evaluation below, when evaluated against a **human evaluator** on the exact same 200 cases, actual operational agreement is **38.00%** ($\kappa = -0.0562$, *Very Low / Below Chance-Adjusted Agreement*), with **106 human-escalation cases auto-handled** due to contextual nuance.
>
> **Safety Hazard Recall**: **Acute Safety Hazard Recall: 100% on the evaluated explicit hazard-trigger subset** (explicit swollen batteries, thermal runaway, smoke triggers). This metric applies strictly to the evaluated explicit acute-hazard subset and does not guarantee zero unsafe auto-handling across unmodeled or borderline safety cases (such as subtle thermal degradation without explicit hazard keywords, as documented in the human routing audit).

### Dense Retrieval Performance & Intent-Conditioning Disclosure
Evaluated on the 200 Golden Set samples against the FAISS index:

| Retrieval Configuration | Recall@1 | Recall@3 | Recall@5 | MRR | Mean Top-1 Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dense FAISS (Unconditioned)** | 65.5% | 77.5% | 83.5% | 0.723 | 0.660 |
| **Dense FAISS (Intent-Conditioned)** | **96.5%** | **96.5%** | **96.5%** | **0.965** | **0.634** |

> **Intent-Conditioned Retrieval Disclosure**: The **96.5% Recall@3** metric is strictly conditioned on the predicted intent partition (candidate retrieval is partitioned to search only precedents matching the upstream intent). It should **NOT** be treated as an independent retrieval-only number. If upstream intent classification were to misclassify, retrieval would be bounded by that domain error. Across the unpartitioned global corpus, dense FAISS **Unconditioned Recall@3 is 77.5%** (MRR = 0.723).

### Human Evaluation & Judge Calibration Results

A central finding of this project is that **in this evaluation, automated metrics and the LLM judge diverged substantially from human judgment.** The tables below document the empirical audit results.

#### Human vs. LLM Response Quality ($N=100$)
Source: [`experiments/human_judge_agreement.json`](experiments/human_judge_agreement.json)

| Evaluation Dimension | Human Evaluator Mean | LLM Judge Mean | Discrepancy (MAE) | Exact Match | Within-1-Pt | Spearman $\rho$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Groundedness & Faithfulness** | **3.35** | **4.03** | 1.44 pts | 26.0% | 59.0% | 0.0051 | 0.960 |
| **Answer Relevance & Actionability** | **3.01** | **3.57** | **0.90 pts** | 33.0% | **80.0%** | 0.0396 | 0.695 |
| **Brand Voice & Empathy** | **2.94** | **4.37** | **1.73 pts** | 13.0% | 49.0% | **-0.2022** | 0.044 |
| **Safety & Policy Compliance** | **3.53** | **5.00** | 1.47 pts | 40.0% | 61.0% | 0.0000* | 1.000 |
| **Overall Mean Quality Score** | **3.21 / 5.00** | **4.24 / 5.00** | **1.39 pts** | — | — | **0.0565** | 0.260 |

*\*Note: Safety Spearman correlation is mathematically 0.0000 due to zero variance in LLM judge outputs (saturated at 5.00).*

#### Human vs. System Routing Agreement ($N=200$)
Source: [`experiments/human_routing_benchmarks.json`](experiments/human_routing_benchmarks.json) & [`experiments/human_routing_audit.json`](experiments/human_routing_audit.json)

- **Observed Agreement ($P_o$)**: **38.00%**
- **Expected Agreement by Chance ($P_e$)**: **41.30%**
- **Cohen's Kappa ($\kappa$)**: **-0.0562** (*Very Low / Below Chance-Adjusted Agreement*)
- **Per-Class Breakdown**:
  - `AUTO_HANDLE` (Support: 42): Precision: **0.1846**, Recall: **0.5714**, F1: **0.2791**
  - `HUMAN_ESCALATION` (Support: 158): Precision: **0.7429**, Recall: **0.3291**, F1: **0.4561**

##### Operational Routing Confusion Matrix:
| Ground Truth (Human Evaluator) \ System Decision | System: `AUTO_HANDLE` | System: `HUMAN_ESCALATION` | Total (Human) |
| :--- | :---: | :---: | :---: |
| **Human: `AUTO_HANDLE`** | **24** | 18 | 42 |
| **Human: `HUMAN_ESCALATION`** | **106** *(Under-escalations)* | **52** | 158 |
| **Total (System)** | 130 | 70 | 200 |

#### Historical System Policy Concordance vs. Human Evaluation

It is essential to distinguish between the two different routing concordance metrics:
- **Historical System Policy Concordance (89.5% agreement, $\kappa = 0.8118$)**: Measures agreement between the agent's deterministic escalation rules and the *automated heuristic triage policy* modeled on historical Twitter support practices. This indicates strong engineering fidelity to the historical rule set.
- **Human Routing Evaluation (38.00% agreement, $\kappa = -0.0562$)**: Measures agreement against a *human evaluator* reviewing each inquiry for real-world complexity, customer distress, and diagnostic uncertainty. The human evaluator deemed 158 inquiries to require human escalation, whereas the automated policy auto-handled 106 of them.

**Key Insight**: High automated policy concordance does **NOT** mean the system agrees with human operational judgments. Automated thresholds based on confidence and similarity regularly miss subtle contextual frustration and diagnostic edge cases.

---

## Top 5 Empirical Failure Modes

### Primary Empirical Human-Evaluation Failure Modes

As documented in [`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md), auditing the system through human evaluation uncovered 5 primary empirical failure modes:

1. **Failure Mode 1: LLM Judge Overestimates Overall Response Quality**
   - *Empirical Evidence*: Human Evaluator Mean **3.21 / 5.00** vs. Automated LLM Judge Mean **4.24 / 5.00** (MAE = 1.39 pts, Spearman $\rho = 0.0565, p = 0.260$).
   - *Why It Matters*: An automated judge reporting 4.24/5 creates a dangerous false confidence that the system is production-ready, whereas human evaluators rate the exact same responses at a mediocre 3.21/5. Relying on the automated judge without human calibration risks deploying an agent that frustrates real users.
   - *Working Hypothesis*: Prompted LLM judges are biased toward surface-level fluency, grammatically polite structures, and standard customer support pleasantries. It rarely penalizes responses unless blatant toxicity or hallucinations occur, whereas human evaluators demand concise, genuinely helpful problem resolution.
   - *Proposed One-Week Mitigation*: Train an ordinal calibration adapter or tune few-shot judge prompts using the 100 human evaluation samples as few-shot calibration anchors, establishing explicit scoring penalties for evasive, non-committal answers.

2. **Failure Mode 2: Brand Voice & Empathy Mismatch**
   - *Empirical Evidence*: Human Evaluator Mean **2.94 / 5.00** vs. LLM Judge Mean **4.37 / 5.00** (Discrepancy MAE = 1.73 pts; Exact Match = 13.0%; Spearman $\rho = \mathbf{-0.2022}, p = 0.044$).
   - *Why It Matters*: The observed negative association ($\rho = -0.2022$, $p = 0.044$) is consistent with a mismatch between the automated brand-voice rubric and human judgments, with canned/polite phrasing appearing to receive higher automated scores than human scores. Causal explanations remain working hypotheses.
   - *Real Artifact Example* (Case `gold_001` / `conv_38882`): Customer reported crackling sound on iPhone 6 on calls and speaker. The model produced a polite greeting and immediately redirected to DM (*"We'd be happy to help with the sound on your iPhone 6. To clarify, does this happen on all calls? Send us a DM."*). The LLM judge scored Tone 5/5 for brand politeness, whereas human evaluators scored 2/5 or 3/5 due to repetitive, scripted DM redirection without addressing hardware diagnostics.
   - *Working Hypothesis*: Generic, canned support phrasing (*"We're here to help! DM us your details"*) may satisfy the automated judge's surface politeness checklist, but human raters perceive it as robotic, evasive, or lacking genuine technical empathy.
   - *Proposed One-Week Mitigation*: Incorporate anti-canned-response penalties into the response generator prompt and implement a specialized empathy scoring rubric that rewards specific diagnostic guidance and penalizes generic redirection language.

3. **Failure Mode 3: Routing Under-Escalation (106 False-Auto Cases)**
   - *Empirical Evidence*: Human-vs-System Routing Agreement is only **38.00%** (Cohen's $\kappa = \mathbf{-0.0562}$, *Very Low / Below Chance-Adjusted Agreement*). Out of 158 cases where a human evaluator required human escalation, the system auto-handled **106 cases** (System Escalation Recall on Human Escalations = 32.91%).
   - *Why It Matters*: Under-escalation is a critical operational failure mode. Attempting to auto-handle inquiries involving severe hardware degradation, complex account lockouts, or repeated troubleshooting failures risks trapping customers in conversational dead-ends.
   - *Real Artifact Example* (Case `gold_003` / `conv_44498`): Customer's iPhone 6s battery percentage dropped wildly from 40% to 1% in seconds. The system decided `AUTO_HANDLE` (Intent: `BATTERY_POWER_HARDWARE`, Confidence: 0.96, Sim: 0.74, no safety violations). The human evaluator assigned `HUMAN_ESCALATION` because severe non-linear voltage drops indicate hardware battery failure requiring authorized physical repair.
   - *Supported Hypotheses*: (a) The system operates on the inbound message ($t=0$) rather than the full conversational lifecycle; (b) explicit escalation keyword triggers (`fraud`, `sue`, `lawyer`) miss contextual technical severity; (c) high intent confidence and retrieval similarity suppress escalation for complex scenarios.
   - *Proposed One-Week Mitigation*: Evaluate a tiered escalation policy conditioned on intent category, conversational severity, repeated troubleshooting failures, and explicit private-channel requirements.

4. **Failure Mode 4: Safety & Policy Compliance Judge Saturation**
   - *Empirical Evidence*: Automated LLM Judge scored **5.00 / 5.00** across all 100 cases (variance = 0.00), whereas Human Evaluator Mean was **3.53 / 5.00** (Discrepancy MAE = 1.47 pts; Spearman $\rho = 0.0000, p = 1.000$).
   - *Why It Matters*: A metric that evaluates 100% of outputs at ceiling is completely non-discriminative. It provides zero visibility into subtle policy edge cases, minor boundary crossings, or unhelpful advisories.
   - *Working Hypothesis*: The automated safety rubric prompt is calibrated exclusively to detect severe catastrophic failures (hate speech, public PII leaks, explicit explosion hazards). It fails to penalize borderline compliance issues (such as suggesting unverified third-party workarounds) that human raters penalize.
   - *Proposed One-Week Mitigation*: Re-engineer the safety judge prompt into a multi-tier deduction rubric that starts at 5.00 and deducts points for unverified links, ungrounded assertions, or missing mandatory safety disclaimers.

5. **Failure Mode 5: Groundedness & Factuality Disagreement**
   - *Empirical Evidence*: Human Evaluator Mean **3.35 / 5.00** vs. LLM Judge Mean **4.03 / 5.00** (Discrepancy MAE = 1.44 pts; Exact Match = 26.0%; Spearman $\rho = 0.0051, p = 0.960$).
   - *Why It Matters*: Grounding is the central architectural premise of retrieval-augmented generation. The near-zero rank correlation ($\rho = 0.0051$) shows that automated lexical overlap checks do not measure whether a response is actually grounded in a way that resolves the user's specific problem.
   - *Real Artifact Example* (Case `gold_005` / `conv_46261`): Customer wrote *"iOS 11.0.3 broke my Wi-Fi toggle switch, it's greyed out in settings."* The system retrieved generic Wi-Fi connection troubleshooting (Reset Network Settings) and advised resetting network settings. The LLM judge rated Groundedness 5/5 because the draft matched retrieved text. The human evaluator rated Groundedness 2/5 because a greyed-out Wi-Fi toggle is a known hardware chip failure that network resets cannot fix, meaning the retrieved precedent was factually mismatched to the root cause.
   - *Proposed One-Week Mitigation*: Replace superficial lexical grounding prompts with an NLI-based (Natural Language Inference) premise-hypothesis entailment check verifying directional logical support.

### Secondary Engineering & Environmental Failure Modes

In addition to the primary human evaluation failure modes above, the system incorporates explicit guards against 5 operational engineering failure modes:

6. **Failure Mode 6: Historical URL Drift**
   - *Real Example*: Retrieved 2017 historical precedent referencing `apple.co/2xyz` which is now a dead redirect.
   - *Root Cause*: Social support documentation URLs evolve over time; static historical answers become stale.
   - *Remedy*: Implemented URL canonicalization layer dynamically remapping legacy shortcut URLs to active `support.apple.com` documentation.
7. **Failure Mode 7: Multi-Turn Context Truncation**
   - *Real Example*: Customer writes *"It didn't work"*, referencing an earlier troubleshooting step in an unlinked tweet.
   - *Root Cause*: Inbound single-turn social messages lack context without conversation graph resolution.
   - *Remedy*: Low-confidence fallback triggers human escalation when inquiry length $< 20$ characters without clear intent.
8. **Failure Mode 8: Sarcasm and Negative Sentiment Masking**
   - *Real Example*: *"Oh fantastic, my phone updated and now it's a very expensive brick. Thanks Apple!"*
   - *Root Cause*: Lexical classifiers interpret words like "fantastic" and "thanks" as positive sentiment.
   - *Remedy*: Irony and risk phrase detection ("expensive brick") triggers immediate human escalation.
9. **Failure Mode 9: Hardware Generation Ambiguity**
   - *Real Example*: *"My iPad won't connect to the Apple Pencil"* (omitting whether it is Pencil 1st vs 2nd Gen).
   - *Root Cause*: Customers frequently omit device generation details required for accurate hardware troubleshooting.
   - *Remedy*: Disambiguation prompt asks customer for exact device generation before prescribing troubleshooting steps.
10. **Failure Mode 10: Partial Precedent Coverage for Multi-Intent Inquiries**
   - *Real Example*: *"My phone battery died during the iOS 11 update and now my screen is black."*
   - *Root Cause*: Vector search matches either the battery issue or the update issue, rarely both in a single precedent.
   - *Remedy*: Multi-intent priority disambiguation routes inquiry to `BATTERY_POWER_HARDWARE` to ensure physical safety first.

---

## What is Misleading About My Headline Number? / Limitations

> **Critical Engineering Audit**: Headline metrics in customer support benchmarks can easily mask production failure risks if accepted uncritically.

1. **Rule-Based 100% Accuracy is a Labeling Artifact**: The rule-based domain taxonomy achieved 100% on initial preprocessed splits because the training data was originally partitioned using the taxonomy definitions. When audited against human annotations on the Golden Set ($N=200$), rule-based accuracy is **98.50%** (Macro F1 = 0.9877), while statistical TF-IDF generalization is **69.50%** (84.92% on held-out test split).
2. **Intent Conditioning Artificially Boosts Retrieval (96.5% vs 77.5%)**: Intent-conditioned retrieval achieves 96.5% Recall@3 by restricting search candidates to the predicted intent partition. If upstream intent classification misclassifies, the retriever searches the wrong partition. True unconditioned retrieval recall across the global index is **77.50% Recall@3** (Recall@1 = 65.5%, Recall@5 = 83.5%, MRR = 0.723).
3. **LLM Judge 4.24 / 5.00 is Not Equivalent to Human Satisfaction**: While the automated LLM judge scored response quality at 4.24 / 5.00, human evaluation rated the exact same responses at **3.21 / 5.00** (MAE = 1.39 points). The automated judge has weak correlation with human judgment ($\rho = 0.0565, p = 0.260$) and systematically overlooks repetitive, robotic customer service phrasing.
4. **Historical Policy Concordance (89.5%, $\kappa = 0.8118$) Masks Human Routing Disagreement**: The high headline concordance metric measures agreement against automated heuristic rules. When evaluated against human routing decisions, actual agreement drops to **38.00%** ($\kappa = -0.0562$, *Very Low / Below Chance-Adjusted Agreement*), with **106 human-escalation cases auto-handled** due to single-turn visibility and narrow keyword triggers.
5. **The Private Channel Escalation Paradox**: Historical human support reps frequently escalated 79% of conversations to Direct Message (`PRIVATE_CHANNEL_HANDOFF`) simply to move traffic off public timelines, even for routine informational inquiries. An AI system configured to deliver direct public troubleshooting will naturally diverge from historical agent behavior that favored private DM handoffs, depressing apparent routing agreement on routine diagnostic inquiries.

---

## Engineering Trade-Offs

1. **Exact FAISS `IndexFlatIP` vs Approximate `IVF`/`HNSW`**:
   - *Chosen*: Exact inner product (`IndexFlatIP`).
   - *Sacrificed*: Sub-linear index scaling at multi-million vector scale.
   - *Rationale*: At 2,245 vectors, exact search takes **< 1 ms** with 100% recall. Approximate indexing would introduce false-negative recall drops for zero practical latency benefit.
2. **Rule-Based Priority Classifier vs Few-Shot LLM Classifier**:
   - *Chosen*: Rule-based taxonomy with priority disambiguation.
   - *Sacrificed*: Ability to understand esoteric slang without explicit pattern updates.
   - *Rationale*: Rule classifiers execute in **~2 ms** with 0 token cost, 100% predictable latency, and zero cold-start failures.
3. **Deterministic Safety Barriers vs LLM Self-Correction**:
   - *Chosen*: Hardcoded regex and set validation barriers.
   - *Sacrificed*: Nuanced contextual understanding of edge-case phrasing.
   - *Rationale*: LLM prompt directives ("do not quote prices") fail under adversarial probes. Deterministic barriers provide a **predictable, auditable first-line defense** against pricing and hazard liabilities.
4. **Precedent Fallback vs Failing Fast**:
   - *Chosen*: Graceful nearest-neighbor precedent fallback when external LLM APIs fail.
   - *Sacrificed*: Novel conversational phrasing under complete external API outages.
   - *Rationale*: In customer support, returning a proven historical resolution is infinitely better than throwing an HTTP 500 error or stalling the customer queue.

---

## How to Run / Reproducibility (< 15 Minutes)

### Fast-Track Quickstart (< 3 Minutes)

> 🌐 **Live Cloud Deployment**: Try the deployed system directly in your browser at **[https://grounded-customer-support-agent.onrender.com/](https://grounded-customer-support-agent.onrender.com/)** (zero installation required).
```bash
# 1. Clone & enter repository
git clone https://github.com/NISHAKAR06/customer-support-agent.git
cd customer-support-agent

# 2. Virtual environment setup
python -m venv venv
# Windows: .\venv\Scripts\activate | macOS/Linux: source venv/bin/activate

# 3. Install dependencies
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -r requirements-dev.txt

# 4. Run the ENTIRE pipeline in a SINGLE command (typically under 3 minutes on a standard CPU environment)
# (Trains baselines, builds FAISS index, runs smoke test, and executes full evaluation)
python scripts/run_all.py

# 5. Launch interactive web dashboard
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

### Detailed Step-by-Step Reproduction

#### 1. System Requirements
- **Python**: 3.11 or 3.12
- **Memory**: Minimum 4 GB RAM (8 GB recommended)
- **OS**: Windows, macOS, or Linux (cross-platform validated)
- **Git**: Installed and accessible on PATH

#### 2. Environment Setup & Configuration
```bash
# Clone the repository
git clone https://github.com/NISHAKAR06/customer-support-agent.git
cd customer-support-agent

# Create Python virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (cmd.exe):
.\venv\Scripts\activate.bat
# Linux / macOS:
source venv/bin/activate

# Copy environment configuration (optional API keys for Groq/OpenAI/Gemini/Claude)
cp .env.example .env
```
> **Offline Operation**: If no API keys are configured in `.env`, the system automatically falls back to `GroundedPrecedentProvider`, ensuring 100% functionality without network credentials.

#### 3. Dependency Installation
```bash
# Install PyTorch CPU wheel (uses extra-index-url to resolve PyPI dependencies correctly)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

# Install application and development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

#### 4. Unified One-Command Pipeline Runner (`scripts/run_all.py`)
To execute the entire project lifecycle across all scripts sequentially with real-time timers and a consolidated summary matrix:
```bash
# Master pipeline: Trains baselines + builds FAISS index + runs CLI smoke test + evaluates benchmarks
python scripts/run_all.py

# Master pipeline including full pytest test suite (122 tests with coverage report):
python scripts/run_all.py --with-tests

# Additional modular flags:
python scripts/run_all.py --train       # Run only baseline models & FAISS index build
python scripts/run_all.py --smoke       # Run only 4-scenario end-to-end CLI smoke test
python scripts/run_all.py --eval        # Run only headless evaluation benchmarks
python scripts/run_all.py --tests       # Run only pytest test suite
python scripts/run_all.py --with-data   # Also execute full raw data preprocessing pipeline
```

#### 5. Individual Script Execution (Manual Step-by-Step)
Alternatively, you can run each script independently:

**A. Build Baseline Models & FAISS Vector Index (~20 Seconds)**:
```bash
# Train Majority Class and TF-IDF Logistic Regression baselines
python scripts/training/train_baselines.py

# Build dense vector index using sentence-transformers/all-MiniLM-L6-v2
python scripts/training/build_faiss_index.py
```

**B. Instant End-to-End Pipeline Smoke Test**:
```bash
# Run 4 live customer scenarios via CLI
python scripts/utilities/check_e2e_flow.py
```

#### 6. Run Automated Test Suite & Code Quality Checks
```bash
# Run all 122 unit, integration, and API tests with code coverage report
pytest tests/ -v --cov=app --cov-report=term

# Verify code formatting and lint rules
ruff check .
black --check .
```

#### 7. Run Headless Evaluation Harness
Execute the complete evaluation suite against the 200 hand-verified Golden Set samples:
```bash
# Master evaluation script (executes all 4 stages and generates JSON artifacts)
python scripts/run_evaluation.py

# Human evaluation study scripts:
python scripts/reproduce_headline.py                        # Reproduce headline metrics (<3 min)
python scripts/evaluation/evaluate_human_judge_agreement.py # Human-vs-judge response quality evaluation
python scripts/evaluation/evaluate_human_routing.py         # Human routing agreement audit

# Or run individual benchmark modules independently:
python scripts/evaluation/evaluate_intent_models.py   # Intent classification & baseline comparison
python scripts/evaluation/evaluate_retrieval.py       # Dense retrieval Recall@K & MRR
python scripts/evaluation/evaluate_generation.py      # Response generation & routing policy
python scripts/evaluation/evaluate_judge.py           # LLM-as-a-Judge heuristic agreement
```

#### 8. Launch Interactive Web Application

**Hosted Live Environment**:
- **Live Deployment Base**: [https://grounded-customer-support-agent.onrender.com/](https://grounded-customer-support-agent.onrender.com/)
- **Live Simulator**: [grounded-customer-support-agent.onrender.com/](https://grounded-customer-support-agent.onrender.com/)
- **Live Support Inbox**: [grounded-customer-support-agent.onrender.com/inbox](https://grounded-customer-support-agent.onrender.com/inbox)
- **Live Evaluation Dashboard**: [grounded-customer-support-agent.onrender.com/evaluation](https://grounded-customer-support-agent.onrender.com/evaluation)
- **Live Failure Analysis**: [grounded-customer-support-agent.onrender.com/failures](https://grounded-customer-support-agent.onrender.com/failures)
- **Live Decision Log**: [grounded-customer-support-agent.onrender.com/decisions](https://grounded-customer-support-agent.onrender.com/decisions)
- **Live OpenAPI Docs**: [grounded-customer-support-agent.onrender.com/docs](https://grounded-customer-support-agent.onrender.com/docs)

**Local Deployment via Uvicorn**:
```bash
# Start FastAPI application via Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Navigate locally to:
- **Simulate Workspace**: `http://localhost:8000/`
- **Customer Support Inbox**: `http://localhost:8000/inbox`
- **Evaluation Dashboard**: `http://localhost:8000/evaluation`
- **Failure Analysis**: `http://localhost:8000/failures`
- **Decision Log & Metrics**: `http://localhost:8000/decisions`
- **Interactive OpenAPI Docs**: `http://localhost:8000/docs`

#### 9. Optional: Full Data Preprocessing Pipeline from Scratch
Pre-split training and evaluation datasets are already committed in `data/splits/`. If you want to re-execute the entire pipeline from raw Kaggle Twitter data:
```bash
# 1. Analyze raw dataset distribution across 108 brands
python scripts/data/explore_dataset.py

# 2. Profile @AppleSupport conversation threads
python scripts/data/profile_brand.py

# 3. Reconstruct multi-turn customer-brand conversation trees
python scripts/data/preprocess_conversations.py

# 4. Classify intents and apply priority disambiguation
python scripts/data/classify_intents.py

# 5. Generate stratified Train / Val / Test splits (70 / 15 / 15)
python scripts/data/split_dataset.py

# 6. Curate the 200 hand-verified Golden Evaluation Set
python scripts/evaluation/curate_golden_set.py
```

#### 10. Optional: Containerized Execution with Docker
```bash
# Build the production Docker image (compiles models on assembly)
docker build -t customer-support-agent .

# Run container on port 8000
docker run -d -p 8000:8000 --name support-agent customer-support-agent
```

---

## Real End-to-End Example Walkthrough

```
1. Customer Inquiry (Input)
   "@AppleSupport My iPhone 7 battery drops from 80% to 15% in less than an hour, and the back is getting very hot."

2. Intent Classification
   Predicted Intent: BATTERY_POWER_HARDWARE (Confidence: 0.95)
   Signals Detected: ["battery"]

3. Retrieved Precedent Evidence (Top-1 FAISS match, Cosine Sim = 0.6786)
   Case ID: conv_105402
   Historical Inquiry: "Yo @AppleSupport - my wifes iPhone 6s drops from 80% battery to 20% in seconds. 5 minutes later it'll be at 5%. You plug it back in and it'll immediately jump back up to 75%. What's up with that?"
   Historical Resolution: "We'd like to help with this battery issue. Let us know which version of iOS is installed on this phone via DM to get started. We'll go from there. https://t.co/GDrqU22YpT"

4. Generated Draft Reply (Runtime Active Coordinator)
   "Hi @alex_tester — I’m sorry you’re experiencing rapid battery loss and heat on your iPhone 7. For safety, please stop charging the device and avoid heavy use until we can assess it. Could you send us a DM with the iOS version you’re running and the Battery Health reading (Settings > Battery > Battery Health) along with your device’s serial number? We’ll review the details and help arrange a battery service if needed. You can also see our battery service information here: https://support.apple.com/battery-service"

5. Deterministic Safety Validation
   - Non-Empty & Length Check: PASSED
   - Grounding Evidence Overlap Check: PASSED (Grounding score: 0.238)
   - Ungrounded Pricing Check: PASSED (No fabricated pricing claims)
   - URL Allowlist Check: PASSED (support.apple.com authorized)
   - Public PII Security Check: PASSED (No public password solicitation)
   - Hazardous Hardware Safety Check: PASSED (Precautionary advice to stop charging was included)

6. Policy Routing Decision & Runtime Stated Reasons
   Routing Decision: AUTO_HANDLE
   Reasons Stated:
   - "High intent confidence (0.95 >= 0.80)"
   - "Sufficient historical evidence retrieved and verified"
   - "No human escalation triggers or risk keywords detected"
   - "All grounding and policy checks successfully passed"
```

> **Audited Policy Tension & Observed Grounding Failure Disclosure**:
> 1. **Observed Grounding Leakage / Hallucination**: While the retrieved historical precedent (*conv_105402*) only requests the customer's iOS version via DM, the generated draft reply introduces ungrounded instructions not present in the retrieved evidence: soliciting the device serial number, navigating to *Settings > Battery > Battery Health*, arranging battery service, and citing `https://support.apple.com/battery-service`. The deterministic validator passed this reply with a grounding overlap score of 0.238 because the check relies on lexical token overlap rather than strict semantic entailment. This example directly illustrates **Failure Mode 5 (Groundedness & Factuality Disagreement)** and highlights the limitation of shallow lexical validation.
> 2. **The Routing Tension**: While the customer reported thermal degradation (*"the back is getting very hot"*), the policy engine routed this inquiry to `AUTO_HANDLE` because intent confidence (0.95) and retrieval similarity (0.6786) were above thresholds, and `hazardous_safety_check` passed because the generated text included safety precautions (*"stop charging"*).
> 3. **The Empirical Contradiction**: While the system achieves **Acute Safety Hazard Recall: 100% on the evaluated explicit hazard-trigger subset** containing trigger keywords (*"swollen"*, *"smoke"*, *"fire"*, *"explode"*), subtle non-acute thermal degradation without swelling triggers satisfies automated heuristic thresholds and is auto-handled. In contrast, a human evaluator reviewing the operational risk of a hot device with rapid battery drops would flag this inquiry for `HUMAN_ESCALATION` (as documented in Failure Mode 3). This transparently demonstrates why heuristic routing achieves 89.5% agreement against automated triage rules, but drops to 38.00% agreement against human judgment.

---

## Testing & CI/CD Pipeline

The repository is protected by an automated GitHub Actions CI/CD pipeline ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)):

- **Lint & Code Style**: Enforced with `black --check .` (100-char line length) and `ruff check .` with zero errors.
- **Automated Test Matrix**: Executed across **Python 3.11 and Python 3.12** on `ubuntu-latest`.
- **Zero-Credential Resilience**: Tests execute offline using pre-indexed training splits and `GroundedPrecedentProvider`, requiring zero live API keys in CI runners.
- **Security Audit**: Dependency scanning via `pip-audit`.
- **Static Analysis (SAST)**: GitHub CodeQL security analysis scanning Python AST.

---

## Future Work Tied to Actual Limitations

1. **Multi-Turn Dialogue State Machine**:
   - *Observed Bottleneck*: 40% of Golden Set conversations are multi-turn, but single-tweet processing treats turns as isolated events.
   - *Planned Improvement*: Implement a dialogue memory graph storing previously attempted troubleshooting steps to prevent repetitive advice.
2. **Dynamic Knowledge Base Crawler (`support.apple.com/kb`)**:
   - *Observed Bottleneck*: Historical Twitter precedents reflect iOS 11; operating system documentation drifts over time.
   - *Planned Improvement*: Connect an automated crawler to index official Apple Support documentation alongside historical tweets.
3. **Fine-Tuned Small Language Model (SLM)**:
   - *Observed Bottleneck*: Cloud LLMs introduce 400–600 ms latency over the network.
   - *Planned Improvement*: Fine-tune a lightweight Llama-3.2-3B model on `@AppleSupport` resolutions for ultra-low latency local inference (< 100 ms).
4. **Sentiment Velocity Priority Queue**:
   - *Observed Bottleneck*: Frustrated customers writing sarcastic complaints are escalated, but placed into an unranked queue.
   - *Planned Improvement*: Calculate emotional velocity across thread turns to prioritize severely distressed users at the top of the human queue.

---

## AI Tools Used

- **Ideation & Brainstorming**: ChatGPT and Google Gemini were used for ideation, brainstorming edge cases, and conceptual exploration of domain taxonomy boundaries.
- **Human Engineering & Governance**: All architectural design decisions, domain taxonomy boundaries, priority disambiguation logic, deterministic validation regexes, evaluation harness, and test suites were designed, implemented, verified, and audited by the engineer. Every test in the 122-test suite was executed and verified locally.
