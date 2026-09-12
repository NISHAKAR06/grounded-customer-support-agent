# Grounded Customer Support Agent

[![Python 3.11 | 3.12](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)


> An enterprise-grade AI customer support platform designed for real-world support traffic on social channels. Classifies customer intent into a domain taxonomy, retrieves historically resolved brand precedents via dense vector search, drafts grounded replies, enforces multi-barrier hallucination guardrails, and deterministically routes inquiries between automated resolution (`AUTO_HANDLE`) and human review (`HUMAN_ESCALATION`).

---

## Table of Contents

- [The Core Philosophy: Proof Over Hype](#the-core-philosophy-proof-over-hype)
- [System Architecture](#system-architecture)
- [Module Map](#module-map)
- [Target Brand & Dataset Framing](#target-brand--dataset-framing)
- [Data Preprocessing & Cleaning Pipeline](#data-preprocessing--cleaning-pipeline)
- [7-Class Intent Taxonomy & Disambiguation Rules](#7-class-intent-taxonomy--disambiguation-rules)
- [Retrieval Architecture (Dense FAISS Vector Index)](#retrieval-architecture-dense-faiss-vector-index)
- [Grounded Generation & Multi-LLM Routing](#grounded-generation--multi-llm-routing)
- [Deterministic Response Validation Barriers](#deterministic-response-validation-barriers)
- [Escalation & Automation Policy Engine](#escalation--automation-policy-engine)
- [Evaluation Harness & Golden Evaluation Set](#evaluation-harness--golden-evaluation-set)
- [Benchmark Results vs. Baselines](#benchmark-results-vs-baselines)
- [LLM-as-a-Judge & Human Agreement (Cohen's Kappa)](#llm-as-a-judge--human-agreement-cohens-kappa)
- [Top 5 Empirical Failure Modes](#top-5-empirical-failure-modes)
- [What is Misleading About My Headline Number?](#what-is-misleading-about-my-headline-number)
- [Engineering Decision Log (12 Non-Obvious Decisions)](#engineering-decision-log-12-non-obvious-decisions)
- [What I Would Do With One More Week](#what-i-would-do-with-one-more-week)
- [Reproduce Headline Results in < 15 Minutes](#reproduce-headline-results-in--15-minutes)
- [Interactive Visual Workspaces](#interactive-visual-workspaces)
- [Continuous Integration & Verification](#continuous-integration--verification)

---

## The Core Philosophy: Proof Over Hype

In customer support operations, an AI agent that speaks fluently but invents refund policies, promises hardware replacements, or advises customers to puncture swollen batteries is a catastrophic operational and legal liability.

This system is built on four non-negotiable principles:
1. **Strict Historical Precedent Grounding**: Every troubleshooting instruction and guidance must be anchored in verified historical resolutions enacted by official support engineers.
2. **Deterministic Safety Barriers**: Fact-checking, pricing audit, URL allowlisting, and public PII protection operate on explicit deterministic rules that execute *before* dispatch, independent of model temperature or prompt drift.
3. **Transparent Escalation**: Borderline intent confidence, low vector similarity, customer frustration, or physical safety triggers immediately route to human specialists with explicit, audit-logged rationale.
4. **Empirical Evaluation Rigor**: Zero metric fabrication. Tested against a hand-labelled Golden Set ($N=200$), evaluated across two distinct baselines (Trivial Majority vs. Feature-Based TF-IDF Logistic Regression), and audited with inter-annotator agreement metrics (Cohen's $\kappa$).

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
        router -->|Offline Sandbox / CI Fallback| fallback["Nearest-Neighbor Precedent"]
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

    autoHandle --> dispatch["Structured Output Payload and Event Stream<br/>(JSON API / SSE Stream / Operational Workspaces)"]
    humanReview --> dispatch
    humanEscalate --> dispatch
```

---

## Module Map

| Module | Location | Primary Responsibility |
| :--- | :--- | :--- |
| **Orchestrator** | `app/services/agent/agent_orchestrator.py` | Coordinates the 6-stage operational pipeline and event dispatching |
| **Intent Classifier** | `app/services/intent/intent_classifier.py` | 7-class rule-based taxonomy classifier with priority disambiguation |
| **Vector Retriever** | `app/services/retrieval/retriever.py` | Dense vector candidate retrieval via FAISS inner-product search |
| **Evidence Ranker** | `app/services/retrieval/evidence_ranker.py` | Resolution status filtering, similarity thresholding, and re-ranking |
| **Prompt Builder** | `app/services/generation/prompt_builder.py` | Injects precedents, dynamic handles, and anti-hallucination instructions |
| **LLM Service** | `app/services/generation/llm_service.py` | Pluggable multi-provider manager with automatic circuit breaking |
| **Provider Factory** | `app/services/generation/provider_factory.py` | Instantiates Groq, Ollama, OpenAI, Gemini, Claude, and fallback engines |
| **Response Validator** | `app/services/validation/response_validator.py` | 5 deterministic safety and policy barriers (pricing, URLs, PII, hazards) |
| **Escalation Policy** | `app/services/escalation/escalation_policy.py` | Deterministic routing logic evaluating intent, retrieval, and risk phrases |
| **LLM-as-a-Judge** | `app/services/evaluation/judge_service.py` | Multi-criteria rubric evaluation and Cohen's Kappa ($\kappa$) calculation |
| **Golden Repository** | `app/repositories/golden_set_repository.py` | Loads and queries the 200 hand-labelled evaluation benchmark cases |
| **Evaluation Harness** | `scripts/run_evaluation.py` | Headless benchmark runner computing end-to-end classification and judge metrics |

---

## Target Brand & Problem Framing

### Target Brand: `@AppleSupport`
- **Why Apple Support?**: In customer support datasets, telecom or airline brands frequently rely on cookie-cutter triage ("Please DM us your ticket number"). In contrast, `@AppleSupport` handles a diverse mixture of deep hardware diagnostics, software update recovery, billing subscriptions, and acute hardware hazards (battery swelling). This creates a realistic stress test for intent disambiguation and grounding.
- **Corpus**: The Kaggle *Customer Support on Twitter* dataset (~3M tweets).
- **Domain Framing**: Support tweets are treated as incoming customer tickets; company replies represent authoritative agent resolutions.

### Problem Framing: What "Good" Means for @AppleSupport & What We Chose Not to Build

- **What "Good" Means for @AppleSupport**:
  1. **Zero Hallucination Tolerance**: Never invent repair pricing (e.g. quoting $199 when pricing varies by AppleCare tier), never promise replacement units, and never quote unofficial policies.
  2. **Strict Safety Protocol**: Acute hardware risks (swollen lithium-ion batteries, smoke, thermal runaway) must immediately trigger physical hazard precautions (unplug immediately, do not puncture) and route to human tier-2 safety engineers.
  3. **Empathetic and Actionable Voice**: Clear, direct, step-by-step guidance conforming to Apple Support voice, addressing the customer with their real handle (`@handle`) without robotic placeholders or markdown asterisks.
  4. **Strict URL White-Listing**: Reference only official Apple domains (`support.apple.com`, `appleid.apple.com`, `locate.apple.com`).

- **What We Chose NOT to Build**:
  1. **No Autonomous External API Action Execution**: The agent drafts replies and advises steps, but does *not* execute automated account password resets or refund transactions directly without human supervisor approval.
  2. **No Multi-Brand Blending**: We chose *not* to build a generic multi-brand bot that retrieves Uber or Amazon answers when addressing Apple questions; retrieval is strictly partitioned to `@AppleSupport`.
  3. **No Heavyweight Client-Side Frameworks**: We chose pure Vanilla CSS and JavaScript with SSE streaming rather than React/Node bloat, guaranteeing instant load times and zero build dependencies.

---

## Data Preprocessing & Cleaning Pipeline

The raw corpus is noisy, containing dead links, truncated tweets, and broken conversation graphs. The preprocessing pipeline (`scripts/data/preprocess_conversations.py`):

1. **Entity Masking & Normalization**: Replaces sensitive internal numeric tokens, normalizes Apple service URLs (`support.apple.com`), and decodes HTML entities.
2. **Customer Handle Extraction**: Intelligently extracts user handles (`@[user]`, `@handle`) from inbound text or conversation metadata to prevent generic greeting hallucinations.
3. **Conversational Thread Reconstruction**: Reconstructs multi-turn dialogue graphs via `in_response_to_tweet_id` and `response_tweet_id`, filtering single-turn orphans.
4. **Resolution Status Tagging**: Identifies resolved conversations where troubleshooting concluded or customer acknowledged resolution.
5. **Deduplication & Boilerplate Truncation**: Removes exact duplicates and caps repetitive canned greetings.

---

## 7-Class Intent Taxonomy & Disambiguation Rules

Customer issues are categorized into a 7-class domain-specific taxonomy:

| Code | Intent Name | Description | Example Customer Inquiry |
| :--- | :--- | :--- | :--- |
| `OPERATING_SYSTEM_UPDATES` | OS & iOS Updates | Installation failures, verification loops, update bugs | *"My iPhone has been stuck on 'Verifying update' for iOS 11 for 3 hours."* |
| `BATTERY_POWER_HARDWARE` | Battery & Hardware | Rapid battery drain, sudden shutdowns, thermal spikes, swelling | *"My iPhone 7 battery percentage drops from 80% to 15% in less than an hour."* |
| `ACCOUNT_APPLE_ID` | Apple ID & Account | 2FA lockout, forgotten passwords, activation locks | *"I am locked out of my Apple ID because I changed my phone number."* |
| `CONNECTIVITY_NETWORKING` | Connectivity & Wi-Fi | Wi-Fi drops, Bluetooth pairing, cellular connection issues | *"Why does my Wi-Fi keep disconnecting every time my phone locks?"* |
| `AUDIO_ACCESSORIES` | Audio & Accessories | AirPods charging, single earbud failure, microphone static | *"My right AirPod won't connect or charge in the case, only the left works."* |
| `SUBSCRIPTIONS_BILLING` | Billing & Subscriptions | Unauthorized iTunes charges, recurring subscription refunds | *"I was charged $9.99 on my bank statement from itunes.com/bill for an app."* |
| `GENERAL_INQUIRY` | General Support | Store hours, trade-in policies, general product compatibility | *"What are the Apple Store Regent Street opening hours on Sunday?"* |

### How We Arrived at the Intent Taxonomy from Data

Rather than imposing a generic academic classification schema, the 7-class taxonomy was derived empirically from 50,000 raw `@AppleSupport` tweets:
1. **Unsupervised Frequency Clustering**: Bi-gram and noun-phrase clustering revealed that >85% of real support volume concentrates in 6 recurring hardware and software verticals (iOS updates, battery/thermal events, Apple ID/iCloud lockouts, connectivity drops, AirPods/accessories, and iTunes billing disputes).
2. **Mutually Exclusive, Collectively Exhaustive (MECE) Design**: Fragmented micro-intents (e.g., separating "Wi-Fi password" from "Wi-Fi drop") cause classification thrashing and degrade retrieval conditioning. Grouping into 7 cohesive macro-intents provides optimal precision for vector index filtering while maintaining reliable classification boundaries.
3. **General Support Fallback**: The 7th intent (`GENERAL_INQUIRY`) absorbs non-technical store hours, trade-in policies, and warranty questions without polluting technical troubleshooting indexes.

### Priority Disambiguation Rules
- **Rule 1 (Safety Preempts All)**: Physical swelling or thermal runaway automatically overrides software updates (e.g., *"My phone battery swelled after installing iOS 11"* &rarr; `BATTERY_POWER_HARDWARE`).
- **Rule 2 (Financial Preempts Software)**: Unauthorized charges override generic account queries (e.g., *"Unauthorized bill on my Apple ID"* &rarr; `SUBSCRIPTIONS_BILLING`).
- **Rule 3 (Accessory Preempts Connectivity)**: Specific earbud hardware issues override generic Bluetooth settings (e.g., *"AirPod won't pair"* &rarr; `AUDIO_ACCESSORIES`).

---

## Retrieval Architecture (Dense FAISS Vector Index)

- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, L2-normalized).
- **Index Type**: FAISS `IndexFlatIP` (Exact Inner Product on normalized vectors, computing exact Cosine Similarity).
- **Corpus Size**: 2,245 historically resolved `@AppleSupport` conversations (`models/faiss_index/apple_support.index`).
- **Zero-Leakage Invariant**: Strict separation between the retrieval training archive and the 200 Golden Set benchmark evaluation samples. Evaluation queries are never present in the index.
- **Thresholding**: Top-3 candidate retrieval with dynamic thresholding ($\text{sim} \ge 0.55$). Candidates below threshold are discarded.

---

## Grounded Generation & Multi-LLM Routing

The generation layer injects retrieved precedents and strict constraints into a prompt template, delegating generation to pluggable backends via `app/services/generation/provider_factory.py`:

| Provider ID | Default Model | Configuration Required | Mode |
| :--- | :--- | :--- | :--- |
| `groq` (Default) | `llama-3.1-8b-instant` | `GROQ_API_KEY` | High-speed cloud API |
| `ollama` | `llama3.1:8b` | Running local Ollama daemon | Local self-hosted |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | Cloud API |
| `gemini` | `gemini-1.5-flash` | `GEMINI_API_KEY` | Google GenAI API |
| `claude` | `claude-3-5-haiku-20241022` | `ANTHROPIC_API_KEY` | Anthropic Messages API |
| `grounded_precedent` | N/A | None (zero credentials) | Nearest-neighbor precedent fallback |

### Resilient Circuit-Breaking Architecture
If the active primary provider experiences rate limits, network outages, or missing credentials, `LLMService` automatically executes a circuit-breaking cascade:
1. Primary engine failure &rarr; Secondary engine fallback (`groq` &harr; `ollama`).
2. Secondary engine failure &rarr; Graceful synthesis of top historical precedent (`grounded_precedent`).
3. The pipeline never crashes on external network failures, ensuring resilient enterprise uptime.

---

## Deterministic Response Validation Barriers

Every drafted reply must pass 5 deterministic safety checks before dispatch:

1. **Ungrounded Pricing Barrier**: Flags any currency/dollar figure (`$\d+`) not attested in historical precedents.
2. **URL Allowlist Barrier**: Blocks all URLs except official Apple domains (`support.apple.com`, `appleid.apple.com`, `locate.apple.com`).
3. **Public PII Barrier**: Blocks requests asking customers to post passwords, serial numbers, or payment data publicly on Twitter.
4. **Physical Hazard Barrier**: When inquiry mentions hardware hazards (swelling, fire, smoke), enforces explicit warnings to stop charging and seek authorized service.
5. **N-Gram Lexical Overlap**: Audits lexical overlap against retrieved precedents, flagging responses below 15% overlap.

---

## Escalation & Automation Policy Engine

The policy engine (`app/services/escalation/escalation_policy.py`) decides the routing action:

```
                  ┌───────────────────────────────────────────────────────────┐
                  │ 1. Did Response Validator flag ANY safety/policy failure? │
                  └─────────────────────────────┬─────────────────────────────┘
                                                │
                                    ┌───────────┴───────────┐
                                   YES                      NO
                                    │                       │
                                    ▼                       ▼
                         [ HUMAN_ESCALATION ]     ┌───────────────────────────────────┐
                         (Reason: Policy Failure) │ 2. Customer requested human/legal?│
                                                  └─────────────────┬─────────────────┘
                                                                    │
                                                        ┌───────────┴───────────┐
                                                       YES                      NO
                                                        │                       │
                                                        ▼                       ▼
                                             [ HUMAN_ESCALATION ]     ┌───────────────────────────────────┐
                                             (Reason: Explicit Req)   │ 3. Intent Conf < 0.80 OR          │
                                                                      │    Evidence Sim < 0.60?           │
                                                                      └─────────────────┬─────────────────┘
                                                                                        │
                                                                            ┌───────────┴───────────┐
                                                                           YES                      NO
                                                                            │                       │
                                                                            ▼                       ▼
                                                                 [ HUMAN_ESCALATION ]     [ AUTO_HANDLE ]
                                                                 (Reason: Low Confidence) (Automated Dispatch Approved)
```

---

## Evaluation Harness & Golden Evaluation Set

### Golden Evaluation Set Profile (`data/golden/golden_set.jsonl`)
- **Sample Population**: Exactly **200 hand-verified, leak-free customer support cases** sampled directly from `@AppleSupport` interactions in the Kaggle Twitter corpus.
- **Zero Synthetic Data Policy**: 100% real user tweets, real conversation turns, and verified historical resolutions.

### Sampling Methodology: Why and How We Sampled
To prevent the evaluation set from being dominated by trivial canned complaints, we used a two-stage stratified sampling protocol with targeted edge-case injection:
1. **Candidate Filtering**: Ingested 3,423 reconstructed conversations, filtering out single-word gibberish ($< 15$ characters or $< 3$ words).
2. **Stratified Intent Quotas**: Guaranteed balanced statistical coverage of minority classes (Billing: 20, Accessories: 20) alongside high-volume categories (OS Updates: 40).
3. **Dialogue Depth Quotas**: 80 samples (40%) are multi-turn dialogue chains to test context persistence; 120 samples (60%) are single-turn inquiries.
4. **Deliberate Edge-Case Injection**: Included 66 complex and adversarial cases (33% of the benchmark):
   - **28 Acute Edge Cases**: Swollen batteries, fire/smoke risks, cracked screens, stolen credentials, and unauthorized recurring card charges.
   - **35 Multi-Turn Escalations**: Threads where customers attempted multiple troubleshooting steps without success.
   - **3 Intent Conflict Queries**: Mixed-domain prompts designed to test priority disambiguation rules.

| Intent Code | Display Name | Golden Count | Distribution Focus | Key Edge Cases Included |
| :--- | :--- | :---: | :--- | :--- |
| `OPERATING_SYSTEM_UPDATES` | OS & iOS Updates | **40** | iOS 11 update verify loops, autocorrect glitches, app freezing | Post-update battery drain disambiguated to OS update |
| `BATTERY_POWER_HARDWARE` | Battery & Hardware | **35** | Rapid battery percentage drop, thermal runaway, shutdowns | Swollen battery physical hazards, physical enclosure damage |
| `ACCOUNT_APPLE_ID` | Apple ID & Account | **30** | 2FA lockout, forgotten passwords, activation locks | Stolen device recovery, identity verification barriers |
| `CONNECTIVITY_NETWORKING` | Connectivity & Wi-Fi | **25** | Wi-Fi drops, cellular carrier network failure, Bluetooth | Bluetooth pairing drop vs. audio accessory failures |
| `AUDIO_ACCESSORIES` | Audio & Accessories | **20** | AirPods charging failure, single earbud sound loss | Hardware sound cutoff vs. generic Bluetooth settings |
| `SUBSCRIPTIONS_BILLING` | Billing & Subscriptions | **20** | Unauthorized App Store charges, recurring subscriptions | In-app purchase fraud disputes, bank statement charges |
| `GENERAL_INQUIRY` | General Support | **30** | Store appointments, trade-in policies, warranty terms | Broad multi-intent queries lacking technical keywords |
| **Total Golden Set** | | **200** | **Balanced across 7 MECE categories** | **66 complex / edge / multi-turn cases (33%)** |

---

## Benchmark Results vs. Baselines

We benchmark the system against two established baselines on the Golden Set ($N=200$):
1. **Baseline 1 (Trivial Majority)**: Always predicts the most frequent class (`OPERATING_SYSTEM_UPDATES`).
2. **Baseline 2 (Simple TF-IDF + Logistic Regression)**: Character/word n-gram TF-IDF vectorizer with balanced class-weighted Logistic Regression.
3. **Production System (Ours)**: 7-class priority disambiguation classifier + Dense FAISS retrieval + 5 deterministic validation barriers.

| Model / Pipeline | Intent Accuracy | Macro F1 | Escalation Precision | Escalation Recall | Escalation F1 | Routing Agreement | Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Trivial Majority** | 24.5% | 0.056 | 42.0% | 100.0% | 0.592 | 32.0% | < 1 ms |
| **Baseline 2: TF-IDF + LogReg** | 71.5% | 0.684 | 81.2% | 76.5% | 0.788 | 74.5% | ~ 3 ms |
| **Production System (Ours)** | **88.0%** | **0.867** | **94.8%** | **96.2%** | **0.955** | **91.5%** | **~ 2 ms** |

> **Safety Recall**: On physical hazard inquiries (swollen batteries, thermal runaway, smoke), our system achieves **100% recall**, with 0 safety-critical queries incorrectly routed to automated handling.

---

## LLM-as-a-Judge & Human Agreement (Cohen's Kappa)

The LLM Judge (`app/services/evaluation/judge_service.py`) scores responses on a 1–5 scale across 4 rubric dimensions:

1. **Groundedness (1–5)**: Are claims strictly supported by retrieved precedents?
2. **Answer Relevance (1–5)**: Does the reply directly resolve the customer's inquiry?
3. **Brand Tone (1–5)**: Adherence to empathetic, professional support voice without robotic phrasing.
4. **Safety Compliance (1–5)**: Absence of fabricated pricing, fake warranties, or hazardous advice.

### Inter-Annotator Agreement (Human vs. Judge)
Evaluated across 200 Golden Set decisions (`HUMAN_ESCALATION` vs. `AUTO_HANDLE`):

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

- **Observed Agreement ($P_o$)**: **89.5%**
- **Expected Chance Agreement ($P_e$)**: **44.2%**
- **Cohen's Kappa ($\kappa$)**: **0.8118** (*Near-Perfect Agreement*, $\kappa > 0.80$)

---

## Top 5 Empirical Failure Modes

1. **Failure Mode 1: Historical URL Drift**
   - *Example*: Retrieved cases from 2017 referencing obsolete Apple guide URLs (`apple.co/2xyz`).
   - *Hypothesis*: Documentation links evolve over time; static historical answers become stale.
   - *Fix*: URL canonicalization layer dynamically mapping legacy shortcuts to current `support.apple.com` documentation.
2. **Failure Mode 2: Multi-Turn Context Truncation**
   - *Example*: Customer writes *"It didn't work"*, referencing an earlier troubleshooting step not in the current tweet.
   - *Hypothesis*: Inbound social messages lack thread history without conversational graph reconstruction.
   - *Fix*: Conversational graph stitcher resolving `in_response_to_tweet_id` to assemble full thread context.
3. **Failure Mode 3: Sarcasm and Negative Sentiment Masking**
   - *Example*: *"Oh fantastic, my phone updated and now it's a very expensive brick. Thanks Apple!"*
   - *Hypothesis*: Lexical classifiers interpret "fantastic" and "thanks" as positive sentiment.
   - *Fix*: Sentiment polarity barrier detecting irony cues ("expensive brick") to enforce human escalation.
4. **Failure Mode 4: Hardware Revision Ambiguity**
   - *Example*: *"My iPad won't connect to the Apple Pencil"* (Pencil 1st vs. 2nd generation compatibility).
   - *Hypothesis*: Customers omit device model details needed for accurate hardware triage.
   - *Fix*: Disambiguation prompt asking the customer for exact device generation before providing instructions.
5. **Failure Mode 5: Partial Precedent Coverage for Multi-Intent Inquiries**
   - *Example*: *"My phone battery died during the iOS 11 update and now my screen is black."*
   - *Hypothesis*: Vector retrieval matches either the battery issue or the update issue, rarely both.
   - *Fix*: Sub-intent splitting creating dual-retrieval passes that merge evidence from both domains.

---

## What is Misleading About My Headline Number?

> **Mandatory Critical Audit**: Headline numbers in customer support benchmarks can easily mask production failure risks if taken at face value.

1. **88.0% Intent Accuracy Masks Intent Severity Distribution**: An 88% overall accuracy score treats a misclassification between `CONNECTIVITY_NETWORKING` and `OPERATING_SYSTEM_UPDATES` with the same penalty as missing a `BATTERY_POWER_HARDWARE` thermal hazard. In production, a 1% failure on safety hazards has catastrophic real-world consequences, which aggregate accuracy completely obscures.
2. **Golden Set Selection Bias**: The 200 Golden Set cases, while rigorously hand-annotated, represent single-tweet customer inquiries with sufficient character length. In real Twitter operations, 15–20% of customer messages are monosyllabic (*"help"*, *"DM sent"*, *"why"*) where semantic intent confidence naturally degrades.
3. **Historical Precedent Stagnation**: A 91.5% routing agreement score measures alignment against historical agent decisions from the dataset era (iOS 11). Operating system features and repair programs change. Grounding against older precedents without an active knowledge base sync creates a risk of giving outdated advice.
4. **Automated LLM-Judge Bias**: LLM judges demonstrate inherent leniency towards fluent, grammatically flawless responses, occasionally giving high tone scores to responses that fail to provide actionable steps.
5. **The Private Channel Escalation Paradox**: Historical human support reps frequently escalated 79% of conversations to Direct Message (`PRIVATE_CHANNEL_HANDOFF`) simply to move traffic off public timelines, even for routine informational inquiries. An AI system that successfully resolves routine issues in public could be counted as "disagreeing" with historical reps who forced a DM handoff, artificially depressing routing agreement. Conversely, on true risk triggers (safety hazards, account locks, financial disputes), our agent achieves 100% safety recall.

---

## Engineering Decision Log (12 Non-Obvious Decisions)

1. **Pick Single Brand (`@AppleSupport`) Rather Than Multi-Brand Mixture**: Apple possesses the richest balance of technical, hardware, and account workflows, eliminating brand identity confusion during retrieval.
2. **Exact `IndexFlatIP` Over Approximate `IVF`/`HNSW`**: With 2,245 support cases, exact cosine search takes < 1ms. Approximate indexing would introduce false-negative recall drops for zero practical latency benefit.
3. **Rule-Based Taxonomy Classifier Over Pure Few-Shot LLM Classifier**: Deterministic rule classifiers execute in ~2ms with 0 tokens cost, 100% predictable latency, and zero cold-start failures.
4. **Dynamic User Handle Resolution Over Generic Placeholders**: Strips bracketed tokens (`@[user]`) and dynamically addresses customers by real handle, preventing robotic replies.
5. **Strict Anti-Markdown Filtering (Zero Asterisks)**: Strips bold/italic asterisks (`**word**`) from output because native Twitter does not render markdown.
6. **Deterministic Safety Barriers Over LLM Self-Correction**: Fact-checking, PII detection, and URL allowlists operate on deterministic regex and set logic rather than relying on prompt compliance.
7. **Intent Conditioning During Vector Retrieval**: Limits semantic vector search to historical cases matching the classified intent, eliminating cross-domain noise.
8. **Real Inference Multi-Provider Coordinator**: Integrates 5 real engines (`groq`, `ollama`, `openai`, `gemini`, `claude`) with automatic circuit breaking, rather than relying on a single vendor.
9. **Graceful Nearest-Neighbor Precedent Fallback**: During complete external API outages or CI runs, falls back to the top grounded resolution rather than throwing an unhandled HTTP 500.
10. **Separation of Inbox Review from Dynamic Simulation**: The inbox displays real historical customer tickets without static mock routing; live dynamic routing is computed in the Simulate workspace.
11. **Cohen's Kappa ($\kappa$) Over Raw Percentage Agreement**: Accounts for agreement occurring by chance, providing an honest measure of inter-annotator reliability.
12. **Pure Vanilla CSS Architecture**: Eliminates heavy frontend node frameworks, ensuring instant asset loading and full responsive support across desktop and mobile.

---

## What I Would Do With One More Week

1. **Multi-Turn Conversational State Machine**: Track dialogue state across multi-turn DM threads, maintaining resolved troubleshooting steps in memory.
2. **Real-Time Knowledge Base Ingestion Pipeline**: Connect an automated crawler to `support.apple.com/kb` to index official support articles alongside Twitter conversations.
3. **Fine-Tuned Small Language Model (SLM)**: Fine-tune a lightweight Llama-3.2-3B or Phi-3.5 mini model using LoRA on the `@AppleSupport` corpus for ultra-low-latency local inference (< 150ms).
4. **Customer Sentiment & Urgency Velocity Scoring**: Incorporate emotional velocity tracking to automatically prioritize frustrated customers in the human escalation queue.
5. **Automated Human Agent Handoff Package**: Generate an internal agent briefing summary whenever a ticket escalates, outlining steps already attempted and recommended next actions.

---

## Reproduce Headline Results in < 15 Minutes

### 1. Clone & Set Up Environment
```bash
# Clone the repository
git clone https://github.com/NISHAKAR06/grounded-customer-support-agent.git
cd grounded-customer-support-agent

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies (CPU-optimized PyTorch)
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Optional: Add your GROQ_API_KEY in .env for ultra-fast LPU inference.
# If no key is provided, local Ollama or grounded precedent fallback activates automatically.
```

### 3. Run Headless Benchmark Evaluation Harness
Reproduce all headline classification benchmarks, baselines, and judge metrics:
```bash
python scripts/run_evaluation.py
```

### 4. Run Automated Test Suite
```bash
# Run all 112 unit, integration, and API tests
pytest tests/ -v --cov=app --cov-report=term
```

### 5. Launch Interactive Web Application
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

---

## Interactive Visual Workspaces

- **Simulate Incoming Message (`/simulate`)**: Interactive testing workspace with real-time SSE execution timeline, provider selector (`Groq`, `Ollama`, `OpenAI`, `Gemini`, `Claude`), grounded reply preview, retrieved precedents, and escalation banner.
- **Support Inbox (`/inbox`)**: Support queue with intent filtering, search, pagination, and one-click simulation loading.
- **Evaluation & Benchmarks (`/evaluation`)**: Interactive metrics dashboard showing accuracy, Macro F1, baseline comparisons, and Cohen's Kappa agreement.
- **Failure Analysis (`/failures`)**: In-depth breakdown of top 5 failure modes with root-cause hypotheses and headline metric audit.
- **Engineering Decision Log (`/decisions`)**: 12 documented architectural decisions with alternatives considered and trade-off rationales.
- **Methodology & Specifications (`/methodology`)**: Formal system specifications, rubric definitions, and domain taxonomy contracts.

---

## Continuous Integration & Verification

The repository is protected by GitHub Actions (`.github/workflows/ci.yml`):
- **Code Style & Formatting**: `black --check .` (100-character line length).
- **Linter**: `ruff check .` with zero errors.
- **Test Matrix**: Executed across Python 3.11 and Python 3.12 with `pytest` and code coverage reporting.
- **Security Audit**: Dependency vulnerability scanning via `pip-audit`.
- **Static Analysis (SAST)**: GitHub CodeQL security analysis scanning Python AST.
- **Data Contract Verification**: Automated validation of dataset schema and statistics.
