# Architecture Specification: Grounded Customer Support Agent

## 1. System Overview

The **Grounded Customer Support Agent** is designed as a modular, decoupled support intelligence engine. It mirrors real-world internal B2B customer support workflows (such as modern shared inbox automation and email/ticket copilot systems).

The system operates strictly on a **"Grounding First, Automation Second"** paradigm:
1. Every customer query is classified into a well-defined domain intent.
2. The agent queries a dense vector store of verified historical resolved support interactions from the same brand.
3. A response is generated solely bounded by the customer query and retrieved historical resolutions.
4. An explainable validation engine checks for unsupported claims and policy violations.
5. An escalation policy deterministically evaluates confidence, evidence sufficiency, and risk signals before choosing between automated resolution or human routing.

```
Incoming Customer Message
           │
           ▼
   [AgentOrchestrator]
           │
  ┌────────┴───────────────────────────────────┐
  │ 1. Context Assembly                        │
  │    (Conversation history, user context)    │
  │                                            │
  │ 2. Intent Classification                   │
  │    [IntentClassifier]                      │
  │    (Transformer / Classical ML Model)      │
  │                                            │
  │ 3. Historical Retrieval                    │
  │    [Retriever] + [EvidenceRanker]          │
  │    (Sentence-Transformers + FAISS Vector)  │
  │                                            │
  │ 4. Grounded Reply Generation               │
  │    [LLMService]                            │
  │    (Gemini Provider ──► Local LLM Fallback)│
  │                                            │
  │ 5. Response Validation                     │
  │    [ResponseValidator]                     │
  │    (Fact checking, policy, claim bounds)   │
  │                                            │
  │ 6. Escalation Policy Evaluation            │
  │    [EscalationPolicy]                      │
  │    (Confidence + Retrieval + Risk triggers)│
  └────────┬───────────────────────────────────┘
           │
           ▼
     AgentRunResult
  ┌────────────────────────────────────────────┐
  │ • Intent & Model Confidence                │
  │ • Top Retrieved Historical Cases           │
  │ • Grounded Draft Reply                     │
  │ • Validation Status & Checks Passed        │
  │ • Decision: AUTO_HANDLE / HUMAN_ESCALATION │
  │ • Explainable Decision Reasons             │
  └────────────────────────────────────────────┘
```

---

## 2. Directory Structure & Layer Responsibilities

```
app/
├── main.py                      # FastAPI application entrypoint & middleware
├── api/
│   └── routes/
│       ├── agent.py             # Agent execution (live simulation & streaming)
│       ├── conversations.py     # Inbox views and thread management
│       ├── evaluation.py        # Benchmark results and judge metrics
│       └── health.py            # Liveness and dependency readiness probes
├── core/
│   ├── config.py                # Pydantic Settings loaded from .env
│   ├── logging.py               # Structured JSON/Console logging with run_id
│   └── exceptions.py            # Domain-specific custom exceptions
├── models/
│   ├── domain_models.py         # Internal entities (Conversation, Message, Evidence)
│   ├── request_models.py        # API inputs (SimulateRequest, RunRequest)
│   └── response_models.py       # API outputs (AgentRunResult, HealthResponse)
├── services/
│   ├── agent/
│   │   ├── agent_orchestrator.py # Master pipeline coordinator
│   │   └── agent_context.py      # Working memory & run state
│   ├── intent/
│   │   ├── intent_classifier.py  # Intent inference interface & implementations
│   │   └── model_loader.py       # Thread-safe model artifact caching
│   ├── retrieval/
│   │   ├── retriever.py          # Vector query & candidate fetcher
│   │   └── evidence_ranker.py    # Resolution scoring & similarity filtering
│   ├── generation/
│   │   ├── llm_service.py        # Provider-agnostic generation interface
│   │   ├── gemini_provider.py    # Google Gemini primary implementation
│   │   ├── local_llm_provider.py # Local HuggingFace/heuristic fallback
│   │   └── prompt_builder.py     # Grounded evidence injection prompt constructor
│   ├── validation/
│   │   └── response_validator.py # Fact-checking & hallucination barrier
│   ├── escalation/
│   │   └── escalation_policy.py  # Deterministic rule engine for routing
│   └── evaluation/
│       └── evaluation_service.py # Metrics aggregation & benchmark harness
├── repositories/
│   ├── conversation_repository.py# Data access for reconstructed conversations
│   └── model_repository.py       # Data access for model metadata & checkpoints
├── infrastructure/
│   ├── vector_store/             # FAISS index wrapper and query abstractions
│   ├── llm/                      # Underlying HTTP/client connections
│   └── persistence/              # Local storage for golden set and runs
└── templates/                    # Jinja2 server-rendered templates
```

---

## 3. Component Details & Design Contracts

### 3.1 `AgentOrchestrator`
The orchestrator implements the primary pipeline. It receives an `AgentRunContext`, coordinates services sequentially, measures per-step latency, emits telemetry events, and builds the final `AgentRunResult`.

### 3.2 Intent Classification
- **Abstract Base Class**: `BaseIntentClassifier`
- **Implementations**:
  - `MajorityIntentClassifier` (Baseline 1)
  - `TfidfLogRegIntentClassifier` (Baseline 2)
  - `TransformerIntentClassifier` (Final Model)

### 3.3 Historical Evidence Retrieval
- Uses dense vector embeddings (`all-MiniLM-L6-v2`) mapped over resolved historical support cases.
- Indexing engine: FAISS `IndexFlatIP` with normalized cosine similarity.

### 3.4 Grounded Generation & Provider Fallback
- `LLMProvider` interface declares `generate(prompt: str) -> str`.
- `GeminiProvider`: Primary API using `google-genai`.
- `LocalLLMProvider`: Fallback system operating offline with zero external API dependencies.

### 3.5 Response Validation (`ResponseValidator`)
Screens draft replies against non-empty checks, unsupported claim phrases, and policy adherence.

### 3.6 Escalation Policy (`EscalationPolicy`)
Deterministic routing matrix evaluating intent confidence, retrieval similarity, complaint triggers, and validation results.
