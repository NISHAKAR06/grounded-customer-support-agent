# Grounded Customer Support Agent

[![CI/CD Pipeline](https://github.com/NISHAKAR06/customer-support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/NISHAKAR06/customer-support-agent/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Design: B2B SaaS](https://img.shields.io/badge/UI-B2B--SaaS-slate.svg)]()

> A production-grade AI customer support operations platform that classifies customer intent, retrieves historically resolved support conversations, generates grounded replies, validates against hallucinations, and intelligently decides between automated handling (`AUTO_HANDLE`) and human escalation (`HUMAN_ESCALATION`).

---

## The Core Philosophy: "The Proof is Worth More Than the System"

In customer support automation, an agent that sounds fluent but invents policies, promises unauthorized refunds, or mishandles an angry customer is a liability. 

This project demonstrates an enterprise-grade approach:
1. **Historical Precedent Grounding**: Answers are derived strictly from how the brand previously resolved similar issues.
2. **Deterministic Validation**: Guardrails prevent unsupported promises before customer dispatch.
3. **Transparent Escalation**: When intent confidence is borderline, retrieval evidence is weak, or risk triggers are detected, the system safely routes to human agents with explicit reasons.
4. **Zero Metric Fabrication**: Built upon real customer data from the Kaggle *Customer Support on Twitter* corpus, tested against two baselines (Majority and TF-IDF LogReg) and a 150–250 hand-labelled Golden Set.

```
Customer Message
       ↓
Conversation Context
       ↓
Intent Classification
       ↓
Historical Resolved Retrieval
       ↓
Grounded Reply Generation
       ↓
Response Validation
       ↓
Escalation / Automation Decision
       ↓
Final Result
```

---

## Architecture & Visual Interface

The application interface captures the aesthetic of **modern B2B support operations software** (inspired by clean, high-contrast shared inboxes):
- **Dark Sidebar Navigation**: Focuses on core operational workspaces.
- **Simulate Incoming Message**: The primary workspace where operators and evaluators can test live customer inputs and observe the multi-stage pipeline executing in real-time.
- **Support Inbox**: Operational queue distinguishing automated candidates from human escalation tickets.
- **Evaluation & Failure Deep Dive**: Transparent reporting of model benchmarks, LLM-as-Judge scores, human-judge correlation, and top failure modes.

```
┌─────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Grounded Support AI     │ Breadcrumb: Workspace / Simulate                            │
│ [● System Online]       ├────────────────────────┬────────────────────────────────────┤
├─────────────────────────┤ INCOMING MESSAGE       │ AI AGENT ACTIVITY                  │
│ WORKSPACE               │ [Textarea Composer]    │ ✓ 1. Message Received              │
│   Inbox                 │                        │ ✓ 2. Intent: Address Change (94%)  │
│  ► Simulate             │ Quick-Load Examples:   │ ✓ 3. Retrieved 3 Historical Cases  │
│                         │ [Address Change]       │ ✓ 4. Grounded Reply Drafted        │
│ INSIGHTS                │ [Duplicate Charge]     │ ✓ 5. Validation Passed             │
│   Evaluation            │ [App Crash]            │ ✓ 6. Decision: AUTO_HANDLE         │
│   Failure Analysis      │                        ├────────────────────────────────────┤
│                         │ [ Run AI Agent ]       │ GROUNDED CASE RESULT               │
│ ENGINEERING             │                        │ Evidence + Validation + Routing    │
│   Decision Log          │                        │                                    │
│   Methodology           │                        │                                    │
└─────────────────────────┴────────────────────────┴────────────────────────────────────┘
```

---

## 5-Minute Evaluator Quickstart

You do **NOT** need the raw 3M Twitter dataset to run and inspect the application.

```bash
# 1. Clone repository
git clone https://github.com/NISHAKAR06/customer-support-agent.git
cd customer-support-agent

# 2. Set up virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Copy configuration
cp .env.example .env

# 5. Launch FastAPI server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## Project Documentation (`docs/`)

Detailed technical specifications and logs are maintained inside [`docs/`](docs/):

| Document | Description |
| :--- | :--- |
| **[`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md)** | Phased development roadmap, lifecycle gates, and setup rules |
| **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)** | Component architecture, OOP service contracts, and fallback mechanics |
| **[`docs/DATASET.md`](docs/DATASET.md)** | Kaggle dataset specifications, partitioning, and leakage prevention |
| **[`docs/BRAND_SELECTION.md`](docs/BRAND_SELECTION.md)** | Evidence-backed criteria and candidate brand analysis |
| **[`docs/EVALUATION_PLAN.md`](docs/EVALUATION_PLAN.md)** | Golden set methodology, baselines, LLM-as-Judge rubric, and human agreement |
| **[`docs/UI_SPEC.md`](docs/UI_SPEC.md)** | B2B SaaS monochrome design system, layout, and simulation timeline specs |
| **[`docs/API_SPEC.md`](docs/API_SPEC.md)** | FastAPI REST endpoints, Pydantic schemas, and SSE streaming events |
| **[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md)** | Intent classifier architectures, embeddings, and ethical boundaries |
| **[`docs/DECISION_LOG.md`](docs/DECISION_LOG.md)** | 12 architectural decisions with alternatives and trade-offs |
| **[`docs/RUNBOOK.md`](docs/RUNBOOK.md)** | Evaluator quickstart, operational procedures, and troubleshooting |
| **[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md)** | Code conventions, type hinting, and Git commit permission rules |
| **[`docs/FAILURE_ANALYSIS.md`](docs/FAILURE_ANALYSIS.md)** | Top 5 empirical failure modes with root cause hypotheses and fixes |
| **[`docs/REPORT.md`](docs/REPORT.md)** | Comprehensive 6-page equivalent engineering and evaluation report |

---

## Testing & Quality Assurance

```bash
# Run all automated tests
pytest tests/ -v

# Run linting checks
ruff check .
black --check .
```
