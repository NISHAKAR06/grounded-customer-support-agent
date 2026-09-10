# API Specification (API_SPEC.md)

## 1. Overview & Protocol Conventions

The **Grounded Customer Support Agent API** is implemented via FastAPI. All request and response bodies use JSON encoding conforming to OpenAPI 3.1 standards.

- Base URL: `http://localhost:8000/api`
- Content-Type: `application/json`
- Date-Time format: ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`)

---

## 2. Core Endpoints

### 2.1 Health & Readiness Probes
- `GET /api/health`: Returns system readiness, active LLM provider, and model checkpoint statuses.

### 2.2 Agent Execution
- `POST /api/agent/run`: Executes the full pipeline for an incoming customer message (`SimulateRequest`) and returns `AgentRunResult`.
- `GET /api/agent/events/{run_id}`: Server-Sent Events (SSE) channel for real-time visualization of agent pipeline execution.

### 2.3 Conversations & Evaluation Endpoints
- `GET /api/conversations`: Lists verified customer support conversation records for the Inbox view.
- `GET /api/conversations/{conversation_id}`: Retrieves a single conversation by ID.
- `GET /api/evaluation/metrics`: Returns benchmark metrics for baselines and final system across test and golden sets.
