# Runbook: Operations & Evaluator Quickstart

## 1. Quickstart for Evaluators & Reviewers

This runbook allows an evaluator to run and verify the Grounded Customer Support Agent locally in under 5 minutes without downloading the full 3M Twitter dataset.

### Step 1: Environment Setup
```bash
# Clone the repository
git clone https://github.com/NISHAKAR06/customer-support-agent.git
cd customer-support-agent

# Create and activate Python virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create environment configuration from template
cp .env.example .env
```

### Step 2: Start the Application Server
```bash
# Start FastAPI application with Uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Open [http://localhost:8000](http://localhost:8000) in your web browser.

---

## 2. Navigation & Feature Walkthrough

### 2.1 The Simulate Workspace (`/simulate`)
1. Click **Simulate** in the left sidebar (or navigate to `/simulate`).
2. Click one of the prepared quick-load buttons (e.g., *Address Change* or *Duplicate Charge Dispute*).
3. Click **Run AI Agent**.
4. Observe the live pipeline trace executing:
   - Message Received & Context Assembled
   - Intent Classification (with confidence score)
   - Historical Evidence Retrieval (fetching matching resolved cases)
   - Grounded Response Generation
   - Validation Checks (Hallucination barriers & Policy adherence)
   - Routing Decision (`AUTO_HANDLE` vs. `HUMAN_ESCALATION`)
5. Inspect the generated response, citations, and deterministic decision reasons.

### 2.2 Support Inbox (`/inbox`)
View incoming customer threads categorized by status: `All`, `AI Ready`, `Needs Human`, and `Resolved`.

### 2.3 Evaluation Benchmark (`/evaluation`)
Inspect head-to-head metrics comparing:
- **Baseline 1 (Majority Classifier)**
- **Baseline 2 (TF-IDF + Logistic Regression)**
- **Final System (Dense Support Model)**
- Metrics: Accuracy, Macro F1, Recall@1/3/5, LLM-as-a-Judge scores, Human Agreement.

### 2.4 Failure Analysis (`/failures`)
Review the deep-dive report on real observed failure modes and the analysis: *"What is misleading about my headline number?"*

---

## 3. Running Automated Tests & Quality Gates

```bash
# Run complete test suite
pytest tests/ -v

# Run unit tests only
pytest tests/unit/ -v

# Run API integration tests
pytest tests/api/ -v

# Run linting and code formatting checks
ruff check .
black --check .
```

---

## 4. Troubleshooting & FAQ

### Issue: Gemini API returns 429 or quota exceeded
- **Behavior**: The agent's circuit breaker automatically catches provider exceptions and switches to `LocalLLMProvider`.
- **Result**: The agent completes successfully without failing the user request.

### Issue: Missing FAISS index or embedding model on clean machine
- **Behavior**: The application runs in fixture-grounded mode when vector indices are not yet built, enabling instant UI simulation for reviewers without heavy data downloads.
