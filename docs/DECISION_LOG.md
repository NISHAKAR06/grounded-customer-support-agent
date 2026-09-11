# Engineering Decision Log (DECISION_LOG.md)

This log records the major architectural, algorithmic, and engineering decisions made during the design and development of the Grounded Customer Support Agent.

---

### Decision 1: Single Brand Focus from Customer Support on Twitter
- **Context**: The Kaggle dataset contains ~3M tweets from dozens of distinct global brands across multiple industries.
- **Alternatives Considered**: Multi-brand aggregate training vs. single focused brand model.
- **Choice**: Select exactly ONE brand based on empirical volume, resolution quality, and intent diversity.
- **Reason**: Support guidelines and policies vary drastically between industries. Grounding is only meaningful when tied to a specific brand's verified policies.
- **Trade-off**: The model specializes deeply in one brand rather than acting as a universal support agent across all retail domains.

---

### Decision 2: Conversation-Thread Splitting over Random Message Splitting
- **Context**: Customer support conversations span multiple back-and-forth turns.
- **Alternatives Considered**: Row-level random train/test split vs. thread-level split.
- **Choice**: Partition data strictly by `conversation_id` / root tweet thread.
- **Reason**: Random row splitting causes massive context leakage: the model sees prior turns of a conversation in training and evaluates on subsequent turns in testing, inflating accuracy artificially.
- **Trade-off**: Requires pre-computing the conversational graph before performing splits.

---

### Decision 3: Hand-Labelled Golden Evaluation Set (150–250 Examples)
- **Context**: Evaluating real support agents requires a reliable ground truth benchmark.
- **Alternatives Considered**: Synthetic LLM-generated labels vs. human hand-labelling.
- **Choice**: Personally review and hand-label 150–250 real examples.
- **Reason**: Authentic human verification without automated shortcuts guarantees evaluation integrity.
- **Trade-off**: High manual labeling effort; smaller test set size compared to automated bulk scraping.

---

### Decision 4: Two Meaningful Baselines (Majority + TF-IDF LogReg)
- **Context**: To validate that an advanced model adds real value, comparative baselines are required.
- **Alternatives Considered**: Zero-shot LLM baseline vs. heuristic keyword baseline vs. TF-IDF + Logistic Regression.
- **Choice**: Implement both a trivial Majority Classifier and a classical TF-IDF + Logistic Regression classifier.
- **Reason**: Provides an explainable, deterministic floor.
- **Trade-off**: Requires maintaining and benchmarking three distinct model pipelines.

---

### Decision 5: Dense Retrieval with FAISS Vector Store
- **Context**: Grounded generation requires fetching historical resolved cases matching the incoming query.
- **Alternatives Considered**: Pure keyword search (BM25) vs. dense embeddings (FAISS).
- **Choice**: Sentence-Transformers (`all-MiniLM-L6-v2`) with FAISS index.
- **Reason**: Customer support queries use informal language, typos, and synonyms that fail exact keyword matching.
- **Trade-off**: Requires vector embedding precomputation and vector store indexing.

---

### Decision 6: LLM Provider Abstraction with Offline Local Fallback
- **Context**: Cloud LLM APIs (like Google Gemini) are subject to rate limits, network outages, or key unavailability.
- **Alternatives Considered**: Direct Gemini SDK calls vs. decoupled `LLMProvider` interface with a `LocalLLMProvider` fallback.
- **Choice**: Decoupled provider interface with automatic fallback.
- **Reason**: Enterprise support systems cannot drop customer inquiries when an upstream API experiences downtime.
- **Trade-off**: Additional architectural abstraction layers.

---

### Decision 7: Multi-Check Deterministic Response Validator
- **Context**: Generative models can hallucinate unauthorized discounts, fake policies, or inaccurate status updates.
- **Alternatives Considered**: Trusting the LLM system prompt vs. deterministic post-generation validation.
- **Choice**: Deterministic `ResponseValidator` checking grounding presence, policy violations, and unverified promises.
- **Reason**: A deterministic validation gate guarantees safety before customer presentation.
- **Trade-off**: Minor latency overhead (~5–10ms) per response.

---

### Decision 8: Deterministic Escalation Policy Matrix
- **Context**: The agent must decide whether to automate a response or route it to a human support agent.
- **Alternatives Considered**: Asking the LLM "Should this be escalated?" vs. deterministic rule-based matrix.
- **Choice**: Deterministic multi-factor policy evaluating intent confidence, retrieval similarity, complaint triggers, and validation results.
- **Reason**: Safety-critical routing must be auditable and explainable.
- **Trade-off**: Requires tuning confidence thresholds during validation.

---

### Decision 9: Vanilla HTML/CSS/JS with Jinja2 Templates (No React / Next.js)
- **Context**: The architectural goal is to provide a clean, high-performance B2B SaaS interface without heavy single-page framework overhead.
- **Alternatives Considered**: React/Next.js SPA vs. FastAPI Jinja2 server-rendered templates.
- **Choice**: FastAPI Jinja2 templates styled with custom Vanilla CSS and Vanilla JavaScript.
- **Reason**: Eliminates Node/NPM build toolchains, guarantees instant evaluator startup, and reduces dependency bloat.
- **Trade-off**: Dynamic DOM updates require vanilla DOM manipulation rather than React state management.

---

### Decision 10: High-Contrast Modern B2B SaaS Design Language
- **Context**: The interface must convey production support operations credibility.
- **Alternatives Considered**: Generic Tailwind dashboard vs. dark-mode cyber aesthetic vs. clean B2B SaaS monochrome design.
- **Choice**: Dark sidebar, crisp white card surfaces, thin borders, restrained typography, and functional green/red/amber status indicators.
- **Reason**: Resembles real enterprise software (like modern operational shared inboxes). Avoids flashy AI tropes.
- **Trade-off**: Requires disciplined custom CSS design tokens rather than pre-packaged UI kits.

---

### Decision 11: Real-Time Event Streaming (SSE) for Simulation Pipeline
- **Context**: Evaluators need to see the agent actually executing the multi-step pipeline.
- **Alternatives Considered**: Artificial delay with JavaScript setTimeout vs. Server-Sent Events (SSE).
- **Choice**: Expose actual processing events via SSE (`/api/agent/events/{run_id}`) and granular telemetry in the POST response.
- **Reason**: Proves that the pipeline runs real components in sequence without fake animations.
- **Trade-off**: Requires event pub/sub management in backend services.

---

### Decision 12: Structuring LLM-as-Judge with Strictly Enforced Schema
- **Context**: Evaluating generated response quality across hundreds of samples requires automated scoring.
- **Alternatives Considered**: Free-form qualitative text prompts vs. structured JSON rubric with numeric anchors.
- **Choice**: Structured rubric scoring Groundedness (1–5), Helpfulness (1–5), Resolution Fit (1–5), and Unsupported Claims (Binary).
- **Reason**: Enables automated aggregation, statistical correlation with human raters, and programmatic error analysis.
- **Trade-off**: Requires strict JSON parsing and schema validation on judge outputs.
