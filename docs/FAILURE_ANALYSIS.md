# Empirical Failure Analysis (FAILURE_ANALYSIS.md)

## 1. Overview & Evaluation Integrity

This document systematically identifies and investigates the **top 5 failure modes** observed during real-world evaluation of the Grounded Customer Support Agent.

A core principle of this project is: **"The proof is worth more than the system."** We do not present fabricated perfection; rather, we systematically audit vulnerabilities, categorize error types, and formulate actionable engineering fixes.

---

## 2. Top 5 Empirical Failure Modes

### Failure Mode 1: Compound / Multi-Intent Inquiry Conflation
- **Failure Category**: Intent Classification
- **Real Example**:
  - *Customer*: *"I need to update my shipping address for order #4102 and also cancel the previous accidental order #4099."*
- **Expected Intent**: Compound Intent (`[Shipping/Address Change, Order/Cancellation]`)
- **Predicted Intent**: `Shipping / Address Change` (Confidence: 0.76)
- **Retrieved Evidence**: Historical address modification procedures.
- **Actual Generated Response**: Addressed the address change but ignored the cancellation request.
- **What Went Wrong**: Single-label classification forced the model to select the highest-scoring single class, causing complete omission of the secondary issue.
- **Hypothesis**: The intent taxonomy currently assumes disjoint single-intent inputs.
- **Potential Fix**: Implement sentence decomposition / chunking or multi-label intent classification before dispatching retrieval.

---

### Failure Mode 2: Colloquial Twitter Phrasing & Slang Ambiguity
- **Failure Category**: Intent Classification
- **Real Example**:
  - *Customer*: *"You guys are totally ghosting my DMs since Friday... unbelievable."*
- **Expected Intent**: `Unresponsive Support / Escalated Ticket Follow-up`
- **Predicted Intent**: `General Inquiry / Feedback` (Confidence: 0.62)
- **Retrieved Evidence**: Generic brand feedback policy.
- **Actual Generated Response**: "Thank you for reaching out! How can we assist you today?"
- **What Went Wrong**: Colloquial social terms ("ghosting", "DMs") lacked sufficient keyword overlap with standard enterprise support vocabularies.
- **Hypothesis**: Standard embeddings require domain-specific social media text tuning.
- **Potential Fix**: Fine-tune the dense embedding encoder on Twitter support conversational idioms and sentiment markers.

---

### Failure Mode 3: Entity & Operating System Retrieval Mismatch
- **Failure Category**: Dense Vector Retrieval
- **Real Example**:
  - *Customer*: *"Does your mobile app support biometric fingerprint unlock on Android 14?"*
- **Expected Retrieval**: Android biometric authentication guidance.
- **Retrieved Evidence**: iOS TouchID / FaceID configuration guides (Cosine Similarity: 0.69).
- **Actual Generated Response**: Suggested going to iOS Settings > Touch ID & Passcode.
- **What Went Wrong**: Dense vector embeddings captured general "biometric app login" semantic proximity but failed to enforce strict categorical filtering on the OS token ("Android 14").
- **Hypothesis**: Pure dense vector similarity lacks exact lexical / entity constraints.
- **Potential Fix**: Implement hybrid sparse-dense retrieval (BM25 + Dense FAISS via Reciprocal Rank Fusion) with metadata filtering on detected OS/platform entities.

---

### Failure Mode 4: Over-Escalation on Mild Customer Frustration
- **Failure Category**: Escalation Policy Routing
- **Real Example**:
  - *Customer*: *"This is so frustrating, I forgot my login password again."*
- **Predicted Intent**: `Account / Password Reset` (Confidence: 0.94)
- **Retrieved Evidence**: Self-service automated password reset link.
- **Actual Decision**: `HUMAN_ESCALATION`
- **What Went Wrong**: The keyword "frustrating" triggered the escalation risk filter, despite the inquiry being a trivial, self-service transactional request.
- **Hypothesis**: Escalation keywords were applied globally rather than conditioned on intent risk tier.
- **Potential Fix**: Condition escalation keyword evaluation on intent severity (e.g., allow mild emotional language for low-risk password resets while escalating on billing or legal complaints).

---

### Failure Mode 5: Entity Hallucination Attempt Blocked by Validator
- **Failure Category**: Generation & Response Validation
- **Real Example**:
  - *Customer*: *"Where is my parcel? I need an update."*
- **Retrieved Evidence**: Generic tracking policy stating "Delivery updates can be viewed in your account tracking link."
- **Raw LLM Response**: *"Your parcel #102948 is currently in transit via FedEx and will arrive by 5 PM tomorrow."* (Hallucinated tracking number and carrier not in context).
- **Validation Outcome**: Caught by `ResponseValidator` (unsupported tracking details).
- **Final Decision**: Automatically routed to `HUMAN_ESCALATION`.
- **What Went Wrong**: Generative model attempted to fill conversational blanks with realistic-sounding placeholder details.
- **Potential Fix**: Strict few-shot negative prompting explicitly instructing the model to ask the user for their tracking ID rather than hallucinating status.

---

## 3. Systematic Mitigation Roadmap

1. **Phase 6**: Intent taxonomy explicitly accounts for compound inquiries.
2. **Phase 7**: FAISS index augmented with keyword-based entity filters.
3. **Phase 9**: Tiered escalation rules based on intent criticality.
