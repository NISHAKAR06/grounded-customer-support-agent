# Project Evaluation & Engineering Report (REPORT.md)

## 1. Problem Framing & System Objective

Customer support automation often fails in production because generative models prioritize fluency over factual fidelity. When an agent invents refund policies, misinterprets customer frustration, or attempts to resolve ambiguous requests, it creates brand liability and customer churn.

The **Grounded Customer Support Agent** addresses this operational challenge for enterprise shared inboxes (inspired by modern shared inbox support operations). It implements a strict pipeline where generation is tethered to historically resolved interactions from the brand, screened by an explainable validation gate, and routed deterministically between automated dispatch (`AUTO_HANDLE`) and human escalation (`HUMAN_ESCALATION`).

---

## 2. What "Good" Means for the Selected Brand

For an enterprise support organization, a "good" AI agent is not one that claims to answer 100% of tickets autonomously. A truly good system is defined by:
1. **Zero Policy Hallucination**: Never promising unauthorized financial compensation or speculative delivery timelines.
2. **High-Precision Grounding**: Recommending only resolutions documented in historical resolved precedent cases.
3. **Safe, Explainable Escalation**: Immediately transferring ambiguous, frustrated, or high-risk inquiries to human agents with structured, auditable reasons.
4. **Sub-Second Processing Latency**: Returning complete triage, retrieval, draft reply, and routing decisions in under 1,000ms.

---

## 3. What Was Deliberately Not Built (Scope Discipline)

To preserve engineering focus and evaluation integrity, we explicitly chose not to build:
- **Autonomous Direct Dispatch to Live Customers**: Responses are presented as operator-assisted copilots rather than unmonitored direct replies.
- **Complex Multi-Agent Swarms**: Avoided multi-agent debate frameworks that introduce non-deterministic latency and unexplainable reasoning loops.
- **Generic Multi-Brand Generalization**: Intentionally constrained to a single brand's historical corpus rather than a diluted multi-industry chatbot.

---

## 4. Baseline Comparison & Evaluation Methodology

Three distinct classification systems were benchmarked against a held-out test split ($N=484$) and an independent human-annotated Golden Set ($N=200$):

1. **Baseline 1 (Majority Classifier)**: Trivial floor predicting the dominant intent unconditionally.
2. **Baseline 2 (TF-IDF + Logistic Regression)**: Transparent classical ML baseline with L2 regularization.
3. **Rule-Based Domain Taxonomy**: Priority-ordered expert heuristics capturing critical safety keywords.

### Intent Classification Empirical Benchmarks

| System / Model Architecture | Held-Out Test ($N=484$) Acc / Macro F1 | Human Golden Set ($N=200$) Acc / Macro F1 | Role & Status |
| :--- | :---: | :---: | :--- |
| **1. Majority Baseline** | 47.11% / 0.0915 | 15.00% / 0.0373 | Trivial statistical floor |
| **2. TF-IDF + Logistic Regression** | **84.92% / 0.7154** | **69.50% / 0.6619** | Statistical ML out-of-sample generalization |
| **3. Rule-Based Domain Taxonomy** | 100.0%* / 1.0000* | **98.50% / 0.9877** | Expert heuristic taxonomy |

*\*Disclosure:* The rule-based model scored 100% on the preprocessed splits because the dataset was initially structured using the rule taxonomy definitions. When audited against genuine independent human annotations on the Golden Set, rule-based accuracy is **98.50%** and statistical TF-IDF generalization is **69.50%**.

---

## 5. Independent Human Evaluation & Judge Agreement

The core tenet of this evaluation is: **the human evaluation was conducted as an audit of automated metrics, not a benchmark to inflate.** Rather than presenting uncalibrated automated scores as ground truth, we subjected the system to an independent human evaluation study across both response quality and operational routing.

### A. Human Evaluation Methodology
- **Response Quality Study ($N=100$)**: A stratified sample of 100 agent-generated responses was evaluated independently by a human rater alongside the automated LLM Judge (`judge_service.py`). Neither evaluator was influenced by the other's scores. Both evaluated responses across four standardized dimensions on a 1–5 Likert scale: Groundedness & Faithfulness, Answer Relevance & Helpfulness, Brand Voice & Empathy, and Safety & Policy Compliance.
- **Routing Agreement Study ($N=200$)**: All 200 Golden Set conversation threads were reviewed independently by a human annotator without exposure to system predictions, assigning either `AUTO_HANDLE` or `HUMAN_ESCALATION` based on operational complexity and risk.
- **No Post-Hoc Contamination**: As recorded in [`docs/DECISION_LOG.md`](../docs/DECISION_LOG.md), we strictly refrained from tuning thresholds or prompt rubrics post-hoc to artificially improve agreement scores.

### B. Human-vs-LLM Response Quality Results ($N=100$)
Source: [`experiments/human_judge_agreement.json`](../experiments/human_judge_agreement.json)

| Evaluation Dimension | Human Mean | LLM Judge Mean | Discrepancy (MAE) | Exact Match | Within-1-Pt | Spearman $\rho$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Groundedness & Faithfulness** | **3.35** | **4.03** | 1.44 pts | 26.0% | 59.0% | 0.0051 | 0.960 |
| **Answer Relevance & Helpfulness** | **3.01** | **3.57** | **0.90 pts** | 33.0% | **80.0%** | 0.0396 | 0.695 |
| **Brand Voice & Empathy** | **2.94** | **4.37** | **1.73 pts** | 13.0% | 49.0% | **-0.2022** | 0.044 |
| **Safety & Policy Compliance** | **3.53** | **5.00** | 1.47 pts | 40.0% | 61.0% | 0.0000* | 1.000 |
| **Overall Mean Quality Score** | **3.21 / 5.00** | **4.24 / 5.00** | **1.39 pts** | — | — | **0.0565** | 0.260 |

*\*Note: Safety Spearman correlation is mathematically 0.0000 due to zero variance in LLM judge outputs.*

### C. Human-vs-System Routing Results ($N=200$)
Source: [`experiments/human_routing_benchmarks.json`](../experiments/human_routing_benchmarks.json) & [`experiments/human_routing_audit.json`](../experiments/human_routing_audit.json)

- **Observed Agreement**: **38.00%**
- **Expected Agreement by Chance ($P_e$)**: **41.30%**
- **Cohen's Kappa ($\kappa$)**: **-0.0562** (*Very Low / Below Chance-Adjusted Agreement*)
- **Per-Class Breakdown**:
  - `AUTO_HANDLE` (Support: 42): Precision: **0.1846**, Recall: **0.5714**, F1: **0.2791**
  - `HUMAN_ESCALATION` (Support: 158): Precision: **0.7429**, Recall: **0.3291**, F1: **0.4561**

#### Confusion Matrix:
| Ground Truth vs. Runtime Decision | System: `AUTO_HANDLE` | System: `HUMAN_ESCALATION` | Total (Human) |
| :--- | :---: | :---: | :---: |
| **Human: `AUTO_HANDLE`** | **24** | 18 | 42 |
| **Human: `HUMAN_ESCALATION`** | **106** | **52** | 158 |
| **Total (System)** | 130 | 70 | 200 |

### D. What the Disagreement Tells Us
1. **The LLM judge is not a reliable proxy for human response-quality assessment in this evaluation.** With an overall correlation of $\rho = 0.0565$ ($p = 0.26$), automated scores provide no statistical guarantee of human customer approval.
2. **Systematic Score Inflation**: The LLM judge scores every dimension higher than humans (overall mean **4.24** vs **3.21**, +1.03 point inflation). The judge rewards polite, structured formatting, while humans demand authentic, concise problem resolution.
3. **Brand Voice as the Primary Point of Friction**: Disagreement is greatest on Brand Voice (Human 2.94 vs Judge 4.37, MAE 1.73, within-1 agreement 49.0%, Spearman $\rho = \mathbf{-0.2022}$, $p = 0.044$). The observed negative association is consistent with a mismatch between the automated brand-voice rubric and human judgments, with canned/polite phrasing appearing to receive higher automated scores than human scores. Causal explanations remain working hypotheses: generic or repetitive customer support phrasing (*"We understand your frustration! DM us to get started..."*) may satisfy surface politeness checks while receiving lower ratings from human evaluators who expect specific, actionable troubleshooting.
4. **Safety Score Saturation**: The LLM judge assigned a constant **5.00 / 5.00** across all 100 cases, failing to discriminate subtle boundary conditions, whereas human evaluators rated safety at **3.53 / 5.00**.
5. **Operational Routing Under-Escalation**: Observed evidence shows exactly 106 cases where human evaluators assigned `HUMAN_ESCALATION` but the automated system selected `AUTO_HANDLE`. Supported hypotheses for this divergence include: (a) single-turn message visibility at $t=0$ without multi-turn conversational history; (b) narrow explicit escalation triggers missing contextual severity; and (c) high intent confidence and retrieval similarity suppressing escalation on complex cases. A tiered escalation policy conditioned on intent category and severity is proposed as the primary next experiment.

### E. Limitations
- **Single Evaluator Horizon**: Human evaluations were conducted by a single technical annotator rather than a multi-annotator crowd, precluding inter-annotator Fleiss' kappa computation.
- **Single-Turn Context Window**: Inbound triage evaluated message turn $t=0$, whereas real support decisions depend on prior multi-turn conversational history.
- **Safety Ceiling Effect**: Automated safety judge prompts lack graduated scoring tiers for minor non-compliance.

### F. What Should Be Improved Next
1. **Intent-Conditioned Escalation Defaults**: Automatically default hardware battery jumps, physical screen damage, and 2FA lockouts to `HUMAN_ESCALATION`.
2. **Few-Shot Calibration for LLM Judge**: Calibrate judge prompts against the 100 human evaluation samples to align scoring distributions.
3. **Anti-Boilerplate Generation Prompts**: Penalize generic DM redirection phrases in synthesis to improve human brand tone ratings.
4. **NLI Entailment Grounding**: Replace lexical overlap checks with natural language inference cross-encoders to ensure directional grounding.

---

## 6. Top 5 Empirical Failure Modes

As documented in [`docs/FAILURE_ANALYSIS.md`](../docs/FAILURE_ANALYSIS.md):
1. **LLM Judge Overestimates Overall Response Quality**: Human mean 3.21 vs Judge mean 4.24 (MAE 1.39, Spearman $\rho = 0.0565$). Prompted judges reward surface fluency over substantive troubleshooting.
2. **Brand Voice & Empathy Mismatch**: Human 2.94 vs Judge 4.37 (MAE 1.73, Spearman $\rho = -0.2022$, $p = 0.044$). Observed negative association consistent with rubric mismatch; canned phrasing appears to score higher automatically than with humans.
3. **Routing Under-Escalation (106 False-Auto Cases)**: 106 human-escalation cases auto-handled by the system ($\kappa = -0.0562$, *Very Low / Below Chance-Adjusted Agreement*). Hypothesized to stem from single-turn visibility and confidence suppression over contextual severity.
4. **Safety & Policy Compliance Judge Saturation**: LLM judge saturated at 5.00 across all 100 cases, failing to discriminate subtle policy risks.
5. **Groundedness & Factuality Disagreement**: Human mean 3.35 vs Judge 4.03 (MAE 1.44, Spearman $\rho = 0.0051$). Lexical overlap with retrieved precedents does not ensure the precedent actually resolves the user's specific problem.

---

## 7. Mandatory Disclosure: "What is Misleading About My Headline Number?"

Reporting a headline metric in enterprise customer support AI requires rigorous engineering transparency:
1. **Rule-Based 100% Accuracy is a Labeling Artifact**: The rule-based taxonomy achieved 100% on initial preprocessed splits because the data was originally curated using the same taxonomy rules. Against independent human ground truth, rule-based accuracy is **98.50%**, and statistical TF-IDF generalization on the Golden Set is **69.50%** (84.92% on held-out test split).
2. **Intent Conditioning Artificially Boosts Retrieval Metrics (96.5% Recall@3)**: Intent-conditioned retrieval achieves 96.5% Recall@3 by restricting search to the predicted intent partition. If upstream classification misclassifies, the retriever searches the wrong partition. True unconditioned retrieval recall is **77.50% Recall@3**.
3. **LLM Judge 4.24 / 5.00 is Not Equivalent to Human Satisfaction**: While the automated LLM judge scored response quality at 4.24 / 5.00, independent human evaluation rated the exact same responses at **3.21 / 5.00**. The automated judge has weak correlation with human judgment ($\rho = 0.0565$) and systematically overlooks repetitive, robotic customer service phrasing.
4. **Heuristic Policy Concordance (89.5%, $\kappa = 0.8118$) Masks Human Routing Disagreement**: The high headline concordance metric measures agreement against automated heuristic rules. When evaluated against independent human routing decisions, actual agreement drops to **38.00%** ($\kappa = -0.0562$, *Very Low / Below Chance-Adjusted Agreement*), with **106 human-escalation cases incorrectly auto-handled**, hypothesized to stem from single-turn message visibility and narrow keyword triggers.
5. **Historical Twitter Reps Heavily Favored Direct Message Redirection**: Real historical `@AppleSupport` agents frequently escalated customer conversations to Direct Message simply to transition ticket volume off public timelines. An automated agent configured to deliver direct public troubleshooting will naturally diverge from historical agent behavior that favored private DM handoffs, depressing apparent routing agreement on routine diagnostic inquiries.

---

## 8. What Should Be Done With One More Week?

1. **Hybrid Retrieval (BM25 + Dense FAISS)**: Implement Reciprocal Rank Fusion (RRF) with metadata hard-filters on platforms, carriers, and product categories.
2. **Tiered Escalation Policy Calibration**: Overhaul the escalation policy engine using the 106 audited failure cases, establishing intent-risk defaults for hardware defects and 2FA lockouts.
3. **Human-Calibrated LLM Judge**: Re-anchor the judge rubric using few-shot ordinal anchors from the 100 human evaluations.
4. **NLI Grounding Gate**: Replace lexical overlap checks in `ResponseValidator` with natural language inference cross-encoder checks.

---

## 9. Limitations & Ethical Safeguards

- Social media data contains noisy, incomplete turns.
- Off-platform private message transitions limit observability of final customer satisfaction.
- The system enforces a strict circuit breaker, falling back to local heuristic replies whenever upstream cloud APIs fail.

---

## 10. Reproducibility & Timing Verification

### Can someone reproduce the headline number quickly?

**Yes.** Across three timed benchmark runs, total wall-clock runtime was **2.18–3.19 minutes, with a mean of 2.62 minutes** (130.54s, 191.43s, and 149.21s).

The automated reproduction script:
```bash
python scripts/reproduce_headline.py
```
evaluates the TF-IDF + Logistic Regression baseline on the held-out test split (Accuracy: 84.92%, Macro F1: 0.7154), evaluates dense FAISS vector retrieval across the 200 Golden Set cases (Unconditioned Recall@3: 77.50%, Intent-Conditioned Recall@3: 96.50%), runs the 6-barrier deterministic safety validation gate (81.0% pass rate), and executes all 122 automated pytest tests. Complete empirical evidence is documented in [`docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md).

---

## 11. AI Tools Disclosure

In accordance with academic and professional transparency standards:
- AI coding assistants were used during development for brainstorming, implementation assistance, documentation refinement, debugging, and test-case ideation.
- All evaluation data, hand-curated Golden Set annotations ($N=200$), and independent human ratings ($N=100$) were directly provided by human evaluation and were **not** fabricated, synthesized, or generated by AI tools.
- The evaluated system was not tuned or re-calibrated post-hoc using human ratings, preserving test-set independence.
