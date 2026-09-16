# Empirical Failure Analysis (FAILURE_ANALYSIS.md)

## 1. Overview & Evaluation Integrity

This document systematically identifies and investigates the **top 5 empirical failure modes** discovered during the independent human evaluation of the Grounded Customer Support Agent across the 200 Golden Set cases and 100 human-rated response samples.

A core principle of this project is: **"The proof is worth more than the system."** We do not present fabricated perfection; rather, we systematically audit vulnerabilities, categorize error types, and formulate actionable engineering experiments.

---

## 2. Top 5 Empirical Failure Modes

### Failure Mode 1: LLM Judge Overestimates Overall Response Quality
- **Failure Category**: Automated Quality Evaluation & Calibration
- **Empirical Evidence**:
  - Independent Human Mean Score: **3.21 / 5.00**
  - Automated LLM Judge Mean Score: **4.24 / 5.00**
  - Overall Mean Absolute Error (MAE): **1.39 points**
  - Overall Spearman Rank Correlation: $\rho = 0.0565$ ($p = 0.26$)
- **Why It Matters**:
  - An automated judge reporting 4.24/5 creates a dangerous false confidence that the system is production-ready, whereas human evaluators rate the exact same responses at a mediocre 3.21/5. Relying on the automated judge without human calibration risks deploying an agent that frustrates real users.
- **Hypothesis**:
  - The automated LLM judge is biased toward surface-level fluency, grammatically polite structures, and standard customer support pleasantries. It rarely penalizes responses unless blatant toxicity or hallucinations occur, whereas human evaluators demand concise, genuinely helpful problem resolution.
- **Proposed One-Week Experiment**:
  - Train an ordinal calibration adapter or tune few-shot judge prompts using the 100 human evaluation samples as few-shot calibration anchors, establishing explicit scoring penalties for evasive, non-committal answers.

---

### Failure Mode 2: Brand Voice & Empathy Mismatch
- **Failure Category**: Tone & Conversational Quality
- **Empirical Evidence**:
  - Independent Human Mean Score: **2.94 / 5.00**
  - Automated LLM Judge Mean Score: **4.37 / 5.00**
  - Discrepancy (MAE): **1.73 points** (largest of all dimensions)
  - Exact Agreement: **13.0%** | Within-1-Point Agreement: **49.0%**
  - Spearman Rank Correlation: $\rho = -0.2022$ ($p = 0.044$, statistically significant negative correlation)
- **Why It Matters**:
  - The observed negative association ($\rho = -0.2022$, $p = 0.044$) is consistent with a mismatch between the automated brand-voice rubric and human judgments, with canned/polite phrasing appearing to receive higher automated scores than human scores. Causal explanations remain working hypotheses.
- **Real Artifact Example** (Case `gold_001` / `conv_38882`):
  - *Customer*: *"My iphone 6 has crackling sound coming on call and speaker too"*
  - *Generated Response*: *"We'd be happy to help with the sound on your iPhone 6. To clarify, does this happen on all calls? Send us a DM."*
  - *Discrepancy*: The LLM judge scored Tone 5/5 for courtesy and brand standard, while human evaluators scored 2/5 or 3/5 due to repetitive, scripted DM redirection without addressing hardware diagnostics.
- **Working Hypothesis**:
  - Generic, canned support phrasing (*"We're here to help! DM us your details"*) may satisfy the automated judge's surface politeness checklist, but human raters may perceive it as robotic, evasive, or lacking genuine technical empathy.
- **Proposed One-Week Experiment**:
  - Incorporate anti-canned-response penalties into the response generator prompt and implement a specialized empathy scoring rubric that rewards specific diagnostic guidance and penalizes generic redirection language.

---

### Failure Mode 3: Routing Under-Escalation (106 False-Auto Cases)
- **Failure Category**: Escalation Policy Engine Calibration
- **Empirical Evidence**:
  - Human vs. System Routing Agreement: **38.00%** (Cohen's $\kappa = -0.0562$, *Very Low / Below Chance-Adjusted Agreement*)
  - Human `HUMAN_ESCALATION` Ground Truth: **158 cases**
  - System `AUTO_HANDLE` Decisions on Human Escalations: **106 cases**
  - System `HUMAN_ESCALATION` Recall on Human Escalations: only **32.91%** (52 / 158)
- **Why It Matters**:
  - Under-escalation is a critical operational failure mode in customer support. Attempting to automate inquiries that require human intervention risks forcing customers into conversational dead-ends on issues requiring specialist diagnostic or administrative support.
- **Real Artifact Example** (Case `gold_003` / `conv_44498`):
  - *Customer*: *"I have an iPhone 6s and my battery percentage jumps around wildly and drops from 40% to 1% in seconds."*
  - *System Decision*: `AUTO_HANDLE` (Intent: `BATTERY_POWER_HARDWARE`, Confidence: 0.96, Similarity: 0.74, no safety violations).
  - *Human Ground Truth*: `HUMAN_ESCALATION` (Severe hardware degradation requiring Genius Bar battery replacement diagnosis).
- **Supported Hypotheses**:
  - *Inbound Message Scope*: The system operates on the inbound message ($t=0$) rather than the full conversational lifecycle.
  - *Narrow Explicit Triggers*: Explicit escalation keyword triggers (`fraud`, `sue`, `lawyer`) miss contextual technical severity or multi-turn conversational deadlocks.
  - *Confidence Suppression*: High intent classification confidence and high retrieved similarity can suppress escalation for complex scenarios that nevertheless require specialist human intervention.
- **Proposed One-Week Experiment**:
  - Evaluate a tiered escalation policy conditioned on intent category, conversational severity, repeated troubleshooting failures, and explicit private-channel requirements, re-evaluating recall on the 106 audited false-auto cases.

---

### Failure Mode 4: Safety & Policy Compliance Judge Saturation
- **Failure Category**: Automated Evaluation Rubric Discriminability
- **Empirical Evidence**:
  - Automated LLM Judge Mean Score: **5.00 / 5.00** across all 100 cases (variance = 0.00)
  - Independent Human Mean Score: **3.53 / 5.00**
  - Discrepancy (MAE): **1.47 points**
  - Spearman Rank Correlation: $\rho = 0.0000$ ($p = 1.000$, degenerate due to zero variance)
  - Exact Agreement: **40.0%**
- **Why It Matters**:
  - A metric that evaluates every single output as 5.00 is completely non-discriminative. It provides zero visibility into subtle policy edge cases, minor boundary crossings, or unhelpful advisories.
- **Hypothesis**:
  - The automated safety rubric prompt is calibrated exclusively to detect severe catastrophic failures (hate speech, public PII leaks, explicit hazardous battery explosion instructions). It fails to penalize borderline compliance issues (such as suggesting third-party workarounds or providing generic advice without necessary diagnostic disclaimers) that human raters penalize.
- **Proposed One-Week Experiment**:
  - Re-engineer the safety judge prompt into a multi-tier deduction rubric that starts at 5.00 and deducts points for unverified external links, ungrounded diagnostic assertions, or missing mandatory safety disclaimers.

---

### Failure Mode 5: Groundedness & Factuality Disagreement
- **Failure Category**: Retrieval Grounding Verification
- **Empirical Evidence**:
  - Independent Human Mean Score: **3.35 / 5.00**
  - Automated LLM Judge Mean Score: **4.03 / 5.00**
  - Discrepancy (MAE): **1.44 points**
  - Exact Agreement: **26.0%** | Within-1-Point Agreement: **59.0%**
  - Spearman Rank Correlation: $\rho = 0.0051$ ($p = 0.960$)
- **Why It Matters**:
  - Grounding is the central architectural premise of retrieval-augmented generation. The near-zero rank correlation ($\rho = 0.0051$) shows that automated lexical overlap checks do not measure whether a response is actually grounded in a way that resolves the user's specific problem.
- **Real Artifact Example** (Case `gold_005` / `conv_46261`):
  - *Customer*: *"iOS 11.0.3 broke my Wi-Fi toggle switch, it's greyed out in settings."*
  - *Retrieved Precedent*: Generic Wi-Fi connection troubleshooting (Reset Network Settings).
  - *Generated Response*: Advised resetting network settings and reconnecting to Wi-Fi.
  - *Discrepancy*: LLM judge rated Groundedness 5/5 because response matched retrieved text. Human rater scored 2/5 because greyed-out Wi-Fi is a known hardware chip failure on iPhone 6/6s that cannot be fixed by network reset, meaning the retrieved precedent was factually mismatched to the root cause.
- **Hypothesis**:
  - The presence of precedent-derived text satisfies the automated judge's surface grounding prompt, but human evaluators evaluate semantic suitability—whether the retrieved precedent actually addresses the customer's specific technical problem.
- **Proposed One-Week Experiment**:
  - Replace superficial lexical grounding prompts with an NLI-based (Natural Language Inference) premise-hypothesis entailment check that verifies directional logical support between retrieved facts and generated claims.

---

## 3. Systematic Mitigation Roadmap

| Priority | Targeted Failure Mode | Proposed Action | Target Milestone |
| :---: | :--- | :--- | :---: |
| **P0** | Mode 3 (Routing Under-Escalation) | Implement intent-risk tiering defaulting hardware/account lockouts to escalation | Week 1 |
| **P0** | Mode 1 & 4 (Judge Calibration & Saturation) | Deploy few-shot human anchor calibration and tiered deduction safety rubric | Week 1 |
| **P1** | Mode 2 (Brand Voice & Empathy) | Add anti-canned-response penalties and diagnostic prompt templates | Week 2 |
| **P1** | Mode 5 (Groundedness Verification) | Integrate NLI cross-encoder entailment gate into `ResponseValidator` | Week 2 |
