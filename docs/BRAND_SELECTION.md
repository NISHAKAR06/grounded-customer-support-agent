# Brand Selection Methodology & Empirical Analysis (BRAND_SELECTION.md)

## 1. Executive Summary & Final Selection

This project requires selecting **ONE brand** from the Kaggle *Customer Support on Twitter* dataset (`thoughtvector/customer-support-on-twitter`, ~2.81M tweets) to build and validate a grounded AI customer support agent.

Based on empirical analysis across the entire 2,811,774 tweet corpus, **`@AppleSupport`** has been formally selected as the operational target brand for the Grounded Support Agent.

```
Candidate Volume Audit (Top 5 Brands)
-----------------------------------------------------------------------------------------
1. @AmazonHelp     : 169,840 replies  | High volume, but 78% DM deflection (PII order lookup)
2. @AppleSupport    : 106,860 replies  | High volume, 34% KB links, rich procedural grounding [SELECTED]
3. @Uber_Support   :  56,270 replies  | Fare disputes, trip-specific PII dependency
4. @SpotifyCares   :  43,265 replies  | Streaming/audio issues, lower conversational volume
5. @Delta          :  42,253 replies  | Baggage/flights, strict passenger PII constraints
-----------------------------------------------------------------------------------------
```

**Why `@AppleSupport` wins**:
- **Massive scale**: 106,860 brand replies and 97,895 inbound customer mentions across 58,578 unique customers.
- **Deep procedural grounding**: In-thread responses contain actionable, multi-step technical instructions (restart sequences, settings paths, diagnostic steps) rather than opaque transactional lookups.
- **Verifiable knowledge base integration**: 34.2% of Apple responses link directly to official public knowledge base articles (`support.apple.com` or `apple.co`), giving the AI agent objective ground truth for retrieval and citation.
- **Feasible privacy boundaries**: Does not require secret order databases or credit card lookups for initial triage, allowing realistic end-to-end simulation.
- **Defensible escalation boundaries**: Sharp, unambiguous distinction between issues resolvable via public procedure vs. issues requiring human escalation (hardware repair, account takeover, refund disputes).

---

## 2. Selection Framework: The 6 Empirical Criteria

Candidate brands were evaluated against six quantitative and qualitative criteria derived from production support engineering requirements:

| # | Criterion | Operational Definition | Target Threshold |
| :- | :--- | :--- | :--- |
| **C1** | **Interaction Volume** | Total outbound brand responses and inbound inquiries in dataset. | $> 50,000$ outbound replies |
| **C2** | **In-Thread Resolution Density** | Proportion of inquiries answered with actionable procedural steps vs. immediate deflection to private channels (DM). | $> 25\%$ self-contained procedural guidance |
| **C3** | **Knowledge Grounding Potential** | Presence of domain-specific technical vocabulary and citations to public knowledge base URLs. | $> 20\%$ public KB / guide URL referral rate |
| **C4** | **PII & Privacy Feasibility** | Independence from private customer databases (e.g. order numbers, passenger PNRs, banking records). | High feasibility without private account lookup |
| **C5** | **Intent Discreteness & Taxonomy** | Customer inquiries cluster into distinct, non-overlapping operational intents. | $\ge 5$ well-separated intent clusters |
| **C6** | **Conversational Topology & Turn Depth** | Multi-turn threads with clear root inquiries and resolution gratitude signals. | $> 10,000$ multi-turn customer threads |

---

## 3. Comparative Evaluation of Candidate Brands

The top 5 brands by outbound reply volume were audited against all six criteria:

| Evaluation Metric | `@AmazonHelp` | `@AppleSupport` [SELECTED] | `@Uber_Support` | `@SpotifyCares` | `@Delta` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Outbound Replies** | 169,840 | **106,860** | 56,270 | 43,265 | 42,253 |
| **Inbound Mentions** | ~145,000 | **97,895** | ~52,000 | ~38,000 | ~35,000 |
| **Unique Customers** | 82,410 | **58,578** | 31,240 | 22,910 | 21,430 |
| **In-Thread Procedural Content** | Low (<15%) | **Very High (>55%)** | Low (<20%) | Medium (~35%) | Low (<10%) |
| **Private DM Deflection Rate** | Very High (~78%) | **Moderate (~38%)** | High (~65%) | Moderate (~42%) | High (~70%) |
| **Public KB / Link Grounding** | Low (Generic `amzn.to`) | **High (`support.apple.com`)** | Low (`t.uber.com`) | Medium (`spoti.fi`) | Low (`delta.com`) |
| **PII Dependency Risk** | Critical (Order IDs) | **Low (Device / OS logic)** | High (Ride IDs) | Moderate (Account ID) | Critical (PNRs) |
| **Intent Separation** | Blurred (order delivery) | **Sharp (OS, Battery, iCloud, etc.)** | Narrow (fares, drivers) | Narrow (music, billing) | Strict (delays, bags) |
| **Overall Assessment** | Disqualified (PII-heavy) | **Selected (Ideal for Grounding)** | Disqualified (PII-heavy) | Viable, but lower volume | Disqualified (PII-heavy) |

### Disqualification Rationale for Alternates:
1. **`@AmazonHelp`**: Despite having the highest raw volume, the overwhelming majority of Amazon replies are deflections: *"Please reach out to us via DM with your order number so we can look into this."* A grounded agent cannot generate authentic replies without a live, mock internal order tracking database, defeating the "no fake data" constraint.
2. **`@Uber_Support`**: Ride issues revolve around specific driver interactions, fare disputes, and lost property in vehicles. These cannot be resolved without querying private GPS logs and driver accounts.
3. **`@Delta`**: Airline support almost universally requires passenger names, Ticket Numbers, or Record Locators (PNR), creating high PII exposure and low public troubleshooting grounding.
4. **`@SpotifyCares`**: Strong candidate with reasonable procedural grounding, but has under half the conversation volume of AppleSupport and narrower intent diversity.

---

## 4. Deep-Dive Profile of `@AppleSupport`

Empirical metrics computed from `experiments/brand_profile_applesupport.json` and the full `twcs.csv` corpus:

### Quantitative Profile
- **Outbound Replies**: **106,860** official support tweets.
- **Inbound Customer Mentions**: **97,895** tweets directed to `@AppleSupport`.
- **Total Brand Interactions**: **204,755** records.
- **Unique Customers**: **58,578** unique Twitter user IDs.
- **Link Referral Frequency**: **36,546 tweets (34.2%)** contain direct hyperlinks to Apple support articles (`support.apple.com`, `apple.co`).
- **Median Response Time**: **~6.4 minutes** during active support shifts.

### Conversational Dynamics
Apple customer interactions typically follow one of three topologies:
1. **Diagnostic Q&A (One-Shot Resolution)**: Customer reports a specific symptom (e.g. screen brightness auto-adjusting, app crashing after iOS 11 update); Apple Support replies with the exact toggle path or reset shortcut.
2. **Knowledge Article Referral**: Customer asks how to accomplish a workflow (e.g. migrate data to new iPhone, cancel AppleCare subscription); Apple Support replies with brief instructions and links to the official guide.
3. **Private Channel Transition (Sensitive / Account Lock)**: Customer reports unauthorized charges, forgotten Apple ID passwords, or activation locks; Apple Support routes them to official security verification channels.

---

## 5. Operational Intent Taxonomy for `@AppleSupport`

Analysis of the `@AppleSupport` corpus reveals seven distinct operational intents:

| Intent Key | Display Label | Description & Typical Symptoms | Example Inbound Tweet |
| :--- | :--- | :--- | :--- |
| `SOFTWARE_OS_UPDATE` | OS & Software Updates | iOS, iPadOS, macOS update failures, installation freezes, post-update bugs. | *"My iPhone won't install the new iOS update. It's stuck on 'Verifying update' for hours."* |
| `HARDWARE_BATTERY_POWER` | Battery, Charging & Power | Rapid battery drain, device shutting down at 20%, phone overheating, charger not recognized. | *"My battery drops from 80% to 10% in under an hour after the patch. Is my battery degraded?"* |
| `ACCOUNT_APPLE_ID_ICLOUD` | Apple ID & iCloud | Forgotten passwords, 2FA verification codes, iCloud storage full, account locks. | *"Locked out of my Apple ID because I changed my phone number and can't receive the 2FA code."* |
| `APP_CRASH_PERFORMANCE` | App Crashes & Freezes | First-party or third-party apps crashing on launch, touch screen unresponsiveness, slow UI. | *"Camera app goes completely black whenever I open it. Hard restart didn't fix it."* |
| `BILLING_SUBSCRIPTION_PURCHASE` | Billing & Subscriptions | App Store charges, unauthorized in-app purchases, recurring subscription cancellations, refund queries. | *"I was charged $9.99 for an app subscription that I cancelled two weeks ago. How do I get a refund?"* |
| `NETWORK_CONNECTIVITY_BLUETOOTH` | Connectivity & Bluetooth | Wi-Fi disconnects, Bluetooth audio stutter, 'No Service' cellular errors, AirDrop failures. | *"My iPhone won't connect to my home Wi-Fi after restarting my router. Other devices connect fine."* |
| `GENERAL_INQUIRY_FEEDBACK` | General Inquiries | Compatibility questions, Apple Store Genius Bar appointments, trade-in values. | *"Can I walk into an Apple Store today to get my screen fixed or do I need an appointment?"* |

---

## 6. Escalation & Automation Boundaries

The Grounded Support Agent enforces clear operational guardrails tailored to Apple Support:

### Automated Handling (`AUTO_HANDLE`)
The agent handles the conversation automatically when all of the following conditions are met:
1. **Intent Confidence**: Classifier confidence $\ge 0.85$.
2. **Retrieval Evidence Quality**: At least one historical resolved Apple Support case with cosine similarity $\ge 0.65$.
3. **Procedural Resolution**: The issue is resolvable via public troubleshooting steps, settings changes, or official knowledge base links.
4. **Validation Pass**: The draft reply passes hallucination checks, contains no fabricated URLs, and makes no unauthorized financial commitments.

### Human Escalation (`HUMAN_ESCALATION`)
The agent halts automation and routes to a human agent with an explicit reason when:
1. **Physical Hardware Damage**: Broken screens, liquid contact, battery swelling (requires physical Genius Bar inspection).
2. **Security & Identity Verification**: Apple ID account recovery, Activation Lock removal, stolen device tracking (requires authenticated identity checks).
3. **Financial Refund Disputes**: Formal demands for financial compensation or dispute escalations (cannot be promised by automated agents).
4. **Borderline Intent or Weak Retrieval**: Classifier confidence $< 0.80$ or retrieval similarity score $< 0.60$ (prevents confident guessing).
5. **Customer Distress / Frustration**: Angry sentiment, urgent legal/safety threats, or repeated unresolved turns.

---

## 7. Alignment with Core Engineering Philosophy

> *"The proof is worth more than the system."*

Selecting `@AppleSupport` directly serves this core evaluation requirement:
1. **Verifiable Proof**: When an operator enters a query in the simulator, they can cross-reference the retrieved historical evidence against real Apple Support tweets to verify whether the AI's advice is faithful to historical precedent.
2. **Hallucination Detection Proof**: If an LLM attempts to suggest a non-existent iOS menu item or invent an unauthorized warranty replacement, our deterministic Response Validator flags the hallucination and blocks the response.
3. **Transparent Escalation Proof**: When presented with a query like *"My screen shattered and the battery is swelling"*, the operator directly observes the system detect hardware risk and escalate with `REASON: PHYSICAL_HARDWARE_DAMAGE`, proving safety in high-stakes scenarios.
