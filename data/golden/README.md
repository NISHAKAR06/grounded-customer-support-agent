# Golden Evaluation Set (Curated `@AppleSupport` Benchmark)

## 1. Overview & Dataset Profile

The **Golden Evaluation Set** is a rigorously curated, leak-free benchmark of **200 real customer support conversations** sampled directly from `@AppleSupport` interactions in the Kaggle Twitter support dataset (`thoughtvector/customer-support-on-twitter`).

In strict adherence to engineering integrity standards:
- **100% Real Customer Data**: Every customer message, turn history, and resolution is from real historical tweets.
- **Zero Mock or Fabricated Data**: No synthetic records or invented agent behaviors.
- **Leak-Free Partition**: The dataset is fully isolated under `data/golden/golden_set.jsonl` with fixed random seed (42) for deterministic reproducibility.
- **Comprehensive Stratification**: Samples are balanced across all 7 domain intent classes, containing both single-turn inquiries (60%) and multi-turn exchanges (40%).

```
Golden Set Summary:
Total Samples: 200 (Target: 150–250)
Status: Verified & Validated
Auto-Handle Candidates: 42 (21.0%)
Human Escalation Required: 158 (79.0%)
Multi-Turn Dialogue Chains: 80 (40.0%)
Single-Turn Root Inquiries: 120 (60.0%)
```

---

## 2. Intent Stratification Breakdown

| Intent Code | Human-Readable Name | Golden Samples | Stratification Focus | Key Edge Cases Included |
| :--- | :--- | :---: | :--- | :--- |
| `OPERATING_SYSTEM_UPDATES` | OS & iOS Updates | **40** | iOS 11 update failures, typing/autocorrect bugs, app freezing, post-update battery drain | Battery drain immediately following OS update (disambiguated to OS) |
| `BATTERY_POWER_HARDWARE` | Battery & Hardware | **35** | Rapid battery percentage drop, thermal overheating, cracked screens, swollen batteries | Swollen battery safety hazards, physical impact damage |
| `ACCOUNT_APPLE_ID` | Apple ID & Account | **30** | Two-Factor Authentication (2FA) lockout, forgotten Apple ID credentials, iCloud sync | Security verification, stolen/hacked account recovery |
| `CONNECTIVITY_NETWORKING` | Connectivity & Wi-Fi | **25** | Wi-Fi connection drops, cellular carrier network failure, Bluetooth pairing | Bluetooth audio disconnection vs. networking |
| `AUDIO_ACCESSORIES` | Audio & Accessories | **20** | AirPods sound cutoff, charging cables, Beats hardware, Apple Watch accessories | Accessory hardware defects vs. general Bluetooth |
| `SUBSCRIPTIONS_BILLING` | Billing & Subscriptions | **20** | Unauthorized App Store charges, recurring subscriptions, refund disputes | Accidental in-app purchases, credit card fraud disputes |
| `GENERAL_INQUIRY` | General Support | **30** | Store appointments, trade-in questions, warranty status, ambiguous inquiries | Multi-intent customer queries without specific hardware keywords |
| **Total** | | **200** | | |

---

## 3. Sampling Methodology

The dataset was curated using a two-stage stratified sampling pipeline (`scripts/evaluation/curate_golden_set.py`):
1. **Candidate Filtering**:
   - Ingested all 3,423 reconstructed conversations from `data/processed/applesupport_conversations.jsonl`.
   - Filtered out single-word gibberish or empty noise ($length < 15$ chars or $< 3$ words).
2. **Stratified Selection**:
   - Quotas enforced per intent to guarantee statistical representation of minority classes (e.g., Billing & Accessories) alongside high-volume classes (OS Updates).
   - Enforced a 40% multi-turn dialogue quota (turn count $> 2$) to evaluate the agent's ability to maintain context over multi-turn exchanges.
3. **Edge Case Injection**:
   - Deliberately sampled conflict queries (e.g. swollen battery during update, iTunes billing on store app) to test priority disambiguation rules.

---

## 4. Annotation Guidelines & Decision Rules

Every sample in `golden_set.jsonl` has been verified against the following formal annotation guidelines:

### 4.1 Intent Disambiguation Rules
- **Rule 1 (Hardware Safety Precedence)**: When a customer mentions both software symptoms and physical battery swelling or thermal smoke, the intent is classified as `BATTERY_POWER_HARDWARE` due to physical safety priority.
- **Rule 2 (Financial Precedence)**: When an App Store app download issue mentions unauthorized payment charges or credit card disputes, the intent is classified as `SUBSCRIPTIONS_BILLING`.
- **Rule 3 (Account Security Precedence)**: If a customer cannot access their device due to an Apple ID 2FA lockout, the intent is classified as `ACCOUNT_APPLE_ID`.
- **Rule 4 (Accessory Audio Precedence)**: Inquiries regarding AirPods, Beats, or external speakers are assigned to `AUDIO_ACCESSORIES` rather than `CONNECTIVITY_NETWORKING`.
- **Rule 5 (OS Upgrade Battery Drain)**: Complaints stating that battery life degraded specifically after installing an iOS update are classified under `OPERATING_SYSTEM_UPDATES`.

### 4.2 Routing Decision Criteria
A conversation is classified as `AUTO_HANDLE` only when:
1. The issue is a standard procedural or informational inquiry.
2. The turn count is low ($\le 3$).
3. The historical precedent did not require escalating to a private Direct Message (DM) or requesting sensitive personal credentials.

A conversation is classified as `HUMAN_ESCALATION` if any of the following triggers are present:
1. `CRITICAL_HARDWARE_SAFETY`: Physical swelling, fire/smoke risk, or cracked enclosure.
2. `ACCOUNT_SECURITY_RISK`: Password reset, 2FA bypass, or account compromise.
3. `FINANCIAL_DISPUTE`: Disputed transactions or refund requests.
4. `REPEATED_TROUBLESHOOTING_FAILURE`: Customer already attempted multiple troubleshooting steps without success (thread turns $\ge 5$).
5. `CUSTOMER_DISTRESS`: Expressed legal threats, extreme dissatisfaction, or churn intent.
6. `PRIVATE_CHANNEL_HANDOFF`: Brand support agent determined that private verification was mandatory.

---

## 5. Record Schema

Each line in `data/golden/golden_set.jsonl` adheres to the following JSON schema:

```json
{
  "sample_id": "gold_001",
  "conversation_id": "conv_43370",
  "ticket_id": "TICK-43370",
  "customer_message": "why is my iphone (6s, recent update) showing symbols and stuff as i️ type when autocorrect is on? @AppleSupport",
  "turn_count": 4,
  "conversation_turns": [
    {
      "turn_id": 1,
      "tweet_id": "43370",
      "author_id": "125615",
      "author_role": "CUSTOMER",
      "text": "why is my iphone (6s, recent update) showing symbols and stuff as i️ type when autocorrect is on? @AppleSupport",
      "raw_text": "why is my iphone (6s, recent update) showing symbols and stuff as i️ type when autocorrect is on? @AppleSupport",
      "created_at": "Wed Nov 01 15:43:54 +0000 2017"
    },
    {
      "turn_id": 2,
      "tweet_id": "43368",
      "author_id": "AppleSupport",
      "author_role": "BRAND",
      "text": "We're glad to help. To start, what version of iOS are you running? Find that in Settings > General > About.",
      "raw_text": "@125615 We're glad to help. To start, what version of iOS are you running? Find that in Settings &gt; General &gt; About.",
      "created_at": "Wed Nov 01 16:04:31 +0000 2017"
    }
  ],
  "gold_intent": "OS & iOS Updates",
  "gold_intent_code": "OPERATING_SYSTEM_UPDATES",
  "gold_routing": "HUMAN_ESCALATION",
  "gold_escalation_reason": "PRIVATE_CHANNEL_HANDOFF: Brand support agent escalated thread to secure direct message channel.",
  "gold_resolution": "Thank you. Let's get to DM to continue. https://t.co/GDrqU22YpT",
  "historical_grounding": {
    "root_tweet_id": "43370",
    "has_dm": true,
    "has_kb_link": false,
    "has_resolution": false
  },
  "complexity": "MULTI_TURN",
  "annotator_notes": "Classified as OPERATING_SYSTEM_UPDATES (MULTI_TURN)..."
}
```
