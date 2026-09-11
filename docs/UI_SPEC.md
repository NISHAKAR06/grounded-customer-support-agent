# User Interface Specification (UI_SPEC.md)

## 1. Design Principles & Aesthetic Identity

The user interface for the **Grounded Customer Support Agent** is designed as a serious, professional B2B SaaS customer support operations tool inspired by high-contrast, clean modern email and ticket management platforms.

### Key Visual Tenets:
1. **High-Contrast Monochrome Palette**: A dark navy/charcoal sidebar (`#0f172a`) paired with clean white cards (`#ffffff`) over an off-white workspace canvas (`#f8fafc`).
2. **Restrained Typography**: Built with modern sans-serif typography (`Inter`), crisp line-heights, and distinct hierarchy.
3. **Thin Borders & Clean Separation**: Subtle 1px borders (`#e2e8f0`) defining structure without heavy drop-shadows or gradients.
4. **Functional Semantic Coloration**:
   - **Green (`#10b981`)**: `AUTO_HANDLE`, high confidence, verified grounding checks passed.
   - **Red (`#ef4444`)**: `HUMAN_ESCALATION`, critical alerts, risk flags.
   - **Amber (`#f59e0b`)**: Borderline confidence, review required.
5. **No AI Gimmicks**: No glowing neon borders or floating orbs. Every visual transition corresponds to actual backend execution.

---

## 2. Shell Layout Architecture

```
┌─────────────────────────┬─────────────────────────────────────────────────────────────┐
│ Grounded Support AI     │ Top Navigation Bar: Breadcrumbs / Active Environment Status │
│ [● System Online]       ├─────────────────────────────────────────────────────────────┤
├─────────────────────────┤                                                             │
│ WORKSPACE               │                                                             │
│   Inbox                 │                                                             │
│  ► Simulate (Primary)   │                      MAIN WORKSPACE                         │
│                         │               (White Cards / Off-White Canvas)              │
│ INSIGHTS                │                                                             │
│   Evaluation            │                                                             │
│   Failure Analysis      │                                                             │
│                         │                                                             │
│ ENGINEERING             │                                                             │
│   Decision Log          │                                                             │
│   Methodology           │                                                             │
└─────────────────────────┴─────────────────────────────────────────────────────────────┘
```

---

## 3. Primary Page: Simulate Incoming Message (`/simulate`)

### 3.1 Layout Split
- **Left Panel (Composer & Controls)**:
  - Textarea: Customer message composer.
  - Quick-Load Example Buttons (Address Change, Duplicate Charge, App Crash, Human Escalation Trigger).
  - Primary Action Button: `Run AI Agent`.
- **Right Panel (Real-Time Agent Activity Timeline)**:
  - Header: `AI AGENT ACTIVITY`
  - Step items updated dynamically: Message Received → Intent Classification → Historical Retrieval → Grounded Reply Generation → Response Validation → Escalation Policy Routing.

### 3.2 Case Details View
- **AI Analysis**: Primary intent name, confidence percentage, extracted signals.
- **Historical Evidence**: Top retrieved resolved cases with case ID, similarity score, customer text, brand response.
- **Grounded Reply**: Draft response with validation checklist badges.
- **Routing & Escalation**: Decision banner (`AUTO_HANDLE` or `HUMAN_ESCALATION`) with deterministic reasons.
