# Domain-Specific Intent Taxonomy Specification (@AppleSupport)

## 1. Executive Summary & Purpose

In production customer support operations, intent classification serves as the foundational decision gate for all downstream workflows: context assembly, historical case retrieval, grounded response formulation, and automated vs. human escalation routing.

This document formalizes the **7-class Mutually Exclusive, Collectively Exhaustive (MECE)** intent taxonomy designed specifically for `@AppleSupport` customer interactions. It is derived from empirical frequency analysis of 15,000+ `@AppleSupport` tweets in the Kaggle *Customer Support on Twitter* dataset.

---

## 2. Taxonomy Overview

| Intent Code | User-Facing Label | Scope & Core Problems | Typical Routing | Empirical Freq |
| :--- | :--- | :--- | :---: | :---: |
| `OPERATING_SYSTEM_UPDATES` | **OS & iOS Updates** | iOS/macOS version glitches, update verification freezes, keyboard autocorrect bugs, lag | `AUTO_HANDLE` | ~32% |
| `BATTERY_POWER_HARDWARE` | **Battery & Hardware** | Rapid battery discharge, thermal overheating, charging port failure, physical display damage | `HUMAN_ESCALATION` | ~21% |
| `ACCOUNT_APPLE_ID` | **Apple ID & Account** | Two-factor authentication (2FA) lockout, password resets, Apple ID region changes, iCloud storage | `HUMAN_ESCALATION` | ~16% |
| `CONNECTIVITY_NETWORKING` | **Connectivity & Wi-Fi** | Wi-Fi disconnects, Bluetooth pairing failures, cellular data dropouts, AirDrop / Home Sharing | `AUTO_HANDLE` | ~12% |
| `AUDIO_ACCESSORIES` | **Audio & Accessories** | AirPods pairing/volume, headphone jack adapters, microphone distortion, Apple Watch bands | `AUTO_HANDLE` | ~8% |
| `SUBSCRIPTIONS_BILLING` | **Billing & Subscriptions** | App Store unauthorized charges, refund disputes, recurring subscription cancellations, receipts | `HUMAN_ESCALATION` | ~6% |
| `GENERAL_INQUIRY` | **General Support** | Retail store hours, device launch reservations, feedback, ambiguous or unclassified queries | `HUMAN_ESCALATION` | ~5% |

---

## 3. Detailed Class Specifications

### 3.1 `OPERATING_SYSTEM_UPDATES`
- **Scope**: Inquiries regarding system software updates (iOS 11, macOS High Sierra, watchOS), update installation freezes, post-update performance degradation, UI glitches, autocorrect anomalies (e.g. the historical iOS 11 letter "I" bug), and crash loops.
- **Key Signals & Terms**: `ios`, `update`, `verifying update`, `freeze`, `stuck`, `glitch`, `autocorrect`, `keyboard`, `reboot`, `lag`, `slow after update`.
- **Default Routing**: `AUTO_HANDLE` (Standard troubleshooting steps: force restart, check storage, reinstall via iTunes, toggle keyboard settings).
- **Canonical Examples**:
  - *"My iPhone has been stuck on 'Verifying update' for iOS 11 for over 3 hours. How do I fix this?"*
  - *"Why does my keyboard keep changing the letter 'I' into an 'A' with a symbol?"*
  - *"Updated to iOS 11.0.3 and now my apps crash every time I swipe up."*

---

### 3.2 `BATTERY_POWER_HARDWARE`
- **Scope**: Inquiries regarding power delivery, rapid battery depletion, unexpected shutdowns, severe thermal throttling/overheating, physical screen fractures, swelling batteries, broken buttons (Home button, power toggle), or liquid immersion.
- **Key Signals & Terms**: `battery`, `drain`, `percentage drops`, `heat`, `burning hot`, `swollen`, `charger`, `won't charge`, `cable`, `shattered screen`, `cracked display`, `home button`, `hardware defect`.
- **Default Routing**: `HUMAN_ESCALATION` (Safety hazard, physical repair intake, Genius Bar appointment scheduling).
- **Canonical Examples**:
  - *"My iPhone 7 battery percentage drops from 80% to 15% in less than an hour and the back gets burning hot."*
  - *"My phone fell down stairs, screen shattered completely and battery is visibly swollen."*
  - *"My lightning cable won't charge my phone unless I hold it at a specific angle."*

---

### 3.3 `ACCOUNT_APPLE_ID`
- **Scope**: Inquiries regarding user authentication, Apple ID recovery, Two-Factor Authentication (2FA) verification code delivery, locked accounts, password reset loops, iCloud account syncing, or Apple ID country/region changes.
- **Key Signals & Terms**: `apple id`, `icloud`, `2fa`, `two-factor`, `verification code`, `locked out`, `password reset`, `region change`, `country change`, `login error`.
- **Default Routing**: `HUMAN_ESCALATION` (Identity verification required, security risk, privacy policy boundaries).
- **Canonical Examples**:
  - *"I'm locked out of my Apple ID because I changed my mobile number and cannot receive the two-factor code."*
  - *"Need help changing the region on my Apple ID so I can download UK apps."*
  - *"Forgot my Apple ID password and the recovery email is no longer accessible."*

---

### 3.4 `CONNECTIVITY_NETWORKING`
- **Scope**: Inquiries regarding wireless connectivity protocols including Wi-Fi network disconnection, Bluetooth peripheral discovery, cellular reception/SIM errors, AirDrop failures, Home Sharing media streaming, and hotspot drops.
- **Key Signals & Terms**: `wifi`, `wi-fi`, `bluetooth`, `bt`, `cellular`, `no service`, `searching`, `airdrop`, `hotspot`, `home sharing`, `lte`, `disconnects`, `pairing`.
- **Default Routing**: `AUTO_HANDLE` (Standard network reset, toggle airplane mode, forget Wi-Fi network, renew DHCP lease).
- **Canonical Examples**:
  - *"Why does my Wi-Fi keep disconnecting every time my phone locks?"*
  - *"My iPhone won't connect to my car Bluetooth after updating to iOS 11."*
  - *"Phone shows 'Searching...' for cellular service for past 2 hours even with full SIM."*

---

### 3.5 `AUDIO_ACCESSORIES`
- **Scope**: Inquiries involving audio hardware and peripherals, including AirPods (pairing, case charging, single earbud failure), EarPods, Lightning-to-3.5mm headphone jack adapters, microphone muffled audio, speaker crackling, and Apple Watch accessories.
- **Key Signals & Terms**: `airpods`, `earpods`, `headphone`, `jack`, `adapter`, `dongle`, `audio`, `sound`, `speaker`, `microphone`, `crackling`, `stutter`, `volume`, `earbud`.
- **Default Routing**: `AUTO_HANDLE` (Reset AirPods case, inspect connector lint, clean speaker grilles, check Balance slider).
- **Canonical Examples**:
  - *"My right AirPod won't connect or charge in the case, only the left one works."*
  - *"The lightning headphone adapter has static noise when I move the cord."*
  - *"Callers say my voice sounds muffled when using speakerphone on iPhone 8."*

---

### 3.6 `SUBSCRIPTIONS_BILLING`
- **Scope**: Inquiries regarding financial transactions, unauthorized credit card charges from ITUNES.COM/BILL, accidental in-app purchases, recurring subscription cancellation, refund requests, and family sharing billing.
- **Key Signals & Terms**: `subscription`, `refund`, `charged`, `billing`, `itunes.com/bill`, `receipt`, `invoice`, `unauthorized purchase`, `cancel subscription`, `app store purchase`.
- **Default Routing**: `HUMAN_ESCALATION` (Financial liability, refund approval limits, fraud detection protocols).
- **Canonical Examples**:
  - *"I was charged $9.99 on my bank statement from Apple for an app I never downloaded. I want a refund."*
  - *"How do I cancel my Apple Music trial before it auto-renews tomorrow?"*
  - *"My child made accidental in-app purchases on Roblox, how can I reverse the charges?"*

---

### 3.7 `GENERAL_INQUIRY`
- **Scope**: Inquiries that do not fall into specific technical or billing categories, including Apple Store operating hours, product reservation status (e.g. iPhone X pre-order deliveries), marketing questions, general feedback, or unintelligible inquiries.
- **Key Signals & Terms**: `store hours`, `reservation`, `order status`, `delivery`, `iphone x pre-order`, `genius bar appointment`, `feedback`, `help`, `anyone there`.
- **Default Routing**: `HUMAN_ESCALATION` / Direct referral to Apple Store or support website.
- **Canonical Examples**:
  - *"Hello, are all the phone support lines closed for tonight?"*
  - *"I got an email saying my iPhone X is reserved for the 3rd, how do I confirm pickup?"*
  - *"Where is the nearest Apple Store in London open on Sunday?"*

---

## 4. Disambiguation & Tie-Breaking Rules

When a customer message exhibits signals spanning multiple intents, annotators and classifiers MUST apply the following priority hierarchy:

1. **Safety & Financial Priority**:
   - Swelling battery or physical hazard -> `BATTERY_POWER_HARDWARE` (Overrides OS or General).
   - Unauthorized charge or money dispute -> `SUBSCRIPTIONS_BILLING` (Overrides App Store or OS).
   - Account lockout or 2FA failure -> `ACCOUNT_APPLE_ID` (Overrides OS).
2. **Peripheral vs. Core System**:
   - If an audio device (AirPods, headphones) fails to connect via Bluetooth, classify under `AUDIO_ACCESSORIES` rather than `CONNECTIVITY_NETWORKING`.
3. **Causal Update Trigger**:
   - If an update caused battery drain ('battery drains fast after iOS 11 update'), classify under `BATTERY_POWER_HARDWARE` because power management is the core issue requiring resolution.
4. **Ambiguity Fallback**:
   - If confidence across all domain classes is below threshold (< 0.50) or query is devoid of technical detail, classify as `GENERAL_INQUIRY`.

---

## 5. Annotation Guidelines for Golden Set Curation

In Phase 5, human annotators will curate a 150-250 dialogue Golden Evaluation Set. Annotators must adhere to:
- **Zero Mock Ground Truth**: Only annotate real customer inquiries reconstructed from `twcs.csv`.
- **Context Inspection**: Always examine the full multi-turn conversation thread to verify the true root cause before confirming the label.
- **Inter-Annotator Agreement**: Discrepancies between annotators must be resolved using the Disambiguation Hierarchy in Section 4.
