"""Intent classification and labeling pipeline for @AppleSupport conversations."""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.domain_models import INTENT_LABELS, SupportIntent  # noqa: E402

# Linguistic rule patterns for each canonical support intent
INTENT_PATTERNS: Dict[SupportIntent, List[Tuple[str, float]]] = {
    SupportIntent.OPERATING_SYSTEM_UPDATES: [
        (r"\b(ios\s*\d+|macos|high\s*sierra|watchos|tvos)\b", 0.95),
        (
            r"\b(verifying\s*update|installing\s*update|software\s*update|actualizaci[oó]n)\b",
            0.95,
        ),
        (r"\b(autocorrect|auto\s*correct|keyboard\s*glitch)\b", 0.92),
        (
            r"\b(letter\s*['\"]?i['\"]?|changes?\s*to\s*['\"]?a['\"]?|capital\s*['\"]?i['\"]?)\b",
            0.96,
        ),
        (r"\b(freeze|frozen|stuck|glitchy?|lag|laggy|reboot(ing)?)\b", 0.85),
        (
            r"\b(slow\s*after\s*update|apps?\s*crashing|crash\s*loop|stutter(s|ing)?)\b",
            0.88,
        ),
        (r"\b(update|updated|updating|upgrade|upgraded|firmware)\b", 0.78),
        (r"\b(app\s*store|apps?\s*won'?t\s*open|mail\s*app)\b", 0.82),
    ],
    SupportIntent.BATTERY_POWER_HARDWARE: [
        (
            r"\b(battery|percentage|drain(ing)?|drops?|battery\s*life|bater[ií]a)\b",
            0.95,
        ),
        (r"\b(burning\s*hot|overheat(ing)?|too\s*hot|warm|calienta)\b", 0.94),
        (r"\b(swollen|swelling|bulg(e|ing))\b", 0.98),
        (
            r"\b(won'?t\s*charge|not\s*charging|charger|charging\s*port|cable|cargador)\b",
            0.92,
        ),
        (
            r"\b(shatter(ed)?|cracked?\s*screen|broken\s*screen|display\s*flicker|pantalla)\b",
            0.95,
        ),
        (r"\b(home\s*button|power\s*button|volume\s*rocker|bot[oó]n)\b", 0.90),
        (r"\b(water\s*damage|dropped\s*in\s*water|liquid)\b", 0.95),
    ],
    SupportIntent.ACCOUNT_APPLE_ID: [
        (r"\b(apple\s*id|icloud|itunes\s*account)\b", 0.95),
        (r"\b(2fa|two-?factor|verification\s*code|auth\s*code)\b", 0.96),
        (r"\b(locked\s*out|account\s*disabled|password\s*reset)\b", 0.95),
        (r"\b(region\s*change|country\s*change|change\s*region)\b", 0.95),
        (r"\b(forgot\s*password|cannot\s*log\s*in|login\s*error)\b", 0.90),
        (r"\b(icloud\s*storage|backup\s*failed|restore\s*from\s*icloud)\b", 0.88),
    ],
    SupportIntent.CONNECTIVITY_NETWORKING: [
        (r"\b(wi-?fi|wifi|wireless|router|hotspot)\b", 0.95),
        (r"\b(bluetooth|bt\s*connection|pair(ing)?)\b", 0.92),
        (r"\b(cellular|lte|4g|no\s*service|searching\.\.\.|sim\s*card)\b", 0.94),
        (r"\b(airdrop|home\s*sharing|airplay|apple\s*tv)\b", 0.92),
        (r"\b(disconnect(s|ing|ed)?|drops?\s*connection|connect(ion)?)\b", 0.82),
    ],
    SupportIntent.AUDIO_ACCESSORIES: [
        (r"\b(airpods?|earpods?|earbuds?)\b", 0.96),
        (r"\b(headphones?|headset|earphones?)\b", 0.94),
        (r"\b(headphone\s*jack|lightning\s*adapter|dongle)\b", 0.95),
        (r"\b(microphone|mic\s*not\s*working|muffled|speaker\s*crackl(e|ing))\b", 0.92),
        (r"\b(audio\s*stutter|no\s*sound|low\s*volume|distort(ed|ion))\b", 0.88),
        (r"\b(apple\s*watch\s*band|watch\s*strap|lock\s*screen\s*audio)\b", 0.90),
    ],
    SupportIntent.SUBSCRIPTIONS_BILLING: [
        (r"\b(subscription|subscriptions|auto-?renew(al)?)\b", 0.95),
        (r"\b(refund|refunds|charge\s*back|dispute)\b", 0.96),
        (r"\b(charged|charge\s*on\s*card|itunes\.com/bill|unauthorized)\b", 0.95),
        (r"\b(billing|invoice|receipt|bank\s*statement|cobro)\b", 0.90),
        (r"\b(cancel\s*subscription|cancel\s*apple\s*music)\b", 0.96),
        (r"\b(accidental\s*purchase|in-?app\s*purchase|paid\s*for)\b", 0.92),
    ],
    SupportIntent.GENERAL_INQUIRY: [
        (
            r"\b(store\s*hours|opening\s*hours|store\s*open|apple\s*store\s*appointment)\b",
            0.92,
        ),
        (r"\b(genius\s*bar|reservation|pre-?order|delivery\s*date|reserved)\b", 0.92),
        (r"\b(iphone\s*x\s*reserve|pickup|order\s*status|lines\s*closed)\b", 0.92),
        (r"\b(phone\s*lines|customer\s*service\s*number|representative)\b", 0.85),
        (r"\b(shipping\s*address|change\s*address|delivery\s*address)\b", 0.90),
    ],
}


def classify_text_intent(text: str) -> Tuple[SupportIntent, float, List[str]]:
    """
    Classify customer message text into one of 7 domain intents with confidence score.
    Returns (intent_enum, confidence, list_of_matched_signals).
    """
    if not text or not text.strip():
        return SupportIntent.GENERAL_INQUIRY, 0.50, ["empty_payload"]

    clean = text.lower()
    intent_scores: Dict[SupportIntent, float] = {}
    intent_signals: Dict[SupportIntent, List[str]] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = 0.0
        signals = []
        for pattern, weight in patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                matched_term = match.group(0)
                signals.append(matched_term)
                score = max(score, weight)
        if signals:
            # Multi-signal reinforcement (caps at 0.98)
            reinforced_score = min(0.98, score + 0.03 * (len(signals) - 1))
            intent_scores[intent] = reinforced_score
            intent_signals[intent] = signals

    if not intent_scores:
        return SupportIntent.GENERAL_INQUIRY, 0.65, ["no_specific_technical_keyword"]

    # Apply Disambiguation & Tie-Breaking Hierarchy
    # 1. Safety & Hardware Priority
    if SupportIntent.BATTERY_POWER_HARDWARE in intent_scores:
        hw_signals = intent_signals[SupportIntent.BATTERY_POWER_HARDWARE]
        if any(
            w in hw_signals for w in ["swollen", "swelling", "shatter", "cracked", "burning", "hot"]
        ):
            return (
                SupportIntent.BATTERY_POWER_HARDWARE,
                intent_scores[SupportIntent.BATTERY_POWER_HARDWARE],
                hw_signals,
            )

    # 2. Financial / Billing Priority
    if SupportIntent.SUBSCRIPTIONS_BILLING in intent_scores:
        bill_signals = intent_signals[SupportIntent.SUBSCRIPTIONS_BILLING]
        if any(w in bill_signals for w in ["refund", "charged", "itunes.com/bill", "cancel"]):
            return (
                SupportIntent.SUBSCRIPTIONS_BILLING,
                intent_scores[SupportIntent.SUBSCRIPTIONS_BILLING],
                bill_signals,
            )

    # 3. Account / 2FA Priority
    if SupportIntent.ACCOUNT_APPLE_ID in intent_scores:
        acc_signals = intent_signals[SupportIntent.ACCOUNT_APPLE_ID]
        if any(w in acc_signals for w in ["2fa", "two-factor", "locked", "verification", "region"]):
            return (
                SupportIntent.ACCOUNT_APPLE_ID,
                intent_scores[SupportIntent.ACCOUNT_APPLE_ID],
                acc_signals,
            )

    # 4. Highest scoring intent
    best_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
    return best_intent, intent_scores[best_intent], intent_signals[best_intent]


def load_intent_taxonomy(taxonomy_path: Optional[Path] = None) -> Any:
    """Load and validate machine-readable intent taxonomy specification."""
    from app.models.taxonomy_models import IntentTaxonomySchema

    path = taxonomy_path or Path("data/taxonomy/intent_taxonomy.json")
    if not path.exists():
        path = Path("models/intent_classifier/intent_taxonomy.json")
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy specification not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return IntentTaxonomySchema(**data)


def label_conversations_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    """Tag real conversations with classified intents and compute distribution metrics."""
    conversations = []
    distribution: Dict[str, int] = {}
    routing_breakdown: Dict[str, Dict[str, int]] = {}
    confidence_sums: Dict[str, float] = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Evaluate customer's primary inquiry
            inquiry = record.get("first_inquiry") or record.get("latest_message") or ""
            intent_enum, conf, signals = classify_text_intent(inquiry)

            record["intent"] = INTENT_LABELS[intent_enum]
            record["intent_code"] = intent_enum.value
            record["intent_confidence"] = round(conf, 3)
            record["intent_signals"] = signals

            # Re-evaluate routing with domain intent rules
            if intent_enum in (
                SupportIntent.BATTERY_POWER_HARDWARE,
                SupportIntent.ACCOUNT_APPLE_ID,
                SupportIntent.SUBSCRIPTIONS_BILLING,
            ):
                record["decision"] = "HUMAN_ESCALATION"
                record["status"] = "Needs Human"

            label = INTENT_LABELS[intent_enum]
            distribution[label] = distribution.get(label, 0) + 1
            confidence_sums[label] = confidence_sums.get(label, 0.0) + conf

            decision = record.get("decision", "AUTO_HANDLE")
            if label not in routing_breakdown:
                routing_breakdown[label] = {"AUTO_HANDLE": 0, "HUMAN_ESCALATION": 0}
            routing_breakdown[label][decision] = routing_breakdown[label].get(decision, 0) + 1

            conversations.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c) + "\n")

    total = len(conversations)
    intent_stats = {}
    for label, count in distribution.items():
        intent_stats[label] = {
            "count": count,
            "percentage": round((count / total) * 100, 2) if total else 0.0,
            "avg_confidence": (round(confidence_sums[label] / count, 3) if count else 0.0),
            "routing": routing_breakdown.get(label, {}),
        }

    return {
        "total_conversations": total,
        "distribution": distribution,
        "classes_represented": len(distribution),
        "detailed_statistics": intent_stats,
    }


if __name__ == "__main__":
    import sys

    full_file = Path("data/processed/applesupport_conversations.jsonl")
    sample_file = Path("data/processed/applesupport_sample.jsonl")
    stats_out = Path("experiments/intent_distribution.json")

    primary_file = full_file if full_file.exists() else sample_file

    if full_file.exists():
        print(f"Classifying intents for full dataset: {full_file}...")
        stats = label_conversations_file(full_file, full_file)
        stats_out.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_out, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Done! Labeled {stats['total_conversations']} full conversations.")

    if sample_file.exists():
        print(f"Classifying intents for sample dataset: {sample_file}...")
        sample_stats = label_conversations_file(sample_file, sample_file)
        if not full_file.exists():
            stats_out.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_out, "w", encoding="utf-8") as f:
                json.dump(sample_stats, f, indent=2)
        print(f"Done! Labeled {sample_stats['total_conversations']} sample conversations.")

    if not full_file.exists() and not sample_file.exists():
        print("No processed conversation files found.")
        sys.exit(1)
