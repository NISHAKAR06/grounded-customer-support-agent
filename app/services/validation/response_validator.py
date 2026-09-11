"""Response validator for grounding, anti-hallucination, safety, and policy checks."""

import re
from typing import List, Set
from urllib.parse import urlparse

from app.models.domain_models import HistoricalCase, ValidationResult


class ResponseValidator:
    """Validates draft responses against hallucination barriers, policy bounds, link whitelists, and safety guardrails."""

    def __init__(self):
        # Forbidden hallucination trigger patterns (unverified promises)
        self.unsupported_claim_phrases = [
            "refund has been processed",
            "sent you a full refund",
            "credited $",
            "free replacement is on its way",
            "free new iphone",
            "free new ipad",
            "waived your fee",
            "apple will pay",
            "promise you will receive a replacement",
            "we are sending you a new device",
        ]

        # Allowed official domains for customer reference
        self.allowed_domains = {
            "apple.com",
            "support.apple.com",
            "appleid.apple.com",
            "locate.apple.com",
            "getsupport.apple.com",
            "help.apple.com",
            "apple.co",
            "t.co",
            "twitter.com",
        }

        # Hazardous hardware terms requiring physical safety guidance
        self.hazard_terms = [
            "swollen",
            "swelling",
            "battery expanding",
            "smoke",
            "spark",
            "fire",
            "burning hot",
            "exploded",
        ]

        # Safe physical actions expected if hazard is present
        self.safe_hazard_actions = [
            "stop using",
            "stop charging",
            "unplug",
            "disconnect",
            "do not charge",
            "authorized service",
            "genius bar",
            "service provider",
            "safety",
        ]

    def validate(
        self,
        reply: str,
        evidence: List[HistoricalCase],
        customer_message: str = "",
    ) -> ValidationResult:
        """Execute deterministic multi-barrier validation checks on draft reply."""
        checks = {}
        warnings = []

        # 1. Non-empty & minimum length check
        clean_reply = (reply or "").strip()
        checks["non_empty_check"] = len(clean_reply) >= 15
        if not checks["non_empty_check"]:
            warnings.append(
                "Draft reply is empty or unreasonably brief (<15 characters)."
            )

        # 2. Evidence presence check
        checks["grounding_evidence_present"] = len(evidence) > 0
        if not checks["grounding_evidence_present"]:
            warnings.append(
                "No historical brand evidence available to ground the response."
            )

        reply_lower = clean_reply.lower()

        # 3. Unsupported financial / promise claims check
        has_unsupported_phrase = any(
            phrase in reply_lower for phrase in self.unsupported_claim_phrases
        )

        # Check for ungrounded monetary amounts (e.g. $29, $99, 50 USD) not found in evidence
        has_ungrounded_pricing = self._check_ungrounded_pricing(clean_reply, evidence)

        checks["unsupported_claim_check"] = not (
            has_unsupported_phrase or has_ungrounded_pricing
        )
        if has_unsupported_phrase:
            warnings.append(
                "Response contains unverified financial promises or replacement guarantees."
            )
        if has_ungrounded_pricing:
            warnings.append(
                "Response contains pricing/fee claims not present in retrieved historical evidence."
            )

        # 4. Official URL & Domain Whitelist Check
        invalid_urls = self._find_unauthorized_urls(clean_reply)
        checks["url_whitelist_check"] = len(invalid_urls) == 0
        if invalid_urls:
            warnings.append(
                f"Response contains unauthorized external links: {', '.join(invalid_urls)}."
            )

        # 5. Public PII / Security Protection Check
        has_pii_violation = self._check_public_pii_request(reply_lower)
        checks["pii_security_check"] = not has_pii_violation
        if has_pii_violation:
            warnings.append(
                "Response solicits sensitive credentials (passwords/PINs) over a public channel."
            )

        # 6. Physical Hardware Hazard Safety Check
        msg_lower = (customer_message or "").lower()
        hazard_present = any(term in msg_lower for term in self.hazard_terms)
        if hazard_present:
            provides_safety_advice = any(
                act in reply_lower for act in self.safe_hazard_actions
            )
            checks["hazardous_safety_check"] = provides_safety_advice
            if not provides_safety_advice:
                warnings.append(
                    "Inquiry mentions physical battery/thermal hazard but reply fails to give safety caution."
                )
        else:
            checks["hazardous_safety_check"] = True

        # 7. Grounding overlap score
        grounding_score = self._compute_grounding_overlap(clean_reply, evidence)

        all_passed = all(checks.values())
        return ValidationResult(
            all_passed=all_passed,
            checks=checks,
            warnings=warnings,
            grounding_score=grounding_score,
        )

    def _check_ungrounded_pricing(
        self, reply: str, evidence: List[HistoricalCase]
    ) -> bool:
        """Detect numeric dollar amounts in reply that are not in the evidence context."""
        price_patterns = re.findall(
            r"\$\s*\d+(?:\.\d{2})?|\b\d+\s*(?:dollars|usd)\b", reply, re.IGNORECASE
        )
        if not price_patterns:
            return False

        evidence_text = " ".join(
            f"{c.customer_text} {c.brand_response}" for c in evidence
        )
        for price in price_patterns:
            digits = re.sub(r"[^\d]", "", price)
            if digits and digits not in evidence_text:
                return True
        return False

    def _find_unauthorized_urls(self, text: str) -> List[str]:
        """Extract URLs and ensure domains match allowed official list."""
        url_matches = re.findall(r"https?://[^\s<>\"'()]+", text)
        invalid = []
        for raw_url in url_matches:
            url = raw_url.rstrip(".,;:!?)]}")
            try:
                parsed = urlparse(url)
                netloc = parsed.netloc.lower().split(":")[0].rstrip(".")
                # Check if netloc matches or is a subdomain of allowed domains
                is_allowed = any(
                    netloc == domain or netloc.endswith(f".{domain}")
                    for domain in self.allowed_domains
                )
                if not is_allowed:
                    invalid.append(url)
            except Exception:
                invalid.append(url)
        return invalid

    def _check_public_pii_request(self, text_lower: str) -> bool:
        """Check if reply asks customer to tweet passwords or sensitive secrets."""
        risky_prompts = [
            "reply with your password",
            "send your password",
            "tweet your password",
            "post your password",
            "reply with your pin",
            "tweet your credit card",
            "reply with your cvv",
            "post your full card",
        ]
        return any(phrase in text_lower for phrase in risky_prompts)

    def _compute_grounding_overlap(
        self, reply: str, evidence: List[HistoricalCase]
    ) -> float:
        """Compute Jaccard token overlap between reply and retrieved historical evidence."""
        if not reply or not evidence:
            return 0.0

        def tokenize(t: str) -> Set[str]:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", t.lower())
            stop_words = {
                "the",
                "and",
                "for",
                "with",
                "this",
                "that",
                "you",
                "your",
                "can",
                "our",
                "are",
            }
            return {w for w in words if w not in stop_words}

        reply_tokens = tokenize(reply)
        if not reply_tokens:
            return 0.0

        evidence_tokens = set()
        for c in evidence:
            evidence_tokens.update(tokenize(c.customer_text))
            evidence_tokens.update(tokenize(c.brand_response))

        if not evidence_tokens:
            return 0.0

        overlap = reply_tokens.intersection(evidence_tokens)
        score = len(overlap) / len(reply_tokens)
        return round(min(score, 1.0), 3)
