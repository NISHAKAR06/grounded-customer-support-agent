"""Prompt builder for grounded customer support reply generation."""

from typing import List

from app.models.domain_models import HistoricalCase


class PromptBuilder:
    """Builds strictly constrained prompts grounded in retrieved historical support cases."""

    @staticmethod
    def build_grounded_prompt(
        customer_message: str,
        intent_name: str,
        evidence: List[HistoricalCase],
        brand: str = "AppleSupport",
        customer_handle: str | None = None,
    ) -> str:
        """Construct prompt injecting historical brand resolutions and grounding constraints."""
        handle = (customer_handle or "@Customer").strip()
        if not handle.startswith("@"):
            handle = f"@{handle}"

        evidence_block = ""
        for i, case in enumerate(evidence, 1):
            evidence_block += (
                f"\n[Historical Case #{i}] (ID: {case.case_id}, Similarity: {case.similarity:.2f})\n"
                f"Customer Inquiry: {case.customer_text}\n"
                f"Official {brand} Resolution: {case.brand_response}\n"
            )

        prompt = (
            f"You are an official @{brand} customer support specialist.\n"
            f"Your task is to draft a helpful, professional, and empathetic response to an incoming customer tweet.\n"
            "\n"
            "GROUNDING RULES (STRICT):\n"
            "1. GROUNDING MANDATE: Base your response and troubleshooting steps strictly on the provided historical brand cases.\n"
            "2. NO FABRICATIONS: Never invent repair costs, refund promises, warranty waivers, or policies not attested in the historical evidence.\n"
            "3. SENSITIVE INFO / PRIVACY: Never ask the customer to post passwords, serial numbers, or payment info publicly on Twitter. Direct them to send a DM (Direct Message) if personal account details or diagnostic logs are required.\n"
            "4. OFFICIAL LINKS ONLY: If linking resources, only reference official Apple domains (support.apple.com, appleid.apple.com, locate.apple.com).\n"
            "5. SAFETY & HARDWARE: If the inquiry involves physical swelling, smoke, or shattered glass, advise the customer to safely stop using/charging the device and seek authorized service immediately.\n"
            "6. TONE & LENGTH: Maintain a calm, helpful, and concise tone suitable for Twitter support.\n"
            f"7. GREETING & USERNAME: Always start your reply addressing the customer with their handle '{handle}' (e.g., 'Hi {handle} —'). NEVER output placeholder bracket tokens like '@[user]', '@[username]', '@{{handle}}', or '[user]'. Always greet '{handle}'.\n"
            "8. NO MARKDOWN / NO ASTERISKS: Do NOT use any asterisks (* or **) anywhere in your reply. Twitter does not support markdown bold or italics. Use plain text only (e.g., write 'Settings > Wi-Fi', never '**Settings > Wi-Fi**').\n"
            "\n"
            f"CLASSIFIED INTENT: {intent_name}\n"
            f'CUSTOMER INQUIRY:\n"{customer_message}"\n'
            "\n"
            f"HISTORICAL PRECEDENTS (RESOLVED BY @{brand.upper()}):\n"
            f"{evidence_block or 'No direct historical precedent found. Advise standard safe troubleshooting or invite to DM for diagnostics.'}\n"
            "\n"
            f"DRAFT @{brand} REPLY:"
        )
        return prompt
