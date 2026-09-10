"""Prompt builder for grounded customer support reply generation."""

from typing import List

from app.models.domain_models import HistoricalCase


class PromptBuilder:
    """Builds strictly constrained prompts grounded in retrieved historical support cases."""

    @staticmethod
    def build_grounded_prompt(
        customer_message: str, intent_name: str, evidence: List[HistoricalCase]
    ) -> str:
        """Construct prompt injecting historical brand resolutions and grounding constraints."""
        evidence_block = ""
        for i, case in enumerate(evidence, 1):
            evidence_block += (
                f"\n[Historical Case #{i}] (ID: {case.case_id})\n"
                f"Customer: {case.customer_text}\n"
                f"Brand Resolution: {case.brand_response}\n"
            )

        prompt = (
            "You are a professional customer support agent representing the brand.\n"
            "Your task is to draft a helpful, professional reply to the incoming customer inquiry.\n"
            "\n"
            "GROUNDING RULES (STRICT):\n"
            "1. You MUST rely ONLY on the provided historical brand cases for policy, procedure, and capabilities.\n"
            "2. NEVER invent refunds, account credits, delivery dates, or policies not supported by the evidence.\n"
            "3. If the evidence requires asking for an order ID or account details, ask politely.\n"
            "4. Maintain an empathetic, professional tone.\n"
            "\n"
            f"CUSTOMER INTENT: {intent_name}\n"
            f'CUSTOMER MESSAGE:\n"{customer_message}"\n'
            "\n"
            f"HISTORICAL RESOLVED CASES:{evidence_block or ' No direct historical match available.'}\n"
            "\n"
            "DRAFT REPLY:"
        )
        return prompt
