"""Deterministic offline mock LLM provider."""

import re
from typing import Optional

from app.core.config import get_settings
from app.services.generation.llm_service import BaseLLMProvider


class MockProvider(BaseLLMProvider):
    """Deterministic, zero-latency mock provider grounded in historical context.

    Designed for test suites, CI/CD pipelines, and local development without
    requiring external API keys or third-party cloud services.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.MOCK_MODEL_NAME

    def provider_name(self) -> str:
        return f"mock ({self.model_name})"

    def generate(self, prompt: str) -> str:
        """Generate deterministic grounded reply from the prompt context."""
        # Check if historical brand resolutions are provided in the prompt
        evidence_matches = re.findall(
            r"Brand Resolution:\s*(.+?)(?=\n\[Historical Case|\n\nDRAFT REPLY:|$)",
            prompt,
            re.DOTALL,
        )

        if evidence_matches:
            # Ground response in the top matching historical brand resolution
            top_resolution = evidence_matches[0].strip()
            # Clean leading customer handles if any
            top_resolution = re.sub(r"^@[A-Za-z0-9_]+\s*", "", top_resolution)
            return top_resolution

        # Check for intent cues in the prompt
        intent_match = re.search(r"CUSTOMER INTENT:\s*([A-Za-z0-9_]+)", prompt)
        intent = intent_match.group(1) if intent_match else "GENERAL"

        templates = {
            "OPERATING_SYSTEM_UPDATES": (
                "Thanks for reaching out. Please make sure your device is backed up to iCloud or your computer, "
                "then navigate to Settings > General > Software Update to install the latest iOS version. "
                "DM us if the issue continues."
            ),
            "BATTERY_POWER_HARDWARE": (
                "We'd like to help inspect your hardware and battery health. Please check Settings > Battery > "
                "Battery Health. For physical damage or hardware inspection, please send us a DM so we can "
                "set up a Genius Bar reservation."
            ),
            "ACCOUNT_APPLE_ID": (
                "Your account security is our top priority. Please visit iforgot.apple.com to reset your credentials. "
                "Never share your password or verification codes with anyone. Send us a DM if you still need assistance."
            ),
            "CONNECTIVITY_NETWORKING": (
                "Let's get your connection back up. Please try turning Airplane Mode on for 15 seconds, then off. "
                "You can also reset network settings under Settings > General > Transfer or Reset iPhone > Reset. "
                "DM us if you need more help."
            ),
            "AUDIO_ACCESSORIES": (
                "We're here to help with your accessory audio issue. Try forgetting the device in Settings > Bluetooth "
                "and pairing it again. If the issue persists, reach out via DM with your device model."
            ),
            "SUBSCRIPTIONS_BILLING": (
                "For billing and subscription inquiries, you can review your purchase history and active subscriptions "
                "at reportaproblem.apple.com. Please DM us if you have additional questions about a specific charge."
            ),
            "GENERAL_INQUIRY": (
                "Thank you for contacting customer support. We're here to help. Please send us a direct message "
                "with more details about what you're experiencing, and we'll look into it right away."
            ),
        }

        return templates.get(intent, templates["GENERAL_INQUIRY"])
