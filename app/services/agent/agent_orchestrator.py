import re
import time
from datetime import datetime, timezone
from typing import Callable, Optional

from app.core.config import get_settings
from app.core.logging import logger
from app.models.response_models import (
    AgentRunResult,
    GenerationResult,
    LatencyBreakdown,
    RetrievalResult,
)
from app.services.agent.agent_context import AgentRunContext
from app.services.escalation.escalation_policy import EscalationPolicy
from app.services.generation.llm_service import LLMService
from app.services.generation.prompt_builder import PromptBuilder
from app.services.intent.intent_classifier import IntentClassifier
from app.services.retrieval.evidence_ranker import EvidenceRanker
from app.services.retrieval.retriever import Retriever
from app.services.validation.response_validator import ResponseValidator


class AgentOrchestrator:
    """Master coordinator executing the grounded support pipeline end-to-end."""

    def __init__(
        self,
        intent_classifier: Optional[IntentClassifier] = None,
        retriever: Optional[Retriever] = None,
        evidence_ranker: Optional[EvidenceRanker] = None,
        llm_service: Optional[LLMService] = None,
        validator: Optional[ResponseValidator] = None,
        escalation_policy: Optional[EscalationPolicy] = None,
    ):
        self.settings = get_settings()
        self.intent_classifier = intent_classifier or IntentClassifier()
        self.retriever = retriever or Retriever()
        self.evidence_ranker = evidence_ranker or EvidenceRanker()
        self.llm_service = llm_service or LLMService()
        self.validator = validator or ResponseValidator()
        self.escalation_policy = escalation_policy or EscalationPolicy()

    @classmethod
    def extract_customer_handle(
        cls,
        customer_message: str,
        brand: str = "AppleSupport",
        explicit_handle: Optional[str] = None,
    ) -> str:
        """Extract customer handle or author ID token automatically from tweet text.

        Examples:
            - "@AppleSupport @115858 my phone is slow..." -> "@115858"
            - "@145247 @AppleSupport need help" -> "@145247"
            - "@AppleSupport @john_smith screen is cracked" -> "@john_smith"
            - "My iPhone is broken..." -> "@Customer"
        """
        if explicit_handle and explicit_handle.strip():
            h = explicit_handle.strip()
            return h if h.startswith("@") else f"@{h}"

        if not customer_message:
            return "@Customer"

        # Look for Twitter mentions in the message: @115858, @john_doe, etc.
        mentions = re.findall(r"@([A-Za-z0-9_]+)", customer_message)

        # Disallow brand itself and generic placeholder/system words
        brand_clean = brand.lower().replace("@", "")
        excluded = {
            brand_clean,
            "applesupport",
            "apple",
            "support",
            "help",
            "admin",
            "user",
            "username",
            "here",
        }

        customer_mentions = [m for m in mentions if m.lower() not in excluded]
        if customer_mentions:
            return f"@{customer_mentions[0]}"

        # Check for patterns like "from: @115858" or "ID: 115858" or "author: 115858"
        id_match = re.search(
            r"(?:user|author|customer|id|from)[\s:=-]+@?([A-Za-z0-9_]+)",
            customer_message,
            re.IGNORECASE,
        )
        if id_match:
            candidate = id_match.group(1).strip()
            if candidate.lower() not in excluded:
                return f"@{candidate}"

        return "@Customer"

    @staticmethod
    def _clean_draft_reply(draft_reply: str, customer_handle: Optional[str] = None) -> str:
        """Sanitize AI response to remove markdown asterisks and replace placeholder username tokens."""
        if not draft_reply:
            return ""

        handle = (customer_handle or "@Customer").strip()
        if not handle.startswith("@"):
            handle = f"@{handle}"

        cleaned = draft_reply

        # 1. Replace placeholder username tokens like @[user], @[username], [user], @{user}, etc.
        # Including any narrow or zero-width spaces (\u200b-\u200f, \ufeff, \u202f, \u00a0)
        placeholder_pattern = re.compile(
            r"@?[\u2000-\u200f\ufeff\s]*[\[\{][\u2000-\u200f\ufeff\s]*(?:user|username|handle|customer)[\u2000-\u200f\ufeff\s]*[\]\}]",
            re.IGNORECASE,
        )
        cleaned = placeholder_pattern.sub(handle, cleaned)

        # 2. Also replace standalone [user] or [username]
        cleaned = re.sub(
            r"\[(?:user|username|handle|customer)\]",
            handle,
            cleaned,
            flags=re.IGNORECASE,
        )

        # 3. Clean up double @@ (e.g. if @ was already outside the placeholder: "@@Customer" -> "@Customer")
        cleaned = re.sub(r"@+(" + re.escape(handle.lstrip("@")) + r")", r"@\1", cleaned)

        # 4. Remove bold/italic markdown asterisks while preserving the enclosed text:
        # e.g., **Settings → Wi-Fi** -> Settings → Wi-Fi, *disconnect* -> disconnect
        cleaned = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", cleaned)
        # Remove any remaining stray asterisks
        cleaned = cleaned.replace("*", "")

        # 5. Remove bold markdown underscores if any (__word__ -> word)
        cleaned = re.sub(r"_{2,3}(.*?)_{2,3}", r"\1", cleaned)

        # 6. Normalize multiple horizontal spaces into single space, keeping line breaks
        cleaned = re.sub(r"[ \t\u2000-\u200a\u202f\u00a0]+", " ", cleaned)

        return cleaned.strip()

    def run(
        self,
        customer_message: str,
        conversation_id: Optional[str] = None,
        brand: Optional[str] = None,
        customer_handle: Optional[str] = None,
        provider: Optional[str] = None,
        event_callback: Optional[Callable[[str, dict], None]] = None,
    ) -> AgentRunResult:
        """Execute the real pipeline synchronously, notifying optional event callback for streaming."""
        effective_brand = brand or self.settings.TARGET_BRAND
        effective_handle = self.extract_customer_handle(
            customer_message=customer_message,
            brand=effective_brand,
            explicit_handle=customer_handle,
        )
        ctx = AgentRunContext(
            customer_message=customer_message,
            conversation_id=conversation_id,
            brand=effective_brand,
        )
        logger.info(f"Agent execution initiated for run_id: {ctx.run_id}")

        def notify(event_name: str, payload: dict):
            ctx.log_event(event_name, payload)
            if event_callback:
                try:
                    event_callback(event_name, payload)
                except Exception as ex:
                    logger.warning(f"Event callback failed: {ex}")

        # 1. Message Received
        notify("MESSAGE_RECEIVED", {"customer_message": customer_message})

        # 2. Intent Classification
        t0 = time.time()
        notify("INTENT_CLASSIFICATION_STARTED", {})
        intent_pred = self.intent_classifier.classify(customer_message)
        intent_ms = round((time.time() - t0) * 1000, 2)
        notify(
            "INTENT_CLASSIFICATION_COMPLETED",
            {
                "intent": intent_pred.name,
                "confidence": intent_pred.confidence,
                "elapsed_ms": intent_ms,
            },
        )

        # 3. Historical Retrieval
        t1 = time.time()
        notify("RETRIEVAL_STARTED", {})
        intent_code = intent_pred.code.value if intent_pred.code else None
        candidates = self.retriever.retrieve(customer_message, top_k=3, intent_filter=intent_code)
        ranked_evidence = self.evidence_ranker.rank_and_filter(candidates)
        retrieval_ms = round((time.time() - t1) * 1000, 2)
        notify(
            "RETRIEVAL_COMPLETED",
            {
                "count": len(ranked_evidence),
                "top_similarity": (ranked_evidence[0].similarity if ranked_evidence else 0.0),
                "elapsed_ms": retrieval_ms,
            },
        )

        # 4. Grounded Reply Generation
        t2 = time.time()
        notify("GENERATION_STARTED", {})
        prompt = PromptBuilder.build_grounded_prompt(
            customer_message=customer_message,
            intent_name=intent_pred.name,
            evidence=ranked_evidence,
            brand=effective_brand,
            customer_handle=effective_handle,
        )
        raw_draft_reply, provider_used = self.llm_service.generate_reply(
            prompt, provider_name=provider
        )
        draft_reply = self._clean_draft_reply(raw_draft_reply, customer_handle=effective_handle)
        generation_ms = round((time.time() - t2) * 1000, 2)
        notify(
            "GENERATION_COMPLETED",
            {
                "provider": provider_used,
                "elapsed_ms": generation_ms,
            },
        )

        # 5. Response Validation
        t3 = time.time()
        notify("VALIDATION_STARTED", {})
        validation_result = self.validator.validate(
            reply=draft_reply,
            evidence=ranked_evidence,
            customer_message=customer_message,
        )
        validation_ms = round((time.time() - t3) * 1000, 2)
        notify(
            "VALIDATION_COMPLETED",
            {
                "all_passed": validation_result.all_passed,
                "grounding_score": validation_result.grounding_score,
                "elapsed_ms": validation_ms,
            },
        )

        # 6. Escalation Policy Routing
        notify("ESCALATION_STARTED", {})
        routing_decision = self.escalation_policy.evaluate(
            customer_message=customer_message,
            intent=intent_pred,
            evidence=ranked_evidence,
            validation=validation_result,
        )
        notify(
            "ESCALATION_COMPLETED",
            {
                "decision": routing_decision.decision.value,
                "reasons": routing_decision.reasons,
            },
        )

        total_ms = round((time.time() - ctx.start_time) * 1000, 2)
        notify("AGENT_COMPLETED", {"run_id": ctx.run_id, "total_ms": total_ms})

        return AgentRunResult(
            run_id=ctx.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            brand=effective_brand,
            customer_message=customer_message,
            intent=intent_pred,
            retrieval=RetrievalResult(
                top_k=len(ranked_evidence),
                evidence=ranked_evidence,
            ),
            generation=GenerationResult(
                draft_reply=draft_reply,
                provider=provider_used,
                grounded_evidence_ids=[c.case_id for c in ranked_evidence],
            ),
            validation=validation_result,
            routing=routing_decision,
            latency_ms=LatencyBreakdown(
                intent_ms=intent_ms,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                validation_ms=validation_ms,
                total_ms=total_ms,
            ),
            metadata={"events": ctx.events},
        )
