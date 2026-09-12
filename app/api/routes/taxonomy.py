"""API routes for Intent Taxonomy inspection and live text classification."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.models.domain_models import IntentPrediction
from app.models.taxonomy_models import IntentTaxonomySchema
from app.services.intent.intent_classifier import IntentClassifier
from scripts.data.classify_intents import load_intent_taxonomy

router = APIRouter(prefix="/v1/taxonomy", tags=["Taxonomy"])


class ClassifyTextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Customer text to classify")


class TaxonomyResponse(BaseModel):
    taxonomy: IntentTaxonomySchema
    distribution: Optional[Dict[str, Any]] = None


@router.get("", response_model=TaxonomyResponse)
def get_taxonomy() -> TaxonomyResponse:
    """Return the formal 7-class @AppleSupport intent taxonomy and empirical distribution."""
    try:
        schema = load_intent_taxonomy()
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Failed to load taxonomy specification: {ex}")

    distribution_data = None
    dist_file = Path("experiments/intent_distribution.json")
    if dist_file.exists():
        try:
            with open(dist_file, "r", encoding="utf-8") as f:
                distribution_data = json.load(f)
        except Exception:
            pass

    return TaxonomyResponse(taxonomy=schema, distribution=distribution_data)


@router.post("/classify", response_model=IntentPrediction)
def classify_text(payload: ClassifyTextRequest) -> IntentPrediction:
    """Classify input text against the 7-class domain taxonomy."""
    classifier = IntentClassifier()
    return classifier.classify(payload.text)
