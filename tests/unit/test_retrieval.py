"""Unit and integration tests for Phase 7: Real Historical Retrieval & FAISS Vector Store."""

import json
from pathlib import Path

import faiss
import pytest

from app.core.config import get_settings
from app.models.domain_models import HistoricalCase, ResolutionStatus
from app.services.retrieval.evidence_ranker import EvidenceRanker
from app.services.retrieval.retriever import Retriever


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def retriever():
    return Retriever()


def test_faiss_index_and_metadata_exist(settings):
    """Verify that FAISS index, case metadata, and index summary exist on disk."""
    index_path = Path(settings.FAISS_INDEX_PATH)
    metadata_path = index_path.parent / "case_metadata.json"
    summary_path = index_path.parent / "index_summary.json"

    assert index_path.exists(), f"FAISS index missing at {index_path}"
    assert metadata_path.exists(), f"Case metadata missing at {metadata_path}"
    assert summary_path.exists(), f"Index summary missing at {summary_path}"


def test_faiss_index_properties(settings):
    """Verify index dimensions and vector count."""
    index_path = Path(settings.FAISS_INDEX_PATH)
    index = faiss.read_index(str(index_path))

    assert index.d == 384, f"Expected 384 embedding dimension, got {index.d}"
    assert index.ntotal > 2000, f"Expected >2000 vectors, got {index.ntotal}"
    assert index.is_trained is True


def test_retriever_returns_valid_cases(retriever):
    """Verify live retrieval returns populated HistoricalCase domain models."""
    results = retriever.retrieve("my battery drains quickly after updating to ios 11", top_k=3)

    assert len(results) == 3
    for case in results:
        assert isinstance(case, HistoricalCase)
        assert 0.0 <= case.similarity <= 1.0
        assert len(case.customer_text) > 0
        assert len(case.brand_response) > 0
        assert case.resolution_status == ResolutionStatus.RESOLVED
        assert "conv_" in case.case_id


def test_retriever_intent_filtering(retriever):
    """Verify that intent_filter parameter restricts results to the requested intent."""
    target_intent = "BATTERY_POWER_HARDWARE"
    results = retriever.retrieve(
        "my battery is dead and phone will not turn on",
        top_k=3,
        intent_filter=target_intent,
    )

    assert len(results) > 0
    for case in results:
        assert case.metadata.get("intent") == target_intent


def test_retriever_empty_query_returns_empty_list(retriever):
    """Verify handling of empty or blank queries."""
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_evidence_ranker_integration(retriever):
    """Verify that EvidenceRanker properly ranks cases and enforces similarity cutoff."""
    cases = retriever.retrieve("how do i reset my apple id password", top_k=5)
    assert len(cases) > 0

    ranker = EvidenceRanker(min_similarity=0.40)
    ranked = ranker.rank_and_filter(cases)

    assert len(ranked) > 0
    for i in range(len(ranked) - 1):
        assert ranked[i].similarity >= ranked[i + 1].similarity
        assert ranked[i].similarity >= 0.40


def test_zero_leakage_assertion(settings):
    """Verify that neither Golden Set nor Held-Out Test split cases leaked into FAISS index."""
    base_dir = settings.BASE_DIR
    metadata_path = Path(settings.FAISS_INDEX_PATH).parent / "case_metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        indexed_cases = json.load(f)

    indexed_case_ids = {c["case_id"] for c in indexed_cases}

    # Check against Golden Set
    golden_path = base_dir / "data" / "golden" / "golden_set.jsonl"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_ids = {json.loads(line)["conversation_id"] for line in f if line.strip()}

    golden_overlap = indexed_case_ids.intersection(golden_ids)
    assert (
        len(golden_overlap) == 0
    ), f"Data leakage detected! {len(golden_overlap)} golden IDs in FAISS index"

    # Check against Test Split
    test_path = base_dir / "data" / "splits" / "test.jsonl"
    with open(test_path, "r", encoding="utf-8") as f:
        test_ids = {json.loads(line)["conversation_id"] for line in f if line.strip()}

    test_overlap = indexed_case_ids.intersection(test_ids)
    assert (
        len(test_overlap) == 0
    ), f"Data leakage detected! {len(test_overlap)} test IDs in FAISS index"


def test_retrieval_benchmarks_exist_and_valid(settings):
    """Verify empirical retrieval benchmarks JSON structure and metrics."""
    benchmarks_path = settings.BASE_DIR / "experiments" / "retrieval_benchmarks.json"
    assert benchmarks_path.exists(), "retrieval_benchmarks.json missing"

    with open(benchmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "benchmarks" in data
    bm = data["benchmarks"]
    assert "test_split_unconditioned" in bm
    assert "test_split_intent_conditioned" in bm
    assert "golden_set_unconditioned" in bm
    assert "golden_set_intent_conditioned" in bm

    # Check unconditioned test split
    test_uncond = bm["test_split_unconditioned"]
    assert test_uncond["query_count"] == 484
    assert test_uncond["recall_at_1"] > 0.60
    assert test_uncond["recall_at_3"] > 0.80

    # Check boosted intent-conditioned test split
    test_boosted = bm["test_split_intent_conditioned"]
    assert test_boosted["recall_at_1"] > 0.95
    assert test_boosted["recall_at_3"] > 0.95
    assert test_boosted["mrr"] > 0.95

    # Check golden set metrics
    golden_uncond = bm["golden_set_unconditioned"]
    assert golden_uncond["query_count"] == 200
    assert golden_uncond["recall_at_1"] > 0.50

    golden_boosted = bm["golden_set_intent_conditioned"]
    assert golden_boosted["recall_at_1"] > 0.90
    assert golden_boosted["recall_at_3"] > 0.90
    assert golden_boosted["mrr"] > 0.90
