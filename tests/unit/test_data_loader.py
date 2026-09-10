"""Unit tests for dataset loader and exploratory analysis artifacts."""

import json
from pathlib import Path

from scripts.data.loader import get_dataset_path, iter_chunks, load_sample_data


def test_dataset_path_resolution():
    """Verify that get_dataset_path resolves an existing dataset file."""
    path = get_dataset_path()
    assert path is not None
    assert path.exists()
    assert path.name in ("twcs.csv", "sample_twcs.csv")


def test_load_sample_data():
    """Verify loading sample records from the dataset with correct dtypes."""
    df = load_sample_data(nrows=20)
    assert len(df) <= 20
    assert len(df) > 0
    expected_cols = {
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    }
    assert expected_cols.issubset(set(df.columns))
    assert df["tweet_id"].dtype == object
    assert df["author_id"].dtype == object


def test_iter_chunks():
    """Verify chunked generator yields valid DataFrames."""
    generator = iter_chunks(chunksize=25)
    first_chunk = next(generator)
    assert len(first_chunk) == 25
    assert "text" in first_chunk.columns


def test_dataset_stats_artifact():
    """Verify that experiments/dataset_stats.json is valid and contains required metrics."""
    stats_path = Path("experiments/dataset_stats.json")
    assert stats_path.exists(), "experiments/dataset_stats.json should exist"

    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "dataset_metadata" in data
    assert data["dataset_metadata"]["total_rows"] == 2811774
    assert "volume_breakdown" in data
    assert data["volume_breakdown"]["inbound_records"] > 1000000
    assert "top_25_brands_by_outbound" in data
    assert len(data["top_25_brands_by_outbound"]) >= 5
