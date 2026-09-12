"""Data loader utility for Customer Support on Twitter dataset.

Provides deterministic resolution of dataset paths and memory-efficient
streaming/sampling of the large twcs.csv dataset.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

import pandas as pd


def get_repo_root() -> Path:
    """Return the absolute path to the repository root."""
    return Path(__file__).resolve().parent.parent.parent


def get_dataset_path() -> Path:
    """Locate the twcs.csv dataset file across standard locations.

    Search priority:
    1. Environment variable TWCS_DATASET_PATH
    2. Local repo path data/raw/twcs.csv
    3. User kagglehub cache directory
    4. Auto-download via kagglehub as fallback
    """
    # 1. Environment variable override
    env_path = os.getenv("TWCS_DATASET_PATH")
    if env_path and Path(env_path).is_file():
        return Path(env_path)

    # 2. Local raw directory
    local_path = get_repo_root() / "data" / "raw" / "twcs.csv"
    if local_path.is_file():
        return local_path

    # 3. Known user kagglehub cache location
    user_home = Path.home()
    cached_twcs_dir = (
        user_home
        / ".cache"
        / "kagglehub"
        / "datasets"
        / "thoughtvector"
        / "customer-support-on-twitter"
    )
    if cached_twcs_dir.exists():
        matches = list(cached_twcs_dir.glob("**/twcs.csv"))
        if matches and matches[0].is_file():
            return matches[0]

    # 4. Committed sample dataset (ideal for CI / offline test environments)
    sample_path = get_repo_root() / "data" / "samples" / "sample_twcs.csv"
    if sample_path.is_file() and os.getenv("CI") == "true":
        return sample_path

    # 5. Attempt kagglehub download (if not in CI)
    if os.getenv("CI") != "true":
        try:
            import kagglehub

            download_dir = Path(
                kagglehub.dataset_download("thoughtvector/customer-support-on-twitter")
            )
            matches = list(download_dir.glob("**/twcs.csv"))
            if matches and matches[0].is_file():
                return matches[0]
            direct_csv = download_dir / "twcs.csv"
            if direct_csv.is_file():
                return direct_csv
        except Exception:
            pass

    # 6. Fallback to sample path if available
    if sample_path.is_file():
        return sample_path

    raise FileNotFoundError("twcs.csv could not be located in local, cache, or sample paths.")


def load_sample_data(nrows: int = 10000) -> pd.DataFrame:
    """Load a sample dataframe from the dataset with proper datatypes."""
    path = get_dataset_path()
    df = pd.read_csv(
        path,
        nrows=nrows,
        dtype={
            "tweet_id": "str",
            "author_id": "str",
            "inbound": "bool",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "str",
        },
    )
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )
    return df


def iter_chunks(chunksize: int = 100000) -> Generator[pd.DataFrame, None, None]:
    """Yield successive chunks of twcs.csv as DataFrames."""
    path = get_dataset_path()
    for chunk in pd.read_csv(
        path,
        chunksize=chunksize,
        dtype={
            "tweet_id": "str",
            "author_id": "str",
            "inbound": "bool",
            "response_tweet_id": "str",
            "in_response_to_tweet_id": "str",
        },
    ):
        yield chunk


def load_brand_subset(
    brand_handle: str = "AppleSupport", max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Stream and filter all tweets authored by or directed at a specific brand."""
    frames = []
    total = 0
    mention_pattern = f"@{brand_handle.lstrip('@')}"

    for chunk in iter_chunks(chunksize=100000):
        # Outbound by brand or Inbound mentioning brand
        outbound_mask = chunk["author_id"] == brand_handle
        inbound_mask = chunk["inbound"].fillna(False).astype(bool) & chunk["text"].fillna(
            ""
        ).str.contains(mention_pattern, case=False, regex=False)
        brand_chunk = chunk[outbound_mask | inbound_mask].copy()

        if not brand_chunk.empty:
            frames.append(brand_chunk)
            total += len(brand_chunk)
            if max_rows and total >= max_rows:
                break

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    if max_rows:
        df = df.iloc[:max_rows]
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )
    return df
