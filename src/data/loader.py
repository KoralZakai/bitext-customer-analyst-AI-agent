"""
Load Bitext from Hugging Face (download) or from preprocessed parquet (runtime).
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from src.config import (
    DATA_AUGMENTED_PATH,
    DATA_CONVERSATIONS_PATH,
    DATA_FILLED_PATH,
    DATA_PROCESSED_PATH,
    DATA_RAW_DIR,
    DATASET_HF_CSV,
    DATASET_HF_ID,
)

RAW_SNAPSHOT_PATH = DATA_RAW_DIR / "bitext_raw.parquet"

_dataframe_cache: pd.DataFrame | None = None
_augmented_cache: pd.DataFrame | None = None
_filled_cache: pd.DataFrame | None = None


def _download_via_csv() -> pd.DataFrame:
    """Download the Hub CSV directly (avoids datasets 'generating train split' failures)."""
    from huggingface_hub import hf_hub_download

    print("Trying direct CSV download from Hugging Face Hub...")
    csv_path = hf_hub_download(
        repo_id=DATASET_HF_ID,
        filename=DATASET_HF_CSV,
        repo_type="dataset",
    )
    df = pd.read_csv(csv_path)
    print(f"Loaded CSV: {len(df)} rows, columns={list(df.columns)}")
    return df


def _download_via_load_dataset() -> pd.DataFrame:
    from datasets import load_dataset

    print("Trying datasets.load_dataset...")
    ds = load_dataset(DATASET_HF_ID, split="train")
    return ds.to_pandas()


def download_raw_dataframe(*, force_download: bool = False) -> pd.DataFrame:
    """
    Name: download_raw_dataframe
    Input: force_download — if True, ignore cached raw parquet and fetch from Hub
    Output: pd.DataFrame — full Bitext train split as pandas
    Purpose: Fetch Bitext for preprocessing (CSV first; cached raw optional).
    """
    if not force_download and RAW_SNAPSHOT_PATH.exists():
        print(f"Using cached raw snapshot: {RAW_SNAPSHOT_PATH}")
        return pd.read_parquet(RAW_SNAPSHOT_PATH)

    try:
        return _download_via_csv()
    except Exception as csv_err:
        print(f"CSV download failed: {csv_err}")
        try:
            return _download_via_load_dataset()
        except Exception as ds_err:
            raise RuntimeError(
                "Could not download Bitext. Set HF_TOKEN, check internet, or place "
                f"raw parquet at {RAW_SNAPSHOT_PATH}"
            ) from ds_err


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """
    Name: load_metadata
    Input: None (reads DATA_METADATA_PATH)
    Output: dict — categories, intents, counts, quality block
    Purpose: Fast access to precomputed stats for tools and router.
    """
    import json

    from src.config import DATA_METADATA_PATH

    with DATA_METADATA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def load_processed_dataframe(force_reload: bool = False) -> pd.DataFrame:
    """
    Name: load_processed_dataframe
    Input: force_reload — if True, bypass in-memory cache
    Output: pd.DataFrame — full processed parquet including all source tags
    Purpose: Singleton cache for tool layer; avoids repeated parquet I/O.
    """
    global _dataframe_cache
    if _dataframe_cache is not None and not force_reload:
        return _dataframe_cache

    if not DATA_PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Processed data not found at {DATA_PROCESSED_PATH}. "
            "Run: python -m src.data.preprocess"
        )

    _dataframe_cache = pd.read_parquet(DATA_PROCESSED_PATH)
    return _dataframe_cache


def load_real_dataframe(force_reload: bool = False) -> pd.DataFrame:
    """
    Name: load_real_dataframe
    Input: force_reload — if True, bypass caches
    Output: pd.DataFrame — rows where source is real (excludes synthetic augmentation)
    Purpose: Ensure count and distribution tools reflect true dataset size.
    """
    df = load_processed_dataframe(force_reload=force_reload)
    if "source" in df.columns:
        return df[df["source"].astype(str) == "real"].copy()
    return df


def load_augmented_dataframe(force_reload: bool = False) -> pd.DataFrame:
    """
    Name: load_augmented_dataframe
    Input: force_reload — if True, bypass augmented cache
    Output: pd.DataFrame — synthetic paraphrase rows only (empty if file missing)
    Purpose: Optional search-only rows without polluting analytics counts.
    """
    global _augmented_cache
    if _augmented_cache is not None and not force_reload:
        return _augmented_cache
    if not DATA_AUGMENTED_PATH.exists():
        _augmented_cache = pd.DataFrame()
        return _augmented_cache
    _augmented_cache = pd.read_parquet(DATA_AUGMENTED_PATH)
    return _augmented_cache


def load_search_dataframe(force_reload: bool = False) -> pd.DataFrame:
    """
    Name: load_search_dataframe
    Input: force_reload — if True, reload real and augmented frames
    Output: pd.DataFrame — concatenation of real + synthetic rows
    Purpose: Power search_instructions with paraphrases when augmentation exists.
    """
    real = load_real_dataframe(force_reload=force_reload)
    aug = load_augmented_dataframe(force_reload=force_reload)
    if aug.empty:
        return real
    return pd.concat([real, aug], ignore_index=True)


def load_filled_dataframe(force_reload: bool = False) -> pd.DataFrame:
    """
    Load placeholder-filled rows from bitext_filled.parquet (generated by fill_placeholders.py).

    Falls back to the regular real dataframe if the filled file does not exist yet.
    Use this when you want realistic examples with real-looking names and order IDs.
    """
    global _filled_cache
    if _filled_cache is not None and not force_reload:
        return _filled_cache
    if not DATA_FILLED_PATH.exists():
        return load_real_dataframe(force_reload=force_reload)
    _filled_cache = pd.read_parquet(DATA_FILLED_PATH)
    return _filled_cache


def load_conversations(intent: str | None = None) -> list[dict]:
    """
    Load synthetic multi-turn conversations from bitext_conversations.jsonl.

    Generated by scripts/augment_conversations.py. Returns an empty list if the
    file doesn't exist. Optionally filter to a specific intent.
    """
    if not DATA_CONVERSATIONS_PATH.exists():
        return []
    import json
    records: list[dict] = []
    with DATA_CONVERSATIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if intent is None or rec.get("intent", "").upper() == intent.upper():
                    records.append(rec)
            except json.JSONDecodeError:
                continue
    return records


def dataset_ready() -> bool:
    """
    Name: dataset_ready
    Input: None
    Output: bool — True if processed parquet exists
    Purpose: Gate CLI and scripts before preprocess has run.
    """
    return DATA_PROCESSED_PATH.exists()


def clear_caches() -> None:
    """Reset in-memory and lru caches after preprocess or augmentation."""
    global _dataframe_cache, _augmented_cache, _filled_cache
    _dataframe_cache = None
    _augmented_cache = None
    _filled_cache = None
    load_metadata.cache_clear()
