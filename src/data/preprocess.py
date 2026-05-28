"""
Mandatory preprocessing for Bitext before agent tools run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATA_METADATA_PATH,
    DATA_PROCESSED_PATH,
    DATA_QUALITY_REPORT_PATH,
    DATA_RAW_DIR,
    MIN_EXPECTED_ROWS,
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
)
from src.data.quality import build_quality_report

ALIASES_PATH = PROJECT_ROOT / "data" / "intent_aliases.json"


def _normalize_text(series: pd.Series) -> pd.Series:
    """
    Name: _normalize_text
    Input: series — pandas Series of string-like values
    Output: pd.Series — stripped string values
    Purpose: Remove leading/trailing whitespace from free-text fields.
    """
    return series.astype(str).str.strip()


def _normalize_enum(series: pd.Series) -> pd.Series:
    """
    Name: _normalize_enum
    Input: series — category or intent column
    Output: pd.Series — uppercase stripped labels
    Purpose: Stabilize filters so tool queries match stored enums.
    """
    return _normalize_text(series).str.upper()


def load_aliases() -> dict[str, str]:
    """
    Name: load_aliases
    Input: None (reads data/intent_aliases.json)
    Output: dict[str, str] — lowercase phrase to canonical intent or category
    Purpose: Map user paraphrases (e.g. money back) to dataset labels.
    """
    if not ALIASES_PATH.exists():
        return {}
    with ALIASES_PATH.open(encoding="utf-8") as f:
        raw: dict[str, str] = json.load(f)
    return {k.strip().lower(): v.strip() for k, v in raw.items()}


def apply_keyword_to_intent(keyword: str, aliases: dict[str, str]) -> str | None:
    """
    Name: apply_keyword_to_intent
    Input: keyword — user phrase; aliases — map from load_aliases()
    Output: str | None — uppercase canonical intent, or None if not an intent alias
    Purpose: Resolve natural language to intent before filtering tools run.
             Supports both exact match and longest-substring match for full sentences.
    """
    key = keyword.strip().lower()
    # Exact match first
    if key in aliases:
        val = aliases[key]
        if "_" in val or val != val.upper():
            return val.upper()
        return None
    # Longest substring match — handles full-sentence queries like
    # "Show me examples of people wanting their money back"
    best: str | None = None
    best_len = 0
    for phrase, val in aliases.items():
        if phrase in key and len(phrase) > best_len:
            if "_" in val or val != val.upper():
                best = val.upper()
                best_len = len(phrase)
    return best


def apply_keyword_to_category(keyword: str, aliases: dict[str, str]) -> str | None:
    """
    Name: apply_keyword_to_category
    Input: keyword — user phrase; aliases — map from load_aliases()
    Output: str | None — uppercase category (e.g. REFUND), or None
    Purpose: Resolve natural language to category for structured filters.
             Supports both exact match and longest-substring match for full sentences.
    """
    key = keyword.strip().lower()
    # Exact match first
    if key in aliases:
        val = aliases[key]
        if val.isupper() and "_" not in val:
            return val
        return None
    # Longest substring match
    best: str | None = None
    best_len = 0
    for phrase, val in aliases.items():
        if phrase in key and len(phrase) > best_len:
            if val.isupper() and "_" not in val:
                best = val
                best_len = len(phrase)
    return best


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Name: preprocess
    Input: df — raw Bitext dataframe with required columns
    Output: tuple[pd.DataFrame, dict] — cleaned frame and metadata dict
    Purpose: Normalize, drop bad rows, dedupe, tag source=real, attach quality stats.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    raw_count = len(df)
    out = df[list(REQUIRED_COLUMNS)].copy()

    out["instruction"] = _normalize_text(out["instruction"])
    out["response"] = _normalize_text(out["response"])
    out["flags"] = _normalize_text(out["flags"])
    out["category"] = _normalize_enum(out["category"])
    out["intent"] = _normalize_enum(out["intent"])

    out = out.replace("", pd.NA)
    after_null = out.dropna(subset=["instruction", "category", "intent"])
    dropped_null_rows = len(out) - len(after_null)
    out = after_null

    before = len(out)
    out = out.drop_duplicates(subset=["instruction", "category", "intent"], keep="first")
    deduped = before - len(out)

    out["source"] = "real"
    out["category"] = out["category"].astype("category")
    out["intent"] = out["intent"].astype("category")

    categories = sorted(out["category"].astype(str).unique().tolist())
    intents = sorted(out["intent"].astype(str).unique().tolist())
    by_category: dict[str, dict[str, int]] = {}
    for cat in categories:
        sub = out[out["category"].astype(str) == cat]
        by_category[cat] = (
            sub["intent"].astype(str).value_counts().sort_index().to_dict()
        )

    quality = build_quality_report(
        df,
        out,
        deduped_rows=deduped,
        dropped_null_rows=dropped_null_rows,
    )

    metadata: dict[str, Any] = {
        "row_count": int(len(out)),
        "raw_row_count": int(raw_count),
        "deduped_rows": int(deduped),
        "dropped_null_rows": int(dropped_null_rows),
        "categories": categories,
        "intents": intents,
        "intent_counts": out["intent"].astype(str).value_counts().sort_index().to_dict(),
        "category_counts": out["category"].astype(str).value_counts().sort_index().to_dict(),
        "counts_by_category_intent": by_category,
        "instruction_len_p95": int(out["instruction"].str.len().quantile(0.95)),
        "response_len_p95": int(out["response"].str.len().quantile(0.95)),
        "aliases_path": str(ALIASES_PATH),
        "quality": quality,
    }
    return out, metadata


def save_processed(df: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """
    Name: save_processed
    Input: df — cleaned dataframe; metadata — stats and quality block
    Output: None (writes parquet and JSON files to disk)
    Purpose: Persist artifacts consumed by loader and tools at runtime.
    """
    DATA_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED_PATH, index=False)
    with DATA_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    quality = metadata.get("quality", {})
    with DATA_QUALITY_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)


def preprocess_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """
    Name: preprocess_from_dataframe
    Input: df — raw dataframe
    Output: dict — metadata after preprocess and save
    Purpose: One-call pipeline for scripts and unit workflows.
    """
    processed, metadata = preprocess(df)
    if metadata["row_count"] < MIN_EXPECTED_ROWS:
        print(
            f"WARNING: row_count {metadata['row_count']} < expected {MIN_EXPECTED_ROWS}",
            file=sys.stderr,
        )
    save_processed(processed, metadata)
    return metadata


def main() -> None:
    """
    Name: main
    Input: None (CLI; optional --redownload to force Hub fetch)
    Output: None (prints paths; exits 1 on failure)
    Purpose: Download Hugging Face Bitext and run full preprocessing pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Download and preprocess Bitext data")
    parser.add_argument(
        "--redownload",
        action="store_true",
        help="Ignore data/raw/bitext_raw.parquet and fetch from Hugging Face again",
    )
    args = parser.parse_args()

    try:
        from src.data.loader import download_raw_dataframe
    except ImportError as e:
        print("Missing dependency. Run: pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from e

    print("Loading Bitext dataset...")
    try:
        raw = download_raw_dataframe(force_download=args.redownload)
    except Exception as e:
        print(f"Download failed: {e}", file=sys.stderr)
        print("Set HF_TOKEN, check internet, or run with cached data/raw/bitext_raw.parquet", file=sys.stderr)
        raise SystemExit(1) from e

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW_DIR / "bitext_raw.parquet"
    raw.to_parquet(raw_path, index=False)
    print(f"Saved raw snapshot: {raw_path} ({len(raw)} rows)")

    metadata = preprocess_from_dataframe(raw)
    print(f"Processed: {DATA_PROCESSED_PATH} ({metadata['row_count']} rows)")
    print(f"Metadata: {DATA_METADATA_PATH}")
    print(f"Quality report: {DATA_QUALITY_REPORT_PATH}")
    print(f"Categories ({len(metadata['categories'])}): {metadata['categories']}")
    print(f"Intents ({len(metadata['intents'])}): first 10 = {metadata['intents'][:10]}...")


if __name__ == "__main__":
    main()
