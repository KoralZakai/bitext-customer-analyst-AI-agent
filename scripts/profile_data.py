"""Emit data/processed/quality_report.json from processed parquet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_METADATA_PATH, DATA_PROCESSED_PATH, DATA_QUALITY_REPORT_PATH
from src.data.loader import dataset_ready, load_metadata, load_processed_dataframe
from src.data.quality import build_quality_report


def main() -> None:
    """
    Name: main
    Input: None (requires processed parquet on disk)
    Output: None (writes quality_report.json; exits 1 if data missing)
    Purpose: Regenerate or refresh the standalone quality report artifact.
    """
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        raise SystemExit(1)

    df = load_processed_dataframe()
    meta = load_metadata()
    quality = meta.get("quality")
    if not quality:
        raw_path = ROOT / "data" / "raw" / "bitext_raw.parquet"
        raw_df = __import__("pandas").read_parquet(raw_path) if raw_path.exists() else df
        quality = build_quality_report(
            raw_df,
            df,
            deduped_rows=meta.get("deduped_rows", 0),
            dropped_null_rows=meta.get("dropped_null_rows", 0),
        )

    DATA_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATA_QUALITY_REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2)

    print(f"Wrote {DATA_QUALITY_REPORT_PATH}")
    print(f"  clean rows: {quality.get('clean_row_count', len(df))}")
    print(f"  deduped: {quality.get('deduped_rows')}")
    print(f"  placeholder %: {quality.get('placeholders', {}).get('placeholder_pct')}")
    print(f"  intent imbalance ratio: {quality.get('intent_imbalance', {}).get('ratio')}")


if __name__ == "__main__":
    main()
