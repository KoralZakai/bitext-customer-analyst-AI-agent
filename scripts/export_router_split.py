"""
Export stratified train/val split for optional intent-router fine-tuning (stretch).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import dataset_ready, load_real_dataframe

OUT_DIR = ROOT / "data" / "router_splits"


def main() -> None:
    """
    Name: main
    Input: None (requires processed dataset)
    Output: None (writes train.parquet, val.parquet, split_meta.json)
    Purpose: Prepare optional fine-tuning splits; agent uses LLM router by default.
    """
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        raise SystemExit(1)

    df = load_real_dataframe()
    train_parts = []
    val_parts = []
    for intent, group in df.groupby(df["intent"].astype(str)):
        group = group.sample(frac=1, random_state=42)
        n_val = max(1, int(len(group) * 0.15))
        val_parts.append(group.head(n_val))
        train_parts.append(group.iloc[n_val:])

    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_path = OUT_DIR / "train.parquet"
    val_path = OUT_DIR / "val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    meta = {
        "train_rows": len(train),
        "val_rows": len(val),
        "intents": sorted(train["intent"].astype(str).unique().tolist()),
        "note": "Use for optional fine-tune; agent uses Nebius LLM router by default.",
    }
    with (OUT_DIR / "split_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Train: {train_path} ({len(train)} rows)")
    print(f"Val: {val_path} ({len(val)} rows)")
    print(f"Meta: {OUT_DIR / 'split_meta.json'}")


if __name__ == "__main__":
    main()
