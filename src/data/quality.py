"""Dataset quality metrics for metadata and profile reports."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}")
MIN_EXPECTED_ROWS = 20_000


def count_placeholders(df: pd.DataFrame) -> dict[str, Any]:
    """
    Name: count_placeholders
    Input: df — cleaned dataframe with instruction and response columns
    Output: dict — counts and percentage of rows containing {{placeholder}} tokens
    Purpose: Measure how synthetic templating affects keyword search quality.
    """
    inst = df["instruction"].astype(str).str.contains(PLACEHOLDER_RE, regex=True)
    resp = df["response"].astype(str).str.contains(PLACEHOLDER_RE, regex=True)
    n = len(df)
    either = inst | resp
    return {
        "instruction_with_placeholder": int(inst.sum()),
        "response_with_placeholder": int(resp.sum()),
        "rows_with_placeholder": int(either.sum()),
        "placeholder_pct": round(100.0 * either.sum() / max(n, 1), 2),
    }


def intent_imbalance(intent_counts: dict[str, int]) -> dict[str, Any]:
    """
    Name: intent_imbalance
    Input: intent_counts — mapping of intent label to row count
    Output: dict — min, max, median counts and max/min ratio
    Purpose: Quantify class skew so analytics answers are interpreted correctly.
    """
    if not intent_counts:
        return {"min": 0, "max": 0, "ratio": 0.0, "median": 0}
    vals = sorted(intent_counts.values())
    mn, mx = vals[0], vals[-1]
    mid = vals[len(vals) // 2]
    return {
        "min": mn,
        "max": mx,
        "median": mid,
        "ratio": round(mx / max(mn, 1), 2),
    }


def top_duplicate_instructions(df: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    """
    Name: top_duplicate_instructions
    Input: df — cleaned dataframe; limit — max duplicate instructions to return
    Output: list[dict] — instruction text and duplicate count per entry
    Purpose: Surface synthetic repetition before/after deduplication.
    """
    vc = df["instruction"].value_counts()
    dup = vc[vc > 1].head(limit)
    return [
        {"instruction": str(idx), "count": int(cnt)}
        for idx, cnt in dup.items()
    ]


def build_quality_report(
    raw_df: pd.DataFrame,
    clean_df: pd.DataFrame,
    *,
    deduped_rows: int,
    dropped_null_rows: int,
) -> dict[str, Any]:
    """
    Name: build_quality_report
    Input: raw_df — pre-clean snapshot; clean_df — post-preprocess frame;
           deduped_rows — rows removed by dedupe; dropped_null_rows — rows dropped for nulls
    Output: dict — full quality report for metadata.json and quality_report.json
    Purpose: Document dataset downsides for the agent and grading artifacts.
    """
    intent_counts = clean_df["intent"].astype(str).value_counts().to_dict()
    flags_top = (
        clean_df["flags"].astype(str).value_counts().head(15).to_dict()
        if "flags" in clean_df.columns
        else {}
    )
    return {
        "raw_row_count": int(len(raw_df)),
        "clean_row_count": int(len(clean_df)),
        "deduped_rows": int(deduped_rows),
        "dropped_null_rows": int(dropped_null_rows),
        "placeholders": count_placeholders(clean_df),
        "intent_imbalance": intent_imbalance({k: int(v) for k, v in intent_counts.items()}),
        "top_duplicate_instructions": top_duplicate_instructions(clean_df),
        "flags_top_values": {k: int(v) for k, v in flags_top.items()},
        "below_expected_rows": len(clean_df) < MIN_EXPECTED_ROWS,
    }
