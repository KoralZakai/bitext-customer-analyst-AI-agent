"""Analytics tools over preprocessed Bitext data."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool

from src.data.loader import load_metadata, load_real_dataframe, load_search_dataframe
from src.data.preprocess import apply_keyword_to_category, apply_keyword_to_intent, load_aliases


def _real_df() -> pd.DataFrame:
    """
    Name: _real_df
    Input: None
    Output: pd.DataFrame — non-synthetic rows only
    Purpose: Internal helper so count tools never include augmented paraphrases.
    """
    return load_real_dataframe()


def _resolve_filters(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    """
    Name: _resolve_filters
    Input: category, intent, keyword — optional filter strings from user or agent
    Output: tuple — normalized (category, intent, keyword) after alias resolution
    Purpose: Map natural language to canonical enums before pandas filtering.
    """
    aliases = load_aliases()
    cat = category.strip().upper() if category else None
    it = intent.strip().upper() if intent else None
    kw = keyword.strip() if keyword else None
    if kw:
        if not it:
            mapped = apply_keyword_to_intent(kw, aliases)
            if mapped:
                it = mapped
        if not cat:
            mapped_cat = apply_keyword_to_category(kw, aliases)
            if mapped_cat:
                cat = mapped_cat
    return cat, it, kw


@tool
def list_categories() -> str:
    """
    List all customer-support categories in the dataset.

    Name: list_categories
    Input: None
    Output: str — JSON array of category names
    Purpose: Expose taxonomy without scanning the full parquet file.
    """
    meta = load_metadata()
    return json.dumps(meta.get("categories", []), indent=2)


@tool
def list_intents(category: str | None = None) -> str:
    """
    List intents, optionally filtered by category (e.g. REFUND, ORDER).

    Name: list_intents
    Input: category — optional category name (case-insensitive)
    Output: str — JSON list of intents or per-category intent counts
    Purpose: Help the agent pick valid intent filters for queries.
    """
    meta = load_metadata()
    if category:
        cat = category.strip().upper()
        counts = meta.get("counts_by_category_intent", {}).get(cat, {})
        return json.dumps({"category": cat, "intents": sorted(counts.keys()), "counts": counts}, indent=2)
    return json.dumps(meta.get("intents", []), indent=2)


@tool
def count_rows(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> str:
    """
    Count dataset rows with optional category, intent, or keyword filters.

    Name: count_rows
    Input: category, intent, keyword — optional filters (aliases applied)
    Output: str — JSON with count and resolved filter values
    Purpose: Answer how many rows match structured or text filters (real data only).
    """
    df = _real_df()
    cat, it, kw = _resolve_filters(category, intent, keyword)
    if cat:
        df = df[df["category"].astype(str) == cat]
    if it:
        df = df[df["intent"].astype(str) == it]
    if kw:
        mask = df["instruction"].str.contains(kw, case=False, na=False) | df[
            "response"
        ].str.contains(kw, case=False, na=False)
        df = df[mask]
    return json.dumps(
        {"count": int(len(df)), "category": cat, "intent": it, "keyword": kw},
        indent=2,
    )


@tool
def distribution_by_category() -> str:
    """
    Return row counts per category (real data only).

    Name: distribution_by_category
    Input: None
    Output: str — JSON map of category to row count
    Purpose: Summarize category balance from precomputed metadata.
    """
    meta = load_metadata()
    return json.dumps(meta.get("category_counts", {}), indent=2)


@tool
def distribution_by_intent(category: str | None = None) -> str:
    """
    Return row counts per intent, optionally within one category.

    Name: distribution_by_intent
    Input: category — optional category scope
    Output: str — JSON map of intent to row count
    Purpose: Show intent distribution globally or within a category.
    """
    meta = load_metadata()
    if category:
        cat = category.strip().upper()
        return json.dumps(
            meta.get("counts_by_category_intent", {}).get(cat, {}),
            indent=2,
        )
    return json.dumps(meta.get("intent_counts", {}), indent=2)


@tool
def filter_records(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    limit: int = 5,
) -> str:
    """
    Return sample records matching filters (instruction, category, intent, response snippet).

    Name: filter_records
    Input: category, intent, keyword — filters; limit — max samples (1–20)
    Output: str — JSON with total matches and sample rows
    Purpose: Show concrete examples for exploratory analyst questions.
    """
    df = _real_df()
    cat, it, kw = _resolve_filters(category, intent, keyword)
    if cat:
        df = df[df["category"].astype(str) == cat]
    if it:
        df = df[df["intent"].astype(str) == it]
    if kw:
        mask = df["instruction"].str.contains(kw, case=False, na=False)
        df = df[mask]
    limit = max(1, min(int(limit), 20))
    rows = df.head(limit)
    out: list[dict[str, Any]] = []
    for _, r in rows.iterrows():
        out.append(
            {
                "category": str(r["category"]),
                "intent": str(r["intent"]),
                "instruction": str(r["instruction"])[:300],
                "response": str(r["response"])[:300],
            }
        )
    return json.dumps({"matches": len(df), "samples": out}, indent=2)


@tool
def search_instructions(keyword: str, limit: int = 10) -> str:
    """
    Search instructions (includes augmented paraphrases if generated).

    Name: search_instructions
    Input: keyword — search text; limit — max results (1–25)
    Output: str — JSON with resolved intent and matching instruction samples
    Purpose: Find phrasing variants including optional synthetic paraphrases.
    """
    df = load_search_dataframe()
    kw = keyword.strip()
    cat, it, _ = _resolve_filters(keyword=kw)
    if it:
        sub = df[df["intent"].astype(str) == it]
    else:
        sub = df
        mask = sub["instruction"].str.contains(kw, case=False, na=False)
        sub = sub[mask]
    limit = max(1, min(int(limit), 25))
    rows = sub.head(limit)
    samples = [
        {
            "category": str(r["category"]),
            "intent": str(r["intent"]),
            "instruction": str(r["instruction"])[:400],
            "source": str(r.get("source", "real")),
        }
        for _, r in rows.iterrows()
    ]
    return json.dumps({"keyword": kw, "resolved_intent": it, "samples": samples}, indent=2)


@tool
def dataset_summary() -> str:
    """
    High-level dataset stats and known quality limitations.

    Name: dataset_summary
    Input: None
    Output: str — JSON overview with row counts and quality notes
    Purpose: Orient the agent on dataset scope and known downsides.
    """
    meta = load_metadata()
    quality = meta.get("quality", {})
    return json.dumps(
        {
            "row_count": meta.get("row_count"),
            "categories": len(meta.get("categories", [])),
            "intents": len(meta.get("intents", [])),
            "deduped_rows": meta.get("deduped_rows"),
            "quality_notes": {
                "synthetic_templated": True,
                "not_production_crm": True,
                "placeholder_pct": quality.get("placeholders", {}).get("placeholder_pct"),
                "intent_imbalance_ratio": quality.get("intent_imbalance", {}).get("ratio"),
            },
        },
        indent=2,
    )


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    distribution_by_category,
    distribution_by_intent,
    filter_records,
    search_instructions,
    dataset_summary,
]
