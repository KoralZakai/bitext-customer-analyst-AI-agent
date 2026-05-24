"""Analytics tools over preprocessed Bitext data."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.loader import load_conversations, load_metadata, load_real_dataframe, load_search_dataframe
from src.data.preprocess import apply_keyword_to_category, apply_keyword_to_intent, load_aliases


def _real_df() -> pd.DataFrame:
    return load_real_dataframe()


def _resolve_filters(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> tuple[str | None, str | None, str | None]:
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


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------

class ListIntentsInput(BaseModel):
    category: str | None = Field(
        None,
        description=(
            "Category name to scope the intent list, e.g. 'REFUND' or 'ACCOUNT'. "
            "Leave empty to list all 27 intents across the whole dataset."
        ),
    )


class CountRowsInput(BaseModel):
    category: str | None = Field(
        None,
        description=(
            "Category to filter by, e.g. 'REFUND', 'ORDER', 'SHIPPING'. "
            "Leave empty to count across all categories."
        ),
    )
    intent: str | None = Field(
        None,
        description=(
            "Specific intent to filter by, e.g. 'GET_REFUND', 'CANCEL_ORDER', 'TRACK_ORDER'. "
            "Leave empty to count all intents."
        ),
    )
    keyword: str | None = Field(
        None,
        description=(
            "Free-text keyword to match inside instruction or response text. "
            "Alias resolution maps natural phrases like 'money back' → GET_REFUND."
        ),
    )


class DistributionByIntentInput(BaseModel):
    category: str | None = Field(
        None,
        description=(
            "Limit the distribution to intents within this category, e.g. 'ACCOUNT'. "
            "Leave empty for a global intent distribution across the whole dataset."
        ),
    )


class FilterRecordsInput(BaseModel):
    category: str | None = Field(
        None,
        description="Category to filter by, e.g. 'SHIPPING'. Leave empty for all categories.",
    )
    intent: str | None = Field(
        None,
        description="Intent to filter by, e.g. 'TRACK_ORDER'. Leave empty for all intents.",
    )
    keyword: str | None = Field(
        None,
        description="Keyword to match in the instruction text (case-insensitive).",
    )
    limit: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of sample records to return (1–20). Defaults to 5.",
    )


class SearchInstructionsInput(BaseModel):
    keyword: str = Field(
        ...,
        description=(
            "Search term or phrase to look up in customer instructions. "
            "Alias resolution is applied, e.g. 'money back' finds GET_REFUND examples. "
            "Includes augmented paraphrases if they were generated."
        ),
    )
    limit: int = Field(
        10,
        ge=1,
        le=25,
        description="Maximum number of matching instructions to return (1–25). Defaults to 10.",
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool
def list_categories() -> str:
    """
    List all 11 customer-support categories in the Bitext dataset.

    Use this when the user asks what categories, topics, or sections exist.
    Returns a JSON array of category names (e.g. REFUND, ORDER, ACCOUNT).
    No parameters needed.
    """
    meta = load_metadata()
    return json.dumps(meta.get("categories", []), indent=2)


@tool(args_schema=ListIntentsInput)
def list_intents(category: str | None = None) -> str:
    """
    List intents in the dataset, optionally scoped to one category.

    Use this to discover valid intent names before filtering, or to show the user
    what intents belong to a specific category. Returns intent names and counts.
    Examples: list_intents() for all, list_intents(category='REFUND') for REFUND only.
    """
    meta = load_metadata()
    if category:
        cat = category.strip().upper()
        counts = meta.get("counts_by_category_intent", {}).get(cat, {})
        return json.dumps({"category": cat, "intents": sorted(counts.keys()), "counts": counts}, indent=2)
    return json.dumps(meta.get("intents", []), indent=2)


@tool(args_schema=CountRowsInput)
def count_rows(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> str:
    """
    Count dataset rows matching optional category, intent, or keyword filters.

    Use this to answer 'how many' questions. Alias resolution maps natural language
    phrases to canonical values (e.g. 'refund requests' → intent=GET_REFUND).
    Only counts real (non-synthetic) rows. Filters can be combined.
    Example: count_rows(intent='GET_REFUND') returns the count of refund requests.
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
    Return row counts per category for the entire dataset (real data only).

    Use this for 'how is the data distributed?' or 'which category has the most rows?'
    questions. Returns a JSON map of category → count, precomputed from metadata.
    No parameters needed.
    """
    meta = load_metadata()
    return json.dumps(meta.get("category_counts", {}), indent=2)


@tool(args_schema=DistributionByIntentInput)
def distribution_by_intent(category: str | None = None) -> str:
    """
    Return row counts per intent, optionally scoped to one category.

    Use this to understand intent balance or answer distribution questions.
    With no category: global intent counts across all 27 intents.
    With category='ACCOUNT': only the intents inside the ACCOUNT category.
    """
    meta = load_metadata()
    if category:
        cat = category.strip().upper()
        return json.dumps(
            meta.get("counts_by_category_intent", {}).get(cat, {}),
            indent=2,
        )
    return json.dumps(meta.get("intent_counts", {}), indent=2)


@tool(args_schema=FilterRecordsInput)
def filter_records(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    limit: int = 5,
) -> str:
    """
    Return sample records (instruction + response) matching the given filters.

    Use this when the user wants to see concrete examples: 'show me 3 examples from
    the SHIPPING category' or 'show examples of people wanting their money back'.
    Returns the total match count plus up to `limit` sample rows.
    Alias resolution applies — 'money back' will resolve to GET_REFUND examples.
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


@tool(args_schema=SearchInstructionsInput)
def search_instructions(keyword: str, limit: int = 10) -> str:
    """
    Search customer instructions by keyword, including augmented paraphrases if available.

    Use this for open-ended search or when the user wants to find phrasing variants.
    Alias resolution maps e.g. 'cancel' to CANCEL_ORDER instructions automatically.
    Prefer filter_records for category/intent filtering; use this for free-text search.
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
    Return a high-level overview of the Bitext dataset including quality limitations.

    Use this to orient the user on dataset scope, size, and known downsides before
    answering deeper questions. Also use at the start of complex summarization tasks.
    Covers: row count, category/intent counts, deduplication stats, quality notes
    (placeholder rate, intent imbalance, synthetic origin).
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
                "single_turn_only": True,
            },
        },
        indent=2,
    )


class GetConversationExampleInput(BaseModel):
    intent: str = Field(
        ...,
        description=(
            "Intent to fetch conversations for, e.g. 'GET_REFUND', 'CANCEL_ORDER'. "
            "Must be a valid intent from list_intents."
        ),
    )
    limit: int = Field(
        2,
        ge=1,
        le=10,
        description="Maximum number of conversations to return (1–10). Defaults to 2.",
    )


@tool(args_schema=GetConversationExampleInput)
def get_conversation_example(intent: str, limit: int = 2) -> str:
    """
    Return synthetic multi-turn conversations for a specific intent.

    Use this when the user wants to see how a full customer-agent exchange unfolds
    for an intent — e.g. 'How do agents typically handle refund requests step by step?'
    Requires scripts/augment_conversations.py to have been run first; returns a notice
    if no conversations have been generated yet.

    Prefer filter_records for simple one-turn examples; use this for multi-turn flow.
    """
    records = load_conversations(intent=intent.strip().upper())
    if not records:
        return json.dumps({
            "note": (
                "No synthetic conversations found. "
                "Run: python scripts/augment_conversations.py --max-intents 10"
            ),
            "intent": intent.strip().upper(),
        }, indent=2)

    sample = records[:max(1, min(int(limit), 10))]
    return json.dumps(
        {"intent": intent.strip().upper(), "count": len(records), "conversations": sample},
        indent=2,
        ensure_ascii=False,
    )


class SuggestNextQueryInput(BaseModel):
    discussed_topics: str = Field(
        ...,
        description=(
            "Comma-separated list of categories and/or intents the user has already "
            "asked about in this session, e.g. 'REFUND, GET_REFUND, ORDER'. "
            "Used to avoid repeating suggestions and to pick relevant follow-ups."
        ),
    )


@tool(args_schema=SuggestNextQueryInput)
def suggest_next_query(discussed_topics: str) -> str:
    """
    Generate 1-3 concrete follow-up query suggestions based on what the user has discussed.

    Call this ONLY when the user asks 'What should I query next?', 'What can I explore?',
    or similar recommendation requests.

    IMPORTANT — after calling this tool:
    1. Present the suggestions clearly to the user.
    2. Ask 'Should I go ahead with [primary suggestion]?'
    3. Do NOT call any data tools (count_rows, filter_records, etc.) yet.
    4. Wait for explicit user confirmation ('yes', 'go ahead', 'do it') before executing.
    5. If the user wants to refine ('I'd rather see examples'), update the suggestion
       and ask again — still do not execute until confirmed.
    """
    meta = load_metadata()
    topics = {t.strip().upper() for t in discussed_topics.split(",") if t.strip()}
    all_categories = meta.get("categories", [])
    all_intents = meta.get("intents", [])

    suggestions: list[str] = []

    # Suggest distribution of unexplored categories
    for cat in all_categories:
        if cat not in topics and len(suggestions) < 2:
            suggestions.append(f"What is the distribution of intents in the {cat} category?")

    # Suggest examples for a category the user mentioned
    for topic in topics:
        if topic in all_categories:
            suggestions.append(f"Show me 5 examples from the {topic} category.")
            break

    # Suggest an unexplored intent count if list is short
    for intent in all_intents:
        if intent not in topics and len(suggestions) < 3:
            suggestions.append(f"How many rows have the {intent} intent?")
            break

    if not suggestions:
        suggestions = [
            "What are the top categories by row count?",
            "Show me 5 examples from the REFUND category.",
            "How many rows mention 'password'?",
        ]

    return json.dumps({"suggestions": suggestions[:3]}, indent=2)


ALL_TOOLS = [
    list_categories,
    list_intents,
    count_rows,
    distribution_by_category,
    distribution_by_intent,
    filter_records,
    search_instructions,
    dataset_summary,
    get_conversation_example,
    suggest_next_query,
]
