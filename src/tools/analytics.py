"""Analytics tools over preprocessed Bitext data."""

from __future__ import annotations

import json
import uuid
from typing import Any

import pandas as pd
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.loader import load_conversations, load_metadata, load_real_dataframe, load_search_dataframe
from src.data.preprocess import apply_keyword_to_category, apply_keyword_to_intent, load_aliases


def _real_df() -> pd.DataFrame:
    return load_real_dataframe()


# ---------------------------------------------------------------------------
# Filter result registry — powers the filter -> count multi-step pipeline.
#
# A filter tool (filter_by_intent / filter_by_category) selects rows and stores
# the filter descriptor under an opaque handle. count_rows then counts the rows
# for that handle. This forces genuine multi-step reasoning ("how many refunds?"
# = filter_by_intent then count_rows) instead of a single all-in-one call, and
# keeps counting deterministic (the handle re-applies the same filter).
# ---------------------------------------------------------------------------
_FILTER_REGISTRY: dict[str, dict] = {}


def _apply_descriptor(df: pd.DataFrame, desc: dict) -> pd.DataFrame:
    """Apply a stored filter descriptor (category / intent / keyword) to a frame."""
    if desc.get("category"):
        df = df[df["category"].astype(str) == desc["category"]]
    if desc.get("intent"):
        df = df[df["intent"].astype(str) == desc["intent"]]
    if desc.get("keyword"):
        kw = desc["keyword"]
        mask = df["instruction"].str.contains(kw, case=False, na=False) | df[
            "response"
        ].str.contains(kw, case=False, na=False)
        df = df[mask]
    return df


def count_filtered(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> dict:
    """Direct count with filters — used by the MCP wrapper and programmatic callers.

    The agent-facing count_rows tool is handle-based (see below); this helper keeps
    a simple one-shot counting API available outside the ReAct loop.
    """
    df = _real_df()
    cat, it, kw = _resolve_filters(category, intent, keyword)
    df = _apply_descriptor(df, {"category": cat, "intent": it, "keyword": kw})
    return {"count": int(len(df)), "category": cat, "intent": it, "keyword": kw}


def _resolve_filters(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    aliases = load_aliases()

    def _clean(v: str | None) -> str | None:
        if not v:
            return None
        s = v.strip()
        return None if s.lower() in ("null", "none", "") else s

    cat_raw = _clean(category)
    it_raw = _clean(intent)
    kw = _clean(keyword)

    cat = cat_raw.upper() if cat_raw else None
    it = it_raw.upper() if it_raw else None

    # Apply alias resolution to the intent parameter itself — handles cases where
    # the agent passes a natural word like "complaints" instead of "COMPLAINT".
    if it:
        mapped = apply_keyword_to_intent(it.lower(), aliases)
        if mapped:
            it = mapped

    alias_resolved = False
    if kw:
        # If keyword resolves to an alias, always clear it to avoid double-filtering.
        # Synthetic instruction text won't contain the user's natural phrase literally,
        # so text-searching on top of an intent/category filter always shrinks counts.
        mapped = apply_keyword_to_intent(kw, aliases)
        if mapped:
            if not it:
                it = mapped
            alias_resolved = True
        mapped_cat = apply_keyword_to_category(kw, aliases)
        if mapped_cat:
            if not cat:
                cat = mapped_cat
            alias_resolved = True

    # When the keyword resolved via alias to an intent/category, clear it so we
    # don't also run a text-contains filter. Synthetic data won't contain the
    # user's natural phrase (e.g. "money back") literally in instruction text.
    if alias_resolved:
        kw = None

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


class FilterByIntentInput(BaseModel):
    intent: str = Field(
        ...,
        description=(
            "Intent to select rows for, e.g. 'GET_REFUND', 'CANCEL_ORDER', 'COMPLAINT'. "
            "Natural phrases are resolved via aliases ('refund requests' → GET_REFUND, "
            "'complaints' → COMPLAINT)."
        ),
    )


class FilterByCategoryInput(BaseModel):
    category: str = Field(
        ...,
        description=(
            "Category to select rows for, e.g. 'REFUND', 'ORDER', 'SHIPPING', 'ACCOUNT'. "
            "Natural phrases are resolved via aliases."
        ),
    )


class CountRowsInput(BaseModel):
    result_handle: str | None = Field(
        None,
        description=(
            "The result_handle returned by a previous filter_by_intent or "
            "filter_by_category call. Counts how many rows that filter selected. "
            "Leave empty to count ALL rows in the whole dataset."
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
    offset: int = Field(
        0,
        ge=0,
        description=(
            "Number of matching rows to SKIP before sampling. Use this to page "
            "through examples: for a 'show me more' follow-up, set offset to the "
            "number of examples already shown (e.g. offset=3 after showing 3) so "
            "you return DIFFERENT rows and never repeat earlier examples."
        ),
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


@tool(args_schema=FilterByIntentInput)
def filter_by_intent(intent: str) -> str:
    """
    Select the rows that match a given intent so they can be counted or inspected.

    This is STEP 1 of answering a 'how many <intent>?' question. It resolves the
    intent (aliases like 'refund requests' → GET_REFUND are handled), selects the
    matching rows, and returns a small preview plus a `result_handle`.

    IMPORTANT: this returns only a SAMPLE preview, NOT the total count. To get the
    total number of matching rows, call count_rows with the returned result_handle
    (STEP 2). Use filter_by_category instead when filtering by a whole category.
    """
    df = _real_df()
    _, it, _ = _resolve_filters(intent=intent)
    if not it:
        _, it, _ = _resolve_filters(keyword=intent)
    if not it:
        return json.dumps(
            {
                "error": f"Could not resolve '{intent}' to a known intent. "
                "Call list_intents to see valid intent names.",
            },
            indent=2,
        )
    sub = df[df["intent"].astype(str) == it]
    handle = uuid.uuid4().hex[:12]
    _FILTER_REGISTRY[handle] = {"intent": it}
    preview = [
        {
            "category": str(r["category"]),
            "intent": str(r["intent"]),
            "instruction": str(r["instruction"])[:200],
        }
        for _, r in sub.head(5).iterrows()
    ]
    return json.dumps(
        {
            "resolved_intent": it,
            "result_handle": handle,
            "preview_rows": preview,
            "note": (
                "Preview only — this is NOT the total. To get the total number of "
                f"matching rows, call count_rows(result_handle='{handle}')."
            ),
        },
        indent=2,
    )


@tool(args_schema=FilterByCategoryInput)
def filter_by_category(category: str) -> str:
    """
    Select the rows that belong to a given category so they can be counted.

    This is STEP 1 of answering a 'how many rows in <category>?' question. It
    resolves the category, selects matching rows, and returns a preview plus a
    `result_handle`.

    IMPORTANT: this returns only a SAMPLE preview, NOT the total count. To get the
    total, call count_rows with the returned result_handle (STEP 2).
    """
    df = _real_df()
    cat, _, _ = _resolve_filters(category=category)
    if not cat:
        cat, _, _ = _resolve_filters(keyword=category)
    if not cat:
        return json.dumps(
            {
                "error": f"Could not resolve '{category}' to a known category. "
                "Call list_categories to see valid category names.",
            },
            indent=2,
        )
    sub = df[df["category"].astype(str) == cat]
    handle = uuid.uuid4().hex[:12]
    _FILTER_REGISTRY[handle] = {"category": cat}
    preview = [
        {
            "category": str(r["category"]),
            "intent": str(r["intent"]),
            "instruction": str(r["instruction"])[:200],
        }
        for _, r in sub.head(5).iterrows()
    ]
    return json.dumps(
        {
            "resolved_category": cat,
            "result_handle": handle,
            "preview_rows": preview,
            "note": (
                "Preview only — this is NOT the total. To get the total number of "
                f"matching rows, call count_rows(result_handle='{handle}')."
            ),
        },
        indent=2,
    )


@tool(args_schema=CountRowsInput)
def count_rows(result_handle: str | None = None) -> str:
    """
    Count rows. This is STEP 2 of the counting pipeline.

    To count rows for a specific intent or category, FIRST call filter_by_intent or
    filter_by_category, THEN pass the result_handle it returned to this tool. Leave
    result_handle empty to count every row in the whole dataset.

    Example flow — 'How many refund requests?':
      1. filter_by_intent(intent='refund requests')  -> returns result_handle "abc123"
      2. count_rows(result_handle='abc123')           -> returns {"count": 918}
    """
    df = _real_df()
    if result_handle:
        desc = _FILTER_REGISTRY.get(result_handle.strip())
        if desc is None:
            return json.dumps(
                {
                    "count": None,
                    "error": (
                        "Unknown result_handle. Call filter_by_intent or "
                        "filter_by_category first, then pass the result_handle it returns."
                    ),
                },
                indent=2,
            )
        df = _apply_descriptor(df, desc)
        return json.dumps({"count": int(len(df)), "filter": desc}, indent=2)
    return json.dumps({"count": int(len(df)), "filter": "all rows"}, indent=2)


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
    offset: int = 0,
) -> str:
    """
    Return sample example records (instruction + response) matching the given filters.

    Use this to SHOW the user concrete examples: 'show me 3 examples from the SHIPPING
    category' or 'show examples of people wanting their money back'. Returns up to
    `limit` sample rows starting at `offset`.

    For a 'show me more' follow-up, set `offset` to the number of examples already
    shown so the new examples are DIFFERENT (e.g. after showing 3, call with offset=3).

    This tool is for displaying examples, not counting — to answer 'how many', use
    filter_by_intent / filter_by_category followed by count_rows.
    Alias resolution applies — 'money back' resolves to GET_REFUND examples.
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
    offset = max(0, int(offset))
    rows = df.iloc[offset : offset + limit]
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
    return json.dumps(
        {"category": cat, "intent": it, "offset": offset, "returned": len(out), "samples": out},
        indent=2,
    )


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
    filter_by_intent,
    filter_by_category,
    count_rows,
    distribution_by_category,
    distribution_by_intent,
    filter_records,
    search_instructions,
    dataset_summary,
    get_conversation_example,
    suggest_next_query,
]
