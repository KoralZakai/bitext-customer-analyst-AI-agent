"""FastMCP server exposing Bitext analytics tools."""

from __future__ import annotations

from fastmcp import FastMCP

from src.tools.analytics import (
    count_rows,
    dataset_summary,
    distribution_by_category,
    distribution_by_intent,
    filter_records,
    get_conversation_example,
    list_categories,
    list_intents,
    search_instructions,
    suggest_next_query,
)

mcp = FastMCP("bitext-analyst")


@mcp.tool
def list_categories_tool() -> str:
    """List all customer-support categories in the dataset."""
    return list_categories.invoke({})


@mcp.tool
def list_intents_tool(category: str | None = None) -> str:
    """List intents, optionally filtered by category."""
    return list_intents.invoke({"category": category})


@mcp.tool
def count_rows_tool(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> str:
    """Count rows with optional filters (real data only)."""
    return count_rows.invoke({"category": category, "intent": intent, "keyword": keyword})


@mcp.tool
def distribution_by_category_tool() -> str:
    """Row counts per category."""
    return distribution_by_category.invoke({})


@mcp.tool
def distribution_by_intent_tool(category: str | None = None) -> str:
    """Row counts per intent."""
    return distribution_by_intent.invoke({"category": category})


@mcp.tool
def filter_records_tool(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    limit: int = 5,
) -> str:
    """Sample records matching filters."""
    return filter_records.invoke(
        {"category": category, "intent": intent, "keyword": keyword, "limit": limit}
    )


@mcp.tool
def search_instructions_tool(keyword: str, limit: int = 10) -> str:
    """Search instructions including augmented paraphrases."""
    return search_instructions.invoke({"keyword": keyword, "limit": limit})


@mcp.tool
def dataset_summary_tool() -> str:
    """Dataset overview and quality notes."""
    return dataset_summary.invoke({})


@mcp.tool
def get_conversation_example_tool(intent: str, limit: int = 2) -> str:
    """
    Return synthetic multi-turn conversations for an intent.
    Requires augment_conversations.py to have been run first.
    """
    return get_conversation_example.invoke({"intent": intent, "limit": limit})


@mcp.tool
def suggest_next_query_tool(discussed_topics: str) -> str:
    """
    Generate 1-3 follow-up query suggestions based on discussed topics.
    Present suggestions to the user and wait for confirmation before executing.
    """
    return suggest_next_query.invoke({"discussed_topics": discussed_topics})


def main() -> None:
    """Entry point for python -m src.mcp.server."""
    mcp.run()


if __name__ == "__main__":
    main()

