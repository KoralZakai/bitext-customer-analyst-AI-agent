"""FastMCP server exposing Bitext analytics tools."""

from __future__ import annotations

from fastmcp import FastMCP

from src.tools.analytics import (
    count_rows,
    dataset_summary,
    distribution_by_category,
    distribution_by_intent,
    filter_records,
    list_categories,
    list_intents,
    search_instructions,
)

mcp = FastMCP("bitext-analyst")


@mcp.tool
def list_categories_tool() -> str:
    """
    List all customer-support categories in the dataset.

    Name: list_categories_tool
    Input: None
    Output: str — JSON from list_categories tool
    Purpose: MCP wrapper for external clients (Cursor, etc.).
    """
    return list_categories.invoke({})


@mcp.tool
def list_intents_tool(category: str | None = None) -> str:
    """
    List intents, optionally filtered by category.

    Name: list_intents_tool
    Input: category — optional category filter
    Output: str — JSON from list_intents tool
    Purpose: Expose intent taxonomy over MCP.
    """
    return list_intents.invoke({"category": category})


@mcp.tool
def count_rows_tool(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
) -> str:
    """
    Count rows with optional filters (real data only).

    Name: count_rows_tool
    Input: category, intent, keyword — optional filters
    Output: str — JSON count result
    Purpose: MCP access to row counting without running the full CLI agent.
    """
    return count_rows.invoke({"category": category, "intent": intent, "keyword": keyword})


@mcp.tool
def distribution_by_category_tool() -> str:
    """
    Row counts per category.

    Name: distribution_by_category_tool
    Input: None
    Output: str — JSON category distribution
    Purpose: MCP wrapper for category distribution analytics.
    """
    return distribution_by_category.invoke({})


@mcp.tool
def distribution_by_intent_tool(category: str | None = None) -> str:
    """
    Row counts per intent.

    Name: distribution_by_intent_tool
    Input: category — optional scope
    Output: str — JSON intent distribution
    Purpose: MCP wrapper for intent distribution analytics.
    """
    return distribution_by_intent.invoke({"category": category})


@mcp.tool
def filter_records_tool(
    category: str | None = None,
    intent: str | None = None,
    keyword: str | None = None,
    limit: int = 5,
) -> str:
    """
    Sample records matching filters.

    Name: filter_records_tool
    Input: category, intent, keyword, limit — filter and sample size
    Output: str — JSON samples
    Purpose: MCP access to example rows from the dataset.
    """
    return filter_records.invoke(
        {"category": category, "intent": intent, "keyword": keyword, "limit": limit}
    )


@mcp.tool
def search_instructions_tool(keyword: str, limit: int = 10) -> str:
    """
    Search instructions including augmented paraphrases.

    Name: search_instructions_tool
    Input: keyword — search text; limit — max hits
    Output: str — JSON search results
    Purpose: MCP text search over instructions.
    """
    return search_instructions.invoke({"keyword": keyword, "limit": limit})


@mcp.tool
def dataset_summary_tool() -> str:
    """
    Dataset overview and quality notes.

    Name: dataset_summary_tool
    Input: None
    Output: str — JSON dataset summary
    Purpose: MCP high-level dataset orientation for clients.
    """
    return dataset_summary.invoke({})


def main() -> None:
    """
    Name: main
    Input: None
    Output: None (runs MCP server process)
    Purpose: Entry point for python -m src.mcp.server.
    """
    mcp.run()


if __name__ == "__main__":
    main()
