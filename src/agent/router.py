"""Classify user queries: structured, unstructured, or out-of-scope."""

from __future__ import annotations

import json
import re
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.llm import router_llm
from src.data.loader import load_metadata
from src.data.preprocess import apply_keyword_to_category, apply_keyword_to_intent, load_aliases

RouteType = Literal["structured", "unstructured", "out_of_scope", "recommend", "profile"]

ROUTER_SYSTEM = """You route questions about the Bitext customer-support TRAINING dataset.
Return JSON only: {"route": "structured"|"unstructured"|"out_of_scope"|"recommend"|"profile", "reason": "..."}

structured:   counts, lists, filters, distributions, dataset stats
unstructured: exploratory summaries, comparisons, insights needing multiple tools
recommend:    user asks what to query next, wants suggestions, says 'what should I explore?'
profile:      user shares personal facts/preferences or asks what you remember about them
out_of_scope: unrelated to this dataset (weather, code, other companies, live CRM data)
"""


def resolve_entities(keyword: str | None) -> dict[str, str | None]:
    """
    Name: resolve_entities
    Input: keyword — user text or phrase to resolve
    Output: dict — keys intent and category with resolved values or None
    Purpose: Apply alias map before the ReAct agent chooses tool arguments.
    """
    if not keyword:
        return {"intent": None, "category": None}
    aliases = load_aliases()
    return {
        "intent": apply_keyword_to_intent(keyword, aliases),
        "category": apply_keyword_to_category(keyword, aliases),
    }


def classify_query(user_text: str) -> dict:
    """
    Name: classify_query
    Input: user_text — natural language question from the CLI
    Output: dict — route, reason, and entities (intent/category hints)
    Purpose: Decide routing path and inject resolved filters into agent context.
    """
    meta = load_metadata()
    hints = {
        "categories": meta.get("categories", [])[:15],
        "intent_count": len(meta.get("intents", [])),
        "row_count": meta.get("row_count"),
    }
    llm = router_llm()
    msg = llm.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(
                content=f"Dataset hints: {json.dumps(hints)}\nUser: {user_text}"
            ),
        ]
    )
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    route: RouteType = "structured"
    reason = text
    if match:
        try:
            data = json.loads(match.group())
            r = data.get("route", "structured")
            if r in ("structured", "unstructured", "out_of_scope", "recommend", "profile"):
                route = r  # type: ignore[assignment]
            reason = data.get("reason", reason)
        except json.JSONDecodeError:
            pass
    entities = resolve_entities(user_text)
    return {"route": route, "reason": reason, "entities": entities}
