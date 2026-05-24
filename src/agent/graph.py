"""LangGraph ReAct agent with router preamble."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from typing import Annotated, TypedDict

from src.agent.llm import agent_llm
from src.agent.router import classify_query
from src.config import MAX_ITERATIONS
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Bitext customer-support dataset analyst.
Use tools for factual counts and samples. The data is synthetic training data, not live CRM.
When the user mentions refunds, shipping, etc., map to dataset category/intent fields.
Be concise. Cite numbers from tool outputs.

Query recommendation flow (when route=recommend):
- Call suggest_next_query with the topics discussed so far.
- Present the suggestions clearly and ask 'Should I go ahead with [suggestion]?'
- Do NOT call any data tools (count_rows, filter_records, etc.) until the user confirms.
- If the user refines ('I'd rather see examples'), adjust the suggestion and ask again.
- Only execute the query after explicit confirmation ('yes', 'go ahead', 'do it').

User profile: if asked 'What do you remember about me?', answer from the profile context."""

FALLBACK_MESSAGE = (
    f"I reached the maximum reasoning steps ({MAX_ITERATIONS}) without a conclusive answer. "
    "Please try rephrasing your question or breaking it into smaller parts."
)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_reason: str
    profile_context: str


def _route_node(state: AgentState) -> dict:
    """Classify query; reply directly if OOS, else inject system preamble for agent."""
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    result = classify_query(text)
    route = result["route"]
    if route == "out_of_scope":
        reply = (
            "I only analyze the Bitext customer-support training dataset "
            "(categories, intents, counts, samples). Please ask about that data."
        )
        return {
            "messages": [AIMessage(content=reply)],
            "route": route,
            "route_reason": result["reason"],
        }
    entities = result.get("entities", {})
    extra = ""
    if entities.get("intent") or entities.get("category"):
        extra = f"\nResolved filters: {entities}"

    profile_ctx = state.get("profile_context", "")
    profile_block = f"\n\n{profile_ctx}" if profile_ctx else ""

    preamble = SystemMessage(
        content=f"{SYSTEM_PROMPT}{profile_block}\nRoute: {route}.{extra}\nReason: {result['reason']}"
    )
    return {
        "messages": [preamble],
        "route": route,
        "route_reason": result["reason"],
    }


def build_graph(checkpointer=None):
    """Build and compile the LangGraph router + ReAct agent workflow."""
    llm = agent_llm()
    react = create_react_agent(llm, ALL_TOOLS)

    def run_agent(state: AgentState) -> dict:
        """Invoke prebuilt ReAct executor unless query was out-of-scope."""
        if state.get("route") == "out_of_scope":
            return {}
        out = react.invoke(
            {"messages": state["messages"]},
            config={"recursion_limit": MAX_ITERATIONS},
        )
        return {"messages": out["messages"]}

    workflow = StateGraph(AgentState)
    workflow.add_node("route", _route_node)
    workflow.add_node("agent", run_agent)
    workflow.set_entry_point("route")

    def after_route(state: AgentState) -> str:
        if state.get("route") == "out_of_scope":
            return "end"
        return "agent"  # structured, unstructured, and recommend all go to agent

    workflow.add_conditional_edges("route", after_route, {"agent": "agent", "end": END})
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)


def _extract_reasoning(messages: list[BaseMessage]) -> tuple[list[str], str]:
    """Return (reasoning_steps, final_answer) extracted from the message list."""
    steps: list[str] = []
    final = "(no response)"
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args_str = json.dumps(tc.get("args", {}), ensure_ascii=False)
                    steps.append(f"  -> {tc['name']}({args_str})")
            elif msg.content:
                final = msg.content if isinstance(msg.content, str) else str(msg.content)
        elif isinstance(msg, ToolMessage):
            raw = msg.content if isinstance(msg.content, str) else str(msg.content)
            snippet = raw[:200] + "..." if len(raw) > 200 else raw
            steps.append(f"  <- {getattr(msg, 'name', 'tool')}: {snippet}")
    return steps, final


def run_turn(
    graph,
    session_id: str,
    user_text: str,
    verbose: bool = True,
    profile_context: str = "",
) -> str:
    """
    Execute one conversation turn and return the agent's final answer.

    When verbose=True, prints each tool call and observation to stdout so the
    grader (and user) can follow the agent's reasoning steps.
    Catches GraphRecursionError and returns a graceful fallback message.
    """
    config = {"configurable": {"thread_id": session_id}}
    try:
        result = graph.invoke(
            {
                "messages": [HumanMessage(content=user_text)],
                "profile_context": profile_context,
            },
            config=config,
        )
    except GraphRecursionError:
        return FALLBACK_MESSAGE

    steps, final = _extract_reasoning(result.get("messages", []))

    if verbose and steps:
        print("[Reasoning]")
        for step in steps:
            print(step)

    return final
