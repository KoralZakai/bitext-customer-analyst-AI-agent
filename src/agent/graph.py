"""LangGraph ReAct agent with router preamble."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

from src.agent.llm import agent_llm
from src.agent.router import classify_query
from src.config import MAX_ITERATIONS
from src.tools import ALL_TOOLS

SYSTEM_PROMPT = """You are a Bitext customer-support dataset analyst.
Use tools for factual counts and samples. The data is synthetic training data, not live CRM.
When the user mentions refunds, shipping, etc., map to dataset category/intent fields.
Be concise. Cite numbers from tool outputs."""


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_reason: str


def _route_node(state: AgentState) -> dict:
    """
    Name: _route_node
    Input: state — AgentState with messages list (last item is user turn)
    Output: dict — partial state update (messages, route, route_reason)
    Purpose: Classify query; reply directly if OOS else inject system preamble for agent.
    """
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
    preamble = SystemMessage(
        content=f"{SYSTEM_PROMPT}\nRoute: {route}.{extra}\nReason: {result['reason']}"
    )
    return {
        "messages": [preamble],
        "route": route,
        "route_reason": result["reason"],
    }


def build_graph(checkpointer=None):
    """
    Name: build_graph
    Input: checkpointer — optional LangGraph checkpointer (e.g. SqliteSaver)
    Output: CompiledGraph — LangGraph runnable with route and ReAct nodes
    Purpose: Wire router + tool-calling agent for multi-turn CLI sessions.
    """
    llm = agent_llm()
    react = create_react_agent(llm, ALL_TOOLS)

    def run_agent(state: AgentState) -> dict:
        """
        Name: run_agent
        Input: state — AgentState after routing
        Output: dict — messages from ReAct subgraph
        Purpose: Invoke prebuilt ReAct executor unless query was out-of-scope.
        """
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
        """
        Name: after_route
        Input: state — AgentState with route field set
        Output: str — next node name ("agent" or "end")
        Purpose: Skip ReAct node when router marked query out-of-scope.
        """
        if state.get("route") == "out_of_scope":
            return "end"
        return "agent"

    workflow.add_conditional_edges("route", after_route, {"agent": "agent", "end": END})
    workflow.add_edge("agent", END)
    return workflow.compile(checkpointer=checkpointer)


def run_turn(graph, session_id: str, user_text: str) -> str:
    """
    Name: run_turn
    Input: graph — compiled graph; session_id — thread id; user_text — user message
    Output: str — final assistant text for this turn
    Purpose: Single CLI turn with checkpoint thread_id for conversation memory.
    """
    config = {"configurable": {"thread_id": session_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_text)]},
        config=config,
    )
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return "(no response)"
