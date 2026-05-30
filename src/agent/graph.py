"""LangGraph ReAct agent with router preamble."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict

from src.agent.llm import agent_llm
from src.agent.router import classify_query
from src.config import MAX_ITERATIONS
from src.tools import ALL_TOOLS

DECLINE_MESSAGE = (
    "I only analyze the Bitext customer-support training dataset "
    "(categories, intents, counts, samples). Please ask about that data."
)

SYSTEM_PROMPT = """You are a Bitext customer-support dataset analyst.
Use tools for factual counts and samples. The data is synthetic training data, not live CRM.
When the user mentions refunds, shipping, etc., map to dataset category/intent fields.
Be concise. Cite numbers from tool outputs.

Grounding rules — ALWAYS follow these:
- Base every answer on tool results, never on LLM general knowledge.
- For summaries: open with the exact category or intent name (e.g. "The FEEDBACK category..."),
  then describe what the tool results show — intents found, example instruction text, patterns.
- Once you have the data you need from tools, return your answer immediately — do not call
  additional tools to "verify" results you already have.

Counting pipeline ('how many ...?' questions) — ALWAYS two steps:
- STEP 1: call filter_by_intent (for an intent like refunds/complaints) or filter_by_category
  (for a whole category) to SELECT the rows. It returns a result_handle, not a count.
- STEP 2: call count_rows(result_handle=...) with that handle to get the exact number.
- The filter tools deliberately return only a preview, never the total — you MUST call
  count_rows to get the number. State the exact count from count_rows in your answer.
- To total several counts (e.g. complaints + refunds), run the two-step pipeline for each,
  then add the numbers yourself.

Showing examples:
- Use filter_records to show concrete example rows. Open with the resolved category/intent,
  e.g. "Here are GET_REFUND examples from the REFUND category:" then list instructions.
- For a 'show me more' follow-up, pass offset = number of examples already shown so the new
  examples are DIFFERENT and never repeat earlier ones.

Using memory:
- If a question can be answered from numbers or facts already established earlier in this
  conversation (e.g. "what is the total of the last two counts?"), compute it directly from
  those remembered values. Do NOT re-run tools to re-fetch data you already have.

Query recommendation flow (when route=recommend):
- Call suggest_next_query with the topics discussed so far.
- Present the suggestions clearly and ask 'Should I go ahead with [suggestion]?'
- Do NOT call any data tools (count_rows, filter_records, etc.) until the user confirms.
- If the user refines ('I'd rather see examples'), adjust the suggestion and ask again.
- Only execute the query after explicit confirmation ('yes', 'go ahead', 'do it').

Profile / memory flow (when route=profile):
- If the user shares a personal fact or preference, acknowledge it briefly and naturally.
- If the user asks what you remember about them, answer from the profile context.
- Do NOT call dataset tools for profile or memory turns.

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
    iterations: int


def _route_node(state: AgentState) -> dict:
    """Classify the query and, for in-scope routes, inject the system preamble.

    Out-of-scope queries set route only; the dedicated `decline` node replies.
    """
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    result = classify_query(text)
    route = result["route"]
    if route == "out_of_scope":
        return {"route": route, "route_reason": result["reason"], "iterations": 0}

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
        "iterations": 0,
    }


def _decline_node(state: AgentState) -> dict:
    """Politely decline out-of-scope queries — no LLM general-knowledge answer."""
    return {"messages": [AIMessage(content=DECLINE_MESSAGE)]}


def _profile_node(state: AgentState) -> dict:
    """Answer profile and memory turns directly without dataset tool calls."""
    last = state["messages"][-1]
    text = last.content if isinstance(last.content, str) else str(last.content)
    lower = text.lower()
    profile_ctx = state.get("profile_context", "").strip()

    if "remember about me" in lower or "what do you know about me" in lower:
        if profile_ctx:
            lines = [line.strip() for line in profile_ctx.splitlines()[1:] if line.strip()]
            details = "\n".join(f"- {line}" for line in lines) if lines else "- I do not have any saved details yet."
            return {"messages": [AIMessage(content=f"Here is what I remember about you:\n{details}")]}
        return {
            "messages": [
                AIMessage(
                    content="I do not have any saved details about you yet. Tell me your name, preferences, or topics you care about and I will remember them for this session ID."
                )
            ]
        }

    if "my name is" in lower:
        return {"messages": [AIMessage(content="Noted. I will remember your name for this session ID.")]}
    if "prefer" in lower:
        return {"messages": [AIMessage(content="Noted. I will keep that preference in mind.")]}
    return {"messages": [AIMessage(content="Noted. I will remember that for this session ID.")]}


def _fallback_node(state: AgentState) -> dict:
    """Graceful message when the agent exceeds the max reasoning steps."""
    return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}


def build_graph(checkpointer=None, llm=None):
    """Build and compile the explicit LangGraph ReAct workflow.

    Nodes: route -> (decline | agent), agent <-> tools loop, agent -> fallback when
    the iteration budget (MAX_ITERATIONS) is exhausted. `llm` is injectable for tests.
    """
    chat = llm if llm is not None else agent_llm()
    llm_with_tools = chat.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def agent_node(state: AgentState) -> dict:
        """One LLM reasoning step; counts toward the iteration budget."""
        iterations = state.get("iterations", 0) + 1
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "iterations": iterations}

    def after_route(state: AgentState) -> str:
        route = state.get("route")
        if route == "out_of_scope":
            return "decline"
        if route == "profile":
            return "profile"
        return "agent"

    def after_agent(state: AgentState) -> str:
        """Loop to tools while the LLM requests them; stop at the iteration budget."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            if state.get("iterations", 0) >= MAX_ITERATIONS:
                return "fallback"
            return "tools"
        return "end"

    workflow = StateGraph(AgentState)
    workflow.add_node("route", _route_node)
    workflow.add_node("decline", _decline_node)
    workflow.add_node("profile", _profile_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("fallback", _fallback_node)

    workflow.set_entry_point("route")
    workflow.add_conditional_edges(
        "route",
        after_route,
        {"decline": "decline", "profile": "profile", "agent": "agent"},
    )
    workflow.add_edge("decline", END)
    workflow.add_edge("profile", END)
    workflow.add_conditional_edges(
        "agent", after_agent, {"tools": "tools", "fallback": "fallback", "end": END}
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("fallback", END)

    # LangGraph Studio may inject checkpointer configuration as a dict.
    # compile() expects a saver instance, bool, or None.
    cp = None if isinstance(checkpointer, dict) else checkpointer
    return workflow.compile(checkpointer=cp)


def build_studio_graph():
    """Zero-argument graph factory for LangGraph Studio/dev server loading."""
    return build_graph()


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
    # Each agent step is followed by a tools step, so allow ~2 supersteps per
    # iteration plus headroom for route/decline/fallback. The agent_node counter
    # is the primary guard; this recursion_limit is a backstop.
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": MAX_ITERATIONS * 2 + 5,
    }
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
