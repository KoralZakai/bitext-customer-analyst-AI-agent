"""
Streamlit chat UI for the Bitext data analyst agent (Bonus A).

Run: streamlit run app.py
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Bitext Analyst", page_icon="📊", layout="wide")

_EXAMPLE_QUERIES = [
    "What categories exist in the dataset?",
    "How many refund requests did we get?",
    "Show me 5 examples from the SHIPPING category.",
    "Summarize how agents respond to complaint intents.",
    "Show me examples of people wanting their money back.",
    "What is the distribution of intents in the ACCOUNT category?",
    "What should I query next?",
    "What do you remember about me?",
    "Who is the president of France?",
]


@st.cache_resource
def _get_graph():
    """Build the LangGraph agent once and cache it for all Streamlit reruns."""
    from src.agent.graph import build_graph
    from src.config import CHECKPOINT_DB_PATH
    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    return build_graph(checkpointer=saver)


def _check_ready() -> bool:
    """Return True if the dataset is preprocessed and API key is set."""
    from src.data.loader import dataset_ready
    from src.config import NEBIUS_API_KEY
    return dataset_ready() and bool(NEBIUS_API_KEY) and NEBIUS_API_KEY != "your_nebius_api_key_here"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Bitext Analyst")
    st.markdown("---")
    session_id = st.text_input("Session ID", value="streamlit_demo",
                                help="Change to start a new conversation or resume an old one.")

    # Profile display
    from src.memory.profile import load_profile
    profile = load_profile(session_id)
    has_profile = profile.get("name") or profile.get("interests") or profile.get("notes")
    if has_profile:
        st.markdown("**What I know about you**")
        if profile.get("name"):
            st.markdown(f"Name: **{profile['name']}**")
        if profile.get("interests"):
            st.markdown(f"Interests: {', '.join(profile['interests'])}")
        if profile.get("preferences"):
            st.markdown(f"Preferences: {', '.join(profile['preferences'])}")
        if profile.get("notes"):
            st.caption(profile["notes"])
        st.markdown("---")

    st.markdown("**Example queries**")
    for q in _EXAMPLE_QUERIES:
        st.markdown(f"- {q}")
    st.markdown("---")
    st.caption("Powered by LangGraph + Nebius Token Factory")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.header("Bitext Customer-Service Dataset Analyst")

if not _check_ready():
    st.error(
        "Dataset not ready or NEBIUS_API_KEY missing.  \n"
        "Run: `python main.py --check` then set your API key in `.env`."
    )
    st.stop()

# Per-session chat history stored in Streamlit session state
chat_key = f"chat_{session_id}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []  # list of {role, content, steps}

# Render existing messages
for entry in st.session_state[chat_key]:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])
        if entry.get("steps"):
            with st.expander("🔍 Reasoning steps"):
                st.code("\n".join(entry["steps"]))

# Chat input
user_input = st.chat_input("Ask about the Bitext dataset…")
if user_input:
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state[chat_key].append({"role": "user", "content": user_input, "steps": []})

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            from langchain_core.messages import HumanMessage
            from langgraph.errors import GraphRecursionError
            from src.agent.graph import _extract_reasoning, FALLBACK_MESSAGE
            from src.agent.llm import router_llm
            from src.config import MAX_ITERATIONS
            from src.memory.profile import load_profile, profile_to_context, update_profile

            graph = _get_graph()
            profile_ctx = profile_to_context(load_profile(session_id))
            config = {"configurable": {"thread_id": session_id}}

            try:
                result = graph.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "profile_context": profile_ctx,
                    },
                    config=config,
                )
                steps, final = _extract_reasoning(result.get("messages", []))
            except GraphRecursionError:
                steps, final = [], FALLBACK_MESSAGE
            except Exception as exc:
                steps, final = [], f"Error: {exc}"

        st.markdown(final)
        if steps:
            with st.expander("🔍 Reasoning steps"):
                st.code("\n".join(steps))

        # Update profile (best-effort)
        try:
            update_profile(session_id, user_input, final, router_llm())
        except Exception:
            pass

    st.session_state[chat_key].append({"role": "assistant", "content": final, "steps": steps})
