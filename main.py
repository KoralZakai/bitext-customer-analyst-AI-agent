"""
CLI for the Bitext customer-support data analyst agent.
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def cmd_check() -> int:
    """Verify preprocess artifacts exist and are loadable without calling the LLM."""
    from src.config import DATA_METADATA_PATH, DATA_PROCESSED_PATH, NEBIUS_API_KEY
    from src.data.loader import dataset_ready, load_metadata, load_processed_dataframe

    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("WARNING: Set NEBIUS_API_KEY in .env before running the agent.")

    if not dataset_ready():
        print("Processed dataset missing. Run:")
        print("  python -m src.data.preprocess")
        return 1

    df = load_processed_dataframe()
    meta = load_metadata()
    print(f"OK: loaded {len(df)} rows from {DATA_PROCESSED_PATH}")
    print(f"Categories: {len(meta['categories'])} | Intents: {len(meta['intents'])}")
    print(f"Metadata: {DATA_METADATA_PATH}")
    return 0


def cmd_chat(session: str) -> int:
    """
    Interactive REPL loop — prints reasoning steps (tool calls + observations)
    before each final answer. Profile is loaded, injected, and updated each turn.
    """
    from src.config import NEBIUS_API_KEY
    from src.agent.graph import build_graph, run_turn
    from src.agent.llm import router_llm
    from src.data.loader import dataset_ready
    from src.memory.checkpoint import get_checkpointer
    from src.memory.profile import load_profile, profile_to_context, update_profile

    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("Set NEBIUS_API_KEY in .env", file=sys.stderr)
        return 1
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        return 1

    profile_llm = router_llm()  # reuse the cheap router model for profile extraction

    print(f"Session: {session} | Type 'quit' to exit.")
    profile = load_profile(session)
    if profile.get("name"):
        print(f"Welcome back, {profile['name']}!\n")
    else:
        print()

    with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer=checkpointer)
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if not user:
                continue
            if user.lower() in {"quit", "exit", "q"}:
                break

            profile_ctx = profile_to_context(load_profile(session))

            reply = run_turn(
                graph,
                session_id=session,
                user_text=user,
                verbose=True,
                profile_context=profile_ctx,
            )
            print(f"Agent: {reply}\n")

            # Update profile in background using router LLM (cheap)
            try:
                profile = update_profile(session, user, reply, profile_llm)
            except Exception:
                pass  # profile update failure must never break the conversation


def main() -> None:
    """CLI entry: python main.py [--session NAME] [--check]"""
    parser = argparse.ArgumentParser(description="Bitext data analyst agent")
    parser.add_argument(
        "--session",
        default="demo",
        help="Conversation thread id for checkpoint memory (default: demo)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify dataset artifacts (no LLM chat)",
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(cmd_check())
    if cmd_check() != 0:
        raise SystemExit(1)
    raise SystemExit(cmd_chat(args.session))


if __name__ == "__main__":
    main()
