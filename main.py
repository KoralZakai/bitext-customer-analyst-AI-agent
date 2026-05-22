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
    """
    Name: cmd_check
    Input: None
    Output: int — exit code 0 if dataset loads, 1 if missing
    Purpose: Verify preprocess artifacts without calling the LLM.
    """
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
    Name: cmd_chat
    Input: session — checkpoint thread id for conversation memory
    Output: int — exit code (0 on normal quit)
    Purpose: Interactive REPL loop for the LangGraph analyst agent.
    """
    from src.config import NEBIUS_API_KEY
    from src.agent.graph import build_graph, run_turn
    from src.data.loader import dataset_ready
    from src.memory.checkpoint import get_checkpointer

    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("Set NEBIUS_API_KEY in .env", file=sys.stderr)
        return 1
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        return 1

    print(f"Session: {session} | Type 'quit' to exit.\n")
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
            try:
                reply = run_turn(graph, session, user)
                print(f"Agent: {reply}\n")
            except Exception as e:
                print(f"Error: {e}\n", file=sys.stderr)


def main() -> None:
    """
    Name: main
    Input: None (parses --session and --check from argv)
    Output: None (raises SystemExit with appropriate code)
    Purpose: CLI entry for python main.py --session demo.
    """
    parser = argparse.ArgumentParser(description="Bitext data analyst agent")
    parser.add_argument(
        "--session",
        default="demo",
        help="Conversation thread id for checkpoint memory",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify dataset load (no LLM chat)",
    )
    args = parser.parse_args()

    if args.check:
        raise SystemExit(cmd_check())
    if cmd_check() != 0:
        raise SystemExit(1)
    raise SystemExit(cmd_chat(args.session))


if __name__ == "__main__":
    main()
