"""
Generate synthetic multi-turn conversations from single Q&A pairs.

WHY: The Bitext dataset contains only single-turn exchanges (one question, one answer).
Real customer service involves 3–5 turns: the customer asks, the agent responds, the
customer follows up, the agent resolves. Generating synthetic conversations enriches
the data for analysis — e.g. 'how do agents typically handle escalating refund requests?'
becomes answerable with realistic multi-turn examples.

Writes data/processed/bitext_conversations.jsonl — one JSON object per conversation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_PROCESSED_PATH, NEBIUS_API_KEY
from src.data.loader import dataset_ready, load_real_dataframe

OUTPUT_PATH = DATA_PROCESSED_PATH.parent / "bitext_conversations.jsonl"

_CONV_SYSTEM = """Expand a single customer-support Q&A pair into a realistic 3–5 turn conversation.

Rules:
- Keep the same intent and resolution as the original.
- The customer starts the conversation (turn 1).
- The agent's first response may ask a clarifying question or request an order ID.
- The customer follows up with more detail or a related question.
- The conversation ends with the agent resolving the issue.
- Each turn is short (1–3 sentences).
- Fill any {{placeholder}} tokens with realistic fake values.
- Return ONLY a JSON array of objects: [{"role": "customer"|"agent", "text": "..."}]
  No markdown, no explanation — raw JSON only."""


def _generate_conversation(
    instruction: str, response: str, intent: str, category: str, llm
) -> list[dict] | None:
    """Call the LLM to expand one Q&A pair into a multi-turn conversation."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        f"Intent: {intent} | Category: {category}\n"
        f"Original customer question: {instruction}\n"
        f"Original agent answer: {response}"
    )
    msg = llm.invoke(
        [SystemMessage(content=_CONV_SYSTEM), HumanMessage(content=prompt)]
    )
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    start = text.find("[")
    end = text.rfind("]") + 1
    if start >= 0 and end > start:
        try:
            turns = json.loads(text[start:end])
            if isinstance(turns, list) and len(turns) >= 3:
                return turns
        except json.JSONDecodeError:
            pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic multi-turn conversations")
    parser.add_argument(
        "--per-intent", type=int, default=3,
        help="Conversations to generate per intent (default: 3)",
    )
    parser.add_argument(
        "--max-intents", type=int, default=10,
        help="Max intents to process — caps API cost (default: 10)",
    )
    args = parser.parse_args()

    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("Set NEBIUS_API_KEY in .env", file=sys.stderr)
        raise SystemExit(1)
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        raise SystemExit(1)

    from src.agent.llm import agent_llm
    llm = agent_llm()

    df = load_real_dataframe()
    intents = df["intent"].astype(str).unique()[: args.max_intents]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total = 0

    with OUTPUT_PATH.open("w", encoding="utf-8") as fout:
        for intent in intents:
            subset = df[df["intent"].astype(str) == intent]
            samples = subset.sample(min(args.per_intent, len(subset)), random_state=42)

            for _, row in samples.iterrows():
                try:
                    turns = _generate_conversation(
                        instruction=str(row["instruction"]),
                        response=str(row["response"]),
                        intent=intent,
                        category=str(row["category"]),
                        llm=llm,
                    )
                except Exception as e:
                    print(f"  Skip {intent}: {e}", file=sys.stderr)
                    continue

                if turns:
                    record = {
                        "intent": intent,
                        "category": str(row["category"]),
                        "turns": turns,
                        "source": "synthetic_conversation",
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total += 1
                    print(f"  [{intent}] generated {len(turns)}-turn conversation")

    print(f"\nDone. Wrote {total} conversations to {OUTPUT_PATH}")
    print("Load with: pd.read_json('data/processed/bitext_conversations.jsonl', lines=True)")


if __name__ == "__main__":
    main()
