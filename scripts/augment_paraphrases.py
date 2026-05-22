"""
Generate synthetic instruction paraphrases per intent (optional, uses Nebius API).
Writes data/processed/bitext_augmented.parquet — used only by search_instructions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DATA_AUGMENTED_PATH, NEBIUS_API_KEY
from src.data.loader import dataset_ready, load_real_dataframe


def generate_paraphrases(instruction: str, n: int = 2) -> list[str]:
    """
    Name: generate_paraphrases
    Input: instruction — example customer question; n — number of paraphrases
    Output: list[str] — generated paraphrase strings (may be empty on parse failure)
    Purpose: Create synthetic search rows without changing real row counts.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.agent.llm import agent_llm

    llm = agent_llm()
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the customer support question as short paraphrases. "
                    f"Return JSON: {{\"paraphrases\": [\"...\"]}} with exactly {n} items."
                )
            ),
            HumanMessage(content=instruction),
        ]
    )
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        data = json.loads(text[start:end])
        return list(data.get("paraphrases", []))[:n]
    return []


def main() -> None:
    """
    Name: main
    Input: CLI args --per-intent, --max-intents (see argparse)
    Output: None (writes bitext_augmented.parquet)
    Purpose: Optional data improvement for search_instructions tool only.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-intent", type=int, default=2, help="Paraphrases per intent")
    parser.add_argument("--max-intents", type=int, default=5, help="Cap intents (API cost)")
    args = parser.parse_args()

    if not NEBIUS_API_KEY or NEBIUS_API_KEY == "your_nebius_api_key_here":
        print("Set NEBIUS_API_KEY in .env", file=sys.stderr)
        raise SystemExit(1)
    if not dataset_ready():
        print("Run: python -m src.data.preprocess", file=sys.stderr)
        raise SystemExit(1)

    df = load_real_dataframe()
    rows: list[dict] = []
    intents = df["intent"].astype(str).unique()[: args.max_intents]

    for intent in intents:
        sample = df[df["intent"].astype(str) == intent].iloc[0]
        for _ in range(args.per_intent):
            try:
                paraphrases = generate_paraphrases(str(sample["instruction"]), n=1)
            except Exception as e:
                print(f"Skip {intent}: {e}", file=sys.stderr)
                continue
            for p in paraphrases:
                rows.append(
                    {
                        "instruction": p,
                        "category": str(sample["category"]),
                        "intent": str(intent),
                        "response": str(sample["response"]),
                        "flags": str(sample["flags"]),
                        "source": "synthetic",
                    }
                )

    if not rows:
        print("No paraphrases generated.", file=sys.stderr)
        raise SystemExit(1)

    out = pd.DataFrame(rows)
    DATA_AUGMENTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATA_AUGMENTED_PATH, index=False)
    print(f"Wrote {len(out)} synthetic rows to {DATA_AUGMENTED_PATH}")


if __name__ == "__main__":
    main()
