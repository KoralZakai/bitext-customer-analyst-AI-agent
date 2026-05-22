"""List models enabled for your Nebius API key (copy IDs into .env)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


def main() -> None:
    import os

    from openai import OpenAI

    key = os.getenv("NEBIUS_API_KEY", "")
    if not key or key.startswith("your_"):
        print("Set NEBIUS_API_KEY in .env", file=sys.stderr)
        raise SystemExit(1)

    client = OpenAI(
        api_key=key,
        base_url=os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"),
    )
    models = sorted(m.id for m in client.models.list())
    print(f"Available models ({len(models)}):\n")
    for m in models:
        print(f"  {m}")
    print("\nSuggested .env:")
    router = next((m for m in models if "27b" in m.lower() or "30b" in m.lower()), models[0])
    agent = next((m for m in models if "70b" in m.lower() or "72b" in m.lower()), models[-1])
    print(f"ROUTER_MODEL={router}")
    print(f"AGENT_MODEL={agent}")


if __name__ == "__main__":
    main()
