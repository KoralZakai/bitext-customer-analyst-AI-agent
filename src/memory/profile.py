"""Persistent per-session user profile (Task 2b)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

PROFILES_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "profiles"

_EMPTY_PROFILE: dict = {
    "name": None,
    "interests": [],
    "preferences": [],
    "notes": "",
}

_UPDATE_SYSTEM = """You maintain a concise user profile from conversation excerpts.
Given the current profile JSON and the latest exchange, return an UPDATED profile JSON only.
Rules:
- Extract the user's name if they mention it.
- Add topics/intents they ask about to interests (e.g. "REFUND", "CANCEL_ORDER").
- Note stated preferences (e.g. "prefers examples over counts").
- Keep interests and preferences as short string lists (max 10 items each).
- notes is a single free-text sentence of anything else worth remembering.
- Return ONLY valid JSON matching the schema. No explanation.
Schema: {"name": str|null, "interests": [str], "preferences": [str], "notes": str}"""


def load_profile(session_id: str) -> dict:
    """Load the user profile for a session, returning an empty profile if none exists."""
    path = PROFILES_DIR / f"{session_id}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return dict(_EMPTY_PROFILE)


def save_profile(session_id: str, profile: dict) -> None:
    """Persist a user profile to disk."""
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{session_id}.json"
    path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")


def profile_to_context(profile: dict) -> str:
    """Convert a profile dict to a human-readable context string for the system prompt."""
    has_data = profile.get("name") or profile.get("interests") or profile.get("preferences") or profile.get("notes")
    if not has_data:
        return ""
    lines = ["Known facts about this user:"]
    if profile.get("name"):
        lines.append(f"  Name: {profile['name']}")
    if profile.get("interests"):
        lines.append(f"  Topics of interest: {', '.join(profile['interests'])}")
    if profile.get("preferences"):
        lines.append(f"  Preferences: {', '.join(profile['preferences'])}")
    if profile.get("notes"):
        lines.append(f"  Notes: {profile['notes']}")
    return "\n".join(lines)


def update_profile(session_id: str, user_text: str, agent_reply: str, llm) -> dict:
    """
    Use the LLM to extract facts from the latest exchange and merge into the profile.

    Uses the router-class model (cheap, fast) since this is a simple extraction task.
    Silently falls back to the existing profile if the LLM returns malformed JSON.
    """
    profile = load_profile(session_id)
    exchange = f"User: {user_text}\nAgent: {agent_reply[:500]}"
    prompt = (
        f"Current profile: {json.dumps(profile)}\n\n"
        f"Latest exchange:\n{exchange}"
    )
    try:
        msg = llm.invoke(
            [SystemMessage(content=_UPDATE_SYSTEM), HumanMessage(content=prompt)]
        )
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            updated = json.loads(match.group())
            # Validate expected keys exist before saving
            if "interests" in updated and "preferences" in updated:
                save_profile(session_id, updated)
                return updated
    except Exception:
        pass
    return profile
