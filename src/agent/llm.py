"""Nebius Token Factory LLM clients."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import AGENT_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL, ROUTER_MODEL


def _client(model: str, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        temperature=temperature,
    )


def router_llm() -> ChatOpenAI:
    # Model: google/gemma-3-27b-it (default, set via ROUTER_MODEL)
    #
    # WHY this model for routing:
    #   The router only needs to output a small JSON blob {route, reason, entities}.
    #   Gemma 3 27B is fast and cheap for this narrow classification task and reliably
    #   follows structured-output instructions at temperature=0.
    #
    # PROS:  Low latency (~2-3x faster than 70B), cost-efficient, deterministic at t=0,
    #        strong JSON instruction-following, good enough accuracy for 3-class routing.
    # CONS:  Weaker at nuanced multi-step reasoning; if the router also had to do complex
    #        entity extraction it might miss edge cases that a 70B model would catch.
    #        Smaller context window than Llama 70B.
    return _client(ROUTER_MODEL, temperature=0.0)


def agent_llm() -> ChatOpenAI:
    # Model: meta-llama/Llama-3.3-70B-Instruct (default, set via AGENT_MODEL)
    #
    # WHY this model for the ReAct agent:
    #   The agent must reason about which tool to call, interpret tool outputs, chain
    #   multiple calls, and produce a coherent final answer — all tasks that benefit from
    #   a larger, more capable model.  Llama 3.3 70B has strong tool-use and reasoning
    #   benchmarks and is openly licensed.
    #
    # PROS:  Strong multi-step reasoning and tool-call accuracy, large context window
    #        (128k tokens) handles long tool outputs, good instruction following,
    #        open weights (no vendor lock-in beyond inference provider).
    # CONS:  Higher latency and cost than smaller models; overkill for simple lookups
    #        that the router already resolved; requires a capable inference host
    #        (Nebius Token Factory here — not suitable for a local laptop).
    return _client(AGENT_MODEL, temperature=0.1)
