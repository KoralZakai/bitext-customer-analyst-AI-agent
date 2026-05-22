"""Nebius Token Factory LLM clients."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.config import AGENT_MODEL, NEBIUS_API_KEY, NEBIUS_BASE_URL, ROUTER_MODEL


def _client(model: str, temperature: float = 0.0) -> ChatOpenAI:
    """
    Name: _client
    Input: model — Nebius model id; temperature — sampling temperature
    Output: ChatOpenAI — OpenAI-compatible client pointed at Nebius base URL
    Purpose: Shared factory for router and agent LLM instances.
    """
    return ChatOpenAI(
        model=model,
        api_key=NEBIUS_API_KEY,
        base_url=NEBIUS_BASE_URL,
        temperature=temperature,
    )


def router_llm() -> ChatOpenAI:
    """
    Name: router_llm
    Input: None (uses ROUTER_MODEL from config)
    Output: ChatOpenAI — low-temperature router model
    Purpose: Classify queries as structured, unstructured, or out-of-scope.
    """
    return _client(ROUTER_MODEL, temperature=0.0)


def agent_llm() -> ChatOpenAI:
    """
    Name: agent_llm
    Input: None (uses AGENT_MODEL from config)
    Output: ChatOpenAI — ReAct agent model for tool use and answers
    Purpose: Plan tool calls and summarize results for the user.
    """
    return _client(AGENT_MODEL, temperature=0.1)
