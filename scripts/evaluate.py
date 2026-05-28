"""
EDD Evaluation Runner — scores the Bitext analyst agent against a golden test set.

Usage:
    python scripts/evaluate.py                    # all tests, no LLM judge
    python scripts/evaluate.py --judge            # add LLM-as-judge for unstructured queries
    python scripts/evaluate.py --max-cases 5      # run only first N cases
    python scripts/evaluate.py --out results.json # also save raw results to JSON

What is evaluated per test case:
    route_ok      — did the router classify the query correctly?
    tools_ok      — did the agent call at least one expected tool?
    answer_ok     — does the final answer contain all expected strings?
    no_leak_ok    — does the final answer NOT contain forbidden strings (OOS guard)?
    judge_score   — 0-9 LLM judge score for unstructured queries (--judge flag required)

Overall output:
    pass/fail per case, per-dimension accuracy, total score, failure details.
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Force UTF-8 output on Windows (cp1252 can't encode Unicode symbols)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

GOLDEN_SET_PATH = ROOT / "data" / "eval" / "golden_set.json"
RESULTS_DIR = ROOT / "data" / "eval"

# ─── ANSI colours (gracefully disabled on Windows if needed) ────────────────
try:
    import colorama
    colorama.init()
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
except ImportError:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

_JUDGE_SYSTEM = """You are an impartial evaluator of AI agent responses.

Given a question and the agent's response, score on three dimensions (each 0–3):
  relevance    — Does the response directly address the question?
  correctness  — Are any cited facts, numbers, or examples accurate and appropriate?
  helpfulness  — Is the response useful and sufficiently complete?

Return ONLY valid JSON: {"relevance": int, "correctness": int, "helpfulness": int, "reasoning": str}
No markdown, no explanation outside the JSON."""


def _run_case(case: dict, graph, use_verbose: bool = False) -> dict:
    """Run one test case through the agent and return raw result dict."""
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError
    from src.agent.graph import FALLBACK_MESSAGE, _extract_reasoning
    from src.memory.profile import profile_to_context, load_profile

    # Isolate each test in its own thread so there's no conversation bleed
    session_id = f"eval_{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": session_id}}

    profile_ctx = profile_to_context(load_profile(session_id))

    t0 = time.monotonic()
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=case["query"])], "profile_context": profile_ctx},
            config=config,
        )
        elapsed = time.monotonic() - t0
        steps, final_answer = _extract_reasoning(result.get("messages", []))
        actual_route = result.get("route", "unknown")
    except GraphRecursionError:
        elapsed = time.monotonic() - t0
        steps, final_answer = [], FALLBACK_MESSAGE
        actual_route = "unknown"
    except Exception as exc:
        elapsed = time.monotonic() - t0
        steps, final_answer = [], f"ERROR: {exc}"
        actual_route = "unknown"

    # Extract which tool names were actually called
    called_tools: list[str] = []
    for step in steps:
        if step.startswith("  ->"):
            # Format: "  -> tool_name({...})"
            tool_name = step.strip()[3:].split("(")[0].strip()
            if tool_name:
                called_tools.append(tool_name)

    return {
        "id": case["id"],
        "query": case["query"],
        "type": case["type"],
        "actual_route": actual_route,
        "called_tools": called_tools,
        "final_answer": final_answer,
        "reasoning_steps": steps,
        "elapsed_s": round(elapsed, 2),
    }


def _score_case(case: dict, raw: dict) -> dict:
    """Score a single case against its golden expectations."""
    scores: dict[str, bool | None] = {}

    # 1. Route check
    expected_route = case.get("expected_route")
    scores["route_ok"] = (
        raw["actual_route"] == expected_route if expected_route else None
    )

    # 2. Tool selection check (at least one expected tool was called)
    expected_tools: list[str] = case.get("expected_tool_calls", [])
    if expected_tools:
        scores["tools_ok"] = any(t in raw["called_tools"] for t in expected_tools)
    else:
        scores["tools_ok"] = None  # OOS — no tools expected

    # 3. Answer contains expected strings
    expected_contains: list[str] = case.get("expected_answer_contains", [])
    if expected_contains:
        answer_lower = raw["final_answer"].lower()
        scores["answer_ok"] = all(
            e.lower() in answer_lower for e in expected_contains
        )
    else:
        scores["answer_ok"] = None

    # 4. Must-not-contain check (OOS leak guard)
    must_not: list[str] = case.get("must_not_contain", [])
    if must_not:
        answer_lower = raw["final_answer"].lower()
        scores["no_leak_ok"] = not any(m.lower() in answer_lower for m in must_not)
    else:
        scores["no_leak_ok"] = None

    # Overall pass: all non-None checks must be True
    active = [v for v in scores.values() if v is not None]
    scores["passed"] = all(active) if active else True

    return scores


def _llm_judge(case: dict, raw: dict, llm) -> dict:
    """Use LLM to score an unstructured answer on relevance/correctness/helpfulness."""
    import re
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = f"Question: {case['query']}\n\nAgent response:\n{raw['final_answer'][:1500]}"
    try:
        msg = llm.invoke([SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=prompt)])
        text = msg.content if isinstance(msg.content, str) else str(msg.content)
        m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if m:
            data = json.loads(m.group())
            total = int(data.get("relevance", 0)) + int(data.get("correctness", 0)) + int(data.get("helpfulness", 0))
            return {
                "relevance": data.get("relevance"),
                "correctness": data.get("correctness"),
                "helpfulness": data.get("helpfulness"),
                "total": total,
                "reasoning": data.get("reasoning", ""),
            }
    except Exception as exc:
        return {"error": str(exc), "total": None}
    return {"total": None}


def _fmt_bool(val: bool | None) -> str:
    if val is None:
        return f"{YELLOW}N/A{RESET}"
    return f"{GREEN}PASS{RESET}" if val else f"{RED}FAIL{RESET}"


def run_evaluation(
    max_cases: int | None = None,
    use_judge: bool = False,
    out_path: Path | None = None,
) -> int:
    """Main evaluation loop. Returns exit code (0 = all passed)."""
    from src.config import CHECKPOINT_DB_PATH
    from src.agent.graph import build_graph
    from src.agent.llm import router_llm
    from src.data.loader import dataset_ready
    from langgraph.checkpoint.sqlite import SqliteSaver

    if not dataset_ready():
        print(f"{RED}Dataset not ready. Run: python -m src.data.preprocess{RESET}")
        return 1

    # Load golden set
    cases: list[dict] = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))
    if max_cases:
        cases = cases[:max_cases]

    print(f"\n{BOLD}{'-' * 60}{RESET}")
    print(f"{BOLD}  Bitext Agent -- EDD Evaluation  ({len(cases)} test cases){RESET}")
    print(f"{BOLD}{'-' * 60}{RESET}\n")

    # Build graph with a fresh in-memory checkpoint so eval is isolated
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    graph = build_graph(checkpointer=SqliteSaver(conn))

    judge_llm = router_llm() if use_judge else None

    all_results: list[dict] = []
    passed = 0
    failed_ids: list[str] = []

    for i, case in enumerate(cases, 1):
        print(f"[{i:2}/{len(cases)}] {CYAN}{case['id']}{RESET}  {case['query'][:60]}")

        raw = _run_case(case, graph)
        scores = _score_case(case, raw)

        judge: dict = {}
        if use_judge and case["type"] == "unstructured":
            judge = _llm_judge(case, raw, judge_llm)

        result_row = {**raw, "scores": scores, "judge": judge, "notes": case.get("notes", "")}
        all_results.append(result_row)

        # Print per-case summary
        status_icon = f"{GREEN}✓{RESET}" if scores["passed"] else f"{RED}✗{RESET}"
        route_str = _fmt_bool(scores["route_ok"])
        tools_str = _fmt_bool(scores["tools_ok"])
        answer_str = _fmt_bool(scores["answer_ok"])
        leak_str = _fmt_bool(scores["no_leak_ok"])
        judge_str = (
            f"  judge={judge.get('total', '?')}/9" if judge else ""
        )
        print(
            f"       {status_icon}  route={route_str}  tools={tools_str}  "
            f"answer={answer_str}  no_leak={leak_str}{judge_str}"
            f"  ({raw['elapsed_s']}s)"
        )

        if not scores["passed"]:
            failed_ids.append(case["id"])
            # Show first 200 chars of answer to help debug
            snippet = raw["final_answer"][:200].replace("\n", " ")
            print(f"         Answer snippet: {YELLOW}{snippet}{RESET}")
        else:
            passed += 1

        print()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"{BOLD}{'-' * 60}{RESET}")
    print(f"{BOLD}  Results: {passed}/{len(cases)} passed{RESET}")

    # Per-dimension accuracy
    def _dim_accuracy(key: str) -> str:
        vals = [r["scores"][key] for r in all_results if r["scores"][key] is not None]
        if not vals:
            return "N/A"
        pct = sum(vals) / len(vals) * 100
        colour = GREEN if pct >= 80 else (YELLOW if pct >= 50 else RED)
        return f"{colour}{pct:.0f}%{RESET} ({sum(vals)}/{len(vals)})"

    print(f"  Router accuracy : {_dim_accuracy('route_ok')}")
    print(f"  Tool selection  : {_dim_accuracy('tools_ok')}")
    print(f"  Answer content  : {_dim_accuracy('answer_ok')}")
    print(f"  OOS guard       : {_dim_accuracy('no_leak_ok')}")

    if use_judge:
        judge_scores = [r["judge"].get("total") for r in all_results if r["judge"].get("total") is not None]
        if judge_scores:
            avg = sum(judge_scores) / len(judge_scores)
            print(f"  LLM judge avg   : {avg:.1f}/9 ({len(judge_scores)} unstructured cases)")

    avg_latency = sum(r["elapsed_s"] for r in all_results) / len(all_results)
    print(f"  Avg latency     : {avg_latency:.1f}s per query")

    if failed_ids:
        print(f"\n  {RED}Failed cases: {', '.join(failed_ids)}{RESET}")
    else:
        print(f"\n  {GREEN}All cases passed!{RESET}")

    print(f"{BOLD}{'-' * 60}{RESET}\n")
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(all_results, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"Results saved to {out_path}\n")

    conn.close()
    return 0 if not failed_ids else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Bitext analyst agent (EDD)")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Enable LLM-as-judge scoring for unstructured queries (uses router model, costs tokens)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N test cases (useful for quick smoke tests)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        metavar="FILE",
        help="Save raw results JSON to this file path",
    )
    args = parser.parse_args()

    out_path = Path(args.out) if args.out else RESULTS_DIR / "last_run.json"
    raise SystemExit(
        run_evaluation(
            max_cases=args.max_cases,
            use_judge=args.judge,
            out_path=out_path,
        )
    )


if __name__ == "__main__":
    main()
