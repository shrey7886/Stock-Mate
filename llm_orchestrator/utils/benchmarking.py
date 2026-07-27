from __future__ import annotations

import math
import re
import time
from statistics import mean
from typing import Any

from llm_orchestrator.utils.quantitative_tools import build_quantitative_answer, execute_quantitative_tools


def extract_numeric_value(text: str | None) -> float | None:
    if not text:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def run_benchmark_case(
    *,
    prompt: str,
    user_id: str = "benchmark",
    portfolio_summary: dict | None = None,
    response_agent: Any | None = None,
) -> dict[str, Any]:
    portfolio_summary = portfolio_summary or {
        "account_id": None,
        "holdings": [],
        "total_invested": 0.0,
        "total_current_value": 0.0,
        "total_pnl": 0.0,
        "total_pnl_pct": 0.0,
    }

    tool_start = time.perf_counter()
    tool_results = execute_quantitative_tools(prompt, holdings=portfolio_summary.get("holdings") or [])
    tool_latency_ms = round((time.perf_counter() - tool_start) * 1000.0, 2)

    tool_answer = build_quantitative_answer(prompt, tool_results) or ""
    tool_numeric = extract_numeric_value(tool_answer)
    tool_name = tool_results[0].get("tool") if tool_results else "unknown"

    baseline_reply = None
    baseline_latency_ms = None
    baseline_tokens = None
    baseline_numeric = None

    if response_agent is not None and getattr(response_agent, "is_available", lambda: False)():
        baseline_start = time.perf_counter()
        baseline_reply = response_agent.reply(
            user_message=prompt,
            portfolio_context={"portfolio": portfolio_summary, "rag_context": []},
            conversation_history=[],
            response_mode="quick",
        )
        baseline_latency_ms = round((time.perf_counter() - baseline_start) * 1000.0, 2)
        baseline_tokens = baseline_reply.token_usage or {}
        baseline_numeric = extract_numeric_value(baseline_reply.answer)

    baseline_total_tokens = None
    if baseline_tokens:
        baseline_total_tokens = int(baseline_tokens.get("total_tokens") or 0)

    tool_total_tokens = 0
    if tool_results:
        tool_total_tokens = 0

    if baseline_total_tokens is None:
        token_savings_pct = None
    elif baseline_total_tokens > 0:
        token_savings_pct = round((1 - (tool_total_tokens / baseline_total_tokens)) * 100.0, 2)
    else:
        token_savings_pct = 0.0

    latency_delta_ms = None if baseline_latency_ms is None else round(tool_latency_ms - baseline_latency_ms, 2)

    if tool_numeric is not None and baseline_numeric is not None and not math.isnan(tool_numeric):
        accuracy_proxy = 1.0 if abs(tool_numeric - baseline_numeric) < 1e-6 else 0.0
    else:
        accuracy_proxy = None

    return {
        "prompt": prompt,
        "tool_name": tool_name,
        "tool_answer": tool_answer,
        "tool_numeric": tool_numeric,
        "baseline_answer": getattr(baseline_reply, "answer", None) if baseline_reply else None,
        "baseline_numeric": baseline_numeric,
        "tool_latency_ms": tool_latency_ms,
        "baseline_latency_ms": baseline_latency_ms,
        "latency_delta_ms": latency_delta_ms,
        "tool_tokens": tool_total_tokens,
        "baseline_tokens": baseline_total_tokens,
        "token_savings_pct": token_savings_pct,
        "accuracy_proxy": accuracy_proxy,
        "tool_results": tool_results,
    }


def run_benchmark_suite(cases: list[dict[str, Any]], response_agent: Any | None = None) -> list[dict[str, Any]]:
    return [run_benchmark_case(prompt=case["prompt"], portfolio_summary=case.get("portfolio_summary"), response_agent=response_agent) for case in cases]


def format_benchmark_report(results: list[dict[str, Any]]) -> str:
    lines = ["Quantitative benchmark report", "============================", ""]
    baseline_available = any(item.get("baseline_latency_ms") is not None for item in results)
    for item in results:
        lines.append(f"Prompt: {item['prompt']}")
        lines.append(f"  Tool: {item.get('tool_name', 'unknown')}")
        lines.append(f"  Tool answer: {item['tool_answer'] or 'n/a'}")
        if item.get("baseline_answer"):
            lines.append(f"  Baseline answer: {item['baseline_answer']}")
        elif baseline_available:
            lines.append("  Baseline answer: n/a")
        lines.append(f"  Tool latency: {item['tool_latency_ms']} ms")
        if item.get("baseline_latency_ms") is not None:
            lines.append(f"  Baseline latency: {item['baseline_latency_ms']} ms")
            lines.append(f"  Latency delta: {item['latency_delta_ms']} ms")
        elif baseline_available:
            lines.append("  Baseline latency: n/a")
        lines.append(f"  Tool tokens: {item['tool_tokens']}")
        if item.get("baseline_tokens") is not None:
            lines.append(f"  Baseline tokens: {item['baseline_tokens']}")
        elif baseline_available:
            lines.append("  Baseline tokens: n/a")
        if item.get("token_savings_pct") is not None:
            lines.append(f"  Token savings: {item['token_savings_pct']}%")
        elif baseline_available:
            lines.append("  Token savings: n/a")
        if item.get("accuracy_proxy") is not None:
            lines.append(f"  Accuracy proxy: {item['accuracy_proxy'] * 100:.1f}%")
        elif baseline_available:
            lines.append("  Accuracy proxy: n/a")
        lines.append("")

    lines.append("Summary")
    lines.append("-------")
    lines.append(f"  Average tool latency: {round(mean(item['tool_latency_ms'] for item in results), 2)} ms")
    lines.append(f"  Average tool tokens: {round(mean(item['tool_tokens'] for item in results), 2)}")
    if baseline_available:
        available_baseline = [item for item in results if item.get("baseline_latency_ms") is not None]
        if available_baseline:
            lines.append(f"  Average baseline latency: {round(mean(item['baseline_latency_ms'] for item in available_baseline), 2)} ms")
            lines.append(f"  Average latency delta: {round(mean(item['latency_delta_ms'] for item in available_baseline), 2)} ms")
        baseline_tokens = [item['baseline_tokens'] for item in available_baseline if item.get('baseline_tokens') is not None]
        if baseline_tokens:
            lines.append(f"  Average baseline tokens: {round(mean(baseline_tokens), 2)}")
        token_savings = [item['token_savings_pct'] for item in available_baseline if item.get('token_savings_pct') is not None]
        if token_savings:
            lines.append(f"  Average token savings: {round(mean(token_savings), 2)}%")
        accuracy_values = [item['accuracy_proxy'] for item in available_baseline if item.get('accuracy_proxy') is not None]
        if accuracy_values:
            lines.append(f"  Average accuracy proxy: {round(mean(accuracy_values) * 100, 2)}%")
    else:
        lines.append("  Baseline comparison unavailable. Set GROQ_API_KEY to enable baseline data.")

    return "\n".join(lines)
