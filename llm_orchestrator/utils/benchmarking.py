from __future__ import annotations

import math
import re
import time
from statistics import mean
from typing import Any

from llm_orchestrator.utils.quantitative_tools import build_quantitative_answer, execute_quantitative_tools


# Upstox holdings snapshot used throughout the research benchmark.
UPSTOX_HOLDINGS: list[dict[str, Any]] = [
    {"tradingsymbol": "IDEA", "exchange": "NSE", "quantity": 1, "average_price": 13, "last_price": 15},
    {"tradingsymbol": "YESBANK", "exchange": "NSE", "quantity": 1, "average_price": 23, "last_price": 23},
    {"tradingsymbol": "SUZLON", "exchange": "NSE", "quantity": 1, "average_price": 53, "last_price": 45},
]

# Primary numeric field of each tool result (the value the baseline is compared against).
PRIMARY_RESULT_KEYS = {
    "correlation": "correlation",
    "beta": "beta",
    "volatility": "volatility_pct",
    "drawdown": "max_drawdown_pct",
    "sharpe": "sharpe_ratio",
    "total_return": "total_return_pct",
    "cagr": "cagr_pct",
    "momentum": "momentum_pct",
    "var": "value_at_risk_pct",
    "downside_deviation": "downside_deviation_pct",
    "sortino": "sortino_ratio",
    "moving_average": "moving_average",
    "rsi": "rsi",
    "mean_daily_return": "mean_daily_return_pct",
    "concentration": "largest_weight_pct",
    "portfolio_classification": "total_pnl_pct",
}

DEFAULT_PORTFOLIO_PROMPTS: list[dict[str, Any]] = [
    {
        "id": "correlation",
        "prompt": "What is the correlation between gold and silver?",
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
    {
        "id": "sharpe_ratio",
        "prompt": "What is the Sharpe ratio for the portfolio?",
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
    {
        "id": "var",
        "prompt": "What is the 90-day Value at Risk for this portfolio?",
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
    {
        "id": "drawdown",
        "prompt": "What is the maximum drawdown for this portfolio?",
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
    {
        "id": "rsi",
        "prompt": "What is the RSI for the portfolio?",
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
    {
        "id": "classification",
        "prompt": (
            "Calculate invested value, current value, profit or loss, and percentage profit or loss "
            "for each holding. Classify the portfolio as Buy, Hold, Trim, or Exit using only the supplied arithmetic."
        ),
        "portfolio_summary": {"holdings": UPSTOX_HOLDINGS},
    },
]


def extract_numeric_values(text: str | None) -> list[float]:
    return [float(m.replace(",", "")) for m in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", (text or "").replace("−", "-"))]


def baseline_matches(tool_value: float | None, baseline_text: str | None, prompt: str, rel_tol: float = 0.01) -> float | None:
    """1.0 if the baseline states the tool value (within rel_tol), ignoring numbers echoed from the prompt."""
    if tool_value is None or not baseline_text:
        return None
    prompt_numbers = set(extract_numeric_values(prompt))
    candidates = [n for n in extract_numeric_values(baseline_text) if n not in prompt_numbers]
    return 1.0 if any(math.isclose(n, tool_value, rel_tol=rel_tol, abs_tol=0.005) for n in candidates) else 0.0


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
    tool_name = tool_results[0].get("tool") if tool_results else "unknown"
    tool_result = (tool_results[0].get("result") or {}) if tool_results and tool_results[0].get("status") == "ok" else {}
    tool_numeric = tool_result.get(PRIMARY_RESULT_KEYS.get(tool_name, ""))

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

    tool_total_tokens = 0  # deterministic path makes no LLM call

    if baseline_total_tokens is None:
        token_savings_pct = None
    elif baseline_total_tokens > 0:
        token_savings_pct = round((1 - (tool_total_tokens / baseline_total_tokens)) * 100.0, 2)
    else:
        token_savings_pct = 0.0

    latency_delta_ms = None if baseline_latency_ms is None else round(tool_latency_ms - baseline_latency_ms, 2)

    accuracy_proxy = baseline_matches(tool_numeric, getattr(baseline_reply, "answer", None), prompt)

    return {
        "prompt": prompt,
        "tool_name": tool_name,
        "tool_answer": tool_answer,
        "tool_numeric": tool_numeric,
        "tool_status": tool_results[0].get("status") if tool_results else "no_tool",
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


def summarize_benchmark_results(results: list[dict[str, Any]]) -> dict[str, float | int]:
    numeric_accuracy = [float(item.get("numeric_accuracy", 0.0)) for item in results if "numeric_accuracy" in item]
    field_coverage = [float(item.get("field_coverage", 0.0)) for item in results if "field_coverage" in item]
    unsupported_claims = [int(item.get("unsupported_claims", 0)) for item in results if "unsupported_claims" in item]
    latency = [float(item.get("latency_ms", 0.0)) for item in results if "latency_ms" in item]
    total_tokens = [float(item.get("total_tokens", 0.0)) for item in results if "total_tokens" in item]

    return {
        "mean_numeric_accuracy": round(mean(numeric_accuracy), 2) if numeric_accuracy else 0.0,
        "mean_field_coverage": round(mean(field_coverage), 2) if field_coverage else 0.0,
        "unsupported_claims_total": sum(unsupported_claims),
        "mean_latency_ms": round(mean(latency), 2) if latency else 0.0,
        "mean_total_tokens": round(mean(total_tokens), 2) if total_tokens else 0.0,
    }


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
