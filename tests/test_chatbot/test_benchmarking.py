import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm_orchestrator.utils.benchmarking import (
    DEFAULT_PORTFOLIO_PROMPTS,
    baseline_matches,
    extract_numeric_value,
    run_benchmark_case,
    summarize_benchmark_results,
)
from llm_orchestrator.utils.quantitative_tools import execute_quantitative_tools


class StubAgent:
    def is_available(self) -> bool:
        return False


def test_extract_numeric_value_from_text() -> None:
    assert extract_numeric_value("The correlation is 0.95") == 0.95
    assert extract_numeric_value("No number here") is None


def test_baseline_match_ignores_numbers_echoed_from_prompt() -> None:
    # Regression: "6-month" in both prompt and answer used to count as a correct 6-month return.
    prompt = "What is the 6-month return of AMZN?"
    assert baseline_matches(-2.65, "Over 6 months AMZN returned about 23%.", prompt) == 0.0
    assert baseline_matches(-2.65, "Over 6 months AMZN returned -2.65%.", prompt) == 1.0
    assert baseline_matches(-2.65, None, prompt) is None


def test_run_benchmark_case_reports_tool_metrics() -> None:
    result = run_benchmark_case(prompt="What is the correlation between gold and silver?", response_agent=StubAgent())

    assert result["prompt"] == "What is the correlation between gold and silver?"
    assert result["tool_latency_ms"] >= 0
    assert result["token_savings_pct"] is None or result["token_savings_pct"] >= 0


def test_default_portfolio_prompts_cover_core_finance_tasks() -> None:
    assert len(DEFAULT_PORTFOLIO_PROMPTS) >= 6
    labels = {item["id"] for item in DEFAULT_PORTFOLIO_PROMPTS}
    assert {"correlation", "sharpe_ratio", "var", "drawdown", "rsi", "classification"}.issubset(labels)


def test_execute_quantitative_tools_handles_portfolio_classification_prompt() -> None:
    holdings = [
        {"symbol": "IDEANSE", "quantity": 1, "average_price": 13, "last_price": 15},
        {"symbol": "YESBANKNSE", "quantity": 1, "average_price": 23, "last_price": 23},
        {"symbol": "SUZLONNSE", "quantity": 1, "average_price": 53, "last_price": 45},
    ]

    results = execute_quantitative_tools(
        "Calculate invested value, current value, profit or loss, and percentage profit or loss for each holding. "
        "Classify the portfolio as Buy, Hold, Trim, or Exit using only the supplied arithmetic.",
        holdings=holdings,
    )

    assert results
    assert results[0]["tool"] == "portfolio_classification"
    assert results[0]["status"] == "ok"
    assert results[0]["result"]["total_invested"] == 89.0
    assert results[0]["result"]["total_current_value"] == 83.0
    assert results[0]["result"]["total_pnl"] == -6.0
    assert results[0]["result"]["total_pnl_pct"] == -6.74


def test_summarize_benchmark_results_aggregates_by_metric() -> None:
    runs = [
        {"numeric_accuracy": 100.0, "field_coverage": 100.0, "unsupported_claims": 0, "latency_ms": 10.0, "total_tokens": 100},
        {"numeric_accuracy": 50.0, "field_coverage": 80.0, "unsupported_claims": 1, "latency_ms": 20.0, "total_tokens": 200},
    ]

    summary = summarize_benchmark_results(runs)

    assert summary["mean_numeric_accuracy"] == 75.0
    assert summary["mean_field_coverage"] == 90.0
    assert summary["unsupported_claims_total"] == 1
    assert summary["mean_latency_ms"] == 15.0
    assert summary["mean_total_tokens"] == 150.0
