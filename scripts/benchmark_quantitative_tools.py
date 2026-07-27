from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from llm_orchestrator.agents.response_agent import ResponseAgent
except Exception:  # pragma: no cover - optional dependency fallback
    ResponseAgent = None

from llm_orchestrator.utils.benchmarking import format_benchmark_report, run_benchmark_suite


if __name__ == "__main__":
    agent = ResponseAgent() if os.getenv("GROQ_API_KEY") and ResponseAgent is not None else None
    cases = [
        {"prompt": "What is the correlation between gold and silver?"},
        {"prompt": "What is the beta of AAPL versus NIFTY?"},
        {"prompt": "What is the Sharpe ratio of TSLA?"},
        {"prompt": "What is the annualized return of MSFT over the last year?"},
        {"prompt": "What is the 1-month momentum of NVDA?"},
        {"prompt": "What is the 6-month return of AMZN?"},
        {"prompt": "What is the 90-day value at risk of AAPL?"},
        {"prompt": "What is the Sortino ratio of GOOG?"},
        {"prompt": "What is the current drawdown for NFLX?"},
        {"prompt": "What is the 50-day moving average of META?"},
        {"prompt": "What is the 200-day moving average of IBM?"},
        {"prompt": "What is the RSI of MSFT?"},
        {"prompt": "What is the CAGR of AAPL over the last 5 years?"},
        {"prompt": "What is the downside deviation of INTC?"},
        {"prompt": "What is the total return of DIS over the past year?"},
        {"prompt": "What is the mean daily return of TSLA?"},
    ]
    results = run_benchmark_suite(cases, response_agent=agent)
    print(format_benchmark_report(results))
