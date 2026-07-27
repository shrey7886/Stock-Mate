import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm_orchestrator.utils.benchmarking import extract_numeric_value, run_benchmark_case


class StubAgent:
    def is_available(self) -> bool:
        return False


def test_extract_numeric_value_from_text() -> None:
    assert extract_numeric_value("The correlation is 0.95") == 0.95
    assert extract_numeric_value("No number here") is None


def test_run_benchmark_case_reports_tool_metrics() -> None:
    result = run_benchmark_case(prompt="What is the correlation between gold and silver?", response_agent=StubAgent())

    assert result["prompt"] == "What is the correlation between gold and silver?"
    assert result["tool_latency_ms"] >= 0
    assert result["token_savings_pct"] >= 0
