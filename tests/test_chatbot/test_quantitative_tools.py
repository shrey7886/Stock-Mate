import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from llm_orchestrator.utils.quantitative_tools import (
    _extract_symbols,
    build_quantitative_answer,
    calculate_correlation,
    select_quantitative_tools,
)


def test_select_quantitative_tools_for_correlation_request() -> None:
    tools = select_quantitative_tools("What is the correlation between gold and silver?")
    assert any(tool["name"] == "correlation" for tool in tools)


def test_calculate_correlation_on_synthetic_prices(monkeypatch) -> None:
    def fake_fetches(symbols: list[str], **_: object) -> dict[str, pd.DataFrame]:
        return {
            "AAPL": pd.DataFrame({"AAPL": [100, 102, 104, 106, 108]}),
            "MSFT": pd.DataFrame({"MSFT": [200, 204, 208, 212, 216]}),
        }

    monkeypatch.setattr("llm_orchestrator.utils.quantitative_tools._fetch_price_histories", fake_fetches)

    result = calculate_correlation("AAPL", "MSFT")

    assert result["tool"] == "correlation"
    assert result["correlation"] > 0.8
    assert result["observations"] >= 3


def test_fetch_price_histories_batches_downloads(monkeypatch) -> None:
    call_log: list[tuple] = []

    def fake_download(tickers, **kwargs):
        call_log.append((tickers, kwargs))
        arrays = [
            ["Close", "Close"],
            ["AAPL", "^NSEI"],
        ]
        columns = pd.MultiIndex.from_arrays(arrays)
        return pd.DataFrame(
            [[100, 110], [101, 111], [102, 112], [103, 113], [104, 114]],
            columns=columns,
        )

    monkeypatch.setattr("llm_orchestrator.utils.quantitative_tools.yf.download", fake_download)
    from llm_orchestrator.utils.quantitative_tools import _PRICE_HISTORY_CACHE, _fetch_price_histories

    _PRICE_HISTORY_CACHE.clear()
    prices = _fetch_price_histories(["AAPL", "^NSEI"])

    assert len(call_log) == 1
    assert "AAPL" in prices and "^NSEI" in prices
    assert prices["AAPL"].shape[0] == 5
    assert prices["^NSEI"].shape[0] == 5


def test_fetch_price_history_uses_cache(monkeypatch) -> None:
    def fake_download(tickers, **kwargs):
        raise AssertionError("Download should not be called when cached")

    monkeypatch.setattr("llm_orchestrator.utils.quantitative_tools.yf.download", fake_download)
    from llm_orchestrator.utils.quantitative_tools import _PRICE_HISTORY_CACHE, _fetch_price_history

    _PRICE_HISTORY_CACHE.clear()
    sample = pd.DataFrame({"AAPL": [100, 101, 102]})
    _PRICE_HISTORY_CACHE[("AAPL", "2y", "1d")] = sample

    result = _fetch_price_history("AAPL")
    assert result.equals(sample)


def test_extract_symbols_ignores_metric_keywords() -> None:
    symbols = _extract_symbols("What is the beta of AAPL versus NIFTY?")

    assert "AAPL" in symbols
    assert "^NSEI" in symbols
    assert "BETA" not in symbols


def test_build_quantitative_answer_formats_correlation_result() -> None:
    tool_results = [{
        "tool": "correlation",
        "status": "ok",
        "result": {
            "correlation": 0.95,
            "symbol_a": "gold",
            "symbol_b": "silver",
            "interpretation": "Strong positive relationship",
        },
    }]

    answer = build_quantitative_answer("correlation between gold and silver", tool_results)

    assert answer is not None
    assert "0.9500" in answer
