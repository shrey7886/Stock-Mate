from __future__ import annotations

import inspect
import math
import re
from typing import Any

try:
    import numpy as np
    import pandas as pd
except Exception:  # pragma: no cover - environment fallback
    np = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]

try:
    import yfinance as yf
except Exception:  # pragma: no cover - environment fallback
    yf = None  # type: ignore[assignment]


_SYMBOL_ALIASES = {
    "gold": "GC=F",
    "silver": "SI=F",
    "nifty": "^NSEI",
    "sensex": "^BSESN",
    "banknifty": "^NSEBANK",
    "reliance": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "sbi": "SBIN.NS",
    "icici": "ICICIBANK.NS",
    "tata": "TATASTEEL.NS",
    "mahindra": "M&M.NS",
    "tatasteel": "TATASTEEL.NS",
}

_STOPWORDS = {
    "what",
    "is",
    "the",
    "between",
    "and",
    "of",
    "my",
    "for",
    "a",
    "an",
    "to",
    "tell",
    "me",
    "show",
    "give",
    "please",
    "can",
    "you",
    "does",
    "how",
    "much",
    "value",
    "compare",
    "vs",
    "versus",
    "against",
    "with",
    "about",
    "latest",
    "current",
    "now",
    "over",
    "last",
    "beta",
    "correlation",
    "correlate",
    "correlated",
    "sharpe",
    "volatility",
    "variance",
    "drawdown",
    "risk",
    "standard",
    "deviation",
    "ratio",
    "annual",
    "annualized",
    "annualised",
    "cagr",
    "return",
    "returns",
    "momentum",
    "month",
    "months",
    "day",
    "days",
    "year",
    "years",
    "value",
    "at",
    "var",
    "sortino",
    "average",
    "daily",
    "mean",
    "rsi",
    "moving",
    "total",
    "downside",
    "weekly",
    "monthly",
    "past",
    "last",
    "current",
    "over",
    "period",
    "window",
    "since",
    "through",
    "before",
    "after",
    "into",
    "into",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "nse",
    "bse",
    "inr",
    "usd",
    "pnl",
    "etf",
}


def _normalize_symbol(token: str) -> str | None:
    cleaned = re.sub(r"[^a-zA-Z0-9.\-^=]", "", token or "")
    if not cleaned:
        return None
    key = cleaned.lower()
    if key in _SYMBOL_ALIASES:
        return _SYMBOL_ALIASES[key]
    if key.endswith(".ns") or key.startswith("^") or key.endswith("=f"):
        return cleaned.upper()
    return cleaned.upper()


def _holding_ticker(holding: dict) -> str | None:
    """Map a broker holding (Upstox/Zerodha schema) to its Yahoo ticker, e.g. IDEA on NSE -> IDEA.NS."""
    symbol = str(holding.get("tradingsymbol") or holding.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    if any(ch in symbol for ch in ".^="):
        return symbol
    return symbol + (".BO" if str(holding.get("exchange") or "").upper().startswith("BSE") else ".NS")


def _extract_symbols(message: str, holdings: list[dict] | None = None) -> list[str]:
    """Symbols explicitly named in the message. Plain words ("portfolio", "maximum") are never tickers."""
    held = {}
    for holding in holdings or []:
        ticker = _holding_ticker(holding)
        if ticker:
            held[ticker.split(".")[0].lower()] = ticker

    symbols: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.\-^=]*", message or ""):
        token = token.rstrip(".-")
        key = token.lower()
        if key in _STOPWORDS:
            continue
        if key in held:
            symbol = held[key]
        elif key in _SYMBOL_ALIASES or key.endswith((".ns", ".bo", "=f")):
            symbol = _normalize_symbol(token)
        elif token.isupper() and len(token) >= 2:  # ponytail: bare ALL-CAPS word = ticker (AAPL); lowercase names need an alias
            symbol = token
        else:
            continue
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return symbols


def portfolio_value_series(holdings: list[dict], *, period: str = "2y", interval: str = "1d") -> pd.Series:
    """Daily value of the current holdings held over the window: sum(quantity x close)."""
    quantities: dict[str, float] = {}
    for holding in holdings:
        ticker = _holding_ticker(holding)
        qty = float(holding.get("quantity") or 0.0)
        if ticker and qty:
            quantities[ticker] = quantities.get(ticker, 0.0) + qty
    if not quantities:
        raise ValueError("No holdings with a quantity to build a portfolio series")
    prices = _fetch_price_histories(list(quantities), period=period, interval=interval)
    closes = pd.concat([prices[t].iloc[:, 0].rename(t) for t in quantities], axis=1, join="inner").dropna()
    if len(closes) < 2:
        raise ValueError("Not enough overlapping price history for the portfolio")
    return (closes * pd.Series(quantities)).sum(axis=1)


def _series(symbol: str, period: str, interval: str, prices: pd.Series | None) -> pd.Series:
    return prices if prices is not None else _fetch_price_history(symbol, period=period, interval=interval).iloc[:, 0]


_PRICE_HISTORY_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}


def _fetch_price_histories(symbols: list[str], *, period: str = "2y", interval: str = "1d") -> dict[str, pd.DataFrame]:
    if yf is None or pd is None or np is None:
        raise RuntimeError("pandas/yfinance is not available")

    cached: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for symbol in symbols:
        key = (symbol, period, interval)
        if key in _PRICE_HISTORY_CACHE:
            cached[symbol] = _PRICE_HISTORY_CACHE[key]
        else:
            missing.append(symbol)

    if missing:
        tickers = missing if len(missing) > 1 else missing[0]
        history = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=False)
        if history.empty:
            raise ValueError(f"No price history returned for {', '.join(missing)}")

        if isinstance(history.columns, pd.MultiIndex):
            close = history.loc[:, ("Close", slice(None))]
            close.columns = [ticker for _, ticker in close.columns]
        elif len(missing) == 1:
            if "Close" not in history.columns:
                raise ValueError(f"No Close column returned for {missing[0]}")
            close = history[["Close"]].rename(columns={"Close": missing[0]})
        else:
            raise ValueError(f"Unexpected yfinance response shape for {missing}")

        close = close.dropna()
        for ticker in missing:
            if ticker not in close.columns:
                raise ValueError(f"No close price returned for {ticker}")
            series = close[[ticker]].copy()
            key = (ticker, period, interval)
            _PRICE_HISTORY_CACHE[key] = series
            cached[ticker] = series

    return cached


def _fetch_price_history(symbol: str, *, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    if yf is None or pd is None or np is None:
        raise RuntimeError("pandas/yfinance is not available")

    if not symbol:
        symbol = "^NSEI"
    key = (symbol, period, interval)
    if key in _PRICE_HISTORY_CACHE:
        return _PRICE_HISTORY_CACHE[key]

    result = _fetch_price_histories([symbol], period=period, interval=interval)
    return result[symbol]


def _daily_returns(series: pd.Series) -> pd.Series:
    return series.pct_change().dropna()


def calculate_correlation(symbol_a: str, symbol_b: str, *, period: str = "2y", interval: str = "1d") -> dict[str, Any]:
    if pd is None or np is None:
        raise RuntimeError("pandas/numpy is not available")

    prices = _fetch_price_histories([symbol_a, symbol_b], period=period, interval=interval)
    left = prices[symbol_a]
    right = prices[symbol_b]
    merged = left.join(right, how="inner")
    if merged.empty or len(merged) < 2:
        raise ValueError("Not enough data for a correlation estimate")

    left_returns = _daily_returns(merged.iloc[:, 0])
    right_returns = _daily_returns(merged.iloc[:, 1])
    aligned = pd.concat([left_returns, right_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        raise ValueError("Not enough return observations for correlation")

    corr = float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1])
    return {
        "tool": "correlation",
        "symbol_a": symbol_a,
        "symbol_b": symbol_b,
        "correlation": round(corr, 4),
        "observations": int(aligned.shape[0]),
        "interpretation": _interpret_correlation(corr),
    }


def calculate_beta(symbol: str, *, benchmark: str = "^NSEI", period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    if pd is None or np is None:
        raise RuntimeError("pandas/numpy is not available")

    bench = _fetch_price_history(benchmark, period=period, interval=interval).iloc[:, 0].rename("bench")
    stock = _series(symbol, period, interval, prices).rename("stock")
    merged = pd.concat([stock, bench], axis=1, join="inner").dropna()
    if merged.empty or len(merged) < 2:
        raise ValueError("Not enough data for a beta estimate")

    stock_returns = _daily_returns(merged.iloc[:, 0])
    bench_returns = _daily_returns(merged.iloc[:, 1])
    aligned = pd.concat([stock_returns, bench_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        raise ValueError("Not enough data for beta")

    cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])[0][1])
    var = float(np.var(aligned.iloc[:, 1], ddof=1))  # same ddof as np.cov
    beta = cov / var if var else None
    return {
        "tool": "beta",
        "symbol": symbol,
        "benchmark": benchmark,
        "beta": round(beta, 4) if beta is not None else None,
        "observations": int(aligned.shape[0]),
    }


def calculate_volatility(symbol: str, *, period: str = "2y", interval: str = "1d", window: int = 30, prices: pd.Series | None = None) -> dict[str, Any]:
    if pd is None or np is None:
        raise RuntimeError("pandas/numpy is not available")

    returns = _daily_returns(_series(symbol, period, interval, prices))
    if returns.empty:
        raise ValueError("Not enough data for volatility")

    recent = returns.tail(window)
    annualized_vol = float(recent.std() * math.sqrt(252) * 100.0) if not recent.empty else None
    return {
        "tool": "volatility",
        "symbol": symbol,
        "volatility_pct": round(annualized_vol, 2) if annualized_vol is not None else None,
        "window": window,
        "observations": int(recent.shape[0]),
    }


def calculate_drawdown(symbol: str, *, period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    if pd is None:
        raise RuntimeError("pandas is not available")

    series = _series(symbol, period, interval, prices)
    cum_max = series.cummax()
    drawdown = (series / cum_max - 1.0) * 100.0
    return {
        "tool": "drawdown",
        "symbol": symbol,
        "max_drawdown_pct": round(float(drawdown.min()), 2),
        "current_drawdown_pct": round(float(drawdown.iloc[-1]), 2),
    }


_PERIOD_PATTERNS = [
    (r"\b5[- ]?y(ears?|rs?)?\b|\bfive years?\b", "5y"),
    (r"\b2[- ]?y(ears?|rs?)?\b|\btwo years?\b", "2y"),
    (r"\b6[- ]?mo(nths?)?\b|\bsix months?\b", "6mo"),
    (r"\b3[- ]?mo(nths?)?\b|\b90[- ]?days?\b|\bthree months?\b|\bquarter\b", "3mo"),
    (r"\b1[- ]?mo(nths?)?\b|\bone month\b", "1mo"),
    (r"\b1[- ]?y(ears?|rs?)?\b|\b12[- ]?months?\b|\b(past|last|one) year\b", "1y"),
]


def _detect_period(text: str) -> str | None:
    """Look-back window named in the message, or None so each metric keeps its own default."""
    for pattern, period in _PERIOD_PATTERNS:
        if re.search(pattern, text or "", re.IGNORECASE):
            return period
    return None


def _annualized_rate(series: pd.Series) -> float | None:
    if series.empty or len(series) < 2:
        return None
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start == 0.0:
        return None
    periods = len(series) - 1
    if periods <= 0:
        return None
    annualized = (end / start) ** (252.0 / periods) - 1.0
    return annualized


def calculate_sharpe_ratio(symbol: str, *, risk_free_rate: float = 0.05, period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    if pd is None or np is None:
        raise RuntimeError("pandas/numpy is not available")

    returns = _daily_returns(_series(symbol, period, interval, prices))
    if returns.empty:
        raise ValueError("Not enough data for Sharpe ratio")

    mean_daily = float(returns.mean())
    std_daily = float(returns.std())
    if std_daily == 0:
        sharpe = None
    else:
        sharpe = (mean_daily - (risk_free_rate / 252.0)) / std_daily * math.sqrt(252)
    return {
        "tool": "sharpe",
        "symbol": symbol,
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "risk_free_rate_pct": risk_free_rate * 100.0,
    }


def calculate_total_return(symbol: str, *, period: str = "1y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    series = _series(symbol, period, interval, prices)
    if len(series) < 2:
        raise ValueError("Not enough data for total return")
    total_return = float(series.iloc[-1] / series.iloc[0] - 1.0) * 100.0
    return {
        "tool": "total_return",
        "symbol": symbol,
        "total_return_pct": round(total_return, 2),
        "period": period,
    }


def calculate_cagr(symbol: str, *, period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    series = _series(symbol, period, interval, prices)
    cagr = _annualized_rate(series)
    if cagr is None:
        raise ValueError("Not enough data for CAGR")
    return {
        "tool": "cagr",
        "symbol": symbol,
        "cagr_pct": round(cagr * 100.0, 2),
        "period": period,
    }


def calculate_momentum(symbol: str, *, period: str = "1mo", interval: str = "1d", lookback: int = 21, prices: pd.Series | None = None) -> dict[str, Any]:
    series = _series(symbol, period, interval, prices)
    if len(series) < lookback + 1 and prices is None:
        series = _fetch_price_history(symbol, period="3mo", interval=interval).iloc[:, 0]
    if len(series) < lookback + 1:
        raise ValueError("Not enough data for momentum")
    momentum = float((series.iloc[-1] / series.iloc[-1 - lookback] - 1.0) * 100.0)
    return {
        "tool": "momentum",
        "symbol": symbol,
        "momentum_pct": round(momentum, 2),
        "window_days": lookback,
    }


def calculate_value_at_risk(symbol: str, *, period: str = "3mo", interval: str = "1d", confidence: float = 0.95, prices: pd.Series | None = None) -> dict[str, Any]:
    returns = _daily_returns(_series(symbol, period, interval, prices))
    if returns.empty:
        raise ValueError("Not enough data for VaR")
    var_pct = float(-np.percentile(returns, (1.0 - confidence) * 100.0) * 100.0)
    return {
        "tool": "var",
        "symbol": symbol,
        "confidence": int(confidence * 100),
        "value_at_risk_pct": round(var_pct, 2),
        "period": period,
    }


def calculate_downside_deviation(symbol: str, *, period: str = "2y", interval: str = "1d", target: float = 0.0, prices: pd.Series | None = None) -> dict[str, Any]:
    returns = _daily_returns(_series(symbol, period, interval, prices))
    downside = returns[returns < target]
    if downside.empty:
        downside_dev = 0.0
    else:
        downside_dev = float(downside.std() * math.sqrt(252) * 100.0)
    return {
        "tool": "downside_deviation",
        "symbol": symbol,
        "downside_deviation_pct": round(downside_dev, 2),
        "target_return_pct": target * 100.0,
    }


def calculate_sortino_ratio(symbol: str, *, risk_free_rate: float = 0.05, period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    returns = _daily_returns(_series(symbol, period, interval, prices))
    if returns.empty:
        raise ValueError("Not enough data for Sortino ratio")
    downside = returns[returns < 0.0]
    if downside.empty:
        return {
            "tool": "sortino",
            "symbol": symbol,
            "sortino_ratio": None,
            "risk_free_rate_pct": risk_free_rate * 100.0,
        }
    downside_std = float(downside.std() * math.sqrt(252))
    mean_daily = float(returns.mean())
    sortino = (mean_daily - (risk_free_rate / 252.0)) / downside_std if downside_std != 0 else None
    return {
        "tool": "sortino",
        "symbol": symbol,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "risk_free_rate_pct": risk_free_rate * 100.0,
    }


def calculate_moving_average(symbol: str, *, window: int = 50, period: str = "2y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    series = _series(symbol, period, interval, prices)
    if len(series) < window:
        raise ValueError("Not enough data for moving average")
    ma = float(series.rolling(window).mean().iloc[-1])
    return {
        "tool": "moving_average",
        "symbol": symbol,
        "window": window,
        "moving_average": round(ma, 2),
    }


def calculate_rsi(symbol: str, *, period: str = "1y", interval: str = "1d", window: int = 14, prices: pd.Series | None = None) -> dict[str, Any]:
    series = _series(symbol, period, interval, prices)
    returns = series.diff().dropna()
    gains = returns.clip(lower=0.0)
    losses = -returns.clip(upper=0.0)
    avg_gain = gains.rolling(window).mean().iloc[-1]
    avg_loss = losses.rolling(window).mean().iloc[-1]
    if avg_loss == 0 or pd.isna(avg_gain) or pd.isna(avg_loss):
        return {
            "tool": "rsi",
            "symbol": symbol,
            "rsi": None,
            "window": window,
        }
    rs = float(avg_gain / avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return {
        "tool": "rsi",
        "symbol": symbol,
        "rsi": round(rsi, 2),
        "window": window,
    }


def calculate_mean_daily_return(symbol: str, *, period: str = "1y", interval: str = "1d", prices: pd.Series | None = None) -> dict[str, Any]:
    returns = _daily_returns(_series(symbol, period, interval, prices))
    if returns.empty:
        raise ValueError("Not enough data for mean daily return")
    mean_pct = float(returns.mean() * 100.0)
    return {
        "tool": "mean_daily_return",
        "symbol": symbol,
        "mean_daily_return_pct": round(mean_pct, 4),
    }


def calculate_portfolio_concentration(holdings: list[dict]) -> dict[str, Any]:
    if not holdings:
        return {"tool": "concentration", "total_value": 0.0, "top_holdings": []}

    rows = []
    total_value = 0.0
    for holding in holdings:
        qty = float(holding.get("quantity") or 0.0)
        avg = float(holding.get("average_price") or 0.0)
        last = float(holding.get("last_price") or 0.0)
        value = qty * (last if last > 0 else avg)
        total_value += value
        rows.append({"symbol": holding.get("tradingsymbol"), "value": value})

    rows = [row for row in rows if row["value"] > 0]
    rows.sort(key=lambda row: row["value"], reverse=True)
    top_holdings = []
    for row in rows[:5]:
        weight = round((row["value"] / total_value * 100.0), 2) if total_value > 0 else 0.0
        top_holdings.append({"symbol": row["symbol"], "weight_pct": weight})

    return {
        "tool": "concentration",
        "total_value": round(total_value, 2),
        "top_holdings": top_holdings,
        "largest_weight_pct": top_holdings[0]["weight_pct"] if top_holdings else 0.0,
    }


def calculate_portfolio_classification(holdings: list[dict]) -> dict[str, Any]:
    if not holdings:
        return {
            "tool": "portfolio_classification",
            "holdings": [],
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "decision": "Hold",
        }

    holding_rows = []
    total_invested = 0.0
    total_current_value = 0.0
    for holding in holdings:
        symbol = holding.get("symbol") or holding.get("tradingsymbol") or "UNKNOWN"
        qty = float(holding.get("quantity") or 0.0)
        avg_price = float(holding.get("average_price") or 0.0)
        last_price = float(holding.get("last_price") or 0.0)
        invested_value = qty * avg_price
        current_value = qty * last_price
        pnl = current_value - invested_value
        pnl_pct = ((current_value - invested_value) / invested_value * 100.0) if invested_value else 0.0
        total_invested += invested_value
        total_current_value += current_value
        holding_rows.append(
            {
                "symbol": symbol,
                "quantity": qty,
                "invested_value": round(invested_value, 2),
                "current_value": round(current_value, 2),
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
            }
        )

    total_pnl = total_current_value - total_invested
    total_pnl_pct = ((total_pnl / total_invested) * 100.0) if total_invested else 0.0

    if total_pnl_pct >= 10.0:
        decision = "Buy"
    elif total_pnl_pct > 0.0:
        decision = "Hold"
    elif total_pnl_pct > -10.0:
        decision = "Trim"
    else:
        decision = "Exit"

    return {
        "tool": "portfolio_classification",
        "holdings": holding_rows,
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "decision": decision,
    }


def _interpret_correlation(correlation: float) -> str:
    if correlation >= 0.7:
        return "Strong positive relationship"
    if correlation >= 0.3:
        return "Moderate positive relationship"
    if correlation <= -0.7:
        return "Strong negative relationship"
    if correlation <= -0.3:
        return "Moderate negative relationship"
    return "Weak or near-zero relationship"


_TOOL_FUNCTIONS = {
    "beta": calculate_beta,
    "sharpe": calculate_sharpe_ratio,
    "sortino": calculate_sortino_ratio,
    "var": calculate_value_at_risk,
    "downside_deviation": calculate_downside_deviation,
    "moving_average": calculate_moving_average,
    "rsi": calculate_rsi,
    "momentum": calculate_momentum,
    "cagr": calculate_cagr,
    "mean_daily_return": calculate_mean_daily_return,
    "total_return": calculate_total_return,
    "volatility": calculate_volatility,
    "drawdown": calculate_drawdown,
}

# First match wins, so order matters (e.g. "value at risk" must win over the generic "risk").
_ROUTES = [
    ("beta", r"\bbeta\b"),
    ("sharpe", r"sharpe|risk[- ]adjusted"),
    ("sortino", r"sortino"),
    ("var", r"\bvar\b|value at risk"),
    ("downside_deviation", r"downside"),
    ("moving_average", r"moving average|\b(50|200)[- ]day\b"),
    ("rsi", r"\brsi\b|relative strength"),
    ("momentum", r"momentum|\b1-month\b"),
    ("cagr", r"cagr|compound annual|annuali[sz]ed"),
    ("mean_daily_return", r"(mean|average) daily return"),
    ("total_return", r"total return|return of|past year|6[- ]month return|annual return"),
    ("volatility", r"volatility|variance|standard deviation|\brisk\b"),
    ("drawdown", r"drawdown|worst decline|crash"),
]

_CLASSIFICATION_KEYWORDS = (
    "invested value",
    "current value",
    "profit or loss",
    "percentage profit or loss",
    "classify the portfolio",
    "classify portfolio",
    "buy, hold, trim, or exit",
    "buy hold trim or exit",
    "portfolio as buy",
)


def select_quantitative_tools(message: str, holdings: list[dict] | None = None) -> list[dict[str, Any]]:
    text = (message or "").lower()
    symbols = _extract_symbols(message, holdings=holdings)
    period = _detect_period(message)

    if any(keyword in text for keyword in ["correlation", "correlate", "co-move", "co move"]) and len(symbols) >= 2:
        return [{"name": "correlation", "kwargs": {"symbol_a": symbols[0], "symbol_b": symbols[1]}}]

    for name, pattern in _ROUTES:
        if not re.search(pattern, text):
            continue
        if symbols:
            kwargs: dict[str, Any] = {"symbol": symbols[0]}
        elif holdings:
            kwargs = {"holdings": holdings}  # no stock named -> the metric is for the whole portfolio
        else:
            continue
        if period:
            kwargs["period"] = period
        if name == "moving_average":
            kwargs["window"] = 200 if "200" in text else 50
        return [{"name": name, "kwargs": kwargs}]

    tools: list[dict[str, Any]] = []
    if holdings and any(keyword in text for keyword in ["concentrat", "allocation", "weight", "overweight", "underweight"]):
        tools.append({"name": "concentration", "kwargs": {"holdings": holdings}})
    if holdings and any(keyword in text for keyword in _CLASSIFICATION_KEYWORDS):
        tools.append({"name": "portfolio_classification", "kwargs": {"holdings": holdings}})
    return tools


def _run_metric(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    fn = _TOOL_FUNCTIONS[name]
    holdings = kwargs.pop("holdings", None)
    period = kwargs.pop("period", None)
    if period is None:
        # Portfolio momentum needs > 21 closes, which the 1mo default cannot give without a refetch.
        period = "3mo" if name == "momentum" and holdings else inspect.signature(fn).parameters["period"].default
    if holdings:
        tickers = [t for t in (_holding_ticker(h) for h in holdings) if t]
        series = portfolio_value_series(holdings, period=period)
        result = fn(f"your portfolio ({', '.join(tickers)})", period=period, prices=series, **kwargs)
        result["scope"] = "portfolio"
    else:
        result = fn(kwargs.pop("symbol"), period=period, **kwargs)
    result.setdefault("period", period)
    return result


def execute_quantitative_tools(message: str, holdings: list[dict] | None = None) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for tool in select_quantitative_tools(message, holdings=holdings):
        name = tool["name"]
        kwargs = dict(tool.get("kwargs", {}))
        try:
            if name == "correlation":
                result = calculate_correlation(kwargs["symbol_a"], kwargs["symbol_b"])
            elif name == "concentration":
                result = calculate_portfolio_concentration(kwargs.get("holdings") or [])
            elif name == "portfolio_classification":
                result = calculate_portfolio_classification(kwargs.get("holdings") or [])
            elif name in _TOOL_FUNCTIONS:
                result = _run_metric(name, kwargs)
            else:
                result = {"tool": name, "status": "unsupported"}

            results.append({"tool": name, "status": "ok", "result": result})
        except Exception as exc:  # pragma: no cover - runtime network fallback
            results.append({"tool": name, "status": "error", "error": str(exc)})

    return results


def build_quantitative_answer(message: str, tool_results: list[dict[str, Any]]) -> str | None:
    answer = _format_quantitative_answer(tool_results)
    if not answer:
        return None
    result = next((item.get("result") or {} for item in tool_results if item.get("status") == "ok"), {})
    if result.get("period") and result.get("tool") not in {"total_return", "cagr", "var"}:
        answer += f" (Based on {result['period']} of daily closing prices.)"
    return answer


def _format_quantitative_answer(tool_results: list[dict[str, Any]]) -> str | None:
    if not tool_results:
        return None
    successful = [item for item in tool_results if item.get("status") == "ok"]
    if not successful:
        return None

    first = successful[0]
    result = first.get("result") or {}
    tool_name = first.get("tool")

    if tool_name == "correlation":
        corr = result.get("correlation")
        if corr is None:
            return None
        return (
            f"The correlation between {result.get('symbol_a')} and {result.get('symbol_b')} is {corr:.4f}. "
            f"{result.get('interpretation', '')}."
        )

    if tool_name == "beta":
        beta = result.get("beta")
        if beta is None:
            return None
        return f"The beta of {result.get('symbol')} versus {result.get('benchmark')} is {beta:.4f}."

    if tool_name == "volatility":
        vol = result.get("volatility_pct")
        if vol is None:
            return None
        return f"The annualized volatility of {result.get('symbol')} is {vol:.2f}%."

    if tool_name == "drawdown":
        max_dd = result.get("max_drawdown_pct")
        if max_dd is None:
            return None
        return f"The maximum drawdown for {result.get('symbol')} is {max_dd:.2f}%."

    if tool_name == "sharpe":
        sharpe = result.get("sharpe_ratio")
        if sharpe is None:
            return None
        return f"The Sharpe ratio for {result.get('symbol')} is {sharpe:.4f}."

    if tool_name == "total_return":
        total_return = result.get("total_return_pct")
        if total_return is None:
            return None
        return f"The total return for {result.get('symbol')} over {result.get('period')} is {total_return:.2f}%."

    if tool_name == "cagr":
        cagr = result.get("cagr_pct")
        if cagr is None:
            return None
        return f"The CAGR for {result.get('symbol')} over {result.get('period')} is {cagr:.2f}%."

    if tool_name == "momentum":
        momentum = result.get("momentum_pct")
        if momentum is None:
            return None
        return f"The {result.get('window_days')}-day momentum for {result.get('symbol')} is {momentum:.2f}% ."

    if tool_name == "var":
        var = result.get("value_at_risk_pct")
        if var is None:
            return None
        return f"The {result.get('confidence')}% Value at Risk for {result.get('symbol')} is {var:.2f}% over {result.get('period')} data."

    if tool_name == "downside_deviation":
        downside = result.get("downside_deviation_pct")
        if downside is None:
            return None
        return f"The downside deviation for {result.get('symbol')} is {downside:.2f}% based on recent returns."

    if tool_name == "sortino":
        sortino = result.get("sortino_ratio")
        if sortino is None:
            return None
        return f"The Sortino ratio for {result.get('symbol')} is {sortino:.4f}."

    if tool_name == "moving_average":
        ma = result.get("moving_average")
        if ma is None:
            return None
        return f"The {result.get('window')}-day moving average of {result.get('symbol')} is {ma:.2f}."

    if tool_name == "rsi":
        rsi = result.get("rsi")
        if rsi is None:
            return None
        return f"The RSI for {result.get('symbol')} is {rsi:.2f}."

    if tool_name == "mean_daily_return":
        mean_return = result.get("mean_daily_return_pct")
        if mean_return is None:
            return None
        return f"The mean daily return for {result.get('symbol')} is {mean_return:.4f}% ."

    if tool_name == "concentration":
        largest = result.get("largest_weight_pct")
        if largest is None:
            return None
        return f"Your largest holding weight is {largest:.2f}%, based on the current portfolio snapshot."

    if tool_name == "portfolio_classification":
        total_invested = result.get("total_invested")
        total_current = result.get("total_current_value")
        pnl = result.get("total_pnl")
        pnl_pct = result.get("total_pnl_pct")
        decision = result.get("decision")
        if total_invested is None or total_current is None or pnl is None or pnl_pct is None:
            return None
        return (
            f"Portfolio summary: invested value = ₹{total_invested:.2f}, current value = ₹{total_current:.2f}, "
            f"profit or loss = ₹{pnl:.2f}, percentage profit or loss = {pnl_pct:.2f}%. Decision: {decision}."
        )

    return None
