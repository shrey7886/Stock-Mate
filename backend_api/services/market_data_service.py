from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import yfinance as yf
    _HAS_YFINANCE = True
except Exception:
    yf = None
    _HAS_YFINANCE = False

PERIOD_MAP = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "1Y": "1y",
}


class MarketDataService:
    def resolve_period(self, period: str) -> str:
        return PERIOD_MAP.get(period.upper(), "1mo")

    def _history_for_symbol(self, base_symbol: str, yf_period: str):
        if not _HAS_YFINANCE:
            return None

        for suffix in (".NS", ".BO"):
            try:
                ticker = yf.Ticker(f"{base_symbol}{suffix}")
                hist = ticker.history(period=yf_period)
                if hist is not None and not hist.empty:
                    return hist
            except Exception as exc:
                logger.warning("yfinance history fetch failed for %s%s: %s", base_symbol, suffix, exc)
        return None

    def fetch_close_series(self, base_symbol: str, period: str) -> dict[str, float]:
        """Returns {"YYYY-MM-DD": close_price} for the given base symbol (no suffix), or {} on failure."""
        yf_period = self.resolve_period(period)
        hist = self._history_for_symbol(base_symbol, yf_period)
        if hist is None:
            return {}

        try:
            series = {}
            for ts, row in hist.iterrows():
                date_key = ts.strftime("%Y-%m-%d")
                close = row.get("Close")
                if close is not None:
                    series[date_key] = float(close)
            return series
        except Exception as exc:
            logger.warning("Failed to parse close series for %s: %s", base_symbol, exc)
            return {}

    def fetch_index_close_series(self, ticker_symbol: str, period: str) -> dict[str, float]:
        """For index/raw tickers that already include their own symbol format (e.g. ^NSEI)."""
        if not _HAS_YFINANCE:
            return {}

        yf_period = self.resolve_period(period)
        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=yf_period)
            if hist is None or hist.empty:
                return {}
            series = {}
            for ts, row in hist.iterrows():
                date_key = ts.strftime("%Y-%m-%d")
                close = row.get("Close")
                if close is not None:
                    series[date_key] = float(close)
            return series
        except Exception as exc:
            logger.warning("yfinance history fetch failed for %s: %s", ticker_symbol, exc)
            return {}

    def fetch_sector(self, base_symbol: str) -> str | None:
        if not _HAS_YFINANCE:
            return None

        for suffix in (".NS", ".BO"):
            try:
                ticker = yf.Ticker(f"{base_symbol}{suffix}")
                info = ticker.info
                sector = info.get("sector") if isinstance(info, dict) else None
                if sector:
                    return sector
            except Exception as exc:
                logger.warning("yfinance sector fetch failed for %s%s: %s", base_symbol, suffix, exc)
        return None

    def fetch_fundamentals(self, base_symbol: str) -> dict | None:
        if not _HAS_YFINANCE:
            return None

        for suffix in (".NS", ".BO"):
            try:
                ticker = yf.Ticker(f"{base_symbol}{suffix}")
                info = ticker.info
                if not isinstance(info, dict) or not info:
                    continue
                return {
                    "long_name": info.get("longName"),
                    "sector": info.get("sector"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "dividend_yield": info.get("dividendYield"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "beta": info.get("beta"),
                    "currency": info.get("currency"),
                }
            except Exception as exc:
                logger.warning("yfinance fundamentals fetch failed for %s%s: %s", base_symbol, suffix, exc)
        return None

    def fetch_index_snapshot(self, ticker_symbol: str, label: str) -> dict | None:
        if not _HAS_YFINANCE:
            return None

        try:
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="5d")
            if hist is None or hist.empty or len(hist) < 2:
                return None
            closes = hist["Close"].dropna()
            if len(closes) < 2:
                return None
            price = float(closes.iloc[-1])
            prior = float(closes.iloc[-2])
            change = price - prior
            change_pct = (change / prior * 100.0) if prior else 0.0
            return {
                "symbol": ticker_symbol,
                "label": label,
                "price": round(price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            }
        except Exception as exc:
            logger.warning("yfinance index snapshot fetch failed for %s: %s", ticker_symbol, exc)
            return None


market_data_service = MarketDataService()
