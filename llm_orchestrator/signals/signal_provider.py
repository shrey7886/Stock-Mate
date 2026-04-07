"""
SignalProvider defines the interface between TFT model outputs and the chatbot.
Swap MockSignalProvider for TFTSignalProvider when your friend completes the TFT.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class StockSignal:
    symbol: str
    action_tag: str                  # Hold | Trim | Add | Watch
    tft_forecast_7d_pct: float | None  # % predicted change over 7 days (None if no TFT yet)
    tft_confidence: float | None       # 0.0 – 1.0 confidence band width (None if no TFT yet)
    sentiment_score: float | None      # 0–100 composite sentiment
    momentum: str | None               # "positive" | "neutral" | "negative"
    rsi_14: float | None               # raw RSI value
    signal_source: str = "mock"        # "tft" | "mock" | "rule_based"
    notes: str = ""


@dataclass
class PortfolioSignals:
    overall_action: str                          # Hold | Trim | Add | Watch | Rebalance
    portfolio_risk_level: str                    # Low | Medium | High
    top_risks: list[str] = field(default_factory=list)
    top_strengths: list[str] = field(default_factory=list)
    stock_signals: list[StockSignal] = field(default_factory=list)
    signals_source: str = "mock"


class SignalProvider(ABC):
    """Abstract base — TFT or any other model plugs in here."""

    @abstractmethod
    def get_portfolio_signals(
        self,
        *,
        holdings: list[dict],
    ) -> PortfolioSignals:
        ...


class MockSignalProvider(SignalProvider):
    """
    Rule-based mock signals derived from live holdings P&L and basic price data.
    Replace with TFTSignalProvider.get_portfolio_signals() when TFT is ready.
    """

    _ACTION_RULES = [
        (lambda pnl_pct: pnl_pct <= -15,  "Trim",  "negative", "Down significantly — consider trimming."),
        (lambda pnl_pct: pnl_pct <= -7,   "Watch", "negative", "Underperforming — monitor closely."),
        (lambda pnl_pct: pnl_pct >= 15,   "Trim",  "positive", "Strong gain — may be time to book partial profits."),
        (lambda pnl_pct: pnl_pct >= 5,    "Hold",  "positive", "Performing well — hold and review."),
        (lambda pnl_pct: True,            "Hold",  "neutral",  "Within normal range."),
    ]

    def get_portfolio_signals(self, *, holdings: list[dict]) -> PortfolioSignals:
        stock_signals: list[StockSignal] = []
        total_invested = 0.0
        total_current = 0.0

        for h in holdings:
            qty = float(h.get("quantity") or 0)
            avg_price = float(h.get("average_price") or 0)
            last_price = float(h.get("last_price") or 0)

            invested = qty * avg_price
            current = qty * last_price
            total_invested += invested
            total_current += current

            pnl_pct = ((current - invested) / invested * 100.0) if invested > 0 else 0.0

            action, momentum, notes = "Hold", "neutral", ""
            for rule, act, mom, note in self._ACTION_RULES:
                if rule(pnl_pct):
                    action, momentum, notes = act, mom, note
                    break

            stock_signals.append(
                StockSignal(
                    symbol=h.get("tradingsymbol", "?"),
                    action_tag=action,
                    tft_forecast_7d_pct=None,
                    tft_confidence=None,
                    sentiment_score=None,
                    momentum=momentum,
                    rsi_14=None,
                    signal_source="mock",
                    notes=notes,
                )
            )

        portfolio_pnl_pct = (
            (total_current - total_invested) / total_invested * 100.0
            if total_invested > 0
            else 0.0
        )

        risk = "High" if portfolio_pnl_pct <= -10 else ("Low" if portfolio_pnl_pct >= 5 else "Medium")
        top_risks = ["Portfolio in drawdown — review allocation."] if portfolio_pnl_pct < -5 else []
        top_strengths = ["Portfolio showing positive returns."] if portfolio_pnl_pct > 5 else []

        return PortfolioSignals(
            overall_action="Watch" if portfolio_pnl_pct < 0 else "Hold",
            portfolio_risk_level=risk,
            top_risks=top_risks,
            top_strengths=top_strengths,
            stock_signals=stock_signals,
            signals_source="mock",
        )


# Singleton — swap to TFTSignalProvider when TFT is ready
signal_provider: SignalProvider = MockSignalProvider()
