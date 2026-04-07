"""
Unit tests for portfolio analytics calculations.
Run: pytest tests/test_chatbot/test_portfolio_analytics.py -v
"""
import pytest

from llm_orchestrator.utils.portfolio_analytics import build_portfolio_analytics

# ── Fixtures ─────────────────────────────────────────────────────────────────

HOLDINGS_SINGLE = [
    {"tradingsymbol": "GOLDBEES", "quantity": 225, "average_price": 54.0, "last_price": 50.0},
]

HOLDINGS_MULTI = [
    {"tradingsymbol": "RELIANCE",  "quantity": 10,  "average_price": 2500.0, "last_price": 2700.0},
    {"tradingsymbol": "INFY",      "quantity": 20,  "average_price": 1400.0, "last_price": 1350.0},
    {"tradingsymbol": "HDFCBANK",  "quantity": 15,  "average_price": 1600.0, "last_price": 1620.0},
    {"tradingsymbol": "TCS",       "quantity": 5,   "average_price": 3800.0, "last_price": 4000.0},
    {"tradingsymbol": "GOLDBEES",  "quantity": 200, "average_price": 54.0,   "last_price": 50.0},
]

HOLDINGS_EMPTY: list = []


def _make_summary(holdings: list[dict]) -> dict:
    total_invested = sum(float(h["average_price"]) * float(h["quantity"]) for h in holdings)
    total_current  = sum(float(h["last_price"])    * float(h["quantity"]) for h in holdings)
    total_pnl      = total_current - total_invested
    total_pnl_pct  = (total_pnl / total_invested * 100.0) if total_invested else 0.0
    return dict(
        holdings=holdings,
        total_invested=total_invested,
        total_current_value=total_current,
        total_pnl=total_pnl,
        total_pnl_pct=total_pnl_pct,
    )


# ── Empty portfolio ──────────────────────────────────────────────────────────

def test_empty_portfolio_returns_no_holdings_flag() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_EMPTY))
    assert "no_holdings" in result["risk_flags"]
    assert result["largest_position"] is None
    assert result["top_winner"] is None


# ── Single holding ───────────────────────────────────────────────────────────

def test_single_holding_concentration_is_100() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_SINGLE))
    assert result["concentration_top1_pct"] == pytest.approx(100.0, abs=0.1)
    assert result["concentration_top3_pct"] == pytest.approx(100.0, abs=0.1)


def test_single_holding_flag_single_stock_concentration() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_SINGLE))
    assert "single_stock_concentration_high" in result["risk_flags"]


# ── Multi holding ────────────────────────────────────────────────────────────

def test_top_winner_is_correct() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    assert result["top_winner"]["symbol"] == "RELIANCE"


def test_top_loser_has_negative_pnl() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    assert result["top_loser"]["pnl"] < 0


def test_concentration_sums_are_valid() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    assert 0.0 < result["concentration_top1_pct"] <= 100.0
    assert result["concentration_top3_pct"] >= result["concentration_top1_pct"]


def test_diversification_score_in_range() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    assert 0.0 <= result["diversification_score"] <= 100.0


def test_diversification_higher_for_more_holdings() -> None:
    single = build_portfolio_analytics(**_make_summary(HOLDINGS_SINGLE))
    multi  = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    assert multi["diversification_score"] > single["diversification_score"]


def test_negative_portfolio_flags_risk() -> None:
    # Force total_pnl_pct = -15 to trigger drawdown flag
    result = build_portfolio_analytics(
        holdings=HOLDINGS_SINGLE,
        total_invested=100_000.0,
        total_current_value=85_000.0,
        total_pnl=-15_000.0,
        total_pnl_pct=-15.0,
    )
    assert "portfolio_drawdown_high" in result["risk_flags"]


# ── Return structure ─────────────────────────────────────────────────────────

def test_result_has_all_required_keys() -> None:
    result = build_portfolio_analytics(**_make_summary(HOLDINGS_MULTI))
    for key in [
        "largest_position", "smallest_position", "top_winner", "top_loser",
        "concentration_top1_pct", "concentration_top3_pct",
        "diversification_score", "risk_flags",
    ]:
        assert key in result, f"Missing key: {key}"
