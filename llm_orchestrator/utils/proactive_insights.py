"""
ProactiveInsightsEngine: generates unsolicited, data-driven alerts and insights
from live portfolio data — similar to Cleo's proactive nudges.
"""
from __future__ import annotations


def generate_proactive_insights(
    *,
    holdings: list[dict],
    total_invested: float,
    total_current_value: float,
    total_pnl: float,
    total_pnl_pct: float,
    analytics: dict,
) -> list[dict]:
    """
    Returns a list of proactive insight dicts:
    {
        "priority": "high" | "medium" | "low",
        "category": "risk" | "opportunity" | "action" | "info",
        "title": str,
        "message": str,
        "action_tag": str,  # Hold | Trim | Add | Watch | Rebalance | None
    }
    Sorted by priority (high → low).
    """
    insights: list[dict] = []

    if not holdings:
        insights.append({
            "priority": "high",
            "category": "action",
            "title": "No holdings found",
            "message": (
                "Your broker account appears to have no holdings. "
                "If you've recently invested, it may take a day to reflect. "
                "Or link a broker account to get started."
            ),
            "action_tag": "None",
        })
        return insights

    risk_flags = analytics.get("risk_flags", [])
    health = analytics.get("portfolio_health", {})
    health_score = health.get("score", 50.0)
    top_winner = analytics.get("top_winner")
    top_loser = analytics.get("top_loser")
    concentration_top1 = analytics.get("concentration_top1_pct", 0.0)
    concentration_top3 = analytics.get("concentration_top3_pct", 0.0)
    largest = analytics.get("largest_position")

    # ── Risk alerts ────────────────────────────────────────────────────────
    if "single_stock_concentration_high" in risk_flags and largest:
        insights.append({
            "priority": "high",
            "category": "risk",
            "title": f"⚠️ Heavy concentration in {largest['symbol']}",
            "message": (
                f"**{largest['symbol']}** makes up **{largest['weight_pct']:.1f}%** of your portfolio. "
                f"This is high — if it moves against you, your whole portfolio feels it. "
                f"Consider trimming to below 30–35%."
            ),
            "action_tag": "Trim",
        })

    if "portfolio_drawdown_high" in risk_flags:
        insights.append({
            "priority": "high",
            "category": "risk",
            "title": f"📉 Portfolio down {abs(total_pnl_pct):.1f}%",
            "message": (
                f"Your portfolio is in a drawdown of **{abs(total_pnl_pct):.1f}%** "
                f"(₹{abs(total_pnl):,.0f} unrealized loss). "
                f"This may be a good time to review your allocation and consider rebalancing."
            ),
            "action_tag": "Watch",
        })

    if "overall_portfolio_negative" in risk_flags and "portfolio_drawdown_high" not in risk_flags:
        insights.append({
            "priority": "medium",
            "category": "risk",
            "title": "Portfolio in the red",
            "message": (
                f"You're currently down ₹{abs(total_pnl):,.0f} ({abs(total_pnl_pct):.2f}%). "
                f"Market downturns are normal — are you investing for the long term or is this a concern?"
            ),
            "action_tag": "Watch",
        })

    if "top3_concentration_high" in risk_flags:
        insights.append({
            "priority": "medium",
            "category": "risk",
            "title": "Your top 3 holdings dominate the portfolio",
            "message": (
                f"Your top 3 positions account for **{concentration_top3:.1f}%** of your portfolio. "
                f"Adding 2–3 more quality positions would improve diversification significantly."
            ),
            "action_tag": "Rebalance",
        })

    # ── Profit-booking opportunities ───────────────────────────────────────
    if top_winner and top_winner.get("pnl_pct", 0) >= 20:
        insights.append({
            "priority": "medium",
            "category": "opportunity",
            "title": f"📈 {top_winner['symbol']} is up {top_winner['pnl_pct']:.1f}% — consider booking partial profits",
            "message": (
                f"**{top_winner['symbol']}** has gained **{top_winner['pnl_pct']:.1f}%** "
                f"(₹{top_winner['pnl']:,.0f}). At this level, trimming 20–30% of the position "
                f"locks in real gains while staying invested."
            ),
            "action_tag": "Trim",
        })
    elif top_winner and top_winner.get("pnl_pct", 0) >= 10:
        insights.append({
            "priority": "low",
            "category": "opportunity",
            "title": f"✅ {top_winner['symbol']} is performing well",
            "message": (
                f"**{top_winner['symbol']}** is up {top_winner['pnl_pct']:.1f}% — solid performance. "
                f"No action needed unless it becomes overly concentrated."
            ),
            "action_tag": "Hold",
        })

    # ── Underperformer alerts ─────────────────────────────────────────────
    if top_loser and top_loser.get("pnl_pct", 0) <= -15:
        insights.append({
            "priority": "high",
            "category": "action",
            "title": f"⚠️ {top_loser['symbol']} is your biggest drag",
            "message": (
                f"**{top_loser['symbol']}** is down **{abs(top_loser['pnl_pct']):.1f}%** "
                f"(₹{abs(top_loser['pnl']):,.0f} unrealized loss). "
                f"Is this a temporary dip or a structural issue? Consider reviewing the thesis."
            ),
            "action_tag": "Watch",
        })

    # ── Health score nudge ─────────────────────────────────────────────────
    if health_score < 50:
        insights.append({
            "priority": "medium",
            "category": "action",
            "title": f"Portfolio health score: {health_score:.0f}/100 — room to improve",
            "message": (
                f"Your portfolio health is **{health.get('grade', '?')} ({health_score:.0f}/100)**. "
                f"{health.get('verdict', '')} "
                f"Ask me for specific steps to improve it."
            ),
            "action_tag": "Rebalance",
        })

    # ── All green ─────────────────────────────────────────────────────────
    if not risk_flags and total_pnl_pct >= 5 and health_score >= 70:
        insights.append({
            "priority": "low",
            "category": "info",
            "title": "🎯 Portfolio looking healthy",
            "message": (
                f"No major risk flags, your portfolio is up {total_pnl_pct:.2f}%, "
                f"and health score is {health_score:.0f}/100. Keep it up — "
                f"review allocation quarterly to stay on track."
            ),
            "action_tag": "Hold",
        })

    # Sort: high → medium → low
    priority_order = {"high": 0, "medium": 1, "low": 2}
    insights.sort(key=lambda i: priority_order.get(i["priority"], 3))
    return insights
