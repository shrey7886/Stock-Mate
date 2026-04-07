from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Intent rules: order matters — more specific patterns first
# ---------------------------------------------------------------------------
INTENT_RULES: dict[str, list[str]] = {
    # ── Core portfolio intents ───────────────────────────────────────────────
    "portfolio_summary": [
        "summary", "overview", "how is my portfolio", "snapshot",
        "how am i doing", "give me a snapshot", "portfolio status",
    ],
    "performance": [
        "return", "performance", "pnl", "p&l", "gain", "loss", "profit",
        "how much have i made", "how much have i lost", "today's change",
        "day change", "up today", "down today",
    ],
    "allocation": [
        "allocation", "weight", "concentrat", "diversif", "sector",
        "exposed", "exposure", "weightage", "portion", "percentage",
    ],
    "position_analysis": [
        "best holding", "worst holding", "biggest position",
        "smallest position", "top winner", "top loser",
        "which stock", "which holding",
        "position", "trim", "my holding",
    ],
    "risk": [
        "risk", "drawdown", "volatility", "safe", "how risky",
        "am i over-exposed", "am i overexposed",
    ],
    "rebalancing": [
        "rebalance", "rebalancing", "redistribute", "should i shift",
        "how to balance", "restructure",
    ],
    "next_action": [
        "what should i do", "next step", "what now", "advise me",
        "what action", "recommend", "suggest", "what do you recommend",
        "should i buy", "should i sell", "should i invest", "should i put",
        "is it worth", "is it good to", "is it safe to", "good investment",
        "crypto", "bitcoin", "gold", "real estate", "fd", "fixed deposit",
        "nps", "ppf", "mutual fund", "etf",
    ],
    # ── New Cleo-level intents ───────────────────────────────────────────────
    "watchlist": [
        "watchlist", "add to watchlist", "track this stock", "watch",
        "i'm tracking", "stocks i'm watching", "remove from watchlist",
        "my watchlist",
    ],
    "tax_analysis": [
        "tax", "ltcg", "stcg", "capital gains", "tax implication",
        "tax on selling", "how long have i held", "holding period",
        "long term", "short term gain",
    ],
    "goal_tracking": [
        "goal", "target", "corpus", "retirement", "how long to reach",
        "when will i reach", "financial goal", "how many years",
        "savings goal", "set a goal", "my goal is",
    ],
    "stock_comparison": [
        "compare", " vs ", "better stock", "which is better",
        "head to head", "better between", "outperform",
    ],
    "sip_advice": [
        "sip", "monthly investment", "invest monthly", "regular investment",
        "systematic investment", "how much should i invest",
        "monthly sip", "sip amount",
    ],
    "portfolio_health": [
        "health", "health score", "grade my portfolio", "how healthy",
        "portfolio grade", "portfolio score", "rate my portfolio",
        "portfolio rating",
    ],
    "market_overview": [
        "market today", "nifty", "sensex", "market mood", "indices",
        "market up", "market down", "bulls", "bears", "how is the market",
        "market condition",
    ],
}

SUPPORTED_INTENTS = set(INTENT_RULES.keys())

# ── Dynamic follow-up suggestions per intent ────────────────────────────────
NEXT_STEPS_MAP: dict[str, list[str]] = {
    "portfolio_summary": [
        "Which holding is your biggest winner? 🏆",
        "How concentrated is your portfolio?",
        "Get your portfolio health score",
    ],
    "performance": [
        "Which holding is dragging you down?",
        "Compare your return to a benchmark",
        "See today's change by holding",
    ],
    "allocation": [
        "Get your diversification score",
        "See which sector you're most exposed to",
        "Ask for a rebalancing plan",
    ],
    "position_analysis": [
        "Compare two of your holdings",
        "What should you trim or add?",
        "Add a stock to your watchlist",
    ],
    "risk": [
        "See all active risk flags",
        "Ask for a rebalancing action plan",
        "Check if you're over-concentrated",
    ],
    "rebalancing": [
        "What's an ideal diversified portfolio?",
        "What are the tax implications of rebalancing?",
        "Get SIP suggestions to gradually rebalance",
    ],
    "next_action": [
        "Explain why you should hold or trim",
        "Set a financial goal",
        "Explore SIP options for your top picks",
    ],
    "watchlist": [
        "Add another stock to your watchlist",
        "Compare watchlist stocks vs your holdings",
        "Get signals for stocks on your watchlist",
    ],
    "tax_analysis": [
        "Which of your holdings qualify for LTCG?",
        "What's your estimated STCG liability?",
        "Ask about tax-loss harvesting",
    ],
    "goal_tracking": [
        "Set a target corpus amount",
        "Calculate monthly SIP needed for your goal",
        "Check your progress toward your goal",
    ],
    "stock_comparison": [
        "Drill into your best performer",
        "Check risk-adjusted returns by holding",
        "Ask which holding to add to",
    ],
    "sip_advice": [
        "Which stock is best for a SIP?",
        "How long to reach ₹10L with a monthly SIP?",
        "Set a goal to calculate your ideal SIP",
    ],
    "portfolio_health": [
        "What's lowering your health score?",
        "How to improve your allocation?",
        "Ask for specific actions to boost your score",
    ],
    "market_overview": [
        "How is the market affecting your portfolio?",
        "Check your sector exposure vs Nifty",
        "See which holdings are market leaders",
    ],
    "unknown": [
        "Ask for portfolio summary",
        "Check your risk and concentration",
        "Get holding-level performance",
    ],
}


def classify_intent(message: str) -> str:
    text = (message or "").strip().lower()
    if not text:
        return "unknown"

    for intent, patterns in INTENT_RULES.items():
        if any(pattern in text for pattern in patterns):
            return intent

    # broad portfolio/question fallback
    keywords = [
        "portfolio", "holding", "pnl", "stock", "risk", "invest",
        "allocation", "market", "broker", "zerodha", "shares",
    ]
    if any(k in text for k in keywords):
        return "portfolio_summary"

    return "unknown"


# ── Symbol extraction (for watchlist commands) ───────────────────────────────
_SYMBOL_RE = re.compile(r"\b([A-Z]{2,12}(?:BE|NS)?)\b")

def extract_ticker_symbols(message: str) -> list[str]:
    """
    Extract potential NSE/BSE ticker symbols from a message.
    Tickers are 2-12 uppercase letters, optionally ending in BE or NS.
    Common English words are filtered out.
    """
    _STOPWORDS = {
        "I", "MY", "IS", "IT", "AM", "TO", "AT", "IN", "ON", "AN",
        "BE", "BY", "DO", "GO", "IF", "NO", "OF", "OR", "SO", "UP",
        "VS", "WE", "OK", "ALL", "AND", "FOR", "GET", "HOW", "SIP",
        "ADD", "CAN", "THE", "ARE", "WAS", "NOT", "SET", "HIT",
        "BUY", "SELL", "HOLD", "TRIM", "WATCH", "LTCG", "STCG",
        "PNL", "RSI", "TFT", "NSE", "BSE", "ETF", "IPO", "FD",
        "SHOW", "TELL", "GIVE", "TAKE", "WHAT", "WHEN", "WHO", "WHY",
    }
    found = _SYMBOL_RE.findall(message)
    return [s for s in found if s not in _STOPWORDS]


def get_next_steps(intent: str) -> list[str]:
    return NEXT_STEPS_MAP.get(intent, NEXT_STEPS_MAP["unknown"])


def estimate_confidence_score(*, intent: str, holdings_count: int, has_llm: bool) -> float:
    score = 0.55
    if intent in SUPPORTED_INTENTS:
        score += 0.15
    if holdings_count > 0:
        score += 0.15
    if has_llm:
        score += 0.10
    if intent == "unknown":
        score -= 0.15
    return max(0.0, min(1.0, round(score, 2)))


def intent_supported_for_portfolio(intent: str) -> bool:
    return intent in SUPPORTED_INTENTS
