"""
Unit tests for intent classification accuracy.
Run: pytest tests/test_chatbot/test_intent_router.py -v
"""
import pytest

from llm_orchestrator.utils.intent_router import (
    classify_intent,
    estimate_confidence_score,
    intent_supported_for_portfolio,
)

# ── Intent classification ────────────────────────────────────────────────────

@pytest.mark.parametrize("message,expected_intent", [
    # portfolio_summary
    ("How is my portfolio doing?",          "portfolio_summary"),
    ("Give me a quick overview",            "portfolio_summary"),
    ("Portfolio snapshot please",           "portfolio_summary"),
    # performance
    ("What are my returns this month?",     "performance"),
    ("Show me my P&L",                      "performance"),
    ("How much profit have I made?",        "performance"),
    # allocation
    ("How diversified am I?",               "allocation"),
    ("Show my allocation breakdown",        "allocation"),
    ("What is the concentration of my top holding?", "allocation"),
    # position_analysis
    ("Should I trim my GOLDBEES position?", "position_analysis"),
    ("Which holding is my biggest loser?",  "position_analysis"),
    ("Tell me about my positions",          "position_analysis"),
    # risk
    ("What is my portfolio risk?",          "risk"),
    ("Am I too exposed to one sector?",     "allocation"),  # 'sector'+'exposed' => allocation
    ("What's my drawdown?",                 "risk"),
    # rebalancing
    ("Should I rebalance my portfolio?",    "rebalancing"),
    ("Help me rebalance",                   "rebalancing"),
    # next_action
    ("What should I do next?",              "next_action"),
    ("What now with my investments?",       "next_action"),
    # unknown
    ("Tell me a joke",                      "unknown"),
    ("What is the weather today?",          "unknown"),
    ("",                                    "unknown"),
])
def test_classify_intent(message: str, expected_intent: str) -> None:
    result = classify_intent(message)
    assert result == expected_intent, (
        f"classify_intent({message!r}) => {result!r}, expected {expected_intent!r}"
    )


def test_classify_intent_case_insensitive() -> None:
    assert classify_intent("HOW IS MY PORTFOLIO DOING") == classify_intent("how is my portfolio doing")


def test_classify_intent_nonempty_result() -> None:
    intents = {classify_intent(m) for m in ["alpha beta gamma", "???"]}
    assert all(isinstance(i, str) and len(i) > 0 for i in intents)


# ── Confidence score ─────────────────────────────────────────────────────────

def test_confidence_score_bounds() -> None:
    for intent in ["portfolio_summary", "unknown", "risk", "performance"]:
        for holdings_count in [0, 1, 10]:
            for has_llm in [True, False]:
                score = estimate_confidence_score(
                    intent=intent,
                    holdings_count=holdings_count,
                    has_llm=has_llm,
                )
                assert 0.0 <= score <= 1.0, f"Score out of range: {score}"


def test_confidence_higher_with_holdings_and_llm() -> None:
    low = estimate_confidence_score(intent="unknown", holdings_count=0, has_llm=False)
    high = estimate_confidence_score(intent="portfolio_summary", holdings_count=5, has_llm=True)
    assert high > low


# ── Support check ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("intent,supported", [
    ("portfolio_summary", True),
    ("performance",       True),
    ("risk",              True),
    ("rebalancing",       True),
    ("unknown",           False),
])
def test_intent_supported(intent: str, supported: bool) -> None:
    assert intent_supported_for_portfolio(intent) == supported
