"""
ChatService: bridge between the FastAPI route and the LLM pipeline.
Fetches live portfolio summary, delegates to the pipeline, and enriches the reply
with proactive insights and portfolio health score.
"""
from __future__ import annotations

from fastapi import HTTPException, status

from backend_api.services.zerodha_service import ZerodhaService
from llm_orchestrator.pipelines.portfolio_explanation_pipeline import ChatReply, pipeline
from llm_orchestrator.utils.portfolio_analytics import build_portfolio_analytics
from llm_orchestrator.utils.proactive_insights import generate_proactive_insights

_zerodha = ZerodhaService()


def _fetch_portfolio_summary(user_id: str) -> dict:
    """
    Attempt to get live portfolio data.
    Falls back to an empty portfolio if Zerodha is not linked or unavailable.
    """
    from backend_api.database.token_store import get_decrypted_broker_tokens

    tokens = get_decrypted_broker_tokens(user_id=user_id, provider="zerodha")
    if not tokens:
        return {
            "account_id": None,
            "holdings": [],
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        }

    try:
        holdings = _zerodha.fetch_holdings(access_token=tokens.get("access_token", ""))
        total_invested = sum(float(h.get("average_price", 0)) * float(h.get("quantity", 0)) for h in holdings)
        total_current = sum(float(h.get("last_price", 0)) * float(h.get("quantity", 0)) for h in holdings)
        total_pnl = total_current - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

        return {
            "account_id": tokens.get("account_id"),
            "holdings": holdings,
            "total_invested": round(total_invested, 2),
            "total_current_value": round(total_current, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
        }
    except Exception:
        return {
            "account_id": tokens.get("account_id"),
            "holdings": [],
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
        }


def chat(user_id: str, message: str) -> ChatReply:
    if not message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Message cannot be empty.",
        )
    portfolio_summary = _fetch_portfolio_summary(user_id)

    # If no broker is linked, prompt the user to connect their portfolio
    if portfolio_summary.get("account_id") is None:
        return ChatReply(
            answer=(
                "It looks like you haven't connected your portfolio yet. "
                "Head over to the **Broker** page to link your Zerodha account "
                "and I'll be able to give you personalized insights!"
            ),
            action_tag="None",
            why="No portfolio connected.",
            risk_note="",
            confidence="high",
            detected_intent="no_portfolio",
        )

    reply = pipeline.run(
        user_id=user_id,
        message=message,
        portfolio_summary=portfolio_summary,
    )

    # Enrich reply with proactive insights and health score
    holdings = portfolio_summary.get("holdings", [])
    analytics = build_portfolio_analytics(
        holdings=holdings,
        total_invested=portfolio_summary.get("total_invested", 0.0),
        total_current_value=portfolio_summary.get("total_current_value", 0.0),
        total_pnl=portfolio_summary.get("total_pnl", 0.0),
        total_pnl_pct=portfolio_summary.get("total_pnl_pct", 0.0),
    )
    reply.proactive_insights = generate_proactive_insights(
        holdings=holdings,
        total_invested=portfolio_summary.get("total_invested", 0.0),
        total_current_value=portfolio_summary.get("total_current_value", 0.0),
        total_pnl=portfolio_summary.get("total_pnl", 0.0),
        total_pnl_pct=portfolio_summary.get("total_pnl_pct", 0.0),
        analytics=analytics,
    )
    reply.portfolio_health = analytics.get("portfolio_health")
    return reply


def chat_proactive_insights(user_id: str) -> list[dict]:
    """Return proactive insights without requiring a chat message."""
    portfolio_summary = _fetch_portfolio_summary(user_id)
    holdings = portfolio_summary.get("holdings", [])
    analytics = build_portfolio_analytics(
        holdings=holdings,
        total_invested=portfolio_summary.get("total_invested", 0.0),
        total_current_value=portfolio_summary.get("total_current_value", 0.0),
        total_pnl=portfolio_summary.get("total_pnl", 0.0),
        total_pnl_pct=portfolio_summary.get("total_pnl_pct", 0.0),
    )
    return generate_proactive_insights(
        holdings=holdings,
        total_invested=portfolio_summary.get("total_invested", 0.0),
        total_current_value=portfolio_summary.get("total_current_value", 0.0),
        total_pnl=portfolio_summary.get("total_pnl", 0.0),
        total_pnl_pct=portfolio_summary.get("total_pnl_pct", 0.0),
        analytics=analytics,
    )


def clear_chat_session(user_id: str) -> None:
    pipeline.clear_session(user_id)

