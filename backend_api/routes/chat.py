from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend_api.core.security import get_current_user
from backend_api.models.schemas import (
    ChatRequest,
    ChatResponse,
    GoalResponse,
    GoalSetRequest,
    WatchlistAddRequest,
    WatchlistResponse,
)
from backend_api.services.chat_service import chat, chat_proactive_insights, clear_chat_session

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("/message")
def chat_message(payload: ChatRequest, current_user: dict = Depends(get_current_user)) -> ChatResponse:
    """
    Main chatbot endpoint. Supports 14 intents including watchlist, goals,
    tax analysis, SIP advice, portfolio health score, and market overview.
    """
    user_id = current_user.get("sub")
    reply = chat(user_id=user_id, message=payload.message)

    # Extract health score from analytics if available (passed via reply metadata)
    return ChatResponse(
        user_id=user_id,
        message=payload.message,
        answer=reply.answer,
        action_tag=reply.action_tag,
        why=reply.why,
        risk_note=reply.risk_note,
        confidence=reply.confidence,
        detected_intent=reply.detected_intent,
        confidence_score=reply.confidence_score,
        next_steps=reply.next_steps or [],
        proactive_insights=reply.proactive_insights or [],
        portfolio_health=reply.portfolio_health,
    )


@router.post("/clear-session")
def clear_session(current_user: dict = Depends(get_current_user)) -> dict:
    """Clear conversation history for the current user."""
    user_id = current_user.get("sub")
    clear_chat_session(user_id)
    return {"user_id": user_id, "message": "Session cleared."}


@router.get("/proactive-insights")
def get_proactive_insights(current_user: dict = Depends(get_current_user)) -> dict:
    """
    Returns live proactive portfolio insights (risk alerts, profit-booking opportunities,
    health score nudges) without requiring a chat message.
    """
    user_id = current_user.get("sub")
    insights = chat_proactive_insights(user_id=user_id)
    return {"user_id": user_id, "insights": insights}


# ── Watchlist endpoints ─────────────────────────────────────────────────────

@router.post("/watchlist/add")
def watchlist_add(
    payload: WatchlistAddRequest,
    current_user: dict = Depends(get_current_user),
) -> WatchlistResponse:
    """Add a ticker to the user's watchlist."""
    from llm_orchestrator.memory.watchlist_store import watchlist_store

    user_id = current_user.get("sub")
    watchlist_store.add(user_id, payload.symbol)
    wl = watchlist_store.get(user_id)
    return WatchlistResponse(user_id=user_id, watchlist=wl, count=len(wl))


@router.delete("/watchlist/remove/{symbol}")
def watchlist_remove(
    symbol: str,
    current_user: dict = Depends(get_current_user),
) -> WatchlistResponse:
    """Remove a ticker from the user's watchlist."""
    from llm_orchestrator.memory.watchlist_store import watchlist_store

    user_id = current_user.get("sub")
    watchlist_store.remove(user_id, symbol)
    wl = watchlist_store.get(user_id)
    return WatchlistResponse(user_id=user_id, watchlist=wl, count=len(wl))


@router.get("/watchlist")
def watchlist_get(current_user: dict = Depends(get_current_user)) -> WatchlistResponse:
    """Get the user's current watchlist."""
    from llm_orchestrator.memory.watchlist_store import watchlist_store

    user_id = current_user.get("sub")
    wl = watchlist_store.get(user_id)
    return WatchlistResponse(user_id=user_id, watchlist=wl, count=len(wl))


# ── Goal endpoints ──────────────────────────────────────────────────────────

@router.post("/goals/set")
def goal_set(
    payload: GoalSetRequest,
    current_user: dict = Depends(get_current_user),
) -> GoalResponse:
    """Set or update the user's investment goal and get SIP calculation."""
    from llm_orchestrator.memory.goal_store import goal_store
    from llm_orchestrator.utils.portfolio_analytics import calculate_sip_for_goal
    from backend_api.services.chat_service import _fetch_portfolio_summary

    user_id = current_user.get("sub")
    portfolio = _fetch_portfolio_summary(user_id)
    current_amount = portfolio.get("total_current_value", 0.0)

    goal = goal_store.set_goal(
        user_id,
        target_amount=payload.target_amount,
        current_amount=current_amount,
        years=payload.years,
        label=payload.label,
        expected_return_pct=payload.expected_return_pct,
    )
    sip = calculate_sip_for_goal(
        target_amount=payload.target_amount,
        current_amount=current_amount,
        years=payload.years,
        expected_annual_return_pct=payload.expected_return_pct,
    )
    progress = (
        min(100.0, (current_amount / payload.target_amount) * 100.0)
        if payload.target_amount > 0 else 0.0
    )
    return GoalResponse(
        user_id=user_id,
        label=goal.label,
        target_amount=goal.target_amount,
        current_amount=round(current_amount, 2),
        years=goal.years,
        progress_pct=round(progress, 2),
        monthly_sip_needed=sip["monthly_sip_needed"],
        sip_note=sip["note"],
    )


@router.get("/goals")
def goal_get(current_user: dict = Depends(get_current_user)) -> dict:
    """Get the user's active investment goal."""
    from llm_orchestrator.memory.goal_store import goal_store
    from llm_orchestrator.utils.portfolio_analytics import calculate_sip_for_goal
    from backend_api.services.chat_service import _fetch_portfolio_summary

    user_id = current_user.get("sub")
    goal = goal_store.get_goal(user_id)
    if not goal:
        return {"user_id": user_id, "goal": None, "message": "No active goal. Set one with POST /api/chat/goals/set"}

    portfolio = _fetch_portfolio_summary(user_id)
    current_amount = portfolio.get("total_current_value", 0.0)
    goal_store.update_current_amount(user_id, current_amount)

    sip = calculate_sip_for_goal(
        target_amount=goal.target_amount,
        current_amount=current_amount,
        years=goal.years,
        expected_annual_return_pct=goal.expected_return_pct,
    )
    return {
        "user_id": user_id,
        "goal": {
            "label": goal.label,
            "target_amount": goal.target_amount,
            "current_amount": round(current_amount, 2),
            "years": goal.years,
            "progress_pct": round(
                min(100.0, current_amount / goal.target_amount * 100.0), 2
            ) if goal.target_amount > 0 else 0.0,
            "monthly_sip_needed": sip["monthly_sip_needed"],
            "sip_note": sip["note"],
        },
    }


@router.delete("/goals")
def goal_clear(current_user: dict = Depends(get_current_user)) -> dict:
    """Clear the user's active goal."""
    from llm_orchestrator.memory.goal_store import goal_store

    user_id = current_user.get("sub")
    goal_store.clear_goal(user_id)
    return {"user_id": user_id, "message": "Goal cleared."}


