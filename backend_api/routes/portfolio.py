from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from backend_api.core.security import get_current_user
from backend_api.models.schemas import PortfolioSummaryResponse
from backend_api.services.broker_token_service import broker_token_service
from backend_api.services.zerodha_service import zerodha_service

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _to_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _aggregate_holdings(holdings: list[dict]) -> dict:
    total_current_value = 0.0
    total_invested = 0.0

    for item in holdings:
        quantity = _to_float(item.get("quantity"), default=0.0)
        average_price = _to_float(item.get("average_price"), default=0.0)
        last_price = _to_float(item.get("last_price"), default=0.0)

        total_current_value += quantity * last_price
        total_invested += quantity * average_price

    total_pnl = total_current_value - total_invested
    total_pnl_pct = (total_pnl / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        "holdings_count": len(holdings),
        "total_invested": round(total_invested, 2),
        "total_current_value": round(total_current_value, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
    }


def _compute_health_score(holdings: list[dict], aggregate: dict) -> float | None:
    try:
        from llm_orchestrator.utils.portfolio_analytics import build_portfolio_analytics
        analytics = build_portfolio_analytics(
            holdings=holdings,
            total_invested=aggregate["total_invested"],
            total_current_value=aggregate["total_current_value"],
            total_pnl=aggregate["total_pnl"],
            total_pnl_pct=aggregate["total_pnl_pct"],
        )
        health = analytics.get("portfolio_health", {})
        return health.get("score")
    except Exception:
        return None


@router.get("/summary")
def portfolio_summary(current_user: dict = Depends(get_current_user)) -> PortfolioSummaryResponse:
    user_id = str(current_user.get("sub"))
    linked = broker_token_service.has_linked_account(user_id=user_id, provider="zerodha")

    if not linked:
        return PortfolioSummaryResponse(
            user_id=user_id,
            linked=False,
            data_status="not_linked",
            action_required="link_broker",
            link_endpoint="/api/zerodha/start",
            message="Broker not linked. Start Zerodha linking flow.",
        )

    tokens = broker_token_service.get_linked_tokens(user_id=user_id, provider="zerodha")
    if not tokens or not tokens.get("access_token"):
        return PortfolioSummaryResponse(
            user_id=user_id,
            linked=True,
            account_id=tokens.get("account_id") if tokens else None,
            data_status="token_missing",
            action_required="relink_broker",
            link_endpoint="/api/zerodha/start",
            message="Broker linked, but token is unavailable. Reconnect Zerodha.",
        )

    try:
        holdings = zerodha_service.fetch_holdings(access_token=tokens.get("access_token", ""))
        aggregate = _aggregate_holdings(holdings)
        health_score = _compute_health_score(holdings, aggregate)
        return PortfolioSummaryResponse(
            user_id=user_id,
            linked=True,
            account_id=tokens.get("account_id"),
            data_status="live",
            message="Live portfolio summary from Zerodha.",
            holdings=holdings,
            health_score=health_score,
            **aggregate,
        )
    except Exception as exc:
        if zerodha_service.is_relink_required_error(exc):
            return PortfolioSummaryResponse(
                user_id=user_id,
                linked=True,
                account_id=tokens.get("account_id"),
                data_status="relink_required",
                action_required="relink_broker",
                link_endpoint="/api/zerodha/start",
                message="Broker token expired or invalid. Reconnect Zerodha.",
            )

        return PortfolioSummaryResponse(
            user_id=user_id,
            linked=True,
            account_id=tokens.get("account_id"),
            data_status="broker_api_unavailable",
            message=f"Broker linked, but live portfolio fetch is temporarily unavailable: {exc}",
        )



@router.get("/verify-broker")
def verify_broker_link(
    current_user: dict = Depends(get_current_user),
    account_id: str | None = Query(default=None),
) -> dict:
    user_id = str(current_user.get("sub"))
    tokens = broker_token_service.get_linked_tokens(
        user_id=user_id,
        provider="zerodha",
        account_id=account_id,
    )
    if not tokens:
        raise HTTPException(status_code=404, detail="No linked Zerodha account for user")

    access_token = tokens.get("access_token")
    if not access_token:
        raise HTTPException(status_code=400, detail="Missing access token for linked broker account")

    try:
        profile = zerodha_service.fetch_profile(access_token=access_token)
        holdings = zerodha_service.fetch_holdings(access_token=access_token)
    except Exception as exc:
        if zerodha_service.is_relink_required_error(exc):
            return {
                "linked": False,
                "action_required": "relink_broker",
                "link_endpoint": "/api/zerodha/start",
                "message": "Broker token expired or invalid. Reconnect Zerodha.",
                "error": str(exc),
            }
        return {
            "linked": True,
            "broker": "zerodha",
            "selected_account_id": tokens.get("account_id"),
            "is_primary": tokens.get("is_primary", False),
            "data_status": "broker_api_unavailable",
            "message": "Broker linked, but live verification is temporarily unavailable.",
            "error": str(exc),
        }

    sample = holdings[:5]
    return {
        "linked": True,
        "broker": "zerodha",
        "selected_account_id": tokens.get("account_id"),
        "is_primary": tokens.get("is_primary", False),
        "profile": {
            "user_id": profile.get("user_id"),
            "user_name": profile.get("user_name"),
            "email": profile.get("email"),
            "broker": profile.get("broker"),
        },
        "holdings_count": len(holdings),
        "holdings_preview": sample,
    }
