from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND

from backend_api.core.config import settings
from backend_api.core.security import get_current_user
from backend_api.models.schemas import (
    ActionResponse,
    UpstoxStartResponse,
    UpstoxStatusResponse,
)
from backend_api.services.broker_token_service import broker_token_service
from backend_api.services.upstox_service import upstox_service

router = APIRouter(prefix="/api/upstox", tags=["upstox"])

FRONTEND_BROKER_URL = f"{settings.frontend_url.rstrip('/')}/broker"


@router.get("/start", response_model=UpstoxStartResponse)
def upstox_start(current_user: dict = Depends(get_current_user)) -> UpstoxStartResponse:
    login_url, state = upstox_service.build_login_url()
    broker_token_service.save_oauth_state(
        user_id=str(current_user.get("sub")),
        provider="upstox",
        state=state,
        ttl_minutes=settings.oauth_state_ttl_minutes,
    )
    return UpstoxStartResponse(login_url=login_url, state=state)


@router.get("/callback")
def upstox_callback(
    code: str = Query(...),
    state: str | None = Query(default=None),
):
    if state:
        state_user_id = broker_token_service.consume_oauth_state(
            provider="upstox",
            state=state,
        )
    else:
        state_user_id = broker_token_service.consume_latest_oauth_state(provider="upstox")

    if not state_user_id:
        params = urlencode({"error": "Invalid or expired state", "provider": "upstox"})
        return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)

    try:
        token_data = upstox_service.exchange_code(code=code)
        broker_token_service.save_upstox_tokens(
            user_id=str(state_user_id),
            account_id=token_data.get("user_id"),
            access_token=token_data.get("access_token", ""),
        )
    except Exception:
        params = urlencode({"error": "Token exchange failed", "provider": "upstox"})
        return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)

    params = urlencode({
        "success": "true",
        "provider": "upstox",
        "account_id": token_data.get("user_id", ""),
    })
    return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)


@router.get("/status", response_model=UpstoxStatusResponse)
def upstox_status(current_user: dict = Depends(get_current_user)) -> UpstoxStatusResponse:
    accounts = broker_token_service.list_linked_accounts(
        user_id=str(current_user.get("sub")),
        provider="upstox",
    )
    if not accounts:
        return UpstoxStatusResponse(linked=False)

    return UpstoxStatusResponse(
        linked=True,
        linked_accounts_count=len(accounts),
        accounts=accounts,
    )


@router.post("/unlink", response_model=ActionResponse)
def upstox_unlink(
    current_user: dict = Depends(get_current_user),
    account_id: str | None = Query(default=None),
) -> ActionResponse:
    removed = broker_token_service.unlink_account(
        user_id=str(current_user.get("sub")),
        provider="upstox",
        account_id=account_id,
    )
    if not removed:
        return ActionResponse(success=True, message="No linked Upstox account found")

    if account_id:
        return ActionResponse(success=True, message=f"Upstox account '{account_id}' unlinked")

    return ActionResponse(success=True, message="All Upstox accounts unlinked")


@router.get("/accounts")
def upstox_accounts(current_user: dict = Depends(get_current_user)) -> dict:
    accounts = broker_token_service.list_linked_accounts(
        user_id=str(current_user.get("sub")),
        provider="upstox",
    )
    return {
        "provider": "upstox",
        "linked": bool(accounts),
        "count": len(accounts),
        "accounts": accounts,
    }


@router.post("/accounts/primary", response_model=ActionResponse)
def upstox_set_primary_account(
    account_id: str = Query(...),
    current_user: dict = Depends(get_current_user),
) -> ActionResponse:
    updated = broker_token_service.set_primary_account(
        user_id=str(current_user.get("sub")),
        provider="upstox",
        account_id=account_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Upstox account not found for user")

    return ActionResponse(success=True, message=f"Primary Upstox account set to '{account_id}'")
