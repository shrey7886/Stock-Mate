from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND

from backend_api.core.config import settings
from backend_api.core.security import get_current_user
from backend_api.models.schemas import (
    ActionResponse,
    ZerodhaStartResponse,
    ZerodhaStatusResponse,
)
from backend_api.services.broker_token_service import broker_token_service
from backend_api.services.zerodha_service import zerodha_service

router = APIRouter(prefix="/api/zerodha", tags=["zerodha"])

FRONTEND_BROKER_URL = "http://localhost:5173/broker"


@router.get("/start", response_model=ZerodhaStartResponse)
def zerodha_start(current_user: dict = Depends(get_current_user)) -> ZerodhaStartResponse:
    login_url, state = zerodha_service.build_login_url()
    broker_token_service.save_oauth_state(
        user_id=str(current_user.get("sub")),
        provider="zerodha",
        state=state,
        ttl_minutes=settings.oauth_state_ttl_minutes,
    )
    return ZerodhaStartResponse(login_url=login_url, state=state)


@router.get("/callback")
def zerodha_callback(
    request_token: str = Query(...),
    state: str | None = Query(default=None),
):
    if state:
        state_user_id = broker_token_service.consume_oauth_state(
            provider="zerodha",
            state=state,
        )
    else:
        state_user_id = broker_token_service.consume_latest_oauth_state(provider="zerodha")

    if not state_user_id:
        params = urlencode({"error": "Invalid or expired state"})
        return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)

    try:
        token_data = zerodha_service.exchange_request_token(request_token=request_token)
        broker_token_service.save_zerodha_tokens(
            user_id=str(state_user_id),
            account_id=token_data.get("user_id"),
            access_token=token_data.get("access_token", ""),
            public_token=token_data.get("public_token"),
        )
    except Exception:
        params = urlencode({"error": "Token exchange failed"})
        return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)

    params = urlencode({
        "success": "true",
        "account_id": token_data.get("user_id", ""),
    })
    return RedirectResponse(url=f"{FRONTEND_BROKER_URL}?{params}", status_code=HTTP_302_FOUND)


@router.get("/status", response_model=ZerodhaStatusResponse)
def zerodha_status(current_user: dict = Depends(get_current_user)) -> ZerodhaStatusResponse:
    accounts = broker_token_service.list_linked_accounts(
        user_id=str(current_user.get("sub")),
        provider="zerodha",
    )
    if not accounts:
        return ZerodhaStatusResponse(linked=False)

    return ZerodhaStatusResponse(
        linked=True,
        linked_accounts_count=len(accounts),
        accounts=accounts,
    )


@router.post("/unlink", response_model=ActionResponse)
def zerodha_unlink(
    current_user: dict = Depends(get_current_user),
    account_id: str | None = Query(default=None),
) -> ActionResponse:
    removed = broker_token_service.unlink_account(
        user_id=str(current_user.get("sub")),
        provider="zerodha",
        account_id=account_id,
    )
    if not removed:
        return ActionResponse(success=True, message="No linked Zerodha account found")

    if account_id:
        return ActionResponse(success=True, message=f"Zerodha account '{account_id}' unlinked")

    return ActionResponse(success=True, message="All Zerodha accounts unlinked")


@router.get("/accounts")
def zerodha_accounts(current_user: dict = Depends(get_current_user)) -> dict:
    accounts = broker_token_service.list_linked_accounts(
        user_id=str(current_user.get("sub")),
        provider="zerodha",
    )
    return {
        "provider": "zerodha",
        "linked": bool(accounts),
        "count": len(accounts),
        "accounts": accounts,
    }


@router.post("/accounts/primary", response_model=ActionResponse)
def zerodha_set_primary_account(
    account_id: str = Query(...),
    current_user: dict = Depends(get_current_user),
) -> ActionResponse:
    updated = broker_token_service.set_primary_account(
        user_id=str(current_user.get("sub")),
        provider="zerodha",
        account_id=account_id,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Zerodha account not found for user")

    return ActionResponse(success=True, message=f"Primary Zerodha account set to '{account_id}'")
