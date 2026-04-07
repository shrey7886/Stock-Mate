from __future__ import annotations

from fastapi import APIRouter, Depends

from backend_api.core.security import get_current_user

router = APIRouter(prefix="/api/user", tags=["user"])


@router.get("/me")
def me(current_user: dict = Depends(get_current_user)) -> dict:
    return {"user": current_user}
