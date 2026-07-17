from __future__ import annotations

import secrets
from urllib.parse import urlencode

import requests

from backend_api.core.config import settings


class UpstoxService:
    AUTHORIZE_URL = "https://api.upstox.com/v2/login/authorization/dialog"
    TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
    API_BASE = "https://api.upstox.com/v2"

    def build_login_url(self) -> tuple[str, str]:
        state = secrets.token_urlsafe(24)
        params = {
            "response_type": "code",
            "client_id": settings.upstox_api_key,
            "redirect_uri": settings.upstox_redirect_url,
            "state": state,
        }
        return f"{self.AUTHORIZE_URL}?{urlencode(params)}", state

    def exchange_code(self, code: str) -> dict:
        if not settings.upstox_api_key or not settings.upstox_api_secret:
            raise ValueError("UPSTOX_API_KEY / UPSTOX_API_SECRET are not configured")

        payload = {
            "code": code,
            "client_id": settings.upstox_api_key,
            "client_secret": settings.upstox_api_secret,
            "redirect_uri": settings.upstox_redirect_url,
            "grant_type": "authorization_code",
        }
        response = requests.post(
            self.TOKEN_URL,
            data=payload,
            timeout=20,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("access_token"):
            raise ValueError(data.get("error_description") or data.get("errors") or "Failed to exchange Upstox code")

        return data

    def _headers(self, access_token: str) -> dict:
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

    def fetch_profile(self, access_token: str) -> dict:
        resp = requests.get(
            f"{self.API_BASE}/user/profile",
            headers=self._headers(access_token),
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise ValueError(payload.get("errors") or "Upstox profile fetch failed")
        return payload.get("data", {})

    def fetch_holdings(self, access_token: str) -> list[dict]:
        resp = requests.get(
            f"{self.API_BASE}/portfolio/long-term-holdings",
            headers=self._headers(access_token),
            timeout=20,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise ValueError(payload.get("errors") or "Upstox holdings fetch failed")

        raw_holdings = payload.get("data", [])
        if not isinstance(raw_holdings, list):
            return []

        normalized = []
        for item in raw_holdings:
            normalized.append({
                "tradingsymbol": item.get("tradingsymbol") or item.get("trading_symbol"),
                "exchange": item.get("exchange"),
                "quantity": item.get("quantity"),
                "average_price": item.get("average_price"),
                "last_price": item.get("last_price"),
                "pnl": item.get("pnl"),
                "day_change": item.get("day_change"),
                "day_change_percentage": item.get("day_change_percentage"),
                "isin": item.get("isin"),
            })
        return normalized

    def is_relink_required_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        relink_markers = [
            "invalid token",
            "token has expired",
            "expired",
            "invalid api_key or access_token",
            "authentication failed",
            "permission denied",
            "unauthorized",
            "udapi100050",
            "401",
        ]
        return any(marker in text for marker in relink_markers)


upstox_service = UpstoxService()
