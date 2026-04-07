from __future__ import annotations

import hashlib
import secrets
from urllib.parse import urlencode

import requests

from backend_api.core.config import settings

try:
    from kiteconnect import KiteConnect
    from kiteconnect.exceptions import TokenException
    _HAS_KITE_SDK = True
except Exception:
    KiteConnect = None
    TokenException = None
    _HAS_KITE_SDK = False


class ZerodhaService:
    KITE_LOGIN_URL = "https://kite.zerodha.com/connect/login"
    KITE_SESSION_URL = "https://api.kite.trade/session/token"

    def build_login_url(self) -> tuple[str, str]:
        state = secrets.token_urlsafe(24)
        params = {
            "api_key": settings.zerodha_api_key,
            "v": 3,
            "state": state,
        }
        return f"{self.KITE_LOGIN_URL}?{urlencode(params)}", state

    def exchange_request_token(self, request_token: str) -> dict:
        if not settings.zerodha_api_key or not settings.zerodha_api_secret:
            raise ValueError("ZERODHA_API_KEY / ZERODHA_API_SECRET are not configured")

        if KiteConnect is not None:
            kite = KiteConnect(api_key=settings.zerodha_api_key)
            session_data = kite.generate_session(
                request_token=request_token,
                api_secret=settings.zerodha_api_secret,
            )
            if not isinstance(session_data, dict):
                raise ValueError("Kite session generation failed")
            return session_data

        checksum = hashlib.sha256(
            f"{settings.zerodha_api_key}{request_token}{settings.zerodha_api_secret}".encode("utf-8")
        ).hexdigest()

        payload = {
            "api_key": settings.zerodha_api_key,
            "request_token": request_token,
            "checksum": checksum,
        }

        response = requests.post(
            self.KITE_SESSION_URL,
            data=payload,
            timeout=20,
            headers={"X-Kite-Version": "3"},
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "success":
            raise ValueError(data.get("message", "Failed to exchange request_token"))

        return data.get("data", {})

    def _sdk_client(self, access_token: str):
        if KiteConnect is None:
            return None
        kite = KiteConnect(api_key=settings.zerodha_api_key)
        kite.set_access_token(access_token)
        return kite

    def _http_get(self, path: str, access_token: str) -> dict:
        """Fallback HTTP call when SDK is not available."""
        url = f"https://api.kite.trade{path}"
        headers = {
            "X-Kite-Version": "3",
            "Authorization": f"token {settings.zerodha_api_key}:{access_token}",
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            raise ValueError(data.get("message", f"Kite API error on {path}"))
        return data.get("data", {})

    def fetch_profile(self, access_token: str) -> dict:
        kite = self._sdk_client(access_token)
        if kite is not None:
            profile = kite.profile()
            if not isinstance(profile, dict):
                raise ValueError("Invalid profile response from Kite")
            return profile
        return self._http_get("/user/profile", access_token)

    def fetch_holdings(self, access_token: str) -> list[dict]:
        kite = self._sdk_client(access_token)
        if kite is not None:
            holdings = kite.holdings()
            if not isinstance(holdings, list):
                raise ValueError("Invalid holdings response from Kite")
            return holdings
        data = self._http_get("/portfolio/holdings", access_token)
        if isinstance(data, list):
            return data
        return []

    def is_relink_required_error(self, exc: Exception) -> bool:
        if TokenException is not None and isinstance(exc, TokenException):
            return True

        text = str(exc).lower()
        relink_markers = [
            "token is invalid",
            "token has expired",
            "expired",
            "invalid api_key or access_token",
            "authentication failed",
            "permission denied",
            "unauthorized",
            "inputexception",
        ]
        return any(marker in text for marker in relink_markers)


zerodha_service = ZerodhaService()
