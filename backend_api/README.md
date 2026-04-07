# Backend API

FastAPI backend scaffold aligned to your architecture goals:

- API layer (`app.py`) acting as gateway/router
- Auth service (`/api/auth`) issuing JWT
- User service (`/api/user`)
- Portfolio service (`/api/portfolio`)
- Inference service (`/api/inference`)
- Chat service (`/api/chat`)
- Broker integration (`/api/zerodha`) with encrypted token persistence

## Core Endpoints

- `POST /api/auth/login`
- `GET /api/health`
- `GET /api/user/me` (JWT required)
- `GET /api/portfolio/summary` (JWT required)
- `GET /api/portfolio/verify-broker` (JWT required, live Kite profile/holdings check)
	- Optional query: `account_id` to verify a specific linked Zerodha account.
	- Returns `action_required=relink_broker` when token is expired/invalid so frontend can prompt reconnect.
- `GET /api/zerodha/status` (JWT required)
- `GET /api/zerodha/accounts` (JWT required)
- `POST /api/zerodha/accounts/primary?account_id=...` (JWT required)
- `POST /api/zerodha/unlink` (JWT required)
	- Optional query: `account_id` to unlink one account; without it, all Zerodha accounts are unlinked.
- `GET /api/inference/status` (JWT required)
- `POST /api/chat/message` (JWT required)
- `GET /api/zerodha/start` (JWT required)
- `GET /api/zerodha/callback?request_token=...&state=...` (no JWT required; validated via one-time OAuth state)

## Security Model

- App auth: OAuth-style JWT bearer token at `Authorization: Bearer <token>`
- Broker auth: Zerodha OAuth callback + encrypted token storage (SQLite + Fernet scaffold)
- OAuth state: persisted in DB with TTL to protect callback flow
- OAuth state is HMAC-hashed at rest and consumed one-time during callback
- Callback is bound to stored state->user mapping (no browser auth header dependency)
- Kite token exchange uses official `kiteconnect` SDK when available; checksum fallback is also implemented.
- Production recommendation: replace SQLite/Fernet with managed DB + secret manager (AWS Secrets Manager/KMS)

## Environment

- `APP_NAME`, `APP_ENV`, `APP_HOST`, `APP_PORT`
- `ZERODHA_API_KEY`, `ZERODHA_API_SECRET`, `ZERODHA_REDIRECT_URL`
- `JWT_SECRET`, `JWT_ALGORITHM`, `JWT_EXP_MINUTES`
- `BROKER_TOKEN_SECRET`
- `OAUTH_STATE_TTL_MINUTES`

## Run

```bash
cd "c:/ML Projects/STOCKMATE/Stock-Mate"
"C:/ML Projects/STOCKMATE/.venv/Scripts/python.exe" -m pip install -r backend_api/requirements.txt
"C:/ML Projects/STOCKMATE/.venv/Scripts/python.exe" backend_api/main.py
```
