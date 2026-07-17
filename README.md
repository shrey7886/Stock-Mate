# Stock-Mate

Stock-Mate is an AI-powered portfolio dashboard that connects to your Zerodha brokerage account and gives you a live view of your holdings, an AI chat assistant for portfolio questions, and market analytics — performance vs. NIFTY 50, sector allocation with concentration alerts, per-stock fundamentals, a market overview widget, and more.

## How It Works

1. **Connect your portfolio** — link your Zerodha account via Kite Connect OAuth
2. **Live holdings** — the dashboard pulls your current holdings, P&L, and portfolio health score directly from Zerodha
3. **Market analytics** — portfolio performance is benchmarked against NIFTY 50, holdings are broken down by sector with over-concentration warnings, and a market overview widget shows NIFTY 50/SENSEX levels plus your top gainers/losers
4. **Stock financials** — click any holding to see its fundamentals (P/E, market cap, dividend yield, 52-week range, beta) alongside your own position in that stock
5. **AI chat** — an LLM-backed assistant (Groq or OpenAI) answers portfolio questions with buy/hold/trim guidance and proactive insights

## Architecture

| Component | Description |
|---|---|
| `backend_api` | FastAPI REST API — auth, portfolio, Zerodha OAuth, chat, market data |
| `llm_orchestrator` | LLM reasoning layer behind the chat assistant |
| `frontend` | React + Vite web dashboard |

Data is stored in a local SQLite database (`backend_api/database/backend.db`, created automatically on first run) — no external database is required.

## Features

- **Portfolio summary** — live holdings, invested value, P&L, portfolio health score
- **Portfolio vs NIFTY 50 benchmark** — indexed performance chart (1M/3M/6M/1Y) comparing your current holdings against the NIFTY 50, using historical prices via yfinance
- **Sector allocation** — donut chart of holdings by sector (via yfinance), with a warning banner if any sector exceeds 40% of the portfolio
- **Stock financials panel** — click any holding to view its fundamentals (P/E ratio, market cap, dividend yield, 52-week high/low, beta) fetched from yfinance and cached daily
- **Market overview widget** — live NIFTY 50 and SENSEX levels with day change, plus your top 5 gaining and losing holdings
- **AI chat assistant** — natural-language portfolio Q&A with action tags (Hold/Trim/Add/Watch/Rebalance) and proactive insights
- **Zerodha account linking** — OAuth-based linking/unlinking, multiple accounts, primary account selection

## Prerequisites

Install these before running the app:

- **Python 3.10+** (required — `backend_api/models/schemas.py` uses `X | None` type syntax that is not supported on Python 3.9)
- **Node.js 18+** and npm (for the Vite/React frontend)
- A **Zerodha Kite Connect** developer app (API key + secret) — [https://developers.kite.trade](https://developers.kite.trade) — needed to link a real brokerage account
- Optional: a **Groq** or **OpenAI** API key to power the AI chat assistant (the app defaults to Groq, which has a free tier)

No Docker and no external database are required to run the app locally.

## Setup

### 1. Backend

```bash
cd backend_api
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root (or `backend_api/`) with at least:

```bash
# App
APP_HOST=0.0.0.0
APP_PORT=8000
FRONTEND_URL=http://localhost:5174

# Auth (change these in production)
JWT_SECRET=change-me-in-prod
BROKER_TOKEN_SECRET=change-me-too
OAUTH_STATE_HMAC_SECRET=replace-state-hmac-secret

# Zerodha Kite Connect
ZERODHA_API_KEY=your-kite-api-key
ZERODHA_API_SECRET=your-kite-api-secret
ZERODHA_REDIRECT_URL=http://localhost:8000/api/zerodha/callback

# LLM provider for the chat assistant: "groq" (free) or "openai"
LLM_PROVIDER=groq
GROQ_API_KEY=your-groq-api-key
# OPENAI_API_KEY=your-openai-api-key
```

Run the backend:

```bash
python main.py
```

The API is served at `http://localhost:8000` (docs at `/docs`).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard is served at `http://localhost:5174` and proxies API calls to the backend.

## Market data notes

- Historical/benchmark, sector-allocation, fundamentals, and index data are all fetched live from **Yahoo Finance** via `yfinance` (NSE symbols first, BSE as a fallback) — no API key required, but it does need outbound internet access.
- Sector classifications are cached in SQLite and refreshed at most once a week per symbol; stock fundamentals (P/E, market cap, etc.) are cached and refreshed at most once a day per symbol, to minimize external calls.
- If Yahoo Finance is temporarily unreachable, the affected endpoints degrade gracefully (empty chart/panel/`data_status: market_data_unavailable`) instead of failing the whole dashboard.

## Project Structure
```
Stock-Mate/
├── backend_api/         # FastAPI backend (auth, portfolio, Zerodha, chat, market data)
│   ├── routes/          # API route modules
│   ├── services/        # Zerodha, broker token, and market data services
│   ├── database/        # SQLite access layer
│   └── models/          # Pydantic request/response schemas
├── llm_orchestrator/    # LLM chat/recommendation logic
├── frontend/            # React + Vite dashboard
└── tests/               # Test suite
```
