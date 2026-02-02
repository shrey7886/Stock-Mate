# StockMate API Contract v1.0

**Last Updated:** January 15, 2026  
**Status:** Phase 0 - Architecture Definition

---

## 🏗️ Architecture Decisions

### Backend Stack
- **Framework:** FastAPI
- **Language:** Python 3.10+
- **API Style:** REST
- **Data Format:** JSON
- **Authentication:** JWT (JSON Web Tokens)
- **Database:** PostgreSQL + TimescaleDB extension
- **Hosting:** Timescale Cloud (free tier)

### ML Integration Strategy

**Phase 1 (MVP):** Direct Python Import
```python
from ml_service.inference.predict import predict_next_days
```

**Phase 2 (Production):** Microservice
```
Backend (FastAPI) → HTTP → ML Service (port 8001)
```

---

## 📡 API Endpoints Contract

### 1. Forecast Endpoint

**Purpose:** Get price predictions for a stock symbol

**Endpoint:** `POST /api/v1/forecast`

**Request:**
```json
{
  "symbol": "AAPL",
  "horizon": 7
}
```

**Response:**
```json
{
  "symbol": "AAPL",
  "forecast_date": "2026-01-15T10:30:00Z",
  "predictions": [
    {"day": 1, "date": "2026-01-16", "price": 182.4},
    {"day": 2, "date": "2026-01-17", "price": 184.1},
    {"day": 3, "date": "2026-01-18", "price": 183.8},
    {"day": 4, "date": "2026-01-19", "price": 185.2},
    {"day": 5, "date": "2026-01-20", "price": 186.0},
    {"day": 6, "date": "2026-01-21", "price": 185.5},
    {"day": 7, "date": "2026-01-22", "price": 187.3}
  ],
  "confidence": {
    "p10": [180.1, 181.5, 180.9, 182.1, 183.0, 182.3, 183.8],
    "p50": [182.4, 184.1, 183.8, 185.2, 186.0, 185.5, 187.3],
    "p90": [184.9, 186.8, 186.5, 188.2, 189.1, 188.9, 191.0]
  },
  "model_version": "tft_v1.0"
}
```

**Status Codes:**
- `200` - Success
- `400` - Invalid symbol or horizon
- `401` - Unauthorized
- `500` - Model inference error

---

### 2. Authentication Endpoints

#### Register
**Endpoint:** `POST /api/v1/auth/register`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "full_name": "John Doe"
}
```

**Response:**
```json
{
  "id": "uuid-here",
  "email": "user@example.com",
  "full_name": "John Doe",
  "created_at": "2026-01-15T10:30:00Z"
}
```

#### Login
**Endpoint:** `POST /api/v1/auth/login`

**Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

---

### 3. Portfolio Endpoints

#### Create Portfolio
**Endpoint:** `POST /api/v1/portfolio`

**Headers:** `Authorization: Bearer <token>`

**Request:**
```json
{
  "name": "My Tech Portfolio"
}
```

**Response:**
```json
{
  "id": "uuid-here",
  "name": "My Tech Portfolio",
  "created_at": "2026-01-15T10:30:00Z"
}
```

#### Add Stock to Portfolio
**Endpoint:** `POST /api/v1/portfolio/{portfolio_id}/holdings`

**Request:**
```json
{
  "symbol": "AAPL",
  "quantity": 10,
  "avg_price": 180.50
}
```

---

### 4. Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-15T10:30:00Z",
  "version": "1.0.0",
  "services": {
    "database": "connected",
    "ml_model": "loaded"
  }
}
```

---

## 🗄️ Database Schema (TimescaleDB)

### Hypertables (Time-series optimized)

```sql
-- Stock prices (historical)
CREATE TABLE stock_prices (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume BIGINT,
    PRIMARY KEY (timestamp, symbol)
);
SELECT create_hypertable('stock_prices', 'timestamp');

-- Predictions (cached forecasts)
CREATE TABLE predictions (
    created_at TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    horizon INT NOT NULL,
    predictions JSONB NOT NULL,
    confidence JSONB NOT NULL,
    model_version TEXT,
    PRIMARY KEY (created_at, symbol)
);
SELECT create_hypertable('predictions', 'created_at');
```

### Regular Tables

```sql
-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Holdings
CREATE TABLE holdings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID REFERENCES portfolios(id),
    symbol TEXT NOT NULL,
    quantity NUMERIC NOT NULL,
    avg_price NUMERIC NOT NULL,
    added_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 🔒 Security

- **Password Hashing:** bcrypt
- **JWT Secret:** Environment variable `JWT_SECRET_KEY`
- **Token Expiration:** 1 hour (configurable)
- **CORS:** Configured for frontend origin
- **Rate Limiting:** 100 requests/minute per IP (Phase 6)

---

## 🚀 Environment Variables

```env
# Database
DATABASE_URL=postgresql://user:pass@host:port/dbname
TIMESCALE_ENABLED=true

# Security
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

# API
API_V1_PREFIX=/api/v1
PROJECT_NAME=StockMate
VERSION=1.0.0

# ML Model
ML_MODEL_PATH=../ml_service/models/tft_final.ckpt
ML_SERVICE_URL=http://localhost:8001  # For microservice mode
```

---

## 📦 Dependencies

```
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6
pydantic>=2.0.0
pydantic-settings>=2.0.0
```

---

## ✅ Implementation Phases

- [x] **Phase 0:** Architecture & Contracts (This document)
- [ ] **Phase 1:** Backend Skeleton (FastAPI setup)
- [ ] **Phase 2:** Forecast API (Stub)
- [ ] **Phase 3:** Authentication & Users
- [ ] **Phase 4:** Portfolio & Watchlist
- [ ] **Phase 5:** ML Integration (Real TFT)
- [ ] **Phase 6:** Performance & Caching
- [ ] **Phase 7:** Observability & Metrics
- [ ] **Phase 8:** Deployment & CI/CD

---

**Next Step:** Proceed to Phase 1 - Build FastAPI skeleton with folder structure
