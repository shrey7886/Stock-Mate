# StockMate Backend API

Production-ready FastAPI backend for StockMate stock prediction platform.

## 🏗️ Architecture

- **Framework:** FastAPI
- **Database:** PostgreSQL + TimescaleDB
- **Authentication:** JWT
- **API Docs:** Auto-generated at `/docs`

## 📁 Project Structure

```
backend_api/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── api/
│   │   ├── routes/          # API endpoints
│   │   │   ├── health.py    # Health check
│   │   │   ├── forecast.py  # Stock predictions
│   │   │   ├── auth.py      # Authentication
│   │   │   └── portfolio.py # Portfolio management
│   │   └── schemas/         # Request/response models
│   │       ├── request.py
│   │       └── response.py
│   ├── core/
│   │   ├── config.py        # Settings
│   │   └── security.py      # JWT & password hashing
│   ├── db/
│   │   ├── models.py        # SQLAlchemy models
│   │   └── session.py       # Database connection
│   └── services/            # Business logic (future)
├── requirements.txt
├── .env.example
└── API_CONTRACT.md          # API documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend_api
pip install -r requirements.txt
```

### 2. Set Up Environment

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Set Up Database

**Option A: Local PostgreSQL**
```bash
# Install PostgreSQL and TimescaleDB extension
createdb stockmate
psql stockmate -c "CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"
```

**Option B: Timescale Cloud (Recommended)**
1. Sign up at https://www.timescale.com/
2. Create a new service
3. Copy connection string to `.env`

### 4. Run the Server

```bash
# Development mode (auto-reload)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python
python -m app.main
```

### 5. Access API

- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## 📡 API Endpoints

### Public Endpoints
- `GET /health` - Health check
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login and get JWT token

### Protected Endpoints (Require JWT)
- `GET /api/v1/auth/me` - Get current user info
- `POST /api/v1/forecast` - Get stock predictions
- `POST /api/v1/portfolio` - Create portfolio
- `GET /api/v1/portfolio` - List portfolios
- `POST /api/v1/portfolio/{id}/holdings` - Add stock to portfolio

See [API_CONTRACT.md](API_CONTRACT.md) for full documentation.

## 🔐 Authentication

All protected endpoints require a JWT token in the Authorization header:

```bash
# 1. Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!","full_name":"John Doe"}'

# 2. Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!"}'

# 3. Use token
curl -X POST http://localhost:8000/api/v1/forecast \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","horizon":7}'
```

## 🗄️ Database Schema

### Regular Tables
- **users** - User accounts
- **portfolios** - User portfolios
- **holdings** - Stock positions

### Hypertables (TimescaleDB)
- **stock_prices** - Historical price data
- **predictions** - Cached forecasts

## 🔧 Configuration

Edit `.env` file:

```env
DATABASE_URL=postgresql://user:pass@host:port/dbname
JWT_SECRET_KEY=your-secret-key
DEBUG=True
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=app tests/
```

## 📦 Deployment

See [API_CONTRACT.md](API_CONTRACT.md) Phase 8 for deployment instructions.

## 🛣️ Roadmap

- [x] Phase 0: Architecture & Contracts
- [x] Phase 1: Backend Skeleton
- [x] Phase 2: Forecast API (Stub)
- [x] Phase 3: Authentication & Users
- [x] Phase 4: Portfolio Management
- [ ] Phase 5: ML Integration (Real TFT)
- [ ] Phase 6: Performance & Caching
- [ ] Phase 7: Observability & Metrics
- [ ] Phase 8: Deployment & CI/CD

## 📄 License

MIT
