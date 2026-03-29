# Stock-Mate

Stock-Mate is an AI-powered portfolio advisory platform that connects to your brokerage account, analyses sentiment from Twitter and Reddit, and generates personalised buy/sell recommendations based on real-time social sentiment around stocks in your portfolio.

## How It Works

1. **Connect your portfolio** — link your Zerodha or Groww account
2. **Sentiment analysis** — the platform continuously monitors Twitter and Reddit for discussions around your portfolio stocks
3. **ML scoring** — sentiment signals are fed into ML models that score risk and momentum for each holding
4. **LLM reasoning** — an LLM layer synthesises signals into plain-language buy/sell/hold recommendations
5. **Dashboard** — view recommendations and sentiment trends through the web interface

## Architecture

The system is composed of independent services orchestrated via Docker:

| Service | Description |
|---|---|
| `sentiment_service` | Scrapes and analyses Twitter/Reddit posts for stock sentiment |
| `ml_service` | ML models for scoring sentiment signals and predicting trends |
| `llm_orchestrator` | Generates natural language buy/sell recommendations |
| `data_pipeline` | Ingests portfolio data and social media feeds |
| `backend_api` | REST API connecting all services to the frontend |
| `frontend` | Web dashboard for portfolio insights and recommendations |

## Tech Stack

- **Backend:** Python, REST API
- **ML/NLP:** Sentiment analysis, ensemble ML models
- **LLM:** Natural language recommendation generation
- **Orchestration:** Docker, docker-compose
- **Integrations:** Zerodha API, Groww API, Twitter API, Reddit API

## Project Structure
```
Stock-Mate/
├── backend_api/        # REST API layer
├── sentiment_service/  # Twitter/Reddit sentiment pipeline
├── ml_service/         # ML scoring and trend models
├── llm_orchestrator/   # LLM recommendation engine
├── data_pipeline/      # Portfolio and social data ingestion
├── frontend/           # Web dashboard
├── configs/            # Configuration files
├── scripts/            # Utility scripts
├── tests/              # Test suite
└── docker-compose.yml  # Service orchestration
```
