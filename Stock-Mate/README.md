# StockMate AI - Microservices Architecture

A comprehensive AI-powered portfolio management system built with a microservices architecture, featuring Temporal Fusion Transformer (TFT) models for stock prediction and LangChain-based conversational AI.

## 🏗️ Architecture Overview

StockMate AI is designed as a collection of microservices that work together to provide intelligent portfolio management and financial advisory services:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Service    │
│   (React)       │◄──►│   (FastAPI)     │◄──►│   (PyTorch)     │
│   Port: 3000    │    │   Port: 8000    │    │   Port: 8001    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ LLM Orchestrator│    │   Database      │    │   Redis Cache   │
│   (LangChain)   │    │  (PostgreSQL)   │    │   (Sessions)    │
│   Port: 8002    │    │   Port: 5432    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Services

### 1. Backend API (`backend_api/`)
- **Technology**: FastAPI, SQLAlchemy
- **Purpose**: Portfolio management, user authentication, data persistence
- **Features**:
  - RESTful API endpoints
  - User registration and authentication
  - Portfolio CRUD operations
  - Stock position management
  - Integration with ML service

### 2. ML Service (`ml_service/`)
- **Technology**: PyTorch, PyTorch Lightning, PyTorch Forecasting
- **Purpose**: Stock price prediction using Temporal Fusion Transformer
- **Features**:
  - TFT model training and inference
  - Technical indicator calculation
  - Time series data preprocessing
  - Model evaluation and metrics

### 3. LLM Orchestrator (`llm_orchestrator/`)
- **Technology**: LangChain, OpenAI GPT
- **Purpose**: Conversational AI and service coordination
- **Features**:
  - Natural language processing
  - Tool integration for backend services
  - Context-aware conversations
  - Portfolio insights generation

### 4. Frontend Web (`frontend_web/`)
- **Technology**: React, CSS3
- **Purpose**: User interface and chat experience
- **Features**:
  - Real-time chat interface
  - Portfolio visualization
  - Responsive design
  - Modern UI/UX

### 5. Infrastructure (`infra/`)
- **Technology**: Docker, Kubernetes, Nginx
- **Purpose**: Deployment and orchestration
- **Features**:
  - Docker containerization
  - Kubernetes deployment manifests
  - Load balancing and reverse proxy
  - Health checks and monitoring

## 🛠️ Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Node.js 18+
- CUDA-capable GPU (for ML service)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stock-Mate
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start all services with Docker Compose**
   ```bash
   cd infra
   docker-compose up -d
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - ML Service: http://localhost:8001
   - LLM Orchestrator: http://localhost:8002

### Individual Service Development

#### Backend API
```bash
cd backend_api
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

#### ML Service
```bash
cd ml_service
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook model_tft/train_model.ipynb
```

#### LLM Orchestrator
```bash
cd llm_orchestrator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

#### Frontend
```bash
cd frontend_web
npm install
npm start
```

## 📊 Key Features

### AI-Powered Stock Prediction
- **Temporal Fusion Transformer (TFT)** for accurate time series forecasting
- Technical indicators: RSI, MACD, Bollinger Bands
- Multi-stock portfolio analysis
- Confidence intervals and risk assessment

### Conversational AI
- **LangChain-based** natural language processing
- Context-aware portfolio discussions
- Real-time market analysis
- Personalized investment advice

### Microservices Architecture
- **Scalable** and maintainable design
- Independent service deployment
- API-first approach
- Containerized with Docker

### Modern Tech Stack
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **ML**: PyTorch, PyTorch Lightning, PyTorch Forecasting
- **AI**: LangChain, OpenAI GPT
- **Frontend**: React, Modern CSS
- **Infrastructure**: Docker, Kubernetes, Nginx

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API Key (required for LLM orchestrator)
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/stockmate

# Service URLs (for development)
BACKEND_API_URL=http://localhost:8000
ML_SERVICE_URL=http://localhost:8001
LLM_ORCHESTRATOR_URL=http://localhost:8002
```

### Docker Configuration
- All services are containerized
- GPU support for ML service
- Volume mounts for data persistence
- Health checks for reliability

### Kubernetes Deployment
- Production-ready manifests in `infra/k8s/`
- Horizontal pod autoscaling
- Resource limits and requests
- Ingress configuration for external access

## 📈 Usage Examples

### Portfolio Management
```python
# Create a new portfolio
POST /api/v1/portfolio/
{
    "name": "Growth Portfolio",
    "description": "Tech-focused growth investments",
    "initial_balance": 10000.0
}

# Add stock positions
POST /api/v1/portfolio/1/positions
{
    "symbol": "AAPL",
    "quantity": 10,
    "purchase_price": 150.00
}
```

### AI Chat Interface
```
User: "What's the performance of my portfolio?"
StockMate: "Your portfolio is up 12.5% this quarter. Your top performer is AAPL with a 18.2% gain..."

User: "Should I buy Tesla stock?"
StockMate: "Based on our analysis, TSLA shows strong technical indicators but high volatility..."
```

### ML Predictions
```python
# Get stock analysis and predictions
GET /ml/api/v1/analyze/AAPL
{
    "symbol": "AAPL",
    "current_price": 150.25,
    "prediction": {
        "next_30_days": {
            "predicted_price": 155.30,
            "confidence": 0.75,
            "trend": "bullish"
        }
    }
}
```

## 🧪 Testing

### Backend API Tests
```bash
cd backend_api
pytest tests/
```

### ML Model Evaluation
```bash
cd ml_service
python -m pytest tests/
```

### Frontend Tests
```bash
cd frontend_web
npm test
```

## 📚 Documentation

- **API Documentation**: Available at http://localhost:8000/docs when running
- **ML Model Documentation**: See `ml_service/model_tft/train_model.ipynb`
- **Architecture Decisions**: See individual service README files

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch Forecasting team for the TFT implementation
- LangChain team for the conversational AI framework
- FastAPI team for the excellent web framework
- The open-source community for various dependencies

## 📞 Support

For questions and support:
- Create an issue in the repository
- Check the documentation
- Review the example notebooks

---

**Note**: This is a demonstration project showcasing microservices architecture with AI/ML integration. For production use, additional security, monitoring, and scalability considerations should be implemented.
