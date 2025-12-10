#!/bin/bash

# Root project folder


echo "Creating folder structure inside current repo..."


# -----------------------------
# 1. Backend API
# -----------------------------
mkdir -p backend_api/{routes,services,models,database,utils}

touch backend_api/app.py
touch backend_api/main.py

# -----------------------------
# 2. ML Service (TFT)
# -----------------------------
mkdir -p ml_service/{models/data,training,inference,utils}

mkdir -p ml_service/models/saved_models

touch ml_service/models/tft_model.py
touch ml_service/models/model_wrapper.py

touch ml_service/data/feature_engineering.py
touch ml_service/data/dataset_generator.py
touch ml_service/data/timeseries_dataset_builder.py

touch ml_service/training/train_tft.py
touch ml_service/training/trainer_config.yaml
touch ml_service/training/evaluation.py
touch ml_service/training/metrics.py

touch ml_service/inference/predict.py
touch ml_service/inference/forecast_service.py

touch ml_service/utils/scaler.py
touch ml_service/utils/logger.py

# -----------------------------
# 3. LLM Orchestrator
# -----------------------------
mkdir -p llm_orchestrator/{prompts,agents,pipelines,utils}

touch llm_orchestrator/prompts/summary_prompt.txt
touch llm_orchestrator/prompts/risk_prompt.txt
touch llm_orchestrator/prompts/portfolio_prompt.txt

touch llm_orchestrator/agents/response_agent.py
touch llm_orchestrator/agents/risk_agent.py

touch llm_orchestrator/pipelines/tft_to_text_pipeline.py
touch llm_orchestrator/pipelines/portfolio_explanation_pipeline.py

touch llm_orchestrator/utils/formatter.py
touch llm_orchestrator/utils/logger.py

# -----------------------------
# 4. Data Pipeline
# -----------------------------
mkdir -p data_pipeline/{ingestion,scheduler/tasks,database,utils}

touch data_pipeline/ingestion/yahoo_ingestor.py
touch data_pipeline/ingestion/news_ingestor.py
touch data_pipeline/ingestion/reddit_ingestor.py
touch data_pipeline/ingestion/twitter_ingestor.py

touch data_pipeline/scheduler/cron_jobs.py
touch data_pipeline/scheduler/celery_app.py
touch data_pipeline/scheduler/tasks/fetch_prices.py
touch data_pipeline/scheduler/tasks/fetch_news.py
touch data_pipeline/scheduler/tasks/fetch_sentiment.py

touch data_pipeline/database/write_prices.py
touch data_pipeline/database/write_news.py
touch data_pipeline/database/write_sentiment.py

touch data_pipeline/utils/cleaner.py
touch data_pipeline/utils/logger.py

# -----------------------------
# 5. Sentiment Service
# -----------------------------
mkdir -p sentiment_service/{models/finbert/tokenizer,models/finbert/model,inference,utils}

touch sentiment_service/inference/analyze_news.py
touch sentiment_service/inference/analyze_social.py

touch sentiment_service/utils/preprocessor.py
touch sentiment_service/utils/logger.py

# -----------------------------
# 6. Frontend
# -----------------------------
mkdir -p frontend/src/{components,pages,services}
mkdir -p frontend/public

touch frontend/src/services/api.js

touch frontend/src/components/ChatWindow.jsx
touch frontend/src/components/PortfolioCard.jsx
touch frontend/src/components/PredictionChart.jsx

touch frontend/src/pages/Home.jsx
touch frontend/src/pages/Portfolio.jsx

# -----------------------------
# 7. Configs
# -----------------------------
mkdir -p configs
touch configs/env.template
touch configs/docker.env
touch configs/db_config.yaml

# -----------------------------
# 8. Docker
# -----------------------------
mkdir -p docker

touch docker/backend_api.Dockerfile
touch docker/ml_service.Dockerfile
touch docker/llm_orchestrator.Dockerfile
touch docker/sentiment_service.Dockerfile
touch docker/data_pipeline.Dockerfile

touch docker-compose.yml

# -----------------------------
# 9. Scripts
# -----------------------------
mkdir -p scripts
touch scripts/run_all.sh
touch scripts/train_tft.sh
touch scripts/update_data.sh
touch scripts/deploy.sh

# -----------------------------
# 10. Tests
# -----------------------------
mkdir -p tests/{test_ml_service,test_backend,test_sentiment}

# -----------------------------
# Root files
# -----------------------------
touch README.md
touch requirements.txt

echo "Folder structure created successfully!"
