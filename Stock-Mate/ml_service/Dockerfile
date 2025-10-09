# ML Service Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port for ML service API
EXPOSE 8001

# Run Jupyter notebook for development
# In production, this would run a FastAPI service
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8001", "--allow-root", "--no-browser"]
