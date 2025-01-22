# Base image
FROM python:3.12.5-slim

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /CrewAI-Studio

# Copy only requirements first to leverage Docker caching
COPY requirements.txt ./

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files (after dependencies to leverage caching)
COPY ./ /CrewAI-Studio/

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH="/CrewAI-Studio/app"

# Expose necessary ports
EXPOSE 8501 8000

# Start both Streamlit and FastAPI
CMD ["sh", "-c", "streamlit run ./app/app.py --server.port=8501 & uvicorn app:api_app --host 0.0.0.0 --port 8000 --reload"]
