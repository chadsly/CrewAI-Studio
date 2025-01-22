# Baseimage
FROM python:3.12.5-slim

# Install build tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy CrewAI-Studio
RUN mkdir /CrewAI-Studio
COPY ./ /CrewAI-Studio/

# Set working directory
WORKDIR /CrewAI-Studio

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH="/CrewAI-Studio/app"

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8501 8000

# Start both Streamlit and FastAPI
CMD ["sh", "-c", "streamlit run ./app/app.py --server.port=8501 & uvicorn app:api_app --host 0.0.0.0 --port 8000 --reload"]
