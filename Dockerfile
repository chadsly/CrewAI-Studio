# # Baseimage
# FROM python:3.12.5-slim-bookworm

# # Update Packages
# RUN apt update
# RUN apt upgrade -y
# RUN pip install --upgrade pip
# # install git
# RUN apt-get install build-essential -y

# # Copy CrewAI-Studio
# RUN mkdir /CrewAI-Studio
# COPY ./ /CrewAI-Studio/

# # Set working directory
# WORKDIR /CrewAI-Studio

# # Set PYTHONPATH to include the app directory
# ENV PYTHONPATH="/CrewAI-Studio/app"

# # Install required packages
# RUN pip install -r requirements.txt

# EXPOSE 8501 8000

# # Start both Streamlit and FastAPI
# CMD ["sh", "-c", "streamlit run ./app/app.py --server.port=8501 & uvicorn app:api_app --host 0.0.0.0 --port 8000 --reload"]
# Base image
FROM python:3.12.5-alpine

# Install required build dependencies
RUN apk update && apk add --no-cache \
    build-base \
    gcc \
    musl-dev \
    libffi-dev \
    openssl-dev \
    && pip install --upgrade pip

# Set working directory
WORKDIR /CrewAI-Studio

# Copy application files
COPY ./ /CrewAI-Studio/

# Set PYTHONPATH to include the app directory
ENV PYTHONPATH="/CrewAI-Studio/app"

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose necessary ports
EXPOSE 8501 8000

# Start both Streamlit and FastAPI
CMD ["sh", "-c", "streamlit run ./app/app.py --server.port=8501 & uvicorn app:api_app --host 0.0.0.0 --port 8000 --reload"]
