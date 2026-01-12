# Use official Python image
FROM python:3.12-slim

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Set environment variables (override in production!)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REDIS_URL=redis://redis:6379/0 \
    LOG_LEVEL=INFO

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
