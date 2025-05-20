FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    gcc \
    g++ \
    make \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies all at once to save layers
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make Python scripts executable
RUN chmod +x /app/entrypoint.py

# Set default port for better visibility
ENV PORT=8000

# Command to run the application using Python directly
CMD ["python", "/app/entrypoint.py"] 