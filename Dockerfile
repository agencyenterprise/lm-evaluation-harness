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

# Install Python dependencies one by one to better identify issues
RUN pip install --no-cache-dir fastapi==0.88.0
RUN pip install --no-cache-dir uvicorn==0.20.0
RUN pip install --no-cache-dir pydantic==1.10.4
RUN pip install --no-cache-dir requests==2.28.2
RUN pip install --no-cache-dir numpy==1.23.5
RUN pip install --no-cache-dir python-dotenv==0.21.1
RUN pip install --no-cache-dir pymongo==4.3.3
RUN pip install --no-cache-dir regex==2022.10.31
RUN pip install --no-cache-dir python-multipart==0.0.5
RUN pip install --no-cache-dir huggingface_hub==0.13.3
RUN pip install --no-cache-dir tokenizers==0.13.2
RUN pip install --no-cache-dir datasets==2.9.0

# Copy the rest of the application
COPY . .

# Expose the port
EXPOSE $PORT

# Command to run the application
CMD uvicorn api:app --host 0.0.0.0 --port $PORT 