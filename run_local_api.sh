#!/bin/bash

# Check if python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed or not in PATH"
    exit 1
fi

# Check if environment variables are set
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f .env ]; then
        echo "Loading environment variables from .env file"
        export $(grep -v '^#' .env | xargs)
    else
        echo "Warning: OPENAI_API_KEY environment variable not set and no .env file found"
        echo "You will need to set this variable before running evaluations"
    fi
fi

# Install dependencies if not already installed
pip install -r requirements.txt

# Run the API server
echo "Starting API server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload 