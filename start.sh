#!/bin/bash
set -e

# Set default port if not provided
export PORT=${PORT:-8000}

echo "Starting API server on port $PORT"

# Run uvicorn with proper port
exec uvicorn api:app --host 0.0.0.0 --port $PORT 