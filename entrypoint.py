#!/usr/bin/env python3
import os
import sys
import uvicorn

if __name__ == "__main__":
    # Get port from environment or use default
    try:
        port = int(os.environ.get("PORT", 8000))
        print(f"Starting server on port {port}")
    except Exception as e:
        print(f"Error parsing PORT: {e}")
        port = 8000
        print(f"Falling back to default port {port}")
    
    # Print environment variables for debugging (excluding sensitive ones)
    print("Environment variables:")
    for key, value in os.environ.items():
        if not any(secret in key.lower() for secret in ['key', 'token', 'password', 'secret']):
            print(f"  {key}: {value}")
    
    # Run the application
    try:
        uvicorn.run("api:app", host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Failed to start server: {e}")
        sys.exit(1) 