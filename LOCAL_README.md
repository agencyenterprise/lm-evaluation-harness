# Running the Moral Stories Evaluation API Locally

This guide explains how to run and test the API on your local machine before deploying to Railway.

## Prerequisites

1. Python 3.7 or higher
2. OpenAI API key
3. MongoDB URI (optional, for database integration)

## Setup

1. **Create a `.env` file with your API keys:**

```
OPENAI_API_KEY=your_openai_api_key
MONGODB_URI=your_mongodb_connection_string
```

2. **Make the run script executable:**

```bash
chmod +x run_local_api.sh
```

## Running the API

```bash
./run_local_api.sh
```

This will:
- Install all required dependencies
- Start the FastAPI server on http://localhost:8000
- Enable auto-reload for development

## Testing the API

Once the API is running, you can access:

1. **Interactive API documentation:**
   - http://localhost:8000/docs (Swagger UI)
   - http://localhost:8000/redoc (ReDoc)

2. **Test endpoints using curl:**

```bash
# Check if API is running
curl http://localhost:8000/

# Start an evaluation
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "examples": 2,
    "system": "You are an ethical assistant",
    "message_id": "test123"
  }'

# Check results (replace TASK_ID with the task_id from the response above)
curl http://localhost:8000/result/TASK_ID
```

3. **Test with the Python requests library:**

```python
import requests
import time

# Start evaluation
response = requests.post(
    "http://localhost:8000/evaluate",
    json={
        "model": "gpt-3.5-turbo",
        "examples": 2,
        "system": "You are an ethical assistant",
        "message_id": "test123"
    }
)
task_id = response.json()["task_id"]
print(f"Evaluation started with task_id: {task_id}")

# Poll for results
while True:
    result = requests.get(f"http://localhost:8000/result/{task_id}").json()
    if result["status"] != "processing":
        break
    print("Still processing...")
    time.sleep(5)

print("Results:", result)
```

## Monitoring Database Results

If you've configured MongoDB, you can check the results directly in the database:

1. **Using the MongoDB shell:**
```bash
mongosh "your_mongodb_uri"
use test
db.baseline_results.find({})
db.with_context_results.find({})
```

2. **Using MongoDB Compass or similar GUI tool:**
   - Connect to your MongoDB instance
   - Navigate to the `test` database
   - Check the `baseline_results` and `with_context_results` collections

## Common Issues

1. **API Key Not Found**: Make sure your .env file is in the root directory or set the environment variables directly.

2. **Database Connection Failed**: Check your MongoDB URI and ensure your IP is whitelisted if using MongoDB Atlas.

3. **Dataset Download Issues**: If you encounter problems downloading the dataset, you can specify a custom cache directory:
```bash
export DATASETS_CACHE="/path/to/custom/cache"
```
Then run the API again.

## Memory Considerations

The moral_stories dataset can be large, and evaluations with many examples may require significant memory. If you encounter memory issues:

1. Reduce the number of examples in each evaluation
2. Consider setting up swap space if running on a low-memory system
3. Use the `--cache_dir` option to specify a location with sufficient disk space 