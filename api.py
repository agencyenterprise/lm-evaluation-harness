from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import json
import os
import sys
import traceback
from dotenv import load_dotenv

# Import the evaluation function
try:
    from run_moral_stories_eval_gen import evaluate_moral_stories_with_openai, get_mongodb_connection
except ImportError as e:
    print(f"Failed to import from run_moral_stories_eval_gen: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Moral Stories Evaluation API",
    description="API for evaluating language models on moral reasoning tasks",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define request models
class MessageModel(BaseModel):
    role: str
    content: str

class EvaluationRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: Optional[str] = None
    force_download: bool = False
    skip_db: bool = False

class EvaluationResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Store for background tasks
evaluation_results = {}

@app.get("/")
async def root():
    return {
        "message": "Moral Stories Evaluation API is running",
        "version": "1.0.0",
        "endpoints": {
            "/": "This info message",
            "/evaluate": "POST - Start an evaluation",
            "/result/{task_id}": "GET - Get evaluation results",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    import platform
    import sys
    
    # Check OpenAI API key
    api_key_status = "Available" if os.environ.get("OPENAI_API_KEY") else "Missing"
    
    # Check MongoDB connection
    mongo_status = "Not checked"
    if os.environ.get("MONGODB_URI"):
        try:
            db = get_mongodb_connection()
            mongo_status = "Connected" if db else "Failed to connect"
        except Exception as e:
            mongo_status = f"Error: {str(e)}"
    else:
        mongo_status = "No MongoDB URI provided"
    
    return {
        "status": "healthy",
        "python_version": sys.version,
        "platform": platform.platform(),
        "openai_api_key": api_key_status,
        "mongodb": mongo_status,
        "tasks_in_progress": len([k for k, v in evaluation_results.items() if v.get("status") == "processing"])
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
    task_id = f"task_{len(evaluation_results) + 1}"
    
    # Set the evaluation task to run in the background
    background_tasks.add_task(
        run_evaluation,
        task_id=task_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        message_id=request.message_id,
        force_download=request.force_download,
        skip_db=request.skip_db
    )
    
    return EvaluationResponse(
        task_id=task_id,
        status="processing",
        message=f"Evaluation started for model {request.model}"
    )

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    if task_id not in evaluation_results:
        raise HTTPException(status_code=404, detail="Task not found")
    
    result = evaluation_results[task_id]
    
    if result["status"] == "processing":
        return {"status": "processing", "message": "Evaluation in progress"}
    
    return result

@app.get("/debug")
async def debug_info(request: Request):
    """Endpoint for debugging the API environment"""
    import platform
    import sys
    
    # Get environment variables (excluding API keys)
    env_vars = {k: v for k, v in os.environ.items() if not any(secret in k.lower() for secret in ['key', 'token', 'password', 'secret'])}
    
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "environment": env_vars,
        "headers": dict(request.headers),
        "client_host": request.client.host if request.client else None,
        "base_url": str(request.base_url)
    }

def run_evaluation(
    task_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    message_id: Optional[str],
    force_download: bool,
    skip_db: bool
):
    # Set initial status
    evaluation_results[task_id] = {
        "status": "processing",
        "message": f"Processing evaluation for {model}"
    }
    
    try:
        # Connect to DB if not skipping
        db = None
        if not skip_db:
            try:
                db = get_mongodb_connection()
            except Exception as e:
                evaluation_results[task_id] = {
                    "status": "error",
                    "message": f"MongoDB connection error: {str(e)}"
                }
                return
        
        # Process context and system prompt
        if system and context:
            if isinstance(context, list) and not any(msg.get('role') == 'system' for msg in context if isinstance(msg, dict)):
                context.insert(0, {"role": "system", "content": system})
        elif system:
            context = [{"role": "system", "content": system}]
        
        # Run the evaluation
        result = evaluate_moral_stories_with_openai(
            model_name=model,
            num_examples=examples,
            context=context,
            cache_dir="/tmp/hf_cache_moral_stories",
            db=db,
            message_id=message_id
        )
        
        # Store and return the result
        evaluation_results[task_id] = {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        error_trace = traceback.format_exc()
        evaluation_results[task_id] = {
            "status": "error",
            "message": str(e),
            "traceback": error_trace
        }
        print(f"Error in task {task_id}: {e}")
        print(error_trace)

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 