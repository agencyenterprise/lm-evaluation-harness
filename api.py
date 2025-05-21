from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import json
import os
import sys
import traceback
from dotenv import load_dotenv
from datetime import datetime

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
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

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
            "/result/{task_id}": "GET - Get evaluation results (also removes completed tasks)",
            "/tasks": "GET - List all tasks and statuses. Use ?clear_completed=true to clean up",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    import platform
    import sys
    
    # Check API keys
    openai_key_status = "Available" if os.environ.get("OPENAI_API_KEY") else "Missing"
    anthropic_key_status = "Available" if os.environ.get("ANTHROPIC_API_KEY") else "Missing"
    
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
    
    # Get all environment variable names (not values) for debugging
    env_vars = list(os.environ.keys())
    
    return {
        "status": "healthy",
        "python_version": sys.version,
        "platform": platform.platform(),
        "api_keys": {
            "openai": openai_key_status,
            "anthropic": anthropic_key_status
        },
        "env_vars_available": env_vars,
        "mongodb": mongo_status,
        "tasks_in_progress": len([k for k, v in evaluation_results.items() if v.get("status") == "processing"])
    }

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
    # Use message_id as task_id if provided, otherwise generate one
    if request.message_id:
        task_id = request.message_id
        print(f"Using message_id as task_id: {task_id}")
    else:
        task_id = f"task_{len(evaluation_results) + 1}"
        print(f"No message_id provided, generated task_id: {task_id}")
    
    # Log the parameters received from frontend
    print(f"\n=== Received Evaluation Request (Task ID: {task_id}) ===")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print(f"Context Type: {type(request.context)}")
    print(f"Context: {request.context}")
    print(f"System: {request.system}")
    print(f"Message ID: {request.message_id}")
    print(f"Skip DB: {request.skip_db}")
    print(f"Use Local Dataset: {request.use_local_dataset}")
    print("=" * 50)
    
    # Check if system prompt is empty
    system = request.system
    if system is not None and not system.strip():
        print("Empty system prompt received, treating as None")
        system = None
    
    # Set the evaluation task to run in the background
    background_tasks.add_task(
        run_evaluation,
        task_id=task_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=system,
        message_id=request.message_id,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
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
    
    # Task is completed or error, return the result and remove from memory
    response_copy = result.copy()  # Create a copy to return
    
    # Clean up memory by removing the result from evaluation_results
    # We only remove completed or error tasks, not processing ones
    if result["status"] in ["completed", "error"]:
        print(f"Removing task {task_id} from memory (status: {result['status']})")
        evaluation_results.pop(task_id)
    
    return response_copy

@app.get("/tasks")
async def list_tasks(clear_completed: bool = False):
    """Get a list of all evaluation tasks and their statuses
    
    Args:
        clear_completed: If True, removes all completed and error tasks from memory
    """
    tasks_summary = {}
    
    # Track which tasks to remove if clear_completed is True
    tasks_to_remove = []
    
    for task_id, task_info in evaluation_results.items():
        status = task_info.get("status", "unknown")
        model = task_info.get("params", {}).get("model", "unknown")
        started = task_info.get("params", {}).get("start_time", "unknown")
        
        # Get completion or error time
        end_time = (
            task_info.get("completion_time") if status == "completed" 
            else task_info.get("error_time") if status == "error"
            else None
        )
        
        tasks_summary[task_id] = {
            "status": status,
            "model": model,
            "started": started,
            "completed": end_time,
            "message_id": task_info.get("params", {}).get("message_id", None)
        }
        
        # Mark task for removal if it's completed or error and clear_completed is True
        if clear_completed and status in ["completed", "error"]:
            tasks_to_remove.append(task_id)
    
    # Clean up completed/error tasks if requested
    if clear_completed and tasks_to_remove:
        for task_id in tasks_to_remove:
            evaluation_results.pop(task_id)
        print(f"Removed {len(tasks_to_remove)} completed/error tasks from memory")
    
    # Count tasks by status
    status_counts = {}
    for task in tasks_summary.values():
        status = task["status"]
        status_counts[status] = status_counts.get(status, 0) + 1
    
    return {
        "tasks": tasks_summary,
        "total": len(tasks_summary),
        "status_counts": status_counts,
        "cleaned_up": len(tasks_to_remove) if clear_completed else 0
    }

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
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    # Set initial status and store input parameters
    evaluation_results[task_id] = {
        "status": "processing",
        "message": f"Processing evaluation for {model}",
        "params": {
            "model": model,
            "provider": provider,
            "examples": examples,
            "context_type": type(context).__name__,
            "context_sample": str(context)[:100] + "..." if context and len(str(context)) > 100 else str(context),
            "system": system,
            "message_id": message_id,
            "skip_db": skip_db,
            "use_local_dataset": use_local_dataset,
            "start_time": str(datetime.now())
        }
    }
    
    print(f"\n=== Starting Evaluation (Task ID: {task_id}) ===")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Examples: {examples}")
    print(f"Using local dataset: {use_local_dataset}")
    if not use_local_dataset:
        print(f"Using cache_dir: /tmp/hf_cache_moral_stories")
    
    try:
        # Connect to DB if not skipping
        db = None
        if not skip_db:
            try:
                print(f"Connecting to MongoDB...")
                db = get_mongodb_connection()
                print(f"MongoDB connection successful")
            except Exception as e:
                print(f"MongoDB connection error: {str(e)}")
                evaluation_results[task_id] = {
                    "status": "error",
                    "message": f"MongoDB connection error: {str(e)}"
                }
                return
        else:
            print(f"Skipping MongoDB connection (skip_db=True)")
        
        # Process context and system prompt
        print(f"Processing context and system prompt...")
        
        # Convert MessageModel objects to dictionaries if needed
        if isinstance(context, list):
            try:
                # Check if we have MessageModel objects and convert them to dicts
                converted_context = []
                for msg in context:
                    if hasattr(msg, 'role') and hasattr(msg, 'content'):
                        # This is likely a MessageModel - convert to dict
                        converted_context.append({
                            'role': msg.role,
                            'content': msg.content
                        })
                    elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        # Already a proper dict
                        converted_context.append(msg)
                    else:
                        raise ValueError(f"Invalid message format: {msg}")
                
                context = converted_context
                print(f"Converted {len(converted_context)} MessageModel objects to dictionaries")
            except Exception as e:
                print(f"Error converting context: {e}")
                raise ValueError(f"Failed to process context: {e}")
        
        # Check if system prompt is empty and treat as None
        if system is not None and not system.strip():
            print("System prompt is empty, treating as None")
            system = None
        
        if system and context:
            if isinstance(context, list) and not any(msg.get('role') == 'system' for msg in context if isinstance(msg, dict)):
                context.insert(0, {"role": "system", "content": system})
                print(f"Added system message to context list")
        elif system:
            context = [{"role": "system", "content": system}]
            print(f"Created new context with system message")
        
        # Run the evaluation
        print(f"Starting evaluation_moral_stories_with_openai...")
        result = evaluate_moral_stories_with_openai(
            model_name=model,
            num_examples=examples,
            context=context,
            cache_dir="/tmp/hf_cache_moral_stories",
            db=db,
            message_id=message_id,
            use_local_dataset=use_local_dataset,
            provider=provider
        )
        
        print(f"Evaluation completed successfully")
        
        # Store and return the result
        evaluation_results[task_id] = {
            "status": "completed",
            "result": result,
            "params": evaluation_results[task_id].get("params", {}),
            "completion_time": str(datetime.now())
        }
        print(f"=== Evaluation Complete (Task ID: {task_id}) ===")
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in evaluation: {str(e)}")
        print(error_trace)
        evaluation_results[task_id] = {
            "status": "error",
            "message": str(e),
            "traceback": error_trace,
            "params": evaluation_results[task_id].get("params", {}),
            "error_time": str(datetime.now())
        }
        print(f"Error in task {task_id}: {e}")
        print(f"=== Evaluation Failed (Task ID: {task_id}) ===")

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 