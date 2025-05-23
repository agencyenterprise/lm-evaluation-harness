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
import time
import threading

# Import the evaluation functions
try:
    from run_moral_stories_eval_gen import evaluate_moral_stories_with_openai, get_mongodb_connection
    from run_crows_pairs_eval import evaluate_crows_pairs
    from run_truthfulqa_eval import evaluate_truthfulqa
    from eval_utils import get_mongodb_connection as get_db_connection, make_json_serializable
except ImportError as e:
    print(f"Failed to import evaluation functions: {e}")
    traceback.print_exc()
    sys.exit(1)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Language Model Evaluation API",
    description="API for evaluating language models on moral reasoning, bias detection, and truthfulness tasks",
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

# Specific request models for different evaluation types
class CrowsPairsRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    message_id: Optional[str] = None
    provider: str = "openai"  # 'openai' or 'anthropic'

class TruthfulQARequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    message_id: Optional[str] = None
    provider: str = "openai"  # 'openai' or 'anthropic'

# Store for background tasks
evaluation_results = {}

@app.get("/")
async def root():
    return {
        "message": "Language Model Evaluation API is running",
        "version": "1.0.0",
        "endpoints": {
            "/": "This info message",
            "/evaluate": "POST - Start a moral stories evaluation",
            "/evaluate/crows-pairs": "POST - Start a CrowS-Pairs bias evaluation",
            "/evaluate/truthfulqa": "POST - Start a TruthfulQA evaluation",
            "/result/{task_id}": "GET - Get evaluation results (also removes completed tasks)",
            "/tasks": "GET - List all tasks and statuses. Use ?clear_completed=true to clean up",
            "/health": "GET - Check API health"
        },
        "available_evaluations": {
            "moral_stories": "Evaluate moral reasoning and ethical decision making",
            "crows_pairs": "Evaluate social biases and stereotyping",
            "truthfulqa": "Evaluate truthfulness and factual accuracy"
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
            db = get_db_connection()
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

@app.post("/evaluate/crows-pairs", response_model=EvaluationResponse)
async def evaluate_crows_pairs_endpoint(request: CrowsPairsRequest, background_tasks: BackgroundTasks):
    # Use message_id as task_id if provided, otherwise generate one
    if request.message_id:
        task_id = request.message_id
        print(f"Using message_id as task_id: {task_id}")
    else:
        task_id = f"crows_pairs_task_{len(evaluation_results) + 1}"
        print(f"No message_id provided, generated task_id: {task_id}")
    
    # Log the parameters received from frontend
    print(f"\n=== Received CrowS-Pairs Evaluation Request (Task ID: {task_id}) ===")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print(f"Context Type: {type(request.context)}")
    print(f"Context: {request.context}")
    print(f"Message ID: {request.message_id}")
    print("=" * 50)
    
    # Set the evaluation task to run in the background
    background_tasks.add_task(
        run_crows_pairs_evaluation,
        task_id=task_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        message_id=request.message_id,
        provider=request.provider
    )
    
    return EvaluationResponse(
        task_id=task_id,
        status="processing",
        message=f"CrowS-Pairs bias evaluation started for model {request.model}"
    )

@app.post("/evaluate/truthfulqa", response_model=EvaluationResponse)
async def evaluate_truthfulqa_endpoint(request: TruthfulQARequest, background_tasks: BackgroundTasks):
    # Use message_id as task_id if provided, otherwise generate one
    if request.message_id:
        task_id = request.message_id
        print(f"Using message_id as task_id: {task_id}")
    else:
        task_id = f"truthfulqa_task_{len(evaluation_results) + 1}"
        print(f"No message_id provided, generated task_id: {task_id}")
    
    # Log the parameters received from frontend
    print(f"\n=== Received TruthfulQA Evaluation Request (Task ID: {task_id}) ===")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print(f"Context Type: {type(request.context)}")
    print(f"Context: {request.context}")
    print(f"Message ID: {request.message_id}")
    print("=" * 50)
    
    # Set the evaluation task to run in the background
    background_tasks.add_task(
        run_truthfulqa_evaluation,
        task_id=task_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        message_id=request.message_id,
        provider=request.provider
    )
    
    return EvaluationResponse(
        task_id=task_id,
        status="processing",
        message=f"TruthfulQA evaluation started for model {request.model}"
    )

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    # First check if task is in memory
    if task_id in evaluation_results:
        result = evaluation_results[task_id]
        
        if result["status"] == "processing":
            # Add the progress information to the response for processing tasks
            progress = result.get("progress", {"current": 0, "total": 0, "percent": 0})
            return {
                "status": "processing", 
                "message": "Evaluation in progress",
                "progress": progress
            }
        
        # Task is completed or error, return the result and remove from memory
        response_copy = result.copy()  # Create a copy to return
        
        # Clean up memory by removing the result from evaluation_results
        # We only remove completed or error tasks, not processing ones
        if result["status"] in ["completed", "error"]:
            print(f"Removing task {task_id} from memory (status: {result['status']})")
            evaluation_results.pop(task_id)
        
        return response_copy
    
    # If task is not in memory, check MongoDB for the result
    # This is useful for retrieving historical results
    try:
        # Only try to check DB if task_id might be a message_id
        if not task_id.startswith("task_") and len(task_id) > 10:
            print(f"Task {task_id} not found in memory, checking MongoDB...")
            db = get_db_connection()
            
            # Check all result collections
            collection_names = [
                "baseline_results", 
                "with_context_results",
                "crows_pairs_baseline_results",
                "crows_pairs_with_context_results", 
                "truthfulqa_baseline_results",
                "truthfulqa_with_context_results"
            ]
            
            for collection_name in collection_names:
                try:
                    collection = db[collection_name]
                    stored_result = collection.find_one({"message_id": task_id})
                    
                    if stored_result:
                        print(f"Found result for task {task_id} in MongoDB ({collection_name})")
                        # Make the stored result compatible with the API format and JSON serializable
                        serializable_result = make_json_serializable(stored_result)
                        return {
                            "status": "completed",
                            "result": serializable_result,
                            "retrieved_from": "database",
                            "collection": collection_name
                        }
                except Exception as e:
                    print(f"Error checking collection {collection_name}: {e}")
                    continue
            
            # If we reach here, the result was not found in the database
            print(f"Result for task {task_id} not found in MongoDB")
    except Exception as e:
        print(f"Error checking MongoDB for task {task_id}: {e}")
    
    # If task is not found in memory or DB, return 404
    raise HTTPException(status_code=404, detail=f"Task {task_id} not found in memory or database")

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
        
        # Get progress information
        progress = task_info.get("progress", {"current": 0, "total": 0, "percent": 0})
        
        tasks_summary[task_id] = {
            "status": status,
            "model": model,
            "started": started,
            "completed": end_time,
            "message_id": task_info.get("params", {}).get("message_id", None),
            "progress": progress
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
        "progress": {
            "current": 0,
            "total": examples,
            "percent": 0
        },
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
                db = get_db_connection()
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
        
        # Create a progress callback for this task
        def progress_callback(current, total):
            update_progress(task_id, current, total)
        
        result = evaluate_moral_stories_with_openai(
            model_name=model,
            num_examples=examples,
            context=context,
            cache_dir="/tmp/hf_cache_moral_stories",
            db=db,
            message_id=message_id,
            use_local_dataset=use_local_dataset,
            provider=provider,
            progress_callback=progress_callback
        )
        
        print(f"Evaluation completed successfully")
        
        # Create a temporary copy of result to save in memory
        temp_result = {
            "status": "completed",
            "result": result,
            "params": evaluation_results[task_id].get("params", {}),
            "completion_time": str(datetime.now())
        }
        
        # Store result before removing from memory
        if task_id in evaluation_results:
            evaluation_results[task_id] = temp_result
            
            # Remove from memory now that it's completed and saved to DB
            print(f"Task {task_id} completed and saved to DB. Removing from memory.")
            # Add a small delay to ensure any current API requests can access the result
            time.sleep(1.5)
            
            # Only remove if it's still there and hasn't been removed by another process
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed task {task_id} from memory.")
            
        print(f"=== Evaluation Complete (Task ID: {task_id}) ===")
        
        # Return the result even though we've removed it from memory
        return temp_result
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in evaluation: {str(e)}")
        print(error_trace)
        
        # Update the error status
        error_result = {
            "status": "error",
            "message": str(e),
            "traceback": error_trace,
            "params": evaluation_results[task_id].get("params", {}),
            "error_time": str(datetime.now())
        }
        
        # Store in memory briefly
        if task_id in evaluation_results:
            evaluation_results[task_id] = error_result
            
            # Remove from memory after storing the error
            print(f"Task {task_id} encountered error. Removing from memory.")
            time.sleep(0.5)
            
            # Only remove if it's still there
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed errored task {task_id} from memory.")
                
        print(f"Error in task {task_id}: {e}")
        print(f"=== Evaluation Failed (Task ID: {task_id}) ===")
        
        # Return the error result
        return error_result

# Create a progress update function that we can pass to the evaluation function
def update_progress(task_id, current, total):
    """Update the progress of a task"""
    if task_id not in evaluation_results:
        return
    
    evaluation_results[task_id]["progress"] = {
        "current": current,
        "total": total,
        "percent": int((current / total) * 100) if total > 0 else 0
    }
    print(f"Task {task_id}: Progress {current}/{total} ({int((current / total) * 100)}%)")

def run_crows_pairs_evaluation(
    task_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    message_id: Optional[str],
    provider: str = "openai"
):
    # Set initial status and store input parameters
    evaluation_results[task_id] = {
        "status": "processing",
        "message": f"Processing CrowS-Pairs evaluation for {model}",
        "progress": {
            "current": 0,
            "total": examples,
            "percent": 0
        },
        "params": {
            "evaluation_type": "crows_pairs",
            "model": model,
            "provider": provider,
            "examples": examples,
            "context_type": type(context).__name__,
            "context_sample": str(context)[:100] + "..." if context and len(str(context)) > 100 else str(context),
            "message_id": message_id,
            "start_time": str(datetime.now())
        }
    }
    
    print(f"\n=== Starting CrowS-Pairs Evaluation (Task ID: {task_id}) ===")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Examples: {examples}")
    
    try:
        # Get database connection
        try:
            db = get_db_connection()
            print("Database connection established for CrowS-Pairs evaluation")
        except Exception as e:
            print(f"Warning: Could not connect to database: {e}")
            db = None
        
        # Update task status
        evaluation_results[task_id]["status"] = "running"
        evaluation_results[task_id]["progress"] = {"current": 0, "total": examples}
        
        def progress_callback(current, total):
            update_progress(task_id, current, total)
        
        # Run evaluation with database connection
        print(f"Starting CrowS-Pairs evaluation...")
        result = evaluate_crows_pairs(
            model_name=model,
            num_examples=examples,
            context=context,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass database connection
            message_id=message_id  # Use task_id as message_id
        )
        
        print(f"CrowS-Pairs evaluation completed successfully")
        
        # Create a temporary copy of result to save in memory
        temp_result = {
            "status": "completed",
            "result": result,
            "params": evaluation_results[task_id].get("params", {}),
            "completion_time": str(datetime.now())
        }
        
        # Store result before removing from memory
        if task_id in evaluation_results:
            evaluation_results[task_id] = temp_result
            
            # Remove from memory now that it's completed
            print(f"Task {task_id} completed. Removing from memory.")
            time.sleep(1.5)
            
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed task {task_id} from memory.")
            
        print(f"=== CrowS-Pairs Evaluation Complete (Task ID: {task_id}) ===")
        return temp_result
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in CrowS-Pairs evaluation: {str(e)}")
        print(error_trace)
        
        error_result = {
            "status": "error",
            "message": str(e),
            "traceback": error_trace,
            "params": evaluation_results[task_id].get("params", {}),
            "error_time": str(datetime.now())
        }
        
        if task_id in evaluation_results:
            evaluation_results[task_id] = error_result
            time.sleep(0.5)
            
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed errored task {task_id} from memory.")
                
        print(f"=== CrowS-Pairs Evaluation Failed (Task ID: {task_id}) ===")
        return error_result

def run_truthfulqa_evaluation(
    task_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    message_id: Optional[str],
    provider: str = "openai"
):
    # Set initial status and store input parameters
    evaluation_results[task_id] = {
        "status": "processing",
        "message": f"Processing TruthfulQA evaluation for {model}",
        "progress": {
            "current": 0,
            "total": examples,
            "percent": 0
        },
        "params": {
            "evaluation_type": "truthfulqa",
            "model": model,
            "provider": provider,
            "examples": examples,
            "context_type": type(context).__name__,
            "context_sample": str(context)[:100] + "..." if context and len(str(context)) > 100 else str(context),
            "message_id": message_id,
            "start_time": str(datetime.now())
        }
    }
    
    print(f"\n=== Starting TruthfulQA Evaluation (Task ID: {task_id}) ===")
    print(f"Model: {model}")
    print(f"Provider: {provider}")
    print(f"Examples: {examples}")
    
    try:
        # Get database connection
        try:
            db = get_db_connection()
            print("Database connection established for TruthfulQA evaluation")
        except Exception as e:
            print(f"Warning: Could not connect to database: {e}")
            db = None
        
        # Update task status
        evaluation_results[task_id]["status"] = "running"
        evaluation_results[task_id]["progress"] = {"current": 0, "total": examples}
        
        def progress_callback(current, total):
            update_progress(task_id, current, total)
        
        # Run evaluation with database connection
        print(f"Starting TruthfulQA evaluation...")
        result = evaluate_truthfulqa(
            model_name=model,
            num_examples=examples,
            context=context,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass database connection
            message_id=message_id  # Use task_id as message_id
        )
        
        print(f"TruthfulQA evaluation completed successfully")
        
        # Create a temporary copy of result to save in memory
        temp_result = {
            "status": "completed",
            "result": result,
            "params": evaluation_results[task_id].get("params", {}),
            "completion_time": str(datetime.now())
        }
        
        # Store result before removing from memory
        if task_id in evaluation_results:
            evaluation_results[task_id] = temp_result
            
            # Remove from memory now that it's completed
            print(f"Task {task_id} completed. Removing from memory.")
            time.sleep(1.5)
            
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed task {task_id} from memory.")
            
        print(f"=== TruthfulQA Evaluation Complete (Task ID: {task_id}) ===")
        return temp_result
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in TruthfulQA evaluation: {str(e)}")
        print(error_trace)
        
        error_result = {
            "status": "error",
            "message": str(e),
            "traceback": error_trace,
            "params": evaluation_results[task_id].get("params", {}),
            "error_time": str(datetime.now())
        }
        
        if task_id in evaluation_results:
            evaluation_results[task_id] = error_result
            time.sleep(0.5)
            
            if task_id in evaluation_results:
                evaluation_results.pop(task_id)
                print(f"Removed errored task {task_id} from memory.")
                
        print(f"=== TruthfulQA Evaluation Failed (Task ID: {task_id}) ===")
        return error_result

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 