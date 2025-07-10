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
import weakref
from contextlib import contextmanager
import gc
import psutil

# Import the evaluation functions
try:
    from run_moral_stories_eval_gen import evaluate_moral_stories_with_openai, get_mongodb_connection
    from run_crows_pairs_eval import evaluate_crows_pairs
    from run_truthfulqa_eval import evaluate_truthfulqa
    from run_arc_challenge_eval import evaluate_arc_challenge
    from run_sycophancy_eval import evaluate_sycophancy
    from run_air_deception_eval import evaluate_air_deception
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
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

class EvaluationResponse(BaseModel):
    task_id: str
    status: str
    message: str

# Specific request models for different evaluation types
class CrowsPairsRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

class TruthfulQARequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

class ArcChallengeRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

class SycophancyRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

class AirDeceptionRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai', 'anthropic', or 'google'

def get_collection_name(evaluation_type: str, has_context: bool) -> str:
    """Get the collection name based on evaluation type and context."""
    context_suffix = "with_context" if has_context else "baseline"
    
    if evaluation_type == "moral_stories":
        return f"{context_suffix}_results"
    elif evaluation_type == "crows_pairs":
        return f"crows_pairs_{context_suffix}_results"
    elif evaluation_type == "truthfulqa":
        return f"truthfulqa_{context_suffix}_results"
    else:
        return f"{context_suffix}_results"

def find_processing_document(db, message_id: str):
    """Find a processing document by message_id across all result collections."""
    collections = [
        "baseline_results", "with_context_results",
        "crows_pairs_baseline_results", "crows_pairs_with_context_results",
        "truthfulqa_baseline_results", "truthfulqa_with_context_results",
        "arc_challenge_baseline_results", "arc_challenge_with_context_results",
        "sycophancy_baseline_results", "sycophancy_with_context_results",
        "air_deception_baseline_results", "air_deception_with_context_results"
    ]
    
    for collection_name in collections:
        try:
            collection = db[collection_name]
            doc = collection.find_one({"message_id": message_id, "status": "processing"})
            if doc:
                return collection, doc
        except Exception as e:
            print(f"Error checking collection {collection_name}: {e}")
            continue
    
    return None, None

def update_document_status(db, message_id: str, status: str, **kwargs):
    """Update document status in the appropriate collection."""
    collection, doc = find_processing_document(db, message_id)
    
    if collection is not None and doc is not None:
        update_data = {
            "status": status,
            "updated_at": datetime.now()
        }
        update_data.update(kwargs)
        
        result = collection.update_one(
            {"_id": doc["_id"]},
            {"$set": update_data}
        )
        print(f"Updated document {message_id} in {collection.name} with status: {status}")
        return result
    else:
        print(f"No processing document found for message_id: {message_id}")
        return None

# Add connection pool management
_db_connection_pool = weakref.WeakValueDictionary()
_connection_lock = threading.Lock()

@contextmanager
def get_db_connection_managed():
    """Context manager for database connections with proper cleanup."""
    db = None
    try:
        db = get_db_connection()
        yield db
    finally:
        if db is not None:
            # Ensure connection is properly closed
            try:
                db.close()
            except:
                pass

# Add performance monitoring
def log_memory_usage(operation: str):
    """Log current memory usage for monitoring."""
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        print(f"Memory usage after {operation}: {memory_info.rss / 1024 / 1024:.1f} MB")
    except:
        pass

# Add cleanup utilities
def cleanup_old_cache_files(cache_dir: str, max_age_days: int = 7):
    """Clean up old cache files to prevent disk space issues."""
    if not os.path.exists(cache_dir):
        return
    
    try:
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for root, dirs, files in os.walk(cache_dir):
            for file in files:
                file_path = os.path.join(root, file)
                if current_time - os.path.getmtime(file_path) > max_age_seconds:
                    os.remove(file_path)
                    print(f"Cleaned up old cache file: {file_path}")
    except Exception as e:
        print(f"Error cleaning cache: {e}")

# Add database cleanup
def cleanup_old_database_records(db, max_age_days: int = 30):
    """Clean up old database records to prevent unlimited growth."""
    try:
        from datetime import datetime, timedelta
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        collections = [
            "baseline_results", "with_context_results",
            "crows_pairs_baseline_results", "crows_pairs_with_context_results",
            "truthfulqa_baseline_results", "truthfulqa_with_context_results",
            "arc_challenge_baseline_results", "arc_challenge_with_context_results",
            "sycophancy_baseline_results", "sycophancy_with_context_results",
            "air_deception_baseline_results", "air_deception_with_context_results"
        ]
        
        for collection_name in collections:
            try:
                collection = db[collection_name]
                result = collection.delete_many({
                    "status": {"$in": ["completed", "error"]},
                    "updated_at": {"$lt": cutoff_date}
                })
                if result.deleted_count > 0:
                    print(f"Cleaned up {result.deleted_count} old records from {collection_name}")
            except Exception as e:
                print(f"Error cleaning {collection_name}: {e}")
    except Exception as e:
        print(f"Error in database cleanup: {e}")

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
            "/evaluate/arc-challenge": "POST - Start an Arc-Challenge reasoning evaluation",
            "/evaluate/sycophancy": "POST - Start a Sycophancy evaluation",
            "/evaluate/air-deception": "POST - Start an AIR-Deception safety evaluation",
            "/result/{task_id}": "GET - Get evaluation results",
            "/tasks": "GET - List all tasks and statuses (supports ?clear_completed=true&cleanup_old=true)",
            "/health": "GET - Check API health",
            "/metrics": "GET - Get detailed performance metrics and database statistics",
            "/cleanup": "POST - Manually trigger cleanup operations"
        },
        "available_evaluations": {
            "moral_stories": "Evaluate moral reasoning and ethical decision making",
            "crows_pairs": "Evaluate social biases and stereotyping",
            "truthfulqa": "Evaluate truthfulness and factual accuracy",
            "arc_challenge": "Evaluate scientific reasoning and knowledge",
            "sycophancy": "Evaluate resistance to sycophantic behavior",
            "air_deception": "Evaluate safety and refusal of harmful requests"
        },
        "performance_features": {
            "memory_monitoring": "Real-time memory usage tracking",
            "database_cleanup": "Automatic cleanup of old records",
            "cache_management": "Cache file cleanup and monitoring",
            "connection_pooling": "Managed database connections with proper cleanup",
            "garbage_collection": "Automatic memory management"
        }
    }

@app.get("/health")
async def health():
    import platform
    import sys
    
    # Check API keys
    openai_key_status = "Available" if os.environ.get("OPENAI_API_KEY") else "Missing"
    anthropic_key_status = "Available" if os.environ.get("ANTHROPIC_API_KEY") else "Missing"
    gemini_key_status = "Available" if os.environ.get("GEMINI_API_KEY") else "Missing"
    
    # Check MongoDB connection
    mongo_status = "Not checked"
    processing_tasks_count = 0
    
    # Use managed connection for health check
    try:
        with get_db_connection_managed() as db:
            mongo_status = "Connected" if db else "Failed to connect"
            
            # Get count of processing tasks from database
            collections = [
                "baseline_results", "with_context_results",
                "crows_pairs_baseline_results", "crows_pairs_with_context_results",
                "truthfulqa_baseline_results", "truthfulqa_with_context_results",
                "arc_challenge_baseline_results", "arc_challenge_with_context_results",
                "sycophancy_baseline_results", "sycophancy_with_context_results",
                "air_deception_baseline_results", "air_deception_with_context_results"
            ]
            for collection_name in collections:
                try:
                    collection = db[collection_name]
                    count = collection.count_documents({"status": "processing"})
                    processing_tasks_count += count
                except Exception as e:
                    print(f"Error counting in {collection_name}: {e}")
    except Exception as e:
        mongo_status = f"Error: {str(e)}"
    
    # Add memory and performance metrics
    memory_usage = "Unknown"
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_usage = f"{memory_info.rss / 1024 / 1024:.1f} MB"
    except:
        pass
    
    # Get all environment variable names (not values) for debugging
    env_vars = list(os.environ.keys())
    
    return {
        "status": "healthy",
        "python_version": sys.version,
        "platform": platform.platform(),
        "memory_usage": memory_usage,
        "api_keys": {
            "openai": openai_key_status,
            "anthropic": anthropic_key_status,
            "google": gemini_key_status
        },
        "env_vars_available": env_vars,
        "mongodb": mongo_status,
        "tasks_in_progress": processing_tasks_count
    }

@app.post("/evaluate")
async def evaluate(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """Start moral stories evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting Moral Stories Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"Moral stories evaluation started for model {request.model}"
    }

@app.post("/evaluate/crows-pairs")
async def evaluate_crows_pairs_endpoint(request: CrowsPairsRequest, background_tasks: BackgroundTasks):
    """Start CrowS-Pairs evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting CrowS-Pairs Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_crows_pairs_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"CrowS-Pairs evaluation started for model {request.model}"
    }

@app.post("/evaluate/truthfulqa")
async def evaluate_truthfulqa_endpoint(request: TruthfulQARequest, background_tasks: BackgroundTasks):
    """Start TruthfulQA evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting TruthfulQA Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_truthfulqa_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"TruthfulQA evaluation started for model {request.model}"
    }

@app.post("/evaluate/arc-challenge")
async def evaluate_arc_challenge_endpoint(request: ArcChallengeRequest, background_tasks: BackgroundTasks):
    """Start Arc-Challenge evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting Arc-Challenge Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_arc_challenge_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"Arc-Challenge evaluation started for model {request.model}"
    }

@app.post("/evaluate/sycophancy")
async def evaluate_sycophancy_endpoint(request: SycophancyRequest, background_tasks: BackgroundTasks):
    """Start Sycophancy evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting Sycophancy Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_sycophancy_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"Sycophancy evaluation started for model {request.model}"
    }

@app.post("/evaluate/air-deception")
async def evaluate_air_deception_endpoint(request: AirDeceptionRequest, background_tasks: BackgroundTasks):
    """Start AIR-Deception evaluation. Frontend creates document with processing status."""
    message_id = request.message_id
    
    print(f"\n=== Starting AIR-Deception Evaluation ===")
    print(f"Message ID: {message_id}")
    print(f"Model: {request.model}")
    print(f"Provider: {request.provider}")
    print(f"Examples: {request.examples}")
    print("=" * 50)
    
    # Start background task (document already created by frontend)
    background_tasks.add_task(
        run_air_deception_evaluation,
        message_id=message_id,
        model=request.model,
        examples=request.examples,
        context=request.context,
        system=request.system,
        force_download=request.force_download,
        skip_db=request.skip_db,
        use_local_dataset=request.use_local_dataset,
        provider=request.provider
    )
    
    return {
        "task_id": message_id,
        "status": "processing",
        "message": f"AIR-Deception evaluation started for model {request.model}"
    }

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    """Get evaluation result from database."""
    try:
        db = get_db_connection()
        
        # Check all result collections for the message_id
        collections = [
            "baseline_results", "with_context_results",
            "crows_pairs_baseline_results", "crows_pairs_with_context_results",
            "truthfulqa_baseline_results", "truthfulqa_with_context_results",
            "arc_challenge_baseline_results", "arc_challenge_with_context_results",
            "sycophancy_baseline_results", "sycophancy_with_context_results",
            "air_deception_baseline_results", "air_deception_with_context_results"
        ]
        
        for collection_name in collections:
            try:
                collection = db[collection_name]
                doc = collection.find_one({"message_id": task_id})
                
                if doc:
                    print(f"Found document for {task_id} in {collection_name} with status: {doc.get('status', 'unknown')}")
                    
                    status = doc.get("status", "unknown")
                    
                    if status == "processing":
                        return {
                            "status": "processing",
                            "message": "Evaluation in progress"
                        }
                    elif status == "completed":
                        # Remove MongoDB-specific fields for clean response
                        result = make_json_serializable(doc)
                        if "_id" in result:
                            del result["_id"]
                        return {
                            "status": "completed",
                            "result": result
                        }
                    elif status == "error":
                        return {
                            "status": "error",
                            "error": doc.get("error", "Unknown error occurred")
                        }
                    else:
                        return {
                            "status": status,
                            "result": make_json_serializable(doc)
                        }
            except Exception as e:
                print(f"Error checking collection {collection_name}: {e}")
                continue
        
        # Task not found
        return {
            "status": "not_found",
            "error": f"Task {task_id} not found"
        }
        
    except Exception as e:
        print(f"Error getting task result: {e}")
        return {
            "status": "error",
            "error": f"Error retrieving task result: {str(e)}"
        }

@app.get("/tasks")
async def list_tasks(clear_completed: bool = False, cleanup_old: bool = False):
    """List all current evaluation tasks and their status."""
    try:
        with get_db_connection_managed() as db:
            all_tasks = []
            
            # Perform cleanup if requested
            if cleanup_old:
                cleanup_old_database_records(db, max_age_days=30)
                cleanup_old_cache_files("/tmp/hf_cache_moral_stories", max_age_days=7)
            
            collections = [
                "baseline_results", "with_context_results",
                "crows_pairs_baseline_results", "crows_pairs_with_context_results",
                "truthfulqa_baseline_results", "truthfulqa_with_context_results",
                "arc_challenge_baseline_results", "arc_challenge_with_context_results",
                "sycophancy_baseline_results", "sycophancy_with_context_results",
                "air_deception_baseline_results", "air_deception_with_context_results"
            ]
            
            for collection_name in collections:
                try:
                    collection = db[collection_name]
                    
                    # If clear_completed is True, remove completed and error tasks
                    if clear_completed:
                        result = collection.delete_many({
                            "status": {"$in": ["completed", "error"]}
                        })
                        print(f"Cleared {result.deleted_count} completed/error tasks from {collection_name}")
                    
                    # Get tasks with proper pagination and indexing
                    tasks = list(collection.find(
                        {},
                        {"_id": 1, "message_id": 1, "status": 1, "updated_at": 1, "model": 1}
                    ).sort("updated_at", -1).limit(20))  # Reduced limit for performance
                    
                    for task in tasks:
                        if "_id" in task:
                            task["_id"] = str(task["_id"])
                        if "updated_at" in task:
                            task["updated_at"] = task["updated_at"].isoformat() if hasattr(task["updated_at"], "isoformat") else str(task["updated_at"])
                        task["collection"] = collection_name
                        all_tasks.append(task)
                        
                except Exception as e:
                    print(f"Error processing collection {collection_name}: {e}")
                    continue
            
            # Sort all tasks by updated_at
            all_tasks.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            
            # Force garbage collection
            gc.collect()
            log_memory_usage("tasks listing")
            
            return {
                "tasks": all_tasks[:50],  # Limit to 50 most recent
                "total_count": len(all_tasks),
                "cleanup_performed": cleanup_old
            }
        
    except Exception as e:
        print(f"Error listing tasks: {e}")
        return {
            "error": f"Error retrieving tasks: {str(e)}",
            "tasks": [],
            "total_count": 0
        }

@app.get("/metrics")
async def get_metrics():
    """Get detailed performance metrics and database statistics."""
    try:
        with get_db_connection_managed() as db:
            # Memory usage
            memory_info = {}
            try:
                process = psutil.Process()
                memory_info = {
                    "rss": f"{process.memory_info().rss / 1024 / 1024:.1f} MB",
                    "vms": f"{process.memory_info().vms / 1024 / 1024:.1f} MB",
                    "percent": f"{process.memory_percent():.1f}%"
                }
            except:
                memory_info = {"error": "Unable to get memory info"}
            
            # Database statistics
            collections = [
                "baseline_results", "with_context_results",
                "crows_pairs_baseline_results", "crows_pairs_with_context_results",
                "truthfulqa_baseline_results", "truthfulqa_with_context_results",
                "arc_challenge_baseline_results", "arc_challenge_with_context_results",
                "sycophancy_baseline_results", "sycophancy_with_context_results",
                "air_deception_baseline_results", "air_deception_with_context_results"
            ]
            
            db_stats = {}
            total_documents = 0
            
            for collection_name in collections:
                try:
                    collection = db[collection_name]
                    total_count = collection.count_documents({})
                    processing_count = collection.count_documents({"status": "processing"})
                    completed_count = collection.count_documents({"status": "completed"})
                    error_count = collection.count_documents({"status": "error"})
                    
                    db_stats[collection_name] = {
                        "total": total_count,
                        "processing": processing_count,
                        "completed": completed_count,
                        "error": error_count
                    }
                    total_documents += total_count
                except Exception as e:
                    db_stats[collection_name] = {"error": str(e)}
            
            # Cache directory size
            cache_size = "Unknown"
            cache_files = 0
            try:
                cache_dir = "/tmp/hf_cache_moral_stories"
                if os.path.exists(cache_dir):
                    total_size = 0
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            total_size += os.path.getsize(file_path)
                            cache_files += 1
                    cache_size = f"{total_size / 1024 / 1024:.1f} MB"
            except Exception as e:
                cache_size = f"Error: {str(e)}"
            
            return {
                "memory": memory_info,
                "database": {
                    "total_documents": total_documents,
                    "collections": db_stats
                },
                "cache": {
                    "size": cache_size,
                    "files": cache_files
                },
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        return {
            "error": f"Error retrieving metrics: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

@app.post("/cleanup")
async def manual_cleanup(
    max_age_days: int = 30,
    cleanup_cache: bool = True,
    force_gc: bool = True
):
    """Manually trigger cleanup operations."""
    try:
        cleanup_results = {}
        
        # Database cleanup
        with get_db_connection_managed() as db:
            cleanup_old_database_records(db, max_age_days=max_age_days)
            cleanup_results["database"] = f"Cleaned records older than {max_age_days} days"
        
        # Cache cleanup
        if cleanup_cache:
            cleanup_old_cache_files("/tmp/hf_cache_moral_stories", max_age_days=7)
            cleanup_results["cache"] = "Cleaned cache files older than 7 days"
        
        # Force garbage collection
        if force_gc:
            import gc
            collected = gc.collect()
            cleanup_results["garbage_collection"] = f"Collected {collected} objects"
        
        log_memory_usage("manual cleanup")
        
        return {
            "status": "success",
            "cleanup_results": cleanup_results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def run_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run moral stories evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting evaluation {message_id}")
        
        # Clean up cache before starting
        cleanup_old_cache_files("/tmp/hf_cache_moral_stories", max_age_days=7)
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting Moral Stories Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"evaluation progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_moral_stories_with_openai(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                cache_dir="/tmp/hf_cache_moral_stories",
                db=db if not skip_db else None,
                message_id=message_id,
                use_local_dataset=use_local_dataset,
                provider=provider,
                progress_callback=progress_callback
            )
            
            print(f"Moral stories evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in moral stories evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

def run_crows_pairs_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run CrowS-Pairs evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting crows_pairs evaluation {message_id}")
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting CrowS-Pairs Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"crows_pairs progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_crows_pairs(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                system=system,
                provider=provider,
                progress_callback=progress_callback,
                db=db if not skip_db else None,
                message_id=message_id
            )
            
            print(f"CrowS-Pairs evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed crows_pairs evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in CrowS-Pairs evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

def run_truthfulqa_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run TruthfulQA evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting truthfulqa evaluation {message_id}")
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting TruthfulQA Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"truthfulqa progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_truthfulqa(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                system=system,
                provider=provider,
                progress_callback=progress_callback,
                db=db if not skip_db else None,
                message_id=message_id
            )
            
            print(f"TruthfulQA evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed truthfulqa evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in TruthfulQA evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

def run_arc_challenge_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run Arc-Challenge evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting arc_challenge evaluation {message_id}")
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting Arc-Challenge Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"arc_challenge progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_arc_challenge(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                system=system,
                provider=provider,
                progress_callback=progress_callback,
                db=db if not skip_db else None,
                message_id=message_id
            )
            
            print(f"Arc-Challenge evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed arc_challenge evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in Arc-Challenge evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

def run_sycophancy_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run Sycophancy evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting sycophancy evaluation {message_id}")
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting Sycophancy Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"sycophancy progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_sycophancy(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                system=system,
                provider=provider,
                progress_callback=progress_callback,
                db=db if not skip_db else None,
                message_id=message_id
            )
            
            print(f"Sycophancy evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed sycophancy evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in Sycophancy evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

def run_air_deception_evaluation(
    message_id: str,
    model: str,
    examples: int,
    context: Optional[Union[List[Dict[str, str]], str]],
    system: Optional[str],
    force_download: bool,
    skip_db: bool,
    use_local_dataset: bool = True,
    provider: str = "openai"
):
    """Run AIR-Deception evaluation and update the existing document."""
    try:
        log_memory_usage(f"starting air_deception evaluation {message_id}")
        
        with get_db_connection_managed() as db:
            print(f"\n=== Starting AIR-Deception Evaluation (Message ID: {message_id}) ===")
            
            # Convert context if needed
            converted_context = None
            if context:
                if isinstance(context, list):
                    try:
                        converted_context = []
                        for msg in context:
                            if hasattr(msg, 'role') and hasattr(msg, 'content'):
                                converted_context.append({'role': msg.role, 'content': msg.content})
                            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                converted_context.append(msg)
                            else:
                                raise ValueError(f"Invalid message format: {msg}")
                        print(f"Converted {len(converted_context)} context messages")
                    except Exception as e:
                        print(f"Error converting context: {e}")
                        converted_context = context
                else:
                    converted_context = context
            
            # Handle system prompt
            if system and converted_context:
                if isinstance(converted_context, list) and not any(msg.get('role') == 'system' for msg in converted_context if isinstance(msg, dict)):
                    converted_context.insert(0, {"role": "system", "content": system})
                    print(f"Added system message to context list")
            elif system:
                converted_context = [{"role": "system", "content": system}]
                print(f"Created new context with system message")
            
            # Define progress callback (just for logging, no DB updates)
            def progress_callback(current, total):
                percent = (current / total * 100) if total > 0 else 0
                print(f"Progress: {current}/{total} ({percent:.1f}%)")
                if current % 10 == 0:  # Log memory every 10 iterations
                    log_memory_usage(f"air_deception progress {current}/{total}")
            
            # Run evaluation (this will create its own document)
            result = evaluate_air_deception(
                model_name=model,
                num_examples=examples,
                context=converted_context,
                system=system,
                provider=provider,
                progress_callback=progress_callback,
                db=db if not skip_db else None,
                message_id=message_id
            )
            
            print(f"AIR-Deception evaluation completed successfully for {message_id}")
            
            # Force garbage collection after completion
            gc.collect()
            log_memory_usage(f"completed air_deception evaluation {message_id}")
            
            # The evaluation function already updated the document, no need to update again
            print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in AIR-Deception evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            with get_db_connection_managed() as db:
                update_document_status(
                    db, message_id, "error",
                    error=error_msg,
                    error_at=datetime.now()
                )
                print(f"❌ Updated document {message_id} with error status")
        except Exception as db_error:
            print(f"Failed to update error status in database: {db_error}")
        
        # Clean up on error
        gc.collect()

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 