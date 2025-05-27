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
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

class TruthfulQARequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

class ArcChallengeRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

class SycophancyRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

class AirDeceptionRequest(BaseModel):
    model: str
    examples: int = 5
    context: Optional[Union[List[MessageModel], str]] = None
    system: Optional[str] = None
    message_id: str  # Required - frontend must provide this
    force_download: bool = False
    skip_db: bool = False
    use_local_dataset: bool = True
    provider: str = "openai"  # 'openai' or 'anthropic'

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
            "/tasks": "GET - List all tasks and statuses",
            "/health": "GET - Check API health"
        },
        "available_evaluations": {
            "moral_stories": "Evaluate moral reasoning and ethical decision making",
            "crows_pairs": "Evaluate social biases and stereotyping",
            "truthfulqa": "Evaluate truthfulness and factual accuracy",
            "arc_challenge": "Evaluate scientific reasoning and knowledge",
            "sycophancy": "Evaluate resistance to sycophantic behavior",
            "air_deception": "Evaluate safety and refusal of harmful requests"
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
    
    # Get count of processing tasks from database
    processing_tasks_count = 0
    try:
        db = get_db_connection()
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
        print(f"Error getting processing tasks count: {e}")
    
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
async def list_tasks(clear_completed: bool = False):
    """List all current evaluation tasks and their status."""
    try:
        db = get_db_connection()
        all_tasks = []
        
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
                
                # Get all tasks from this collection
                tasks = list(collection.find().sort("updated_at", -1).limit(50))
                
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
        
        return {
            "tasks": all_tasks[:100],  # Limit to 100 most recent
            "total_count": len(all_tasks)
        }
        
    except Exception as e:
        print(f"Error listing tasks: {e}")
        return {
            "error": f"Error retrieving tasks: {str(e)}",
            "tasks": [],
            "total_count": 0
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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_moral_stories_with_openai(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            cache_dir="/tmp/hf_cache_moral_stories",
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id,
            use_local_dataset=use_local_dataset,
            provider=provider,
            progress_callback=progress_callback
        )
        
        print(f"Moral stories evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in moral stories evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_crows_pairs(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            system=system,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id
        )
        
        print(f"CrowS-Pairs evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in CrowS-Pairs evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_truthfulqa(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            system=system,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id
        )
        
        print(f"TruthfulQA evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in TruthfulQA evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_arc_challenge(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            system=system,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id
        )
        
        print(f"Arc-Challenge evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in Arc-Challenge evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_sycophancy(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            system=system,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id
        )
        
        print(f"Sycophancy evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in Sycophancy evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

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
    db = get_db_connection() if not skip_db else None
    
    try:
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
        
        # Run evaluation (this will create its own document)
        result = evaluate_air_deception(
            model_name=model,
            num_examples=examples,
            context=converted_context,
            system=system,
            provider=provider,
            progress_callback=progress_callback,
            db=db,  # Pass the actual database connection so results get saved
            message_id=message_id
        )
        
        print(f"AIR-Deception evaluation completed successfully for {message_id}")
        
        # The evaluation function already updated the document, no need to update again
        print(f"✅ Evaluation function handled document update for {message_id}")
        
    except Exception as e:
        error_msg = f"Error in AIR-Deception evaluation: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        if db is not None:
            update_document_status(
                db, message_id, "error",
                error=error_msg,
                error_at=datetime.now()
            )
            print(f"❌ Updated document {message_id} with error status")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 