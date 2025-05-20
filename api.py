from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Union, Any
import json
import os
from dotenv import load_dotenv

# Import the evaluation function
from run_moral_stories_eval_gen import evaluate_moral_stories_with_openai, get_mongodb_connection

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Moral Stories Evaluation API",
    description="API for evaluating language models on moral reasoning tasks",
    version="1.0.0"
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
    return {"message": "Moral Stories Evaluation API is running"}

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
        db = None if skip_db else get_mongodb_connection()
        
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
            force_download=force_download,
            db=db,
            message_id=message_id
        )
        
        # Store and return the result
        evaluation_results[task_id] = {
            "status": "completed",
            "result": result
        }
    except Exception as e:
        evaluation_results[task_id] = {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    # Get port from environment variable for Railway
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True) 