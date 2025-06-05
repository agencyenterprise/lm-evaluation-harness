#!/usr/bin/env python3

import os
import sys
import json
import requests
import time
import traceback
import numpy as np
from typing import List, Dict, Optional, Union
from dotenv import load_dotenv
from datasets import Dataset, DatasetDict
import pandas as pd
import pymongo
from datetime import datetime

def get_api_key(provider="openai"):
    """Get API key from .env file or environment variables."""
    env_paths = ['.env', '../.env', '~/.env', os.path.expanduser('~/.env')]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            print(f"Loaded environment from {env_path}")
            break
    
    print(f"\nLooking for {provider} API key...")
    env_vars = list(os.environ.keys())
    api_keys_found = []
    if "OPENAI_API_KEY" in env_vars:
        api_keys_found.append("OPENAI_API_KEY")
    if "ANTHROPIC_API_KEY" in env_vars:
        api_keys_found.append("ANTHROPIC_API_KEY")
    if "GEMINI_API_KEY" in env_vars:
        api_keys_found.append("GEMINI_API_KEY")
    
    print(f"API keys found in environment: {', '.join(api_keys_found) if api_keys_found else 'None'}")
    
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print(f"Found Anthropic API key (length: {len(api_key)})")
            return "anthropic", api_key
        raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your environment or .env file.")
    elif provider == "google":
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            print(f"Found Google/Gemini API key (length: {len(api_key)})")
            return "google", api_key
        raise ValueError("Google/Gemini API key not found. Please set GEMINI_API_KEY in your environment or .env file.")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print(f"Found OpenAI API key (length: {len(api_key)})")
            return "openai", api_key
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file.")

def create_chat_completion(provider: str, model: str, messages: List[Dict[str, str]]) -> str:
    """Create a chat completion using OpenAI, Anthropic, or Google based on provider."""
    if provider == "anthropic":
        return create_anthropic_chat_completion(model, messages)
    elif provider == "google":
        return create_google_chat_completion(model, messages)
    else:
        return create_openai_chat_completion(model, messages)

def create_openai_chat_completion(model: str, messages: List[Dict[str, str]]) -> str:
    """Call the OpenAI Chat Completions API with a list of messages."""
    api_type, api_key = get_api_key("openai")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

def create_anthropic_chat_completion(model: str, messages: List[Dict[str, str]]) -> str:
    """Call the Anthropic Chat Completions API with a list of messages."""
    api_type, api_key = get_api_key("anthropic")
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    
    system_content = ""
    converted_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            converted_messages.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            converted_messages.append({"role": "assistant", "content": msg["content"]})
    
    payload = {
        "model": model,
        "messages": converted_messages,
        "system": system_content,
        "max_tokens": 1024
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    return response.json()["content"][0]["text"]

def create_google_chat_completion(model: str, messages: List[Dict[str, str]]) -> str:
    """Call the Google Gemini API with a list of messages."""
    api_type, api_key = get_api_key("google")
    
    # Convert messages to Google's format
    # Google expects an array of contents, each with role and parts
    contents = []
    
    for msg in messages:
        if msg["role"] == "system":
            # Google doesn't support system messages - we'll prepend system message as user message
            contents.append({
                "role": "user",
                "parts": [{"text": f"System instruction: {msg['content']}\n\nPlease follow this instruction in your responses. Keep your response concise and direct."}]
            })
        elif msg["role"] == "user":
            # Add instruction for concise responses to user messages
            user_content = msg["content"]
            if not any(phrase in user_content.lower() for phrase in ["answer with just", "respond with only", "keep it brief", "concise"]):
                user_content += "\n\nPlease provide a direct, concise response."
            
            contents.append({
                "role": "user", 
                "parts": [{"text": user_content}]
            })
        elif msg["role"] == "assistant":
            contents.append({
                "role": "model",  # Google uses 'model' instead of 'assistant'
                "parts": [{"text": msg["content"]}]
            })
    
    # Handle case where we only have system message - add a simple user message
    if len(contents) == 1 and "System instruction:" in contents[0]["parts"][0]["text"]:
        contents.append({
            "role": "user",
            "parts": [{"text": "Please respond with your understanding."}]
        })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,  # Significantly increased to handle thinking process
            "topP": 0.8,
            "topK": 40
        },
        "safetySettings": [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
    }
    
    # Google API endpoint format
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"Google API Error: {response.status_code}, {response.text}")
    
    response_data = response.json()
    
    # Extract text from Google's response format
    if "candidates" in response_data and len(response_data["candidates"]) > 0:
        candidate = response_data["candidates"][0]
        
        # Check finish reason for issues
        finish_reason = candidate.get("finishReason", "UNKNOWN")
        if finish_reason in ["MAX_TOKENS", "RECITATION", "SAFETY"]:
            print(f"Warning: Google API response truncated due to {finish_reason}")
        
        if "content" in candidate:
            content = candidate["content"]
            if "parts" in content and len(content["parts"]) > 0:
                # Normal case - extract text from parts
                first_part = content["parts"][0]
                if "text" in first_part:
                    return first_part["text"]
                else:
                    raise Exception(f"Google API response part missing 'text' field: {first_part}")
            else:
                # Handle case where content exists but parts is missing/empty
                if finish_reason == "MAX_TOKENS":
                    raise Exception("Google API response truncated due to token limit. The model hit maxOutputTokens before completing the response.")
                elif finish_reason == "SAFETY":
                    raise Exception("Google API response blocked due to safety filters.")
                elif finish_reason == "RECITATION":
                    raise Exception("Google API response blocked due to recitation detection.")
                else:
                    raise Exception(f"Google API response content missing 'parts' field. FinishReason: {finish_reason}, Content: {content}")
        else:
            raise Exception(f"Google API response candidate missing 'content' field: {candidate}")
    
    raise Exception(f"Google API response missing candidates or empty candidates list: {response_data}")

def load_dataset_local(dataset_name: str, expected_files: List[str] = None):
    """
    Load a dataset from local files with automatic split detection.
    
    Args:
        dataset_name: Name of the dataset (used for file prefixes)
        expected_files: List of expected file patterns to check for
    
    Returns:
        DatasetDict with the loaded splits
    """
    print(f"Loading {dataset_name} dataset from local files...")
    start_time = time.time()
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        # Auto-detect available splits
        available_files = os.listdir(data_dir)
        
        # Handle different dataset naming patterns
        if dataset_name == "crows_pairs":
            dataset_files = [f for f in available_files if f.startswith("crows_pairs_") and f.endswith('.json') and 'info' not in f]
        elif dataset_name == "truthfulqa_mc":
            dataset_files = [f for f in available_files if f.startswith("truthfulqa_mc_") and f.endswith('.json') and 'info' not in f]
        elif dataset_name == "moral_stories":
            dataset_files = [f for f in available_files if f.startswith("moral_stories_") and f.endswith('.json') and 'info' not in f]
        elif dataset_name == "arc_challenge":
            dataset_files = [f for f in available_files if f.startswith("arc_challenge_") and f.endswith('.json') and 'info' not in f]
        elif dataset_name == "sycophancy":
            dataset_files = [f for f in available_files if f.startswith("sycophancy_") and f.endswith('.json') and 'info' not in f]
        elif dataset_name == "air_deception":
            dataset_files = [f for f in available_files if f.startswith("air_deception_") and f.endswith('.json') and 'info' not in f]
        else:
            # Generic pattern matching
            dataset_files = [f for f in available_files if f.startswith(f"{dataset_name}_") and f.endswith('.json') and 'info' not in f]
        
        if not dataset_files:
            print(f"No dataset files found for {dataset_name} in {data_dir}")
            print(f"Available files: {available_files}")
            print("Please run download_dataset.py first to create the local files")
            sys.exit(1)
        
        dataset_dict = {}
        
        for file in dataset_files:
            # Extract split name from filename
            if dataset_name == "crows_pairs":
                split_name = file.replace("crows_pairs_", "").replace(".json", "")
            elif dataset_name == "truthfulqa_mc":
                split_name = file.replace("truthfulqa_mc_", "").replace(".json", "")
            elif dataset_name == "moral_stories":
                split_name = file.replace("moral_stories_", "").replace(".json", "")
            elif dataset_name == "arc_challenge":
                split_name = file.replace("arc_challenge_", "").replace(".json", "")
            elif dataset_name == "sycophancy":
                split_name = file.replace("sycophancy_", "").replace(".json", "")
            elif dataset_name == "air_deception":
                split_name = file.replace("air_deception_", "").replace(".json", "")
            else:
                split_name = file.replace(f"{dataset_name}_", "").replace(".json", "")
                
            file_path = os.path.join(data_dir, file)
            
            # Load the split - handle different JSON formats
            try:
                # First try JSONL format (lines=True) for existing datasets
                if dataset_name in ["moral_stories", "crows_pairs", "truthfulqa_mc"]:
                    df = pd.read_json(file_path, lines=True)
                else:
                    # For new datasets, try regular JSON format first
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
            except (ValueError, json.JSONDecodeError):
                # Fallback to the other format
                try:
                    if dataset_name in ["moral_stories", "crows_pairs", "truthfulqa_mc"]:
                        # Try regular JSON format as fallback
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        df = pd.DataFrame(data)
                    else:
                        # Try JSONL format as fallback
                        df = pd.read_json(file_path, lines=True)
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
            
            dataset = Dataset.from_pandas(df)
            dataset_dict[split_name] = dataset
            
            print(f"  - Loaded {split_name} split: {len(df)} examples")
        
        dataset = DatasetDict(dataset_dict)
        
        elapsed = time.time() - start_time
        print(f"{dataset_name} dataset loaded from local files in {elapsed:.2f} seconds!")
        return dataset
        
    except Exception as e:
        print(f"Error loading {dataset_name} dataset from local files: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)

def parse_context(context: Optional[Union[str, List, Dict]]) -> List[Dict[str, str]]:
    """Parse context input into a list of message dictionaries."""
    parsed_context = []
    if isinstance(context, list):
        parsed_context = context
    elif isinstance(context, str) and context.strip():
        parsed_context = [{"role": "user", "content": context}]
    return parsed_context

def get_context_type(context: Optional[Union[str, List, Dict]]) -> str:
    """Determine the context type for evaluation naming."""
    parsed_context = parse_context(context)
    return "with_context" if parsed_context else "baseline"

def get_baseline_examples_from_db(db, model_name: str, dataset_type: str = "general") -> Optional[List[Dict]]:
    """
    Fetch examples used in baseline evaluation from database.
    
    Args:
        db: MongoDB database connection
        model_name: Name of the model to find baseline for
        dataset_type: Type of dataset ('general' for moral_stories, 'crows_pairs', 'truthfulqa', etc.)
    
    Returns:
        List of sample dictionaries used in baseline evaluation, or None if not found
    """
    if db is None:
        return None
    
    try:
        # Determine collection name for baseline
        if dataset_type == "general":
            collection_name = "baseline_results"
        else:
            collection_name = f"{dataset_type}_baseline_results"
        
        collection = db[collection_name]
        
        # Find the most recent baseline evaluation for this model
        baseline_doc = collection.find_one(
            {"model": model_name, "context_type": "baseline"},
            sort=[("timestamp", -1)]
        )
        
        if baseline_doc:
            # Extract samples from the baseline evaluation
            if dataset_type == "general" and "moral_stories_gen" in baseline_doc:
                samples = baseline_doc["moral_stories_gen"].get("samples", [])
            elif dataset_type == "crows_pairs" and "crows_pairs" in baseline_doc:
                samples = baseline_doc["crows_pairs"].get("samples", [])
            elif dataset_type == "truthfulqa" and "truthfulqa" in baseline_doc:
                samples = baseline_doc["truthfulqa"].get("samples", [])
            elif dataset_type == "arc_challenge" and "arc_challenge" in baseline_doc:
                samples = baseline_doc["arc_challenge"].get("samples", [])
            elif dataset_type == "sycophancy" and "sycophancy" in baseline_doc:
                samples = baseline_doc["sycophancy"].get("samples", [])
            elif dataset_type == "air_deception" and "air_deception" in baseline_doc:
                samples = baseline_doc["air_deception"].get("samples", [])
            else:
                print(f"Warning: Could not find samples in baseline document for {dataset_type}")
                return None
            
            print(f"Found baseline evaluation for {model_name} with {len(samples)} examples")
            return samples
        else:
            print(f"No baseline evaluation found for model {model_name} in {collection_name}")
            return None
            
    except Exception as e:
        print(f"Error fetching baseline examples: {e}")
        return None

def create_deterministic_sample(dataset_split, num_examples: int, model_name: str, context_type: str, db=None, dataset_type: str = "general"):
    """
    Create a deterministic sample from a dataset split.
    For with-context evaluations, tries to use the same examples as the baseline evaluation.
    """
    split_size = len(dataset_split)
    if num_examples > split_size:
        print(f"Warning: Requested {num_examples} examples but dataset only has {split_size}. Using all available examples.")
        num_examples = split_size
    
    # If this is a with-context evaluation, try to use baseline examples
    if context_type == "with_context" and db is not None:
        baseline_samples = get_baseline_examples_from_db(db, model_name, dataset_type)
        
        if baseline_samples:
            # Extract the identifying information from baseline samples to find matching examples
            if dataset_type == "general":  # moral_stories
                baseline_contexts = [sample.get("context", "") for sample in baseline_samples]
                matching_indices = []
                
                for i, example in enumerate(dataset_split):
                    # Create context from norm, situation, and intention like in moral stories
                    example_context = (
                        example["norm"].capitalize() + " " + 
                        example["situation"].capitalize() + " " + 
                        example["intention"].capitalize()
                    )
                    if example_context in baseline_contexts:
                        matching_indices.append(i)
                
                if len(matching_indices) >= num_examples:
                    # Use the same examples as baseline
                    selected_indices = matching_indices[:num_examples]
                    samples = dataset_split.select(selected_indices)
                    print(f"Using {num_examples} examples from baseline evaluation (matched {len(matching_indices)} total)")
                    return samples
                else:
                    print(f"Could only match {len(matching_indices)}/{num_examples} examples from baseline, falling back to random sampling")
            
            elif dataset_type == "crows_pairs":
                baseline_pairs = [(sample.get("sent_more", ""), sample.get("sent_less", "")) for sample in baseline_samples]
                # Create a mapping from pair to its original position in baseline
                pair_to_baseline_index = {pair: i for i, pair in enumerate(baseline_pairs)}
                
                # Find all examples in dataset that match baseline pairs
                matching_examples = []
                
                for i, example in enumerate(dataset_split):
                    example_pair = (example["sent_more"], example["sent_less"])
                    if example_pair in baseline_pairs:
                        baseline_index = pair_to_baseline_index[example_pair]
                        matching_examples.append((baseline_index, i))
                
                if len(matching_examples) >= num_examples:
                    # Sort by original baseline order and take the first num_examples
                    matching_examples.sort(key=lambda x: x[0])  # Sort by baseline_index
                    selected_indices = [x[1] for x in matching_examples[:num_examples]]  # Get dataset indices
                    samples = dataset_split.select(selected_indices)
                    print(f"Using {num_examples} examples from baseline evaluation (matched {len(matching_examples)} total)")
                    return samples
                else:
                    print(f"Could only match {len(matching_examples)}/{num_examples} examples from baseline, falling back to random sampling")
            
            elif dataset_type in ["truthfulqa", "arc_challenge"]:
                baseline_questions = [sample.get("question", "") for sample in baseline_samples]
                # Create a mapping from question to its original position in baseline
                question_to_baseline_index = {question: i for i, question in enumerate(baseline_questions)}
                
                # Find all examples in dataset that match baseline questions
                matching_examples = []
                
                for i, example in enumerate(dataset_split):
                    if example["question"] in baseline_questions:
                        baseline_index = question_to_baseline_index[example["question"]]
                        matching_examples.append((baseline_index, i))
                
                if len(matching_examples) >= num_examples:
                    # Sort by original baseline order and take the first num_examples
                    matching_examples.sort(key=lambda x: x[0])  # Sort by baseline_index
                    selected_indices = [x[1] for x in matching_examples[:num_examples]]  # Get dataset indices
                    samples = dataset_split.select(selected_indices)
                    print(f"Using {num_examples} examples from baseline evaluation (matched {len(matching_examples)} total)")
                    return samples
                else:
                    print(f"Could only match {len(matching_examples)}/{num_examples} examples from baseline, falling back to random sampling")
            
            elif dataset_type == "sycophancy":
                baseline_questions = [sample.get("question", "") for sample in baseline_samples]
                # Create a mapping from question to its original position in baseline
                question_to_baseline_index = {question: i for i, question in enumerate(baseline_questions)}
                
                # Find all examples in dataset that match baseline questions
                matching_examples = []
                
                for i, example in enumerate(dataset_split):
                    if example["question"] in baseline_questions:
                        baseline_index = question_to_baseline_index[example["question"]]
                        matching_examples.append((baseline_index, i))
                
                if len(matching_examples) >= num_examples:
                    # Sort by original baseline order and take the first num_examples
                    matching_examples.sort(key=lambda x: x[0])  # Sort by baseline_index
                    selected_indices = [x[1] for x in matching_examples[:num_examples]]  # Get dataset indices
                    samples = dataset_split.select(selected_indices)
                    print(f"Using {num_examples} examples from baseline evaluation (matched {len(matching_examples)} total)")
                    return samples
                else:
                    print(f"Could only match {len(matching_examples)}/{num_examples} examples from baseline, falling back to random sampling")
            
            elif dataset_type == "air_deception":
                baseline_prompts = [sample.get("prompt", "") for sample in baseline_samples]
                # Create a mapping from prompt to its original position in baseline
                prompt_to_baseline_index = {prompt: i for i, prompt in enumerate(baseline_prompts)}
                
                # Find all examples in dataset that match baseline prompts
                matching_examples = []
                
                for i, example in enumerate(dataset_split):
                    if example["prompt"] in baseline_prompts:
                        baseline_index = prompt_to_baseline_index[example["prompt"]]
                        matching_examples.append((baseline_index, i, example["prompt"]))
                
                if len(matching_examples) >= num_examples:
                    # Sort by original baseline order and take the first num_examples
                    matching_examples.sort(key=lambda x: x[0])  # Sort by baseline_index
                    selected_indices = [x[1] for x in matching_examples[:num_examples]]  # Get dataset indices
                    samples = dataset_split.select(selected_indices)
                    print(f"Using {num_examples} examples from baseline evaluation (matched {len(matching_examples)} total)")
                    return samples
                else:
                    print(f"Could only match {len(matching_examples)}/{num_examples} examples from baseline, falling back to random sampling")
    
    # Fall back to deterministic random sampling
    import numpy as np
    seed = hash(f"{model_name}_{context_type}") % 10000
    np.random.seed(seed)
    
    indices = np.random.choice(split_size, num_examples, replace=False)
    samples = dataset_split.select(indices)
    
    if context_type == "with_context":
        print(f"Selected {num_examples} examples with random sampling (seed {seed}) - no baseline found or couldn't match examples")
    else:
        print(f"Selected {num_examples} examples with seed {seed}")
    
    return samples

def safe_extract_choice(response: str, valid_choices: List[str]) -> str:
    """
    Safely extract a choice from model response.
    
    Args:
        response: The model's response text
        valid_choices: List of valid choices (e.g., ['A', 'B'] or ['A', 'B', 'C', 'D'])
    
    Returns:
        The extracted choice or 'X' if no valid choice found
    """
    import re
    
    # Create pattern for valid choices
    choice_pattern = f"[{''.join(valid_choices)}]"
    match = re.search(f'(?i)\\b({choice_pattern})\\b', response)
    
    if match:
        return match.group(0).upper()
    else:
        return 'X'

def print_evaluation_header(dataset_name: str, model_name: str, context_type: str, num_examples: int):
    """Print a standardized evaluation header."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Evaluation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Context Type: {context_type}")
    print(f"Examples: {num_examples}")
    print(f"{'='*60}")

def print_evaluation_results(dataset_name: str, model_name: str, context_type: str, metrics: Dict):
    """Print standardized evaluation results."""
    print(f"\n{'='*60}")
    print(f"{dataset_name} Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"Context Type: {context_type}")
    
    # Print dataset-specific metrics
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            if metric_name.endswith('_score') or metric_name == 'accuracy':
                print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.2%}")
            else:
                print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
        else:
            print(f"{metric_name.replace('_', ' ').title()}: {metric_value}")
    
    print(f"{'='*60}")

# Helper function to make data JSON serializable
def make_json_serializable(obj):
    # Handle pydantic models and MessageModel objects
    if hasattr(obj, 'model_dump'):
        # Pydantic v2 models
        return make_json_serializable(obj.model_dump())
    elif hasattr(obj, 'dict'):
        # Pydantic v1 models
        return make_json_serializable(obj.dict())
    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'MessageModel':
        # Specific handling for MessageModel objects
        if hasattr(obj, 'role') and hasattr(obj, 'content'):
            return {"role": str(obj.role), "content": str(obj.content)}
        else:
            return make_json_serializable(obj.__dict__)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'dtype'):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    else:
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)

def get_mongodb_connection():
    """Get MongoDB connection from .env file or environment variables."""
    # Load .env if it exists
    env_paths = [
        '.env',
        '../.env',
        '~/.env',
        os.path.expanduser('~/.env')
    ]
    
    for env_path in env_paths:
        if os.path.exists(env_path):
            load_dotenv(env_path)
            break
    
    mongodb_uri = os.environ.get("MONGODB_URI")
    if not mongodb_uri:
        raise ValueError("No MongoDB URI found. Please set MONGODB_URI in your environment or .env file.")
    
    try:
        client = pymongo.MongoClient(mongodb_uri)
        db = client.test
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

def save_results_to_db(db, results, dataset_type="general"):
    """Save evaluation results to MongoDB with dataset-specific collection naming."""
    try:
        # Determine collection based on context type and dataset
        context_type = results.get('context_type', 'baseline')
        
        if dataset_type in ["crows_pairs", "truthfulqa", "arc_challenge", "sycophancy", "air_deception"]:
            # Use dataset-specific collection names
            collection_name = f"{dataset_type}_{context_type}_results"
        else:
            # Default to original naming for moral stories
            collection_name = f"{context_type}_results"
            
        collection = db[collection_name]
        
        # Add timestamp
        results_copy = results.copy()
        results_copy["timestamp"] = datetime.now()
        
        # Make data serializable
        serializable_results = make_json_serializable(results_copy)
        
        # Check if we should update existing document (when message_id is provided)
        message_id = serializable_results.get("message_id")
        if message_id:
            # Try to find and update existing document with this message_id and status="processing"
            existing_doc = collection.find_one({"message_id": message_id, "status": "processing"})
            if existing_doc:
                # Update the existing document
                serializable_results["status"] = "completed"
                serializable_results["completed_at"] = datetime.now()
                update_result = collection.update_one(
                    {"_id": existing_doc["_id"]},
                    {"$set": serializable_results}
                )
                print(f"Updated existing document with message_id {message_id} in collection '{collection_name}'")
                return existing_doc["_id"]
        
        # If no existing document found or no message_id, create new document
        result = collection.insert_one(serializable_results)
        print(f"Saved {dataset_type} results to database collection '{collection_name}' with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None 