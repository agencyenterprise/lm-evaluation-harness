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
    
    print(f"API keys found in environment: {', '.join(api_keys_found) if api_keys_found else 'None'}")
    
    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key:
            print(f"Found Anthropic API key (length: {len(api_key)})")
            return "anthropic", api_key
        raise ValueError("Anthropic API key not found. Please set ANTHROPIC_API_KEY in your environment or .env file.")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            print(f"Found OpenAI API key (length: {len(api_key)})")
            return "openai", api_key
        raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment or .env file.")

def create_chat_completion(provider: str, model: str, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0) -> str:
    """Create a chat completion using either OpenAI or Anthropic based on provider."""
    if provider == "anthropic":
        return create_anthropic_chat_completion(model, messages, max_tokens, temperature)
    else:
        return create_openai_chat_completion(model, messages, max_tokens, temperature)

def create_openai_chat_completion(model: str, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0) -> str:
    """Call the OpenAI Chat Completions API with a list of messages."""
    api_type, api_key = get_api_key("openai")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    return response.json()["choices"][0]["message"]["content"]

def create_anthropic_chat_completion(model: str, messages: List[Dict[str, str]], max_tokens: int = 100, temperature: float = 0) -> str:
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
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}, {response.text}")
    
    return response.json()["content"][0]["text"]

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
            else:
                split_name = file.replace(f"{dataset_name}_", "").replace(".json", "")
                
            file_path = os.path.join(data_dir, file)
            
            # Load the split
            df = pd.read_json(file_path, lines=True)
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

def create_deterministic_sample(dataset_split, num_examples: int, model_name: str, context_type: str):
    """Create a deterministic sample from a dataset split."""
    split_size = len(dataset_split)
    if num_examples > split_size:
        print(f"Warning: Requested {num_examples} examples but dataset only has {split_size}. Using all available examples.")
        num_examples = split_size
    
    # Use deterministic sampling
    import numpy as np
    seed = hash(f"{model_name}_{context_type}") % 10000
    np.random.seed(seed)
    
    indices = np.random.choice(split_size, num_examples, replace=False)
    samples = dataset_split.select(indices)
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
    if isinstance(obj, (np.integer, np.int64, np.int32)):
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
        
        if dataset_type in ["crows_pairs", "truthfulqa"]:
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
        
        # Insert results
        result = collection.insert_one(serializable_results)
        print(f"Saved {dataset_type} results to database collection '{collection_name}' with ID: {result.inserted_id}")
        return result.inserted_id
    except Exception as e:
        print(f"Error saving to database: {e}")
        return None 