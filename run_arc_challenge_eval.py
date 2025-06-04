#!/usr/bin/env python3

import os
import sys
import argparse
import json
import numpy as np
import re
import time
from typing import List, Dict, Optional, Union

# Import shared functions from eval_utils
from eval_utils import (
    get_api_key, create_chat_completion, load_dataset_local, 
    parse_context, get_context_type, create_deterministic_sample,
    safe_extract_choice, print_evaluation_header, print_evaluation_results,
    get_mongodb_connection, save_results_to_db, make_json_serializable
)

def load_arc_challenge_local():
    """Load the Arc-Challenge dataset from local files."""
    return load_dataset_local("arc_challenge")

def create_arc_challenge_prompt(question: str, choices: List[str], context: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    """Create prompt messages for Arc-Challenge evaluation."""
    messages = context.copy() if context else []
    
    # Format the choices
    choice_text = "\n".join([f"{chr(65+i)}: {choice}" for i, choice in enumerate(choices)])
    
    # Create the reasoning evaluation query
    arc_query = f"""Question: {question}

{choice_text}

Which answer is correct? Answer with just the letter ({chr(65)}-{chr(65+len(choices)-1)})."""
    
    messages.append({"role": "user", "content": arc_query})
    return messages

def evaluate_arc_challenge(
    model_name: str,
    num_examples: int = 5,
    context: Optional[Union[str, List, Dict]] = None,
    system: Optional[str] = None,
    provider: str = "openai",
    progress_callback=None,
    db=None,
    message_id: Optional[str] = None
):
    """Evaluate Arc-Challenge using chat models."""
    
    # Load dataset
    dataset = load_arc_challenge_local()
    
    # Parse context and determine type
    parsed_context = parse_context(context)
    
    # Handle system prompt like moral stories does
    if system and parsed_context:
        if not any(msg.get('role') == 'system' for msg in parsed_context if isinstance(msg, dict)):
            parsed_context.insert(0, {"role": "system", "content": system})
            print(f"Added system message to context list")
    elif system:
        parsed_context = [{"role": "system", "content": system}]
        print(f"Created new context with system message")
    
    context_type = get_context_type(parsed_context)
    
    # Get the dataset split (prefer test, fallback to validation, then train)
    split_name = "test" if "test" in dataset else ("validation" if "validation" in dataset else "train")
    split_data = dataset[split_name]
    
    # Create deterministic sample
    samples = create_deterministic_sample(split_data, num_examples, model_name, context_type)
    print_evaluation_header("Arc-Challenge", model_name, context_type, num_examples)
    
    # Process samples
    all_messages = []
    sample_data = []
    
    print(f"Preparing {num_examples} examples for {context_type} evaluation...")
    for sample in samples:
        question = sample["question"]
        
        # Handle Arc-Challenge format
        if "choices" in sample and "text" in sample["choices"]:
            # Arc format with choices.text and choices.label
            choices = sample["choices"]["text"]
            labels = sample["choices"]["label"]
            answer_key = sample["answerKey"]
            
            # Find the correct answer index
            try:
                correct_answer_idx = labels.index(answer_key)
            except ValueError:
                print(f"Warning: Answer key '{answer_key}' not found in labels {labels}")
                correct_answer_idx = 0
        else:
            print(f"Warning: Unrecognized sample format for question: {question}")
            continue
        
        # Create messages
        messages = create_arc_challenge_prompt(question, choices, parsed_context)
        
        all_messages.append(messages)
        sample_data.append({
            "question": question,
            "choices": choices,
            "labels": labels,
            "answer_key": answer_key,
            "correct_answer_idx": correct_answer_idx,
            "correct_answer": choices[correct_answer_idx] if correct_answer_idx < len(choices) else "Unknown"
        })
    
    # Generate responses
    print(f"Generating responses from {model_name} using {provider} API ({context_type})...")
    responses = []
    model_choices = []
    
    for i, (messages, sample_info) in enumerate(zip(all_messages, sample_data)):
        print(f"\nEvaluating example {i+1}/{num_examples}:")
        print(f"Question: {sample_info['question']}")
        print(f"Choices: {sample_info['choices']}")
        print(f"Correct Answer: {sample_info['answer_key']} - {sample_info['correct_answer']}")
        
        try:
            response = create_chat_completion(provider, model_name, messages)
            responses.append(response)
            print(f"Response: {response}")
            
            # Extract choice letter from response using shared utility
            num_choices = len(sample_info['choices'])
            valid_choices = [chr(65+i) for i in range(num_choices)]
            model_choice = safe_extract_choice(response, valid_choices)
            model_choices.append(model_choice)
            print(f"Extracted choice: {model_choice}")
                
            if progress_callback:
                progress_callback(i+1, num_examples)
        except Exception as e:
            print(f"Error: {e}")
            responses.append(f"ERROR: {str(e)}")
            model_choices.append('X')
            
            if progress_callback:
                progress_callback(i+1, num_examples)
    
    # Calculate accuracy
    correct_count = 0
    total_valid = 0
    
    for choice, sample_info in zip(model_choices, sample_data):
        if choice != 'X':
            total_valid += 1
            choice_idx = ord(choice) - 65  # Convert A=0, B=1, etc.
            if choice_idx == sample_info['correct_answer_idx']:
                correct_count += 1
    
    accuracy = correct_count / total_valid if total_valid > 0 else 0
    
    # Create results
    results = {
        "model": model_name,
        "context_type": context_type,
        "prior_context": make_json_serializable(parsed_context),
        "total_examples": len(model_choices),
        "arc_challenge": {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_valid": total_valid,
            "invalid_choices": len(model_choices) - total_valid,
            "total_examples": len(model_choices),
            "samples": []
        }
    }
    
    # Add message ID if provided
    if message_id:
        results["message_id"] = message_id
    
    # Add samples to results
    for i, (messages, response, choice, sample_info) in enumerate(zip(all_messages, responses, model_choices, sample_data)):
        choice_idx = ord(choice) - 65 if choice != 'X' else -1
        is_correct = choice_idx == sample_info['correct_answer_idx'] if choice != 'X' else False
        
        sample_result = {
            "question": sample_info["question"],
            "choices": sample_info["choices"],
            "labels": sample_info["labels"],
            "correct_answer": sample_info["correct_answer"],
            "correct_choice": sample_info["answer_key"],
            "model_response": make_json_serializable(response),
            "model_choice": choice,
            "is_correct": is_correct,
            "prompt_message": make_json_serializable(messages[-1]) if messages else None
        }
        results["arc_challenge"]["samples"].append(sample_result)
    
    # Print results using shared utility
    metrics = {
        "accuracy": accuracy,
        "correct_count": f"{correct_count}/{total_valid}",
        "valid_responses": f"{total_valid}/{len(model_choices)}"
    }
    print_evaluation_results("Arc-Challenge", model_name, context_type, metrics)
    
    # Save to MongoDB if available
    if db is not None:
        save_results_to_db(db, results, "arc_challenge")
        print("Arc-Challenge results saved to database")
    else:
        print("No database connection provided, results not saved")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run Arc-Challenge evaluation")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", 
                       help="Model name")
    parser.add_argument("--provider", type=str, default="openai", 
                       choices=["openai", "anthropic", "google"], help="AI provider")
    parser.add_argument("--examples", type=int, default=5, 
                       help="Number of examples to evaluate")
    parser.add_argument("--context", type=str, default=None, 
                       help="Prior context (optional)")
    parser.add_argument("--skip-db", action="store_true", 
                       help="Skip database saving")
    
    args = parser.parse_args()
    
    context = None
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            context = args.context
    
    # Establish database connection unless skipped
    db = None
    if not args.skip_db:
        try:
            db = get_mongodb_connection()
            print("✓ Connected to MongoDB")
        except Exception as e:
            print(f"⚠️ Failed to connect to MongoDB: {e}")
            print("Proceeding without database saving...")
    
    evaluate_arc_challenge(
        model_name=args.model,
        num_examples=args.examples,
        context=context,
        provider=args.provider,
        db=db
    )

if __name__ == "__main__":
    main() 