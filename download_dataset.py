#!/usr/bin/env python3

import os
import json
from datasets import load_dataset
import pandas as pd

def download_and_save_dataset(dataset_name, dataset_config=None, save_name=None):
    """
    Generic function to download and save a dataset.
    
    Args:
        dataset_name: The HuggingFace dataset name
        dataset_config: Optional dataset configuration
        save_name: Optional custom name for saving files (defaults to dataset_name)
    """
    if save_name is None:
        save_name = dataset_name.replace("/", "_").replace("-", "_")
    
    print(f"Downloading {dataset_name} dataset...")
    try:
        if dataset_config:
            dataset = load_dataset(dataset_name, dataset_config)
        else:
            dataset = load_dataset(dataset_name)
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        return None
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    print(f"Converting and saving {dataset_name}...")
    
    # Save each split
    splits_saved = []
    for split_name, split_data in dataset.items():
        df = pd.DataFrame(split_data)
        
        # Save as CSV and JSON
        csv_filename = f"data/{save_name}_{split_name}.csv"
        json_filename = f"data/{save_name}_{split_name}.json"
        
        df.to_csv(csv_filename, index=False)
        df.to_json(json_filename, orient="records", lines=True)
        
        splits_saved.extend([csv_filename, json_filename])
        print(f"  - Saved {split_name} split: {len(df)} examples")
    
    # Save dataset info
    dataset_info = {
        "dataset_name": dataset_name,
        "config": dataset_config,
        "splits": {split_name: len(split_data) for split_name, split_data in dataset.items()},
        "features": {split_name: list(split_data.features.keys()) for split_name, split_data in dataset.items()}
    }
    
    info_filename = f"data/{save_name}_info.json"
    with open(info_filename, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    splits_saved.append(info_filename)
    
    print(f"Dataset {dataset_name} saved successfully!")
    print(f"Files created: {', '.join(splits_saved)}")
    
    return dataset

def download_and_save_moral_stories():
    """
    Download the moral_stories dataset once and save it as a local JSON file.
    This only needs to be run once to create the local dataset file.
    """
    return download_and_save_dataset("demelin/moral_stories", "full", "moral_stories")

def download_and_save_truthfulqa():
    """Download TruthfulQA-MC dataset"""
    return download_and_save_dataset("truthful_qa", "multiple_choice", "truthfulqa_mc")

def download_and_save_air_deception():
    """Download AIR-Deception-50 dataset"""
    # Note: This might need adjustment based on the actual dataset name/config
    return download_and_save_dataset("air_bench", "air_deception_50", "air_deception_50")

def download_and_save_crows_pairs():
    """Download CrowS-Pairs dataset"""
    return download_and_save_dataset("crows_pairs", save_name="crows_pairs")

def download_all_datasets():
    """Download all datasets"""
    datasets_to_download = [
        ("Moral Stories", download_and_save_moral_stories),
        ("TruthfulQA-MC", download_and_save_truthfulqa),
        ("AIR-Deception-50", download_and_save_air_deception),
        ("CrowS-Pairs", download_and_save_crows_pairs)
    ]
    
    success_count = 0
    
    print("=" * 60)
    print("Starting download of all datasets...")
    print("=" * 60)
    
    for dataset_display_name, download_func in datasets_to_download:
        try:
            print(f"\n{'='*20} {dataset_display_name} {'='*20}")
            result = download_func()
            if result is not None:
                success_count += 1
                print(f"✓ {dataset_display_name} downloaded successfully")
            else:
                print(f"✗ {dataset_display_name} failed to download")
        except Exception as e:
            print(f"✗ Error downloading {dataset_display_name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Download completed: {success_count}/{len(datasets_to_download)} datasets successful")
    print(f"All files saved to data/ directory")
    print(f"{'='*60}")

if __name__ == "__main__":
    download_all_datasets() 