#!/usr/bin/env python3
"""
Download script for additional evaluation datasets:
- AIR-Deception-50 (from HuggingFace)
- Sycophancy-QA (from HuggingFace) 
- Arc-Challenge (from HuggingFace)

This script downloads and prepares the datasets for evaluation.
"""

import os
import json
from datasets import load_dataset
from pathlib import Path

def download_air_deception():
    """Download AIR-Deception-50 dataset"""
    print("Downloading AIR-Deception-50 dataset...")
    
    # Create directory
    air_dir = Path("datasets/air_deception")
    air_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the dataset - using the correct repository name
        dataset = load_dataset("stanford-crfm/air-bench-2024", split="test")
        
        # Save as JSON
        data = []
        for item in dataset:
            data.append(item)
        
        with open(air_dir / "air_deception_50.json", "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ AIR-Deception-50 downloaded: {len(data)} examples")
        
    except Exception as e:
        print(f"❌ Failed to download AIR-Deception-50: {e}")
        print("Note: This dataset may require manual download or different access method")

def download_sycophancy_qa():
    """Download Sycophancy-QA dataset"""
    print("Downloading Sycophancy-QA dataset...")
    
    # Create directory
    syco_dir = Path("datasets/sycophancy")
    syco_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the main Anthropic model-written-evals dataset
        dataset = load_dataset("Anthropic/model-written-evals")
        
        # Save the main dataset
        for split_name, split_data in dataset.items():
            data = []
            for item in split_data:
                data.append(item)
            
            with open(syco_dir / f"model_written_evals_{split_name}.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Model-written evals downloaded: {len(data)} examples ({split_name})")
        
        # Also try to get the sycophancy-specific dataset
        try:
            dataset_syco = load_dataset("meg-tong/sycophancy-eval")
            for split_name, split_data in dataset_syco.items():
                data = []
                for item in split_data:
                    data.append(item)
                
                with open(syco_dir / f"sycophancy_eval_{split_name}.json", "w") as f:
                    json.dump(data, f, indent=2)
                
                print(f"✅ Sycophancy Eval downloaded: {len(data)} examples ({split_name})")
        except Exception as e:
            print(f"⚠️  Could not download meg-tong/sycophancy-eval: {e}")
        
    except Exception as e:
        print(f"❌ Failed to download Sycophancy-QA: {e}")

def download_arc_challenge():
    """Download Arc-Challenge dataset"""
    print("Downloading Arc-Challenge dataset...")
    
    # Create directory
    arc_dir = Path("datasets/arc_challenge")
    arc_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load the ARC dataset - Challenge subset
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")
        
        # Save each split
        for split_name, split_data in dataset.items():
            data = []
            for item in split_data:
                data.append(item)
            
            with open(arc_dir / f"arc_challenge_{split_name}.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Arc-Challenge downloaded: {len(data)} examples ({split_name})")
        
        # Also download the Easy version for comparison
        dataset_easy = load_dataset("allenai/ai2_arc", "ARC-Easy")
        for split_name, split_data in dataset_easy.items():
            data = []
            for item in split_data:
                data.append(item)
            
            with open(arc_dir / f"arc_easy_{split_name}.json", "w") as f:
                json.dump(data, f, indent=2)
            
            print(f"✅ Arc-Easy downloaded: {len(data)} examples ({split_name})")
        
    except Exception as e:
        print(f"❌ Failed to download Arc-Challenge: {e}")

def main():
    """Main function to download all datasets"""
    print("🚀 Starting download of additional evaluation datasets...")
    print("=" * 60)
    
    # Create main datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Download each dataset
    download_air_deception()
    print()
    
    download_sycophancy_qa()
    print()
    
    download_arc_challenge()
    print()
    
    print("=" * 60)
    print("✅ Dataset download process completed!")
    print("\nDataset locations:")
    print("- AIR-Deception-50: datasets/air_deception/")
    print("- Sycophancy-QA: datasets/sycophancy/")
    print("- Arc-Challenge: datasets/arc_challenge/")
    print("\nNote: Some datasets may require manual download or additional setup.")

if __name__ == "__main__":
    main() 