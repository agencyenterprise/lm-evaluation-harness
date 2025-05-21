#!/usr/bin/env python3

import os
import json
from datasets import load_dataset
import pandas as pd

def download_and_save_moral_stories():
    """
    Download the moral_stories dataset once and save it as a local JSON file.
    This only needs to be run once to create the local dataset file.
    """
    print("Downloading moral_stories dataset...")
    dataset = load_dataset("demelin/moral_stories", "full")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Convert to pandas and save as CSV and JSON
    print("Converting and saving dataset...")
    
    # Save train set
    train_df = pd.DataFrame(dataset["train"])
    train_df.to_csv("data/moral_stories_train.csv", index=False)
    train_df.to_json("data/moral_stories_train.json", orient="records", lines=True)
    
    # Save validation set if it exists
    if "validation" in dataset:
        val_df = pd.DataFrame(dataset["validation"])
        val_df.to_csv("data/moral_stories_validation.csv", index=False)
        val_df.to_json("data/moral_stories_validation.json", orient="records", lines=True)
    
    # Save test set if it exists
    if "test" in dataset:
        test_df = pd.DataFrame(dataset["test"])
        test_df.to_csv("data/moral_stories_test.csv", index=False)
        test_df.to_json("data/moral_stories_test.json", orient="records", lines=True)
    
    # Save dataset info as JSON (converting Dataset objects to lists first)
    print("Saving dataset info...")
    dataset_info = {
        "train_size": len(dataset["train"]),
        "validation_size": len(dataset["validation"]) if "validation" in dataset else 0,
        "test_size": len(dataset["test"]) if "test" in dataset else 0,
        "features": list(dataset["train"].features.keys())
    }
    
    with open("data/moral_stories_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset saved to data/ directory!")
    print(f"Files created:")
    print(f"  - data/moral_stories_train.csv")
    print(f"  - data/moral_stories_train.json")
    print(f"  - data/moral_stories_info.json")
    
    return dataset

if __name__ == "__main__":
    download_and_save_moral_stories() 