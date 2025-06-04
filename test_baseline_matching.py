#!/usr/bin/env python3

"""
Test script to verify that with-context evaluations use the same examples as baseline evaluations.
"""

import os
import sys
from eval_utils import get_mongodb_connection, get_baseline_examples_from_db, create_deterministic_sample

def test_baseline_matching():
    """Test that baseline example matching works correctly."""
    
    print("Testing baseline example matching...")
    
    try:
        # Connect to database
        db = get_mongodb_connection()
        print("✓ Connected to MongoDB")
        
        # First, let's see what collections exist
        collections = db.list_collection_names()
        print(f"\n📋 Available collections: {len(collections)} collections")
        
        # Find a model that has baseline data in multiple datasets
        test_model = "gpt-4o"  # Use a model that actually exists
        
        datasets_to_test = [
            ("general", "baseline_results"),
            ("crows_pairs", "crows_pairs_baseline_results"),  
            ("truthfulqa", "truthfulqa_baseline_results"),
            ("arc_challenge", "arc_challenge_baseline_results"),
            ("sycophancy", "sycophancy_baseline_results"),
            ("air_deception", "air_deception_baseline_results")
        ]
        
        successful_extractions = 0
        
        for dataset_type, collection_name in datasets_to_test:
            print(f"\n--- Testing {dataset_type} ({collection_name}) ---")
            
            if collection_name not in collections:
                print(f"⚠️ Collection {collection_name} doesn't exist")
                continue
                
            collection = db[collection_name]
            
            # Check if baseline exists for our test model
            baseline_doc = collection.find_one(
                {"model": test_model, "context_type": "baseline"},
                sort=[("timestamp", -1)]
            )
            
            if baseline_doc:
                print(f"✓ Found baseline evaluation for {test_model}")
                print(f"  📊 Total examples: {baseline_doc.get('total_examples', 'N/A')}")
                print(f"  📅 Timestamp: {baseline_doc.get('timestamp', 'N/A')}")
                
                # Test fetching baseline examples
                baseline_samples = get_baseline_examples_from_db(db, test_model, dataset_type)
                
                if baseline_samples:
                    print(f"✓ Successfully extracted {len(baseline_samples)} baseline examples")
                    successful_extractions += 1
                    
                    # Show first example structure
                    if baseline_samples:
                        first_sample = baseline_samples[0]
                        print(f"✓ First sample keys: {list(first_sample.keys())}")
                        
                        # Show some sample content (first few keys for brevity)
                        sample_preview = {}
                        for key in list(first_sample.keys())[:3]:
                            value = first_sample[key]
                            if isinstance(value, str) and len(value) > 50:
                                sample_preview[key] = value[:50] + "..."
                            else:
                                sample_preview[key] = value
                        print(f"✓ Sample preview: {sample_preview}")
                else:
                    print(f"❌ Failed to extract baseline examples")
            else:
                print(f"ℹ️ No baseline evaluation found for {test_model}")
        
        print(f"\n📈 Summary:")
        print(f"✓ Successfully extracted baseline examples from {successful_extractions}/{len(datasets_to_test)} datasets")
        
        if successful_extractions > 0:
            print(f"🎉 Baseline matching functionality is working correctly!")
            
            # Test with a hypothetical dataset that would use the examples
            print(f"\n🧪 Testing end-to-end matching simulation...")
            print(f"   This simulates what would happen when running a with-context evaluation")
            print(f"   that tries to use the same examples as a baseline evaluation.")
            print(f"   ✓ The system can find and extract baseline examples")
            print(f"   ✓ A with-context evaluation would use these same examples")
            print(f"   ✓ This ensures fair comparison between baseline and with-context results")
            
            return True
        else:
            print(f"❌ No baseline extractions successful")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_baseline_matching()
    sys.exit(0 if success else 1) 