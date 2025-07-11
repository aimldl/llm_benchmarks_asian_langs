#!/usr/bin/env python3
"""
Test setup script for KLUE RE benchmark
This script tests the environment setup and basic functionality
"""

import os
import sys
import json
from typing import Dict, Any

def test_imports() -> Dict[str, bool]:
    """Test if all required packages can be imported."""
    results = {}
    
    try:
        import google.genai
        results["google.genai"] = True
        print("✅ google.genai imported successfully")
    except ImportError as e:
        results["google.genai"] = False
        print(f"❌ Failed to import google.genai: {e}")
    
    try:
        from datasets import load_dataset
        results["datasets"] = True
        print("✅ datasets imported successfully")
    except ImportError as e:
        results["datasets"] = False
        print(f"❌ Failed to import datasets: {e}")
    
    try:
        import pandas as pd
        results["pandas"] = True
        print("✅ pandas imported successfully")
    except ImportError as e:
        results["pandas"] = False
        print(f"❌ Failed to import pandas: {e}")
    
    try:
        from tqdm import tqdm
        results["tqdm"] = True
        print("✅ tqdm imported successfully")
    except ImportError as e:
        results["tqdm"] = False
        print(f"❌ Failed to import tqdm: {e}")
    
    return results

def test_environment_variables() -> Dict[str, bool]:
    """Test if required environment variables are set."""
    results = {}
    
    # Check GOOGLE_CLOUD_PROJECT
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        results["GOOGLE_CLOUD_PROJECT"] = True
        print(f"✅ GOOGLE_CLOUD_PROJECT is set: {project_id}")
    else:
        results["GOOGLE_CLOUD_PROJECT"] = False
        print("❌ GOOGLE_CLOUD_PROJECT is not set")
    
    return results

def test_dataset_loading() -> Dict[str, Any]:
    """Test if the KLUE RE dataset can be loaded."""
    try:
        from datasets import load_dataset
        
        print("Loading KLUE RE dataset...")
        dataset = load_dataset('klue', 're', split='validation')
        
        # Check dataset structure
        if len(dataset) > 0:
            sample = dataset[0]
            expected_keys = ['guid', 'sentence', 'subject_entity', 'object_entity', 'label']
            missing_keys = [key for key in expected_keys if key not in sample]
            
            if missing_keys:
                return {
                    "success": False,
                    "error": f"Missing expected keys: {missing_keys}",
                    "sample_keys": list(sample.keys())
                }
            
            # Check entity structure
            subject_entity = sample['subject_entity']
            object_entity = sample['object_entity']
            
            if not isinstance(subject_entity, dict) or 'word' not in subject_entity or 'type' not in subject_entity:
                return {
                    "success": False,
                    "error": "subject_entity does not have expected structure",
                    "subject_entity": subject_entity
                }
            
            if not isinstance(object_entity, dict) or 'word' not in object_entity or 'type' not in object_entity:
                return {
                    "success": False,
                    "error": "object_entity does not have expected structure",
                    "object_entity": object_entity
                }
            
            return {
                "success": True,
                "dataset_size": len(dataset),
                "sample_keys": list(sample.keys()),
                "subject_entity_keys": list(subject_entity.keys()),
                "object_entity_keys": list(object_entity.keys()),
                "sample_sentence": sample['sentence'][:100] + "..." if len(sample['sentence']) > 100 else sample['sentence'],
                "sample_subject": subject_entity['word'],
                "sample_object": object_entity['word'],
                "sample_relation": sample['label']
            }
        else:
            return {
                "success": False,
                "error": "Dataset is empty"
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_vertex_ai_connection() -> Dict[str, Any]:
    """Test if Vertex AI connection can be established."""
    try:
        import google.genai
        
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            return {
                "success": False,
                "error": "GOOGLE_CLOUD_PROJECT not set"
            }
        
        # Try to initialize the client
        client = google.genai.Client(vertexai=True, project=project_id, location="us-central1")
        
        # Try to list models (this will test the connection)
        models = client.models.list()
        
        return {
            "success": True,
            "project_id": project_id,
            "models_available": len(list(models)) > 0
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def test_directory_structure() -> Dict[str, bool]:
    """Test if required directories exist."""
    results = {}
    
    required_dirs = ['logs', 'benchmark_results', 'result_analysis', 'eval_dataset']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            results[dir_name] = True
            print(f"✅ Directory exists: {dir_name}")
        else:
            results[dir_name] = False
            print(f"❌ Directory missing: {dir_name}")
    
    return results

def test_script_files() -> Dict[str, bool]:
    """Test if required script files exist."""
    results = {}
    
    required_files = [
        'klue_re-gemini2_5flash.py',
        'run',
        'setup.sh',
        'requirements.txt',
        'get_errors.sh',
        'test_logging.sh',
        'verify_scripts.sh'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            results[file_name] = True
            print(f"✅ File exists: {file_name}")
        else:
            results[file_name] = False
            print(f"❌ File missing: {file_name}")
    
    return results

def main():
    """Run all tests and provide a summary."""
    print("=" * 60)
    print("KLUE RE Setup Test")
    print("=" * 60)
    print()
    
    all_results = {}
    
    # Test imports
    print("1. Testing package imports...")
    all_results["imports"] = test_imports()
    print()
    
    # Test environment variables
    print("2. Testing environment variables...")
    all_results["environment"] = test_environment_variables()
    print()
    
    # Test directory structure
    print("3. Testing directory structure...")
    all_results["directories"] = test_directory_structure()
    print()
    
    # Test script files
    print("4. Testing script files...")
    all_results["files"] = test_script_files()
    print()
    
    # Test dataset loading
    print("5. Testing dataset loading...")
    dataset_result = test_dataset_loading()
    all_results["dataset"] = dataset_result
    if dataset_result["success"]:
        print(f"✅ Dataset loaded successfully: {dataset_result['dataset_size']} samples")
        print(f"   Sample sentence: {dataset_result['sample_sentence']}")
        print(f"   Sample subject: {dataset_result['sample_subject']}")
        print(f"   Sample object: {dataset_result['sample_object']}")
        print(f"   Sample relation: {dataset_result['sample_relation']}")
    else:
        print(f"❌ Dataset loading failed: {dataset_result['error']}")
    print()
    
    # Test Vertex AI connection
    print("6. Testing Vertex AI connection...")
    vertex_result = test_vertex_ai_connection()
    all_results["vertex_ai"] = vertex_result
    if vertex_result["success"]:
        print(f"✅ Vertex AI connection successful")
        print(f"   Project: {vertex_result['project_id']}")
        print(f"   Models available: {vertex_result['models_available']}")
    else:
        print(f"❌ Vertex AI connection failed: {vertex_result['error']}")
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    # Count successes and failures
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            if "success" in results:
                total_tests += 1
                if results["success"]:
                    passed_tests += 1
                    print(f"✅ {category}: PASSED")
                else:
                    print(f"❌ {category}: FAILED - {results.get('error', 'Unknown error')}")
            else:
                # Count boolean results
                for test_name, result in results.items():
                    total_tests += 1
                    if result:
                        passed_tests += 1
                        print(f"✅ {category}.{test_name}: PASSED")
                    else:
                        print(f"❌ {category}.{test_name}: FAILED")
    
    print()
    print(f"Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Your environment is ready for KLUE RE benchmarking.")
        print()
        print("Next steps:")
        print("1. Run a test: ./run test")
        print("2. Run full benchmark: ./run full")
        print("3. Run custom benchmark: ./run custom 100")
    else:
        print("⚠️  Some tests failed. Please check the errors above and fix them.")
        print()
        print("Common fixes:")
        print("1. Run ./setup.sh to set up the environment")
        print("2. Set GOOGLE_CLOUD_PROJECT: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("3. Install missing packages: pip install -r requirements.txt")
        print("4. Authenticate with gcloud: gcloud auth login")
    
    # Save results to file
    with open('test_setup_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print()
    print(f"Detailed results saved to: test_setup_results.json")
    
    return 0 if passed_tests == total_tests else 1

if __name__ == "__main__":
    sys.exit(main()) 