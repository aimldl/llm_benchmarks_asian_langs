#!/usr/bin/env python3
"""
Test setup script for KLUE DST (Dialogue State Tracking) benchmark
This script tests the environment setup and basic functionality.
"""

import os
import sys
import json
from typing import Dict, Any, List

def test_imports():
    """Test if all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import google.genai
        print("✓ google.genai imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import google.genai: {e}")
        return False
    
    try:
        from datasets import load_dataset
        print("✓ datasets imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import datasets: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import pandas: {e}")
        return False
    
    try:
        from tqdm import tqdm
        print("✓ tqdm imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import tqdm: {e}")
        return False
    
    return True

def test_dataset_loading():
    """Test if the KLUE DST dataset can be loaded."""
    print("\nTesting dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Load a small sample to test
        dataset = load_dataset('klue', 'wos', split='validation')
        print(f"✓ Dataset loaded successfully: {len(dataset)} samples")
        
        # Check dataset features
        features = list(dataset.features.keys())
        print(f"✓ Dataset features: {features}")
        
        # Check required features for WOS dataset
        required_features = ['guid', 'dialogue', 'domains']
        missing_features = [f for f in required_features if f not in features]
        
        if missing_features:
            print(f"✗ Missing required features: {missing_features}")
            return False
        else:
            print("✓ All required features present")
        
        # Test sample data
        sample = dataset[0]
        print(f"✓ Sample data structure:")
        print(f"  - guid: {sample['guid']}")
        print(f"  - dialogue length: {len(sample['dialogue'])} turns")
        print(f"  - domains: {sample['domains']}")
        
        # Check dialogue structure
        if sample['dialogue']:
            first_turn = sample['dialogue'][0]
            print(f"  - dialogue turn keys: {list(first_turn.keys())}")
            print(f"  - sample turn: role={first_turn.get('role', 'N/A')}, text={first_turn.get('text', 'N/A')[:50]}...")
            print(f"  - state in turn: {first_turn.get('state', [])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

def test_google_cloud_setup():
    """Test Google Cloud setup."""
    print("\nTesting Google Cloud setup...")
    
    # Check environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        print(f"✓ GOOGLE_CLOUD_PROJECT is set: {project_id}")
    else:
        print("✗ GOOGLE_CLOUD_PROJECT environment variable is not set")
        print("  Please set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return False
    
    # Test Google Cloud authentication
    try:
        import google.auth
        credentials, project = google.auth.default()
        print(f"✓ Google Cloud authentication successful")
        print(f"  Project: {project}")
        print(f"  Credentials type: {type(credentials).__name__}")
        return True
    except Exception as e:
        print(f"✗ Google Cloud authentication failed: {e}")
        print("  Please run: gcloud auth login")
        return False

def test_vertex_ai_access():
    """Test Vertex AI access."""
    print("\nTesting Vertex AI access...")
    
    try:
        from google import genai
        
        # Initialize client
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        client = genai.Client(vertexai=True, project=project_id, location="us-central1")
        print("✓ Vertex AI client initialized successfully")
        
        # Test model access (just check if we can list models)
        try:
            models = client.models.list()
            print("✓ Vertex AI model access successful")
            return True
        except Exception as e:
            print(f"✗ Vertex AI model access failed: {e}")
            return False
            
    except Exception as e:
        print(f"✗ Vertex AI client initialization failed: {e}")
        return False

def test_directory_structure():
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = ['logs', 'benchmark_results', 'result_analysis', 'eval_dataset']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ Directory exists: {dir_name}")
        else:
            print(f"✗ Directory missing: {dir_name}")
            try:
                os.makedirs(dir_name)
                print(f"  Created directory: {dir_name}")
            except Exception as e:
                print(f"  Failed to create directory: {e}")
                return False
    
    return True

def test_script_files():
    """Test if required script files exist and are executable."""
    print("\nTesting script files...")
    
    required_files = [
        'klue_dst-gemini2_5flash.py',
        'run',
        'setup.sh',
        'requirements.txt'
    ]
    
    for file_name in required_files:
        if os.path.exists(file_name):
            print(f"✓ File exists: {file_name}")
            
            # Check if executable
            if file_name in ['run', 'setup.sh']:
                if os.access(file_name, os.X_OK):
                    print(f"  ✓ File is executable: {file_name}")
                else:
                    print(f"  ✗ File is not executable: {file_name}")
                    try:
                        os.chmod(file_name, 0o755)
                        print(f"    Made executable: {file_name}")
                    except Exception as e:
                        print(f"    Failed to make executable: {e}")
        else:
            print(f"✗ File missing: {file_name}")
            return False
    
    return True

def test_sample_prediction():
    """Test a sample prediction (without making actual API calls)."""
    print("\nTesting sample prediction structure...")
    
    try:
        # Import the benchmark class
        sys.path.append('.')
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_dst_module", "klue_dst-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not load klue_dst-gemini2_5flash.py")
        klue_dst_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(klue_dst_module)
        KLUEDialogueStateTrackingBenchmark = klue_dst_module.KLUEDialogueStateTrackingBenchmark
        BenchmarkConfig = klue_dst_module.BenchmarkConfig
        
        # Create a minimal config for testing
        config = BenchmarkConfig(
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            max_samples=1
        )
        
        # Create benchmark instance
        benchmark = KLUEDialogueStateTrackingBenchmark(config)
        print("✓ Benchmark class instantiated successfully")
        
        # Test prompt creation
        sample_dialogue = [
            {"speaker": "user", "utterance": "안녕하세요"},
            {"speaker": "system", "utterance": "안녕하세요! 무엇을 도와드릴까요?"},
            {"speaker": "user", "utterance": "레스토랑을 찾고 있어요"}
        ]
        
        prompt = benchmark.create_prompt(sample_dialogue, ["restaurant"], 3)
        print("✓ Prompt creation successful")
        print(f"  Prompt length: {len(prompt)} characters")
        
        # Test response parsing
        sample_response = """활성 의도: request
요청된 슬롯: [location, cuisine]
슬롯 값: {"location": "서울", "cuisine": "한식"}"""
        
        parsed = benchmark.parse_dst_response(sample_response)
        print("✓ Response parsing successful")
        print(f"  Parsed intent: {parsed['active_intent']}")
        print(f"  Parsed slots: {parsed['requested_slots']}")
        print(f"  Parsed values: {parsed['slot_values']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample prediction test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("KLUE DST Environment Test")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Dataset Loading", test_dataset_loading),
        ("Google Cloud Setup", test_google_cloud_setup),
        ("Vertex AI Access", test_vertex_ai_access),
        ("Directory Structure", test_directory_structure),
        ("Script Files", test_script_files),
        ("Sample Prediction", test_sample_prediction)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Environment is ready for KLUE DST benchmarking.")
        print("\nNext steps:")
        print("1. Run a test benchmark: ./run test")
        print("2. Run the full benchmark: ./run full")
        print("3. Run with custom samples: ./run custom 100")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 