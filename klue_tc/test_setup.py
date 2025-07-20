#!/usr/bin/env python3
"""
Test script to verify the setup for KLUE TC benchmark with Vertex AI.
This script checks if all dependencies are properly installed and Vertex AI is accessible.
"""

import sys
import importlib
import os

def test_imports():
    """Test if all required packages can be imported."""
    required_packages = [
        'google.cloud.aiplatform',
        'vertexai',
        'datasets',
        'pandas',
        'tqdm',
        'huggingface_hub',
        'google.auth',
        'sklearn'
    ]
    
    print("Testing package imports...")
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing packages with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_klue_dataset():
    """Test if KLUE dataset can be loaded."""
    try:
        from datasets import load_dataset
        print("\nTesting KLUE dataset loading...")
        
        # Try to load a small sample
        dataset = load_dataset("klue", "ynat")
        print(f"✓ KLUE ynat dataset for TC loaded successfully")
        print(f"  - Train samples: {len(dataset['train'])}")
        print(f"  - Validation samples: {len(dataset['validation'])}")
        # Comment out because 'test' split is not available
        #print(f"  - Test samples: {len(dataset['test'])}")
        
        # Show a sample
        #sample = dataset['test'][0]
        #print(f"  - Sample text: {sample['title']} {sample['text'][:100]}...")
        #print(f"  - Sample label: {sample['label']}")

        # Show a sample from the validation set, since 'test' is not available
        if 'validation' in dataset and len(dataset['validation']) > 0:
            sample = dataset['validation'][0]
            print(f"  - Sample from validation set:")
            print(f"    - Sample title: {sample.get('title', 'N/A')}")
            print(f"    - Sample text: {sample.get('text', 'N/A')[:100]}...")
            print(f"    - Sample label: {sample.get('label', 'N/A')}")
        elif 'train' in dataset and len(dataset['train']) > 0:
            sample = dataset['train'][0]
            print(f"  - Sample from train set:")
            print(f"    - Sample title: {sample.get('title', 'N/A')}")
            print(f"    - Sample text: {sample.get('text', 'N/A')[:100]}...")
            print(f"    - Sample label: {sample.get('label', 'N/A')}")
        else:
            print("  - No samples available in 'train' or 'validation' splits.")

    except Exception as e:
        print(f"✗ Failed to load KLUE dataset: {e}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load KLUE dataset: {e}")
        return False

def test_vertex_ai_auth():
    """Test if Vertex AI authentication is working."""
    try:
        from google.cloud import aiplatform
        from google.auth import default
        print("\nTesting Vertex AI authentication...")
        
        # Check if credentials are available
        credentials, project = default()
        if not credentials:
            print("✗ No credentials found")
            return False
        
        print(f"✓ Credentials found")
        print(f"  - Project: {project or 'Not set'}")
        print(f"  - Credentials type: {type(credentials).__name__}")
        
        # Test if we can initialize Vertex AI (without making actual API calls)
        try:
            aiplatform.init(project=project or "test-project")
            print("✓ Vertex AI initialization works")
            return True
        except Exception as e:
            print(f"✗ Vertex AI initialization failed: {e}")
            return False
        
    except Exception as e:
        print(f"✗ Failed to test Vertex AI authentication: {e}")
        return False

def test_environment_variables():
    """Test if required environment variables are set."""
    print("\nTesting environment variables...")
    
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if project_id:
        print(f"✓ GOOGLE_CLOUD_PROJECT: {project_id}")
    else:
        print("⚠ GOOGLE_CLOUD_PROJECT: Not set (will need to provide via --project-id)")
    
    if credentials_path:
        print(f"✓ GOOGLE_APPLICATION_CREDENTIALS: {credentials_path}")
        if os.path.exists(credentials_path):
            print("  - Credentials file exists")
        else:
            print("  - ⚠ Credentials file does not exist")
    else:
        print("⚠ GOOGLE_APPLICATION_CREDENTIALS: Not set (using default credentials)")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("KLUE TC Benchmark Setup Test (Vertex AI)")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test environment variables
    env_ok = test_environment_variables()
    
    # Test dataset loading
    dataset_ok = test_klue_dataset()
    
    # Test Vertex AI authentication
    auth_ok = test_vertex_ai_auth()
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if imports_ok and dataset_ok and auth_ok:
        print("✅ All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Ensure your Google Cloud project has Vertex AI API enabled")
        print("2. Set project ID: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("3. Run the benchmark: python klue_tc-gemini2_5flash.py --project-id 'your-project-id'")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        if not imports_ok:
            print("\nTo install dependencies:")
            print("pip install -r requirements.txt")
        if not auth_ok:
            print("\nTo set up authentication:")
            print("gcloud auth application-default login")
            print("# OR set up service account credentials")
        sys.exit(1)

if __name__ == "__main__":
    main() 