#!/usr/bin/env python3
"""
Test script for KLUE NER benchmark setup
This script verifies that all dependencies and components are properly installed and configured.
"""

import sys
import os
import importlib
from typing import List, Dict, Any

def test_imports() -> bool:
    """Test if all required packages can be imported."""
    print("Testing package imports...")
    
    required_packages = [
        'google.genai',
        'datasets',
        'pandas',
        'tqdm',
        'google.cloud.aiplatform'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed to import: {', '.join(failed_imports)}")
        return False
    
    print("  All packages imported successfully!")
    return True

def test_dataset_access() -> bool:
    """Test if the KLUE NER dataset can be accessed."""
    print("\nTesting dataset access...")
    
    try:
        from datasets import load_dataset
        
        # Try to load a small sample of the dataset
        print("  Loading KLUE NER dataset...")
        dataset = load_dataset('klue', 'ner', split='validation')
        
        # Check if we can access the first few samples
        sample = dataset[0]
        print(f"  ✓ Dataset loaded successfully")
        print(f"  ✓ Sample keys: {list(sample.keys())}")
        print(f"  ✓ Tokens: {len(sample['tokens'])} tokens")
        print(f"  ✓ NER tags: {len(sample['ner_tags'])} tags")
        
        # Check for required fields (guid is not available in this dataset version)
        required_fields = ['tokens', 'ner_tags']
        for field in required_fields:
            if field not in sample:
                print(f"  ✗ Missing required field: {field}")
                return False
        
        print("  ✓ All required fields present")
        print("  ⚠ Note: 'guid' field not available, will generate IDs automatically")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load dataset: {e}")
        return False

def test_google_cloud_setup() -> bool:
    """Test Google Cloud setup."""
    print("\nTesting Google Cloud setup...")
    
    # Check if GOOGLE_CLOUD_PROJECT is set
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    if not project_id:
        print("  ⚠ GOOGLE_CLOUD_PROJECT environment variable not set")
        print("  This is required for running the benchmark")
        return False
    
    print(f"  ✓ GOOGLE_CLOUD_PROJECT set to: {project_id}")
    
    # Test genai import
    try:
        from google import genai
        print("  ✓ google.genai imported successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import google.genai: {e}")
        return False
    
    return True

def test_ner_script() -> bool:
    """Test if the NER script can be imported and basic functionality works."""
    print("\nTesting NER script...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, os.getcwd())
        
        # Import the benchmark class (note: filename uses hyphens, not underscores)
        import importlib.util
        spec = importlib.util.spec_from_file_location("klue_ner_gemini2_5flash", "klue_ner-gemini2_5flash.py")
        if spec is None:
            raise ImportError("Could not create module spec for klue_ner-gemini2_5flash.py")
        klue_ner_module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise ImportError("Could not get loader for klue_ner-gemini2_5flash.py")
        spec.loader.exec_module(klue_ner_module)
        KLUENamedEntityRecognitionBenchmark = klue_ner_module.KLUENamedEntityRecognitionBenchmark
        BenchmarkConfig = klue_ner_module.BenchmarkConfig
        
        print("  ✓ NER script imported successfully")
        
        # Test basic configuration
        config = BenchmarkConfig(max_samples=1)
        print("  ✓ Configuration created successfully")
        
        # Test entity type mapping
        expected_entity_types = ["PS", "LC", "OG", "DT", "TI", "QT"]
        actual_entity_types = list(KLUENamedEntityRecognitionBenchmark.ENTITY_TYPES.keys())
        
        if set(expected_entity_types) == set(actual_entity_types):
            print("  ✓ Entity type mapping correct")
        else:
            print(f"  ✗ Entity type mapping mismatch. Expected: {expected_entity_types}, Got: {actual_entity_types}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Failed to import NER script: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Error testing NER script: {e}")
        return False

def test_directory_structure() -> bool:
    """Test if required directories exist."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'logs',
        'benchmark_results',
        'result_analysis',
        'eval_dataset'
    ]
    
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nCreating missing directories: {', '.join(missing_dirs)}")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")
    
    return True

def test_script_permissions() -> bool:
    """Test if scripts have proper permissions."""
    print("\nTesting script permissions...")
    
    scripts = ['run', 'setup.sh', 'install_dependencies.sh', 'get_errors.sh', 'test_logging.sh']
    
    for script in scripts:
        if os.path.exists(script):
            if os.access(script, os.X_OK):
                print(f"  ✓ {script} (executable)")
            else:
                print(f"  ⚠ {script} (not executable)")
        else:
            print(f"  ✗ {script} (missing)")
    
    return True

def main():
    """Main test function."""
    print("KLUE NER Benchmark Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Access", test_dataset_access),
        ("Google Cloud Setup", test_google_cloud_setup),
        ("NER Script", test_ner_script),
        ("Directory Structure", test_directory_structure),
        ("Script Permissions", test_script_permissions)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            if test_func():
                passed += 1
            else:
                print(f"  ❌ {test_name} failed")
        except Exception as e:
            print(f"  ❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Set your Google Cloud project: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        print("2. Run a test: ./run test")
        print("3. Run full benchmark: ./run full")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Run './setup.sh install' to install dependencies")
        print("2. Check VERTEX_AI_SETUP.md for Google Cloud setup instructions")
        print("3. Ensure all required files are present")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 