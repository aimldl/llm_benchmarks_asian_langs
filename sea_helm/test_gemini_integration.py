#!/usr/bin/env python3
"""
Test script for SEA-HELM Gemini 2.5 Flash integration
This script tests the basic functionality of the Gemini serving class.
"""

import os
import sys
import json
from typing import List, Dict, Any

# Add the current directory to the path so we can import sea_helm modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from serving.gemini_serving import GeminiServing
from base_logger import get_logger

logger = get_logger(__name__)


def test_gemini_serving_basic():
    """Test basic Gemini serving functionality."""
    print("Testing basic Gemini serving functionality...")
    
    try:
        # Initialize Gemini serving
        llm = GeminiServing(
            model="gemini-2.5-flash",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1",
            sleep_interval=0.1,  # Slower for testing
            max_tokens=100,      # Small for testing
        )
        
        print("✓ Gemini serving initialized successfully")
        
        # Test simple generation
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        response = llm.generate(messages)
        print(f"✓ Generated response: {response.get('text', '')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in basic test: {e}")
        return False


def test_gemini_serving_conversation():
    """Test conversation-style generation."""
    print("\nTesting conversation-style generation...")
    
    try:
        # Initialize Gemini serving
        llm = GeminiServing(
            model="gemini-2.5-flash",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1",
            sleep_interval=0.1,
            max_tokens=100,
        )
        
        # Test conversation
        messages = [
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "What about 3 + 3?"}
        ]
        
        response = llm.generate(messages)
        print(f"✓ Conversation response: {response.get('text', '')[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in conversation test: {e}")
        return False


def test_gemini_serving_batch():
    """Test batch generation."""
    print("\nTesting batch generation...")
    
    try:
        # Initialize Gemini serving
        llm = GeminiServing(
            model="gemini-2.5-flash",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1",
            sleep_interval=0.1,
            max_tokens=50,
        )
        
        # Test batch generation
        batch_messages = [
            [{"role": "user", "content": "Say hello"}],
            [{"role": "user", "content": "Say goodbye"}],
        ]
        
        responses = llm.batch_generate(batch_messages)
        print(f"✓ Batch generated {len(responses)} responses")
        for i, response in enumerate(responses):
            print(f"  Response {i+1}: {response.get('text', '')[:30]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in batch test: {e}")
        return False


def test_gemini_serving_parse_outputs():
    """Test output parsing."""
    print("\nTesting output parsing...")
    
    try:
        # Initialize Gemini serving
        llm = GeminiServing(
            model="gemini-2.5-flash",
            project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location="us-central1",
            sleep_interval=0.1,
            max_tokens=50,
        )
        
        # Test output parsing
        generated_outputs = [
            {"text": "Hello world"},
            {"text": "Goodbye world"},
        ]
        
        parsed_outputs = llm.parse_outputs(generated_outputs)
        print(f"✓ Parsed outputs: {parsed_outputs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in parse test: {e}")
        return False


def test_environment():
    """Test environment setup."""
    print("Testing environment setup...")
    
    # Check environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if project_id:
        print(f"✓ GOOGLE_CLOUD_PROJECT set to: {project_id}")
    else:
        print("✗ GOOGLE_CLOUD_PROJECT not set")
        return False
    
    # Check imports
    try:
        import google.genai
        print("✓ google-genai imported successfully")
    except ImportError as e:
        print(f"✗ google-genai import failed: {e}")
        return False
    
    try:
        import google.cloud.aiplatform
        print("✓ google-cloud-aiplatform imported successfully")
    except ImportError as e:
        print(f"✗ google-cloud-aiplatform import failed: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("="*60)
    print("SEA-HELM GEMINI 2.5 FLASH INTEGRATION TEST")
    print("="*60)
    
    # Test environment
    if not test_environment():
        print("\n❌ Environment test failed. Please check your setup.")
        return False
    
    # Run tests
    tests = [
        test_gemini_serving_basic,
        test_gemini_serving_conversation,
        test_gemini_serving_batch,
        test_gemini_serving_parse_outputs,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("✅ All tests passed! Gemini integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 