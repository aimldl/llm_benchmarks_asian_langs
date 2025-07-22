#!/usr/bin/env python3
"""
Simple test script for GeminiServing class
"""

import os
import sys
from serving.gemini_serving import GeminiServing

def test_gemini_serving():
    """Test basic Gemini serving functionality."""
    print("Testing GeminiServing class...")
    
    try:
        # Check environment
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
            return False
        
        print(f"✓ Using project: {project_id}")
        
        # Initialize Gemini serving
        llm = GeminiServing(
            model="gemini-2.5-flash",
            project_id=project_id,
            location="us-central1",
            sleep_interval=0.1,
            max_tokens=100,
        )
        
        print("✓ GeminiServing initialized successfully")
        
        # Test get_run_env method
        run_env = llm.get_run_env()
        print(f"✓ get_run_env() returned: {run_env}")
        
        # Test simple generation
        messages = [
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        response = llm.generate(messages)
        print(f"✓ generate() returned: {response}")
        
        # Test parse_outputs method
        outputs = [response]
        parsed = llm.parse_outputs(outputs)
        print(f"✓ parse_outputs() returned: {parsed}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_serving()
    sys.exit(0 if success else 1) 