#!/usr/bin/env python3
"""
Direct test script for Gemini functionality
"""

import os
import sys
import time
import importlib_metadata
from typing import List, Dict, Any, Optional
from google import genai
from google.genai.types import (
    GenerateContentConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)

def test_gemini_direct():
    """Test Gemini functionality directly."""
    print("Testing Gemini functionality directly...")
    
    try:
        # Check environment
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            print("❌ GOOGLE_CLOUD_PROJECT environment variable not set")
            return False
        
        print(f"✓ Using project: {project_id}")
        
        # Initialize Vertex AI client
        client = genai.Client(
            vertexai=True, 
            project=project_id, 
            location="us-central1"
        )
        
        print("✓ Vertex AI client initialized successfully")
        
        # Test simple generation
        content = [
            {"role": "user", "parts": [{"text": "Hello, how are you?"}]}
        ]
        
        generation_config = GenerateContentConfig(
            max_output_tokens=100,
            temperature=0.1,
            top_p=1.0,
            top_k=1,
        )
        
        safety_settings = [
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=HarmBlockThreshold.BLOCK_NONE
            ),
            SafetySetting(
                category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=HarmBlockThreshold.BLOCK_NONE
            ),
        ]
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=content,
            config=generation_config,
        )
        
        print(f"✓ Generated response: {response}")
        
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_text = response.candidates[0].content.parts[0].text
            print(f"✓ Response text: {generated_text}")
        else:
            print("⚠ No response text generated")
            print(f"  Candidates: {len(response.candidates) if response.candidates else 0}")
            if response.candidates:
                print(f"  First candidate content: {response.candidates[0].content}")
                if response.candidates[0].content:
                    print(f"  Parts: {response.candidates[0].content.parts}")
        
        # Test get_run_env equivalent
        try:
            run_env = {
                "google_genai_version": importlib_metadata.version("google-genai"),
                "google_cloud_aiplatform_version": importlib_metadata.version("google-cloud-aiplatform"),
                "model_name": "gemini-2.5-flash",
                "project_id": project_id,
                "location": "us-central1",
            }
            print(f"✓ Environment info: {run_env}")
        except Exception as e:
            print(f"⚠ Could not get version info: {e}")
        
        print("✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_gemini_direct()
    sys.exit(0 if success else 1) 