#!/usr/bin/env python3
"""
Test Gemini API with Correct SDK Usage
=======================================

Uses the proper Google GenAI SDK syntax.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test basic imports
try:
    from google import genai
    print("[OK] google.genai imported successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import google.genai: {e}")
    sys.exit(1)

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"[OK] API Key found: {api_key[:10]}...")
else:
    print("[ERROR] No API key found")
    sys.exit(1)

# Create client with API key
try:
    client = genai.Client(api_key=api_key)
    print("[OK] Client created successfully")
except Exception as e:
    print(f"[ERROR] Failed to create client: {e}")
    sys.exit(1)

# Test simple API call using correct SDK syntax
try:
    print("\nTesting API call...")
    
    # Use the correct method: client.models.generate_content
    response = client.models.generate_content(
        model='gemini-1.5-flash',  # Use faster model
        contents='Say "Hello World" and nothing else'
    )
    
    if response and response.text:
        print(f"[OK] API Response: {response.text.strip()}")
    else:
        print("[ERROR] No response from API")
        
except Exception as e:
    print(f"[ERROR] API call failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Check for specific error types
    if "429" in str(e):
        print("   Issue: Rate limit exceeded")
    elif "403" in str(e):
        print("   Issue: Permission denied - check billing")
    elif "401" in str(e):
        print("   Issue: Invalid API key")
    else:
        print(f"   Full error: {e}")

# Test with code analysis (more complex)
print("\nTesting code analysis...")
try:
    test_code = '''
def add(a, b):
    return a + b
'''
    
    prompt = f"What does this function do? Answer in one sentence:\n{test_code}"
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents=prompt
    )
    
    if response and response.text:
        print(f"[OK] Code analysis works: {response.text.strip()[:100]}")
    else:
        print("[ERROR] Code analysis failed")
        
except Exception as e:
    print(f"[ERROR] Code analysis failed: {e}")

# Test with configuration
print("\nTesting with configuration...")
try:
    from google.genai import types
    
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Write a haiku about coding',
        config=types.GenerateContentConfig(
            temperature=0.5,
            max_output_tokens=100,
        )
    )
    
    if response and response.text:
        print(f"[OK] Configured generation works:\n{response.text.strip()}")
    else:
        print("[ERROR] Configured generation failed")
        
except Exception as e:
    print(f"[ERROR] Configured generation failed: {e}")

print("\nDiagnostic complete!")
print("\nIf all tests passed, the API is working correctly with billing enabled.")