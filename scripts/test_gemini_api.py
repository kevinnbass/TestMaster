#!/usr/bin/env python3
"""
Test Gemini API Connection
===========================

Diagnose why Gemini API calls are failing.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Test basic imports
try:
    import google.generativeai as genai
    print("✓ google.generativeai imported successfully")
except ImportError as e:
    print(f"✗ Failed to import google.generativeai: {e}")
    sys.exit(1)

# Check API key
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    print(f"✓ API Key found: {api_key[:10]}...")
else:
    print("✗ No API key found")
    sys.exit(1)

# Configure API
try:
    genai.configure(api_key=api_key)
    print("✓ API configured successfully")
except Exception as e:
    print(f"✗ Failed to configure API: {e}")
    sys.exit(1)

# Test simple API call
try:
    print("\nTesting API call...")
    model = genai.GenerativeModel('gemini-1.5-flash')  # Use faster model
    
    # Very simple test
    response = model.generate_content("Say 'Hello World' and nothing else")
    
    if response and response.text:
        print(f"✓ API Response: {response.text.strip()}")
    else:
        print("✗ No response from API")
        
except Exception as e:
    print(f"✗ API call failed: {e}")
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
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    test_code = '''
def add(a, b):
    return a + b
'''
    
    prompt = f"What does this function do? Answer in one sentence:\n{test_code}"
    response = model.generate_content(prompt)
    
    if response and response.text:
        print(f"✓ Code analysis works: {response.text.strip()[:100]}")
    else:
        print("✗ Code analysis failed")
        
except Exception as e:
    print(f"✗ Code analysis failed: {e}")

print("\nDiagnostic complete!")