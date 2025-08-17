#!/usr/bin/env python3
"""
Test Single Test Generation
============================

Generate a single test file to verify Gemini 2.5 Pro works.
"""

import os
import sys
import asyncio
from pathlib import Path

# Load environment
def load_env():
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

from google import genai
from google.genai import types

async def generate_single_test():
    """Generate a single test file."""
    print("=" * 70)
    print("SINGLE TEST GENERATION WITH GEMINI 2.5 PRO")
    print("=" * 70)
    
    # Initialize client
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: No API key found")
        return False
    
    print(f"[OK] API key found")
    client = genai.Client(api_key=api_key)
    model_name = "gemini-2.5-pro"
    
    # Read a simple source file
    src_file = Path("src_new/core/domain.py")
    if not src_file.exists():
        print(f"ERROR: {src_file} not found")
        return False
    
    with open(src_file, 'r', encoding='utf-8') as f:
        source_code = f.read()[:1000]  # Reduced to 1000 chars
    
    # Create simpler prompt
    prompt = f"""Write pytest tests for this Python code. Output only the test code, no explanations:

```python
{source_code}
```

Output format:
```python
import pytest
from unittest.mock import Mock

def test_example():
    assert True
```"""
    
    print(f"\n[INFO] Calling Gemini 2.5 Pro...")
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8000  # Increased from 2000
            )
        )
        
        print(f"[DEBUG] Response: {response}")
        print(f"[DEBUG] Has text: {hasattr(response, 'text')}")
        
        if response:
            if hasattr(response, 'text'):
                text = response.text
                print(f"[DEBUG] Text length: {len(text) if text else 0}")
            else:
                print("[DEBUG] No text attribute")
                
        if response and response.text:
            print("[OK] Got response from Gemini 2.5 Pro")
            
            # Extract code
            text = response.text
            if '```python' in text:
                parts = text.split('```python')
                if len(parts) > 1:
                    code_part = parts[1].split('```')[0]
                    
                    # Save test file
                    test_file = Path("tests_new/test_domain_gemini_single.py")
                    test_file.write_text(code_part, encoding='utf-8')
                    
                    print(f"[OK] Saved test to {test_file}")
                    print(f"\nGenerated test preview:")
                    print("-" * 40)
                    print(code_part[:500])
                    print("-" * 40)
                    
                    return True
        else:
            print("ERROR: No response from Gemini")
            
    except Exception as e:
        print(f"ERROR: {e}")
    
    return False

async def main():
    success = await generate_single_test()
    
    if success:
        print("\nSUCCESS: Single test generated!")
        print("\nNow you can run the full generation.")
    else:
        print("\nFAILED: Could not generate test")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))