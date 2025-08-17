#!/usr/bin/env python3
"""
Test Gemini 2.5 Pro Configuration
==================================

Verify that Gemini 2.5 Pro is configured correctly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Load environment variables from .env file
def load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent / '.env'
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"[OK] Loaded environment from {env_file}")

load_env()

# Check for Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    print("[OK] Google GenAI SDK is installed")
except ImportError as e:
    print(f"ERROR: Google GenAI SDK not installed: {e}")
    print("Install with: pip install google-generativeai")
    sys.exit(1)

async def test_gemini_25_pro():
    """Test Gemini 2.5 Pro configuration."""
    print("\n" + "=" * 70)
    print("TESTING GEMINI 2.5 PRO CONFIGURATION")
    print("=" * 70)
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    
    if not api_key:
        print("\nERROR: GOOGLE_API_KEY environment variable not set!")
        print("Please set it with:")
        print("  Windows: set GOOGLE_API_KEY=your-api-key")
        print("  Linux/Mac: export GOOGLE_API_KEY='your-api-key'")
        return False
    
    print(f"\n[OK] Google API key found (length: {len(api_key)})")
    
    # Initialize client
    try:
        client = genai.Client(api_key=api_key)
        print("[OK] Gemini client initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize client: {e}")
        return False
    
    # Test with Gemini 2.5 Pro
    model_name = "gemini-2.5-pro"
    print(f"\n[INFO] Testing model: {model_name}")
    
    try:
        # Simple test prompt
        prompt = """Generate a simple Python function that returns True.
Just the function, no explanation."""
        
        print("[INFO] Sending test request to Gemini 2.5 Pro...")
        
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=1000  # Increased from 100
            )
        )
        
        print(f"[DEBUG] Response type: {type(response)}")
        print(f"[DEBUG] Response: {response}")
        
        if response:
            # Try to access the text
            if hasattr(response, 'text'):
                if response.text:
                    print("[OK] Gemini 2.5 Pro responded successfully!")
                    print(f"\nResponse text: {response.text[:200]}")
                    return True
                else:
                    print("[WARNING] Response.text is empty")
            
            # Check candidates
            if hasattr(response, 'candidates') and response.candidates:
                print(f"[DEBUG] Number of candidates: {len(response.candidates)}")
                candidate = response.candidates[0]
                print(f"[DEBUG] Candidate: {candidate}")
                
                if hasattr(candidate, 'content'):
                    content = candidate.content
                    print(f"[DEBUG] Content type: {type(content)}")
                    print(f"[DEBUG] Content: {content}")
                    
                    # Check if content has parts
                    if hasattr(content, 'parts'):
                        print(f"[DEBUG] Content parts: {content.parts}")
                        if content.parts:
                            text = ''.join(str(part.text) if hasattr(part, 'text') else str(part) for part in content.parts)
                            if text:
                                print("[OK] Gemini 2.5 Pro responded successfully!")
                                print(f"\nExtracted text: {text[:200]}")
                                return True
        
        print("ERROR: Could not extract text from Gemini 2.5 Pro response")
        return False
            
    except Exception as e:
        print(f"ERROR: Gemini API call failed: {e}")
        print("\nPossible issues:")
        print("1. Invalid API key")
        print("2. API key doesn't have access to Gemini 2.5 Pro")
        print("3. Network connectivity issues")
        print("4. Rate limit exceeded")
        return False

async def main():
    """Main test function."""
    success = await test_gemini_25_pro()
    
    print("\n" + "=" * 70)
    if success:
        print("SUCCESS: Gemini 2.5 Pro is configured correctly!")
        print("\nYou can now run the full test generation script:")
        print("  python scripts/test_coverage/achieve_100_percent_coverage.py")
    else:
        print("FAILURE: Gemini 2.5 Pro configuration issues detected")
        print("\nPlease fix the issues above before proceeding")
    print("=" * 70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))