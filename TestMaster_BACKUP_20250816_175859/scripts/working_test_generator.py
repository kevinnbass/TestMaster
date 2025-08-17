#!/usr/bin/env python3
"""
Working Test Generator - Actually generates tests
"""

import os
import sys
import time
from pathlib import Path

# Add parent path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Test imports first
try:
    import google.generativeai as genai
    print("[OK] Google GenAI imported")
except ImportError as e:
    print(f"ERROR: Could not import genai: {e}")
    sys.exit(1)

# Configure API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

print(f"[OK] API key found (length: {len(API_KEY)})")
genai.configure(api_key=API_KEY)
print("[OK] API configured")

def get_first_module():
    """Get first module that needs a test."""
    src_dir = Path("src_new")
    
    # Simple approach - just get first module
    for py_file in src_dir.glob("*.py"):
        if py_file.stem not in ["__init__", "bootstrap"]:
            return py_file
    
    # Try subdirectories
    for py_file in src_dir.rglob("*.py"):
        if ("__pycache__" not in str(py_file) and 
            py_file.stem != "__init__"):
            return py_file
    
    return None

def generate_single_test():
    """Generate a single test to verify everything works."""
    print("\n" + "="*50)
    print("GENERATING SINGLE TEST")
    print("="*50)
    
    module_path = get_first_module()
    if not module_path:
        print("ERROR: No modules found")
        return False
    
    module_name = module_path.stem
    print(f"\nModule: {module_name}")
    print(f"Path: {module_path}")
    
    # Read content
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()[:2000]
        print(f"Content length: {len(content)} chars")
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return False
    
    # Simple prompt
    prompt = f"""Generate a simple pytest test for this Python code:

```python
{content}
```

Output only the test code."""
    
    print("\nCalling Gemini API...")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=2000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if response and response.text:
            print("[OK] Got response")
            print(f"Response length: {len(response.text)} chars")
            
            # Save test
            test_file = Path(f"tests_new/test_{module_name}_working.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            
            print(f"[OK] Saved to {test_file}")
            return True
        else:
            print("ERROR: No response text")
            return False
            
    except Exception as e:
        print(f"ERROR calling API: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = generate_single_test()
    
    print("\n" + "="*50)
    if success:
        print("SUCCESS: Test generated!")
        print("\nNow you can run the full batch generator")
    else:
        print("FAILURE: Could not generate test")
        print("Check the errors above")
    print("="*50)