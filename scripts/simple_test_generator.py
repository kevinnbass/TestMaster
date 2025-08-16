#!/usr/bin/env python3
"""
Simple Test Generator - Generate tests one by one for reliability
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
import json

# Load environment variables
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value
                
import google.generativeai as genai

# Configure API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def get_modules_to_test():
    """Get modules that need tests."""
    modules = []
    src_dir = Path("src_new")
    test_dir = Path("tests_new")
    
    # Get existing tests
    existing = set()
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            name = test_file.stem.replace("test_", "")
            for suffix in ["_coverage", "_gemini", "_100coverage", "_api_test", "_correct_import", "_cov_"]:
                if suffix in name:
                    name = name.split(suffix)[0]
            existing.add(name)
    
    # Find modules without tests
    for py_file in src_dir.rglob("*.py"):
        if ("__pycache__" not in str(py_file) and 
            py_file.stem != "__init__" and
            not py_file.name.startswith("_") and
            py_file.stem not in existing):
            modules.append(py_file)
    
    return modules

def generate_test(module_path):
    """Generate test for a single module."""
    module_name = module_path.stem
    print(f"\nGenerating test for {module_name}...")
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()[:4000]
    except Exception as e:
        print(f"  Error reading: {e}")
        return False
    
    # Build import path
    rel_path = module_path.relative_to("src_new")
    import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
    if import_parts and import_parts != ".":
        import_path = f"{import_parts}.{module_name}"
    else:
        import_path = module_name
    
    # Generate prompt
    prompt = f"""Generate pytest tests for this Python module.

Module: {module_name}.py

Source code:
```python
{content}
```

Generate comprehensive tests with this exact structure:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from {import_path} import *

# Test code here
```

Output ONLY Python test code."""

    # Call API
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=6000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if not response or not response.text:
            print(f"  No response")
            return False
        
        test_code = response.text
        
        # Clean markdown
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
        
        # Save test
        test_file = Path(f"tests_new/test_{module_name}_simple.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        print(f"  [OK] Saved {test_file.name}")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Main function."""
    print("="*70)
    print("SIMPLE TEST GENERATOR")
    print("Model: Gemini-2.5-pro")
    print("Rate: 30 RPM (2 second delay)")
    print("="*70)
    
    modules = get_modules_to_test()
    print(f"\nFound {len(modules)} modules needing tests")
    
    if not modules:
        print("No modules to test!")
        return
    
    # Process first 10 modules
    batch = modules[:10]
    print(f"Processing batch of {len(batch)} modules")
    
    success = 0
    failed = 0
    
    for i, module in enumerate(batch):
        print(f"\n[{i+1}/{len(batch)}] {module.stem}")
        
        if generate_test(module):
            success += 1
        else:
            failed += 1
        
        # Rate limit: 30 RPM = 2 seconds between requests
        if i < len(batch) - 1:
            print("  Waiting 2 seconds for rate limit...")
            time.sleep(2)
    
    print("\n" + "="*70)
    print(f"COMPLETE: {success} success, {failed} failed")
    print("="*70)

if __name__ == "__main__":
    main()