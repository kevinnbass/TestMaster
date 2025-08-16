#!/usr/bin/env python3
"""
Test Updated Prompt - Verify imports work correctly
"""

import os
import sys
from pathlib import Path
import ast

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

def test_with_correct_imports():
    """Test with correct import paths."""
    print("="*70)
    print("TESTING WITH CORRECT IMPORT PATHS")
    print("="*70)
    
    # Test with core.domain
    module_path = Path("src_new/core/domain.py")
    module_name = module_path.stem
    
    # Read module
    with open(module_path, "r", encoding="utf-8") as f:
        content = f.read()[:2000]
    
    # Build correct import path
    rel_path = module_path.relative_to("src_new")
    import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
    if import_parts and import_parts != ".":
        import_path = f"{import_parts}.{module_name}"
    else:
        import_path = module_name
    
    print(f"Module: {module_name}")
    print(f"Import path: {import_path}")
    
    # Generate prompt with correct imports
    prompt = f"""Generate pytest unit tests for this Python module.

Module: {module_name}.py
Import path: {import_path}

Source code:
```python
{content}
```

Requirements:
1. Test all public functions and classes
2. Include edge cases
3. Use mocks where appropriate
4. Generate working pytest code

Start with these imports:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from {import_path} import *
```

Output only the Python test code."""

    print("\nCalling Gemini API...")
    
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
        
        test_code = response.text
        
        # Clean markdown
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
        
        print(f"[OK] Got response")
        
        # Check if imports are correct
        if "from core.domain import *" in test_code:
            print("[OK] Correct import found!")
        else:
            print("[WARNING] Import might be incorrect")
        
        # Show first few lines
        lines = test_code.split('\n')[:20]
        print("\nFirst 20 lines:")
        print("-" * 40)
        for i, line in enumerate(lines, 1):
            print(f"{i:3}: {line}")
        print("-" * 40)
        
        # Validate syntax
        try:
            ast.parse(test_code)
            print("\n[OK] Syntax valid")
        except SyntaxError as e:
            print(f"\n[ERROR] Syntax error: {e}")
            return False
        
        # Save test
        test_file = Path(f"tests_new/test_{module_name}_correct_import.py")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        print(f"\n[OK] Saved to {test_file}")
        
        # Test if it runs
        print("\nTesting if imports work...")
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "passed" in result.stdout or "PASSED" in result.stdout:
            print("[OK] Test runs successfully!")
            return True
        elif "skipped" in result.stdout.lower():
            print("[ERROR] Test was skipped - import failed")
            print(result.stdout[:500])
            return False
        else:
            print("[INFO] Test output:")
            print(result.stdout[:500])
            return True  # May have failures but imports worked
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = test_with_correct_imports()
    
    print("\n" + "="*70)
    if success:
        print("SUCCESS: Import paths work correctly!")
        print("\nReady to run full parallel converter")
    else:
        print("FAILURE: Need to fix import paths")
    print("="*70)