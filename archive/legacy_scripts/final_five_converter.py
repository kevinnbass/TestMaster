#!/usr/bin/env python3
"""
Final Five Converter - Generate tests for the last 5 modules
"""

import os
import sys
import time
from pathlib import Path

# Load environment
env_file = Path(".env")
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

def generate_test_for_module(module_name, module_path):
    """Generate test for a specific module."""
    print(f"\nGenerating test for {module_name}...")
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()[:3000]
    except Exception as e:
        print(f"  Error reading: {e}")
        return False
    
    # Build import path
    try:
        rel_path = module_path.relative_to("src_new")
        import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
        if import_parts and import_parts != ".":
            import_path = f"{import_parts}.{module_name}"
        else:
            import_path = module_name
    except:
        import_path = module_name
    
    print(f"  Import path: {import_path}")
    
    # Generate SIMPLE prompt to avoid safety filters
    prompt = f"""Write a simple pytest test file for this Python module.

Module name: {module_name}.py

Code to test:
```python
{content}
```

Generate a basic test file with:
1. Import statements
2. Simple test functions
3. Basic assertions

Output only Python code."""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=3000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        if response and response.text:
            test_code = response.text
            
            # Clean markdown
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0]
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0]
            
            # Add proper imports
            final_code = f"""import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

{test_code}
"""
            
            # Save test
            test_file = Path(f"tests_new/test_{module_name}_final.py")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(final_code)
            
            print(f"  [SUCCESS] Saved to {test_file.name}")
            return True
        else:
            print(f"  [FAIL] No response")
            return False
            
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False

def main():
    """Generate tests for the final 5 modules."""
    print("="*60)
    print("FINAL FIVE CONVERTER")
    print("Generating tests for the last 5 modules")
    print("="*60)
    
    # The 5 modules that need tests
    final_five = [
        ("async_generator_fixed", Path("src_new/pipeline/core/core/async_generator_fixed.py")),
        ("automated_test_generation", Path("src_new/testing/automated_test_generation.py")),
        ("comprehensive_test_framework", Path("src_new/testing/comprehensive_test_framework.py")),
        ("integration_test_matrix", Path("src_new/testing/integration_test_matrix.py")),
        ("prompt_ab_test", Path("src_new/pipeline/core/utils/prompt_ab_test.py"))
    ]
    
    success = 0
    failed = 0
    
    for module_name, module_path in final_five:
        if generate_test_for_module(module_name, module_path):
            success += 1
        else:
            failed += 1
        
        # Rate limit: 30 RPM
        time.sleep(2)
    
    print("\n" + "="*60)
    print(f"COMPLETE: {success} success, {failed} failed")
    
    if success == 5:
        print("\nALL MODULES NOW HAVE TESTS!")
        print("Run coverage measurement to see final percentage")
    
    print("="*60)

if __name__ == "__main__":
    main()