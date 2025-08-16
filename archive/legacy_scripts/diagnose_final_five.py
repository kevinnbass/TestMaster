#!/usr/bin/env python3
"""
Diagnose why the final 5 modules are failing
"""

import os
import sys
import ast
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

def diagnose_module(module_name, module_path):
    """Diagnose why a module is failing."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSING: {module_name}")
    print(f"Path: {module_path}")
    print("="*60)
    
    # Step 1: Check if file exists and is readable
    if not module_path.exists():
        print("[ERROR] File does not exist!")
        return
    
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"[OK] File readable: {len(content)} chars")
    except Exception as e:
        print(f"[ERROR] reading file: {e}")
        return
    
    # Step 2: Analyze content for problematic patterns
    print("\nContent Analysis:")
    
    # Check for common issues
    issues = []
    
    if len(content) < 50:
        issues.append("File too small (< 50 chars)")
    
    if len(content) > 10000:
        issues.append(f"File very large ({len(content)} chars)")
    
    # Check for imports that might trigger safety filters
    problematic_imports = ["subprocess", "os.system", "eval", "exec", "__import__"]
    for imp in problematic_imports:
        if imp in content:
            issues.append(f"Contains potentially problematic: {imp}")
    
    # Check for test-related code (might confuse the LLM)
    if "pytest" in content or "unittest" in content:
        issues.append("Already contains test code")
    
    # Check syntax validity
    try:
        ast.parse(content)
        print("[OK] Valid Python syntax")
    except SyntaxError as e:
        issues.append(f"Syntax error in source: {e}")
    
    if issues:
        print("[WARNING] Potential issues found:")
        for issue in issues:
            print(f"   - {issue}")
    else:
        print("[OK] No obvious issues in content")
    
    # Step 3: Try minimal test generation
    print("\nAttempting minimal test generation...")
    
    # Use extremely simple prompt to avoid filters
    simple_prompt = f"""Create a minimal test file for a Python module named {module_name}.

Just include:
- An import statement
- One simple test function
- A basic assertion

Output only Python code."""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=500  # Very small
        )
        
        response = model.generate_content(
            simple_prompt,
            generation_config=generation_config
        )
        
        if response and response.text:
            print("[OK] API responded successfully")
            # Try to parse the response
            test_code = response.text
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0]
            
            try:
                ast.parse(test_code)
                print("[OK] Generated valid Python code")
            except SyntaxError as e:
                print(f"[WARNING]  Generated code has syntax error: {e}")
        else:
            print("[ERROR] API returned empty response")
            
    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] API Error: {error_str[:200]}")
        
        # Check for specific error patterns
        if "Invalid operation" in error_str:
            print("   -> Likely hit safety filter")
        elif "finish_reason" in error_str and "2" in error_str:
            print("   -> Safety filter (finish_reason=2)")
        elif "timeout" in error_str.lower():
            print("   -> Request timeout")
    
    # Step 4: Try with actual content (first 500 chars only)
    print("\n[TEST] Attempting with actual content (truncated)...")
    
    content_snippet = content[:500]
    real_prompt = f"""Write a test for this Python code:

```python
{content_snippet}
```

Output a simple pytest test."""

    try:
        response = model.generate_content(
            real_prompt,
            generation_config=generation_config
        )
        
        if response and response.text:
            print("[OK] API responded to content-based prompt")
        else:
            print("[ERROR] Empty response to content-based prompt")
            
    except Exception as e:
        error_str = str(e)
        print(f"[ERROR] Content-based generation failed: {error_str[:100]}")
        
        if "Invalid operation" in error_str or "finish_reason" in error_str:
            print("\n[ALERT] DIAGNOSIS: Content triggers safety filters")
            print("   Possible reasons:")
            print("   - Code contains sensitive operations")
            print("   - Test generation context is problematic")
            print("   - Module name suggests testing/automation")

def main():
    """Diagnose the final 5 modules."""
    print("="*60)
    print("DIAGNOSING FINAL 5 PROBLEMATIC MODULES")
    print("="*60)
    
    # The 5 modules that are failing
    final_five = [
        ("async_generator_fixed", Path("src_new/pipeline/core/core/async_generator_fixed.py")),
        ("automated_test_generation", Path("src_new/testing/automated_test_generation.py")),
        ("comprehensive_test_framework", Path("src_new/testing/comprehensive_test_framework.py")),
        ("integration_test_matrix", Path("src_new/testing/integration_test_matrix.py")),
        ("prompt_ab_test", Path("src_new/pipeline/core/utils/prompt_ab_test.py"))
    ]
    
    for module_name, module_path in final_five:
        diagnose_module(module_name, module_path)
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)
    
    print("\n[SUMMARY] Summary of Issues:")
    print("1. Testing modules (3/5) may confuse LLM about generating tests for tests")
    print("2. Safety filters may be triggered by automation/generation keywords")
    print("3. Complex async/generator code may be problematic")
    print("\n[TIP] Recommendation: These 5 modules may need manual test creation")

if __name__ == "__main__":
    main()