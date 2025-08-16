#!/usr/bin/env python3
"""
Test Single API Call - Verify exact syntax from proven converter
"""

import os
import sys
from pathlib import Path

# Load environment variables EXACTLY like the proven converter
env_file = Path(__file__).parent.parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

import google.generativeai as genai

# Configure API EXACTLY like proven converter
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found")
    sys.exit(1)

genai.configure(api_key=API_KEY)

def test_single_generation():
    """Test single generation with exact proven syntax."""
    print("="*70)
    print("TESTING SINGLE API CALL - EXACT PROVEN SYNTAX")
    print("="*70)
    
    # Read a simple test module
    module_path = Path("src_new/core/domain.py")
    module_name = module_path.stem
    
    if not module_path.exists():
        print(f"ERROR: {module_path} not found")
        return False
    
    # Read module content
    with open(module_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Truncate if needed (EXACTLY like proven converter)
    if len(content) > 5000:
        content = content[:5000] + "\n# ... truncated ..."
    
    # Build import path EXACTLY like proven converter
    try:
        rel_path = module_path.relative_to("src_new")
        import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
        if import_parts and import_parts != ".":
            import_path = f"{import_parts}.{module_name}"
        else:
            import_path = module_name
    except:
        import_path = module_name
    
    print(f"Module: {module_name}")
    print(f"Import path: {import_path}")
    print(f"Content length: {len(content)} chars")
    
    # Generate prompt EXACTLY like proven converter
    prompt = f"""Generate comprehensive pytest test code for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content[:2000]}
```

Requirements:
1. Import using: from {import_path.rsplit('.', 1)[0] if '.' in import_path else import_path} import {module_name}
2. NO MOCKS - test real functionality
3. Test ALL public functions and classes
4. Include edge cases
5. Use pytest
6. Handle import errors with try/except

Generate ONLY Python test code."""

    print(f"\nPrompt length: {len(prompt)} chars")
    print("Calling Gemini API with EXACT proven syntax...")
    
    try:
        # EXACT API call from proven converter
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=6000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        print(f"\n[OK] Got response from Gemini")
        print(f"Response type: {type(response)}")
        
        test_code = response.text
        print(f"Response text length: {len(test_code) if test_code else 0}")
        
        # Clean markdown EXACTLY like proven converter
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
            print("[OK] Extracted code from markdown")
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
            print("[OK] Extracted code from code block")
        
        # Show preview
        print("\nGenerated test preview:")
        print("-" * 40)
        print(test_code[:500])
        print("-" * 40)
        
        # Save raw response for debugging
        debug_file = Path("test_raw_response.txt")
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Original response:\n{response.text}\n\n")
            f.write(f"Cleaned code:\n{test_code}")
        print(f"[DEBUG] Saved raw response to {debug_file}")
        
        # Validate syntax
        import ast
        try:
            ast.parse(test_code)
            print("\n[OK] Syntax validation passed")
        except SyntaxError as e:
            print(f"\n[WARNING] Syntax error (will attempt to fix): {e}")
            # Try to fix common issues
            lines = test_code.split('\n')
            if len(lines) > 280:
                print(f"[DEBUG] Line 278: {lines[277][:100]}")
            # Continue anyway for now
        
        # Save test
        test_file = Path(f"tests_new/test_{module_name}_api_test.py")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        print(f"\n[OK] Test saved to {test_file}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] API call failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    success = test_single_generation()
    
    print("\n" + "="*70)
    if success:
        print("SUCCESS: API call works with proven syntax!")
        print("\nYou can now run the full parallel converter:")
        print("  python scripts/test_coverage/parallel_coverage_converter.py")
    else:
        print("FAILURE: API call failed")
        print("\nCheck the error messages above")
    print("="*70)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())