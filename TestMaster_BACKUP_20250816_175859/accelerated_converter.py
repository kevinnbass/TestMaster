#!/usr/bin/env python3
"""
Accelerated Converter - Push toward 40% completion (105 files).
Uses Gemini-2.5-pro with optimized batch processing.
"""

import os
import time
import ast
from pathlib import Path
from datetime import datetime
import json
import sys
import concurrent.futures
from typing import List, Tuple

# Load environment variables
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

def get_remaining_modules() -> List[Path]:
    """Get all modules without intelligent tests."""
    test_dir = Path("tests/unit")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("*_intelligent.py"):
            module_name = test_file.stem.replace("test_", "").replace("_intelligent", "")
            existing_tests.add(module_name)
    
    remaining = []
    
    # Scan all directories
    base_dir = Path("multi_coder_analysis")
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            remaining.append(py_file)
    
    return remaining

def generate_test(module_path: Path) -> Tuple[str, str]:
    """Generate test for a module."""
    module_name = module_path.stem
    
    # Check if test exists
    test_file = Path(f"tests/unit/test_{module_name}_intelligent.py")
    if test_file.exists():
        return None, "exists"
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if len(content) < 50:
            return None, "too_small"
        
        if len(content) > 4000:
            content = content[:4000] + "\n# ... truncated ..."
    except:
        return None, "read_error"
    
    # Build import path
    try:
        rel_path = module_path.relative_to("multi_coder_analysis")
        import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
        if import_parts and import_parts != ".":
            import_path = f"multi_coder_analysis.{import_parts}.{module_name}"
        else:
            import_path = f"multi_coder_analysis.{module_name}"
    except:
        import_path = f"multi_coder_analysis.{module_name}"
    
    # Generate test
    prompt = f"""Generate comprehensive pytest test code for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content}
```

Requirements:
1. Import using: from {import_path.rsplit('.', 1)[0]} import {module_name}
2. NO MOCKS - test real functionality
3. Test ALL public functions and classes
4. Include edge cases
5. Use pytest
6. Handle import errors with try/except

Generate ONLY Python test code."""

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
        
        # Validate
        try:
            ast.parse(test_code)
        except:
            return None, "syntax_error"
        
        # Save
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        return str(test_file), "success"
        
    except Exception as e:
        return None, f"api_error"

def convert_module_with_delay(module_path: Path, delay: int = 2) -> dict:
    """Convert a single module with rate limiting."""
    module_name = module_path.stem
    test_file, status = generate_test(module_path)
    
    result = {
        "module": str(module_path),
        "module_name": module_name,
        "status": status,
        "test_file": test_file
    }
    
    if status == "success":
        time.sleep(delay)  # Rate limiting
    
    return result

def main():
    """Main accelerated conversion."""
    print("="*70)
    print("ACCELERATED CONVERTER - Target: 40% (105 files)")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status
    current_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    target_count = 105
    needed = target_count - current_count
    
    print(f"\nCurrent: {current_count}/262 ({current_count/262*100:.1f}%)")
    print(f"Target: {target_count}/262 (40.1%)")
    print(f"Need to convert: {needed} more files")
    
    if needed <= 0:
        print("\n✅ Target already achieved!")
        return
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"Found {len(remaining)} modules without tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Convert modules
    results = {
        "start_time": datetime.now().isoformat(),
        "initial_count": current_count,
        "target_count": target_count,
        "conversions": [],
        "success": 0,
        "failed": 0,
        "skipped": 0
    }
    
    print(f"\n{'='*70}")
    print(f"Converting up to {needed} modules")
    print(f"{'='*70}\n")
    
    # Process modules
    for i, module_path in enumerate(remaining[:needed], 1):
        print(f"[{i}/{needed}] {module_path.stem}...", end=" ")
        
        result = convert_module_with_delay(module_path, delay=2)
        results["conversions"].append(result)
        
        if result["status"] == "success":
            print("SUCCESS")
            results["success"] += 1
        elif result["status"] == "exists":
            print("SKIP")
            results["skipped"] += 1
        else:
            print(f"FAILED ({result['status']})")
            results["failed"] += 1
        
        # Progress check
        if i % 10 == 0:
            current = len(list(Path("tests/unit").glob("*_intelligent.py")))
            print(f"\n>>> Progress: {current}/262 ({current/262*100:.1f}%) <<<\n")
    
    # Final summary
    results["end_time"] = datetime.now().isoformat()
    final_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    results["final_count"] = final_count
    
    # Save results
    with open("accelerated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("ACCELERATED CONVERSION COMPLETE")
    print("="*70)
    print(f"Initial: {current_count} files")
    print(f"Final: {final_count} files")
    print(f"Converted: {results['success']} files")
    print(f"Failed: {results['failed']} files")
    print(f"Skipped: {results['skipped']} files")
    print(f"\nFinal Progress: {final_count}/262 ({final_count/262*100:.1f}%)")
    
    if final_count >= target_count:
        print(f"\n🎉 TARGET ACHIEVED! Reached {final_count/262*100:.1f}% completion!")
    else:
        print(f"\nStill need {target_count - final_count} more files to reach 40%")

if __name__ == "__main__":
    main()