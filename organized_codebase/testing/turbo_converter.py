#!/usr/bin/env python3
"""
Turbo Converter - Maximum speed conversion with 1000 RPM rate limit.
Uses Gemini-2.5-pro without unnecessary delays.
"""

import os
import time
import ast
from pathlib import Path
from datetime import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_remaining_modules():
    """Get all modules without intelligent tests."""
    test_dir = Path("tests/unit")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("*_intelligent.py"):
            module_name = test_file.stem.replace("test_", "").replace("_intelligent", "")
            existing_tests.add(module_name)
    
    remaining = []
    base_dir = Path("multi_coder_analysis")
    
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            remaining.append(py_file)
    
    return remaining

def generate_test(module_path):
    """Generate test for a module - FAST version."""
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
        
        # Smaller truncation for speed
        if len(content) > 3000:
            content = content[:3000] + "\n# ... truncated ..."
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
    
    # Concise prompt for speed
    prompt = f"""Generate pytest test code for this module.

MODULE: {module_name}.py
IMPORT: from {import_path.rsplit('.', 1)[0]} import {module_name}

```python
{content}
```

Requirements:
1. Import the real module (no mocks)
2. Test all public functions/classes
3. Use pytest
4. Handle import errors

Generate ONLY Python test code."""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=4000  # Smaller for speed
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
        
        # Quick validation
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
        return None, "api_error"

def process_batch(modules, batch_size=16):
    """Process modules in parallel batches - 1000 RPM = ~16 per second."""
    results = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit batch
        futures = {executor.submit(generate_test, module): module for module in modules[:batch_size]}
        
        # Collect results
        for future in as_completed(futures):
            module = futures[future]
            try:
                test_file, status = future.result()
                results.append({
                    "module": str(module),
                    "status": status,
                    "test_file": test_file
                })
            except Exception as e:
                results.append({
                    "module": str(module),
                    "status": f"error: {e}",
                    "test_file": None
                })
    
    return results

def main():
    """Turbo conversion with 1000 RPM rate limit."""
    print("="*70)
    print("TURBO CONVERTER - Maximum Speed Mode")
    print("Rate Limit: 1000 RPM (16 requests/second)")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status
    current_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    target_count = 262  # Full target
    needed = target_count - current_count
    
    print(f"\nCurrent: {current_count}/262 ({current_count/262*100:.1f}%)")
    print(f"Target: {target_count}/262 (100%)")
    print(f"Need to convert: {needed} more files")
    
    if needed <= 0:
        print("\nTarget already achieved!")
        return
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"Found {len(remaining)} modules without tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Process in rapid batches
    results_all = []
    batch_size = 16  # ~1 second per batch at 1000 RPM
    total_batches = (min(needed, len(remaining)) + batch_size - 1) // batch_size
    
    print(f"\nProcessing {min(needed, len(remaining))} modules in {total_batches} batches")
    print("="*70)
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, min(needed, len(remaining)))
        batch_modules = remaining[start_idx:end_idx]
        
        print(f"\nBatch {batch_num + 1}/{total_batches}: Processing {len(batch_modules)} modules...")
        
        # Process batch
        batch_results = process_batch(batch_modules, len(batch_modules))
        results_all.extend(batch_results)
        
        # Quick summary
        success = sum(1 for r in batch_results if r["status"] == "success")
        print(f"  Success: {success}/{len(batch_modules)}")
        
        # Small delay between batches (0.06 seconds = 1 second per 16 requests)
        if batch_num < total_batches - 1:
            time.sleep(0.06)
        
        # Progress update
        if (batch_num + 1) % 5 == 0:
            current = len(list(Path("tests/unit").glob("*_intelligent.py")))
            print(f"\n>>> PROGRESS: {current}/262 ({current/262*100:.1f}%) <<<")
    
    # Final summary
    final_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    success_count = sum(1 for r in results_all if r["status"] == "success")
    
    # Save results
    with open("turbo_results.json", "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "initial_count": current_count,
            "final_count": final_count,
            "conversions": results_all,
            "success": success_count,
            "failed": len(results_all) - success_count
        }, f, indent=2)
    
    print("\n" + "="*70)
    print("TURBO CONVERSION COMPLETE")
    print("="*70)
    print(f"Initial: {current_count} files")
    print(f"Final: {final_count} files")
    print(f"Converted: {success_count} files")
    print(f"Failed: {len(results_all) - success_count} files")
    print(f"\nFinal Progress: {final_count}/262 ({final_count/262*100:.1f}%)")
    
    if final_count >= 262:
        print("\nCOMPLETE! All 262 files converted!")
    elif final_count >= 200:
        print(f"\nAlmost there! Only {262 - final_count} files remaining!")

if __name__ == "__main__":
    main()