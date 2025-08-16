#!/usr/bin/env python3
"""
Week 7-8 Final Converter - Convert remaining modules to reach 100% completion.
Uses Gemini-2.5-pro with intelligent test generation.
"""

import os
import time
import ast
from pathlib import Path
from datetime import datetime
import json
import sys

# Load environment variables from .env file
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

print(f"Configuring Gemini-2.5-pro API...")
genai.configure(api_key=API_KEY)

def get_remaining_modules():
    """Get all modules that don't have intelligent tests yet."""
    # Get existing intelligent tests
    test_dir = Path("tests/unit")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("*_intelligent.py"):
            module_name = test_file.stem.replace("test_", "").replace("_intelligent", "")
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} existing intelligent tests")
    
    # Find all Python modules
    remaining = []
    priority_dirs = [
        # Core functionality (highest priority)
        "multi_coder_analysis/core",
        "multi_coder_analysis/runtime",
        "multi_coder_analysis/prompt_assembly",
        "multi_coder_analysis/providers",
        "multi_coder_analysis/utils",
        
        # Improvement system core
        "multi_coder_analysis/improvement_system/core",
        "multi_coder_analysis/improvement_system/optimization",
        "multi_coder_analysis/improvement_system/validators",
        "multi_coder_analysis/improvement_system/verifiers",
        
        # Important subsystems
        "multi_coder_analysis/improvement_system/strategies",
        "multi_coder_analysis/improvement_system/factories",
        "multi_coder_analysis/improvement_system/monitoring",
        "multi_coder_analysis/improvement_system/observability",
        
        # Other modules
        "multi_coder_analysis/evaluation",
        "multi_coder_analysis/models",
        "multi_coder_analysis/telemetry"
    ]
    
    for dir_path in priority_dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                if (not py_file.name.startswith("_") and 
                    not py_file.name.startswith("test") and
                    "__pycache__" not in str(py_file)):
                    module_name = py_file.stem
                    if module_name not in existing_tests and module_name != "__init__":
                        remaining.append(py_file)
    
    # Also check root level files
    root_dir = Path("multi_coder_analysis")
    for py_file in root_dir.glob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test")):
            module_name = py_file.stem
            if module_name not in existing_tests and module_name != "__init__":
                remaining.append(py_file)
    
    return remaining

def generate_test(module_path):
    """Generate intelligent test for a module."""
    module_name = module_path.stem
    
    # Check if test already exists
    test_file = Path(f"tests/unit/test_{module_name}_intelligent.py")
    if test_file.exists():
        return None, "exists"
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Skip if too small
        if len(content) < 100:
            return None, "too_small"
        
        # Truncate if too long
        if len(content) > 5000:
            content = content[:5000] + "\n# ... truncated for length ..."
    except Exception as e:
        return None, f"read_error: {e}"
    
    # Get import path
    try:
        rel_path = module_path.relative_to("multi_coder_analysis")
        import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
        if import_parts and import_parts != ".":
            import_path = f"multi_coder_analysis.{import_parts}.{module_name}"
        else:
            import_path = f"multi_coder_analysis.{module_name}"
    except:
        import_path = f"multi_coder_analysis.{module_name}"
    
    # Build comprehensive prompt
    prompt = f"""Generate comprehensive pytest test code for this Python module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content}
```

Requirements:
1. Import the actual module using: from {import_path.rsplit('.', 1)[0]} import {module_name}
2. NO MOCKS for internal code - test real functionality only
3. Test ALL public functions, classes, and methods exhaustively
4. Include edge cases, error conditions, and boundary testing
5. Use pytest framework with fixtures where appropriate
6. Add docstrings explaining what each test validates
7. Handle import errors gracefully with try/except
8. Ensure 100% coverage of all public APIs

Generate ONLY the Python test code, no explanations. Start with imports."""

    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        test_code = response.text
        
        # Clean markdown if present
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
        
        # Validate Python syntax
        try:
            ast.parse(test_code)
        except SyntaxError as e:
            return None, f"syntax_error: {e}"
        
        # Save test file
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        return str(test_file), "success"
        
    except Exception as e:
        return None, f"api_error: {str(e)[:100]}"

def convert_batch(modules, batch_name="batch", max_modules=50):
    """Convert a batch of modules."""
    results = {
        "batch": batch_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(modules),
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "details": []
    }
    
    print(f"\n{'='*70}")
    print(f"Converting {len(modules)} modules for {batch_name}")
    print(f"{'='*70}")
    
    for i, module_path in enumerate(modules[:max_modules], 1):
        module_name = module_path.stem
        print(f"\n[{i}/{min(len(modules), max_modules)}] {module_name}...", end=" ")
        
        test_file, status = generate_test(module_path)
        
        if status == "exists":
            print("SKIP (exists)")
            results["skipped"] += 1
        elif status == "too_small":
            print("SKIP (too small)")
            results["skipped"] += 1
        elif status == "success":
            print(f"SUCCESS")
            results["success"] += 1
            results["details"].append({
                "module": str(module_path),
                "test": test_file,
                "status": "success"
            })
        else:
            print(f"FAILED ({status[:30]})")
            results["failed"] += 1
            results["details"].append({
                "module": str(module_path),
                "status": status
            })
        
        # Rate limiting for Gemini-2.5-pro
        if status == "success":
            time.sleep(3)  # Conservative rate limiting
    
    # Save results
    results_file = Path(f"{batch_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"{batch_name} Complete:")
    print(f"  Success: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Skipped: {results['skipped']}")
    print(f"Results saved to: {results_file}")
    
    return results

def main():
    """Main Week 7-8 conversion process."""
    
    print("="*70)
    print("WEEK 7-8 FINAL CONVERTER")
    print("Target: Complete mock-to-real test conversion")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Get remaining modules
    remaining_modules = get_remaining_modules()
    print(f"\nFound {len(remaining_modules)} modules without intelligent tests")
    
    # Count existing tests
    existing_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    print(f"Current intelligent tests: {existing_count}")
    print(f"Target: 262 total tests")
    print(f"Remaining to convert: {262 - existing_count}")
    
    if not remaining_modules:
        print("\nNo modules to convert!")
        return
    
    # Process in batches
    batch_size = 30  # Conservative batch size
    total_batches = (len(remaining_modules) + batch_size - 1) // batch_size
    
    print(f"\nWill process in {total_batches} batches of up to {batch_size} modules each")
    
    overall_results = {
        "week": "7-8",
        "start_time": datetime.now().isoformat(),
        "total_modules": len(remaining_modules),
        "total_success": 0,
        "total_failed": 0,
        "total_skipped": 0,
        "batches": []
    }
    
    # Process first 3 batches as demonstration (90 modules max)
    max_batches = 3
    
    for batch_num in range(min(total_batches, max_batches)):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(remaining_modules))
        batch_modules = remaining_modules[start_idx:end_idx]
        
        print(f"\n{'='*70}")
        print(f"BATCH {batch_num + 1} of {min(total_batches, max_batches)}")
        print(f"Modules {start_idx + 1} to {end_idx}")
        print(f"{'='*70}")
        
        batch_results = convert_batch(
            batch_modules, 
            f"week_7_8_batch_{batch_num + 1}",
            batch_size
        )
        
        overall_results["total_success"] += batch_results["success"]
        overall_results["total_failed"] += batch_results["failed"]
        overall_results["total_skipped"] += batch_results["skipped"]
        overall_results["batches"].append(batch_results)
        
        # Take a break between batches
        if batch_num < min(total_batches - 1, max_batches - 1):
            print("\nTaking 30-second break between batches...")
            time.sleep(30)
    
    # Final summary
    overall_results["end_time"] = datetime.now().isoformat()
    
    with open("week_7_8_overall_results.json", "w") as f:
        json.dump(overall_results, f, indent=2)
    
    print("\n" + "="*70)
    print("WEEK 7-8 CONVERSION SUMMARY")
    print("="*70)
    print(f"Total modules processed: {overall_results['total_success'] + overall_results['total_failed'] + overall_results['total_skipped']}")
    print(f"Successfully converted: {overall_results['total_success']}")
    print(f"Failed: {overall_results['total_failed']}")
    print(f"Skipped: {overall_results['total_skipped']}")
    
    # Count final test files
    final_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    print(f"\nTotal intelligent tests now: {final_count}")
    print(f"Progress: {final_count}/262 ({final_count/262*100:.1f}%)")
    
    if final_count < 262:
        print(f"\nRemaining work: {262 - final_count} modules still need conversion")
        print("Run this script again to continue conversion")
    else:
        print("\n🎉 CONVERSION COMPLETE! All 262 modules have intelligent tests!")
    
    print("="*70)

if __name__ == "__main__":
    main()