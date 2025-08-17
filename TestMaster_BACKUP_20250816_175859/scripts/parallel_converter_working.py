#!/usr/bin/env python3
"""
Parallel Converter - WORKING VERSION
Based on proven parallel_converter.py from tot_branch_minimal
Optimized for 30 RPM with proper parallelism.
"""

import os
import time
import ast
from pathlib import Path
from datetime import datetime
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load environment variables - EXACTLY like working version
env_file = Path(".env")  # Simple path, not complex parent navigation
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

# Rate limiter for 30 RPM - EXACT COPY
class RateLimiter:
    def __init__(self, calls_per_minute=30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # 2 seconds for 30 RPM
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call = time.time()

rate_limiter = RateLimiter(30)

def get_remaining_modules():
    """Get all modules without tests - SIMPLE version."""
    test_dir = Path("tests_new")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            # Extract module name
            module_name = test_file.stem.replace("test_", "")
            # Remove all known suffixes
            for suffix in ["_coverage", "_gemini", "_100coverage", "_api_test", 
                          "_correct_import", "_simple", "_working", "_cov_",
                          "_edge_cases", "_branches", "_mega"]:
                if suffix in module_name:
                    module_name = module_name.split(suffix)[0]
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} modules with tests")
    
    remaining = []
    base_dir = Path("src_new")
    
    # Simple file listing - no coverage analysis
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            remaining.append(py_file)
    
    # Sort for consistent ordering
    remaining.sort()
    return remaining

def generate_test(module_path):
    """Generate test for a module with rate limiting."""
    module_name = module_path.stem
    
    # Check if test exists
    test_file = Path(f"tests_new/test_{module_name}_coverage.py")
    if test_file.exists():
        return module_path, "exists", None
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if len(content) < 50:
            return module_path, "too_small", None
        
        if len(content) > 4000:
            content = content[:4000] + "\n# ... truncated ..."
    except Exception as e:
        return module_path, f"read_error: {e}", None
    
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
    
    # Generate prompt - SIMPLE AND DIRECT
    prompt = f"""Generate comprehensive pytest test code for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content}
```

Requirements:
1. Import using: from {import_path} import *
2. Test ALL public functions and classes
3. Include edge cases
4. Use pytest
5. Use mocks for external dependencies

Generate ONLY Python test code."""

    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
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
        
        # Validate syntax
        try:
            ast.parse(test_code)
        except:
            return module_path, "syntax_error", None
        
        # Save test
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_code)
        
        return module_path, "success", str(test_file)
        
    except Exception as e:
        return module_path, f"api_error: {str(e)[:50]}", None

def process_modules_parallel(modules, max_workers=5):
    """Process modules in parallel with proper rate limiting."""
    results = []
    total = len(modules)
    completed = 0
    success = 0
    failed = 0
    
    print(f"\nProcessing {total} modules with {max_workers} parallel workers")
    print(f"Rate limit: 30 RPM (2 seconds between requests)")
    print("="*70)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_module = {executor.submit(generate_test, module): module for module in modules}
        
        # Process results as they complete
        for future in as_completed(future_to_module):
            module_path, status, test_file = future.result()
            completed += 1
            
            # Update counters
            if status == "success":
                success += 1
                status_str = "SUCCESS"
            elif status == "exists":
                status_str = "SKIP"
            else:
                failed += 1
                status_str = f"FAIL ({status[:20]})"
            
            # Print progress
            print(f"[{completed}/{total}] {module_path.stem:<40} {status_str}")
            
            # Save result
            results.append({
                "module": str(module_path),
                "status": status,
                "test_file": test_file
            })
            
            # Progress update every 10 files
            if completed % 10 == 0:
                current = len(list(Path("tests_new").glob("test_*.py")))
                print(f"\n>>> PROGRESS: {current} test files | Success: {success} | Failed: {failed} <<<\n")
    
    return results

def main():
    """Main parallel conversion process."""
    print("="*70)
    print("PARALLEL CONVERTER - WORKING VERSION")
    print("Based on proven converter from tot_branch_minimal")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status - SIMPLE counting
    current_count = len(list(Path("tests_new").glob("test_*.py")))
    print(f"\nCurrent: {current_count} test files")
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"Found {len(remaining)} modules without tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Process with parallelism
    start_time = datetime.now()
    
    # Use 5 parallel workers for 30 RPM (each can make 6 requests per minute)
    results = process_modules_parallel(remaining, max_workers=5)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final summary
    final_count = len(list(Path("tests_new").glob("test_*.py")))
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if "fail" in r["status"].lower() or "error" in r["status"])
    skipped_count = sum(1 for r in results if r["status"] == "exists" or r["status"] == "too_small")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "initial_count": current_count,
        "final_count": final_count,
        "duration_seconds": duration,
        "results": results,
        "summary": {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_processed": len(results)
        }
    }
    
    with open("parallel_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("PARALLEL CONVERSION COMPLETE")
    print("="*70)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Initial: {current_count} files")
    print(f"Final: {final_count} files")
    print(f"Converted: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Skipped: {skipped_count} files")
    
    print(f"\nTo measure coverage, run:")
    print(f"  python -m pytest tests_new --cov=src_new --cov-report=term")

if __name__ == "__main__":
    main()