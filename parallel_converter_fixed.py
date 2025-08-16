#!/usr/bin/env python3
"""
Parallel Converter FIXED - With proper error handling and incremental result saving.
Optimized for 30 RPM with proper parallelism.
Uses Gemini-2.5-pro with concurrent processing.
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
import traceback

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

# Rate limiter for 30 RPM
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
    """Get all modules without intelligent tests."""
    test_dir = Path("tests/unit")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("*_intelligent.py"):
            module_name = test_file.stem.replace("test_", "").replace("_intelligent", "")
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} existing intelligent tests")
    
    remaining = []
    base_dir = Path("multi_coder_analysis")
    
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
    test_file = Path(f"tests/unit/test_{module_name}_intelligent.py")
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
        rel_path = module_path.relative_to("multi_coder_analysis")
        import_parts = str(rel_path.parent).replace("\\", ".").replace("/", ".")
        if import_parts and import_parts != ".":
            import_path = f"multi_coder_analysis.{import_parts}.{module_name}"
        else:
            import_path = f"multi_coder_analysis.{module_name}"
    except:
        import_path = f"multi_coder_analysis.{module_name}"
    
    # Generate prompt
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

def save_incremental_results(results, filename="parallel_results_incremental.json"):
    """Save results incrementally to avoid losing data."""
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save incremental results: {e}")

def process_modules_parallel(modules, max_workers=5):
    """Process modules in parallel with proper rate limiting and error handling."""
    results = []
    total = len(modules)
    completed = 0
    success = 0
    failed = 0
    
    print(f"\nProcessing {total} modules with {max_workers} parallel workers")
    print(f"Rate limit: 30 RPM (2 seconds between requests)")
    print("="*70)
    
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_module = {executor.submit(generate_test, module): module for module in modules}
            
            # Process results as they complete
            for future in as_completed(future_to_module):
                try:
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
                    
                    # Save incremental results every 10 files
                    if completed % 10 == 0:
                        current = len(list(Path("tests/unit").glob("*_intelligent.py")))
                        pct = current / 262 * 100
                        print(f"\n>>> PROGRESS: {current}/262 ({pct:.1f}%) | Success: {success} | Failed: {failed} <<<\n")
                        
                        # Save incremental results
                        save_incremental_results({
                            "timestamp": datetime.now().isoformat(),
                            "completed": completed,
                            "results": results,
                            "current_count": current
                        })
                        
                except Exception as e:
                    print(f"Error processing future: {e}")
                    traceback.print_exc()
                    
    except Exception as e:
        print(f"Critical error in thread pool: {e}")
        traceback.print_exc()
    
    return results

def main():
    """Main parallel conversion process with better error handling."""
    print("="*70)
    print("PARALLEL CONVERTER (FIXED) - Optimized for 30 RPM")
    print("Starting from where previous converters left off")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status
    current_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
    target_count = 262
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
    
    # Process with parallelism
    start_time = datetime.now()
    results = []
    
    try:
        # Use 5 parallel workers for 30 RPM (each can make 6 requests per minute)
        results = process_modules_parallel(remaining[:needed], max_workers=5)
    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
    finally:
        # Always try to save results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Final summary
        final_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
        success_count = sum(1 for r in results if r["status"] == "success")
        failed_count = sum(1 for r in results if "fail" in r["status"].lower() or "error" in r["status"])
        skipped_count = sum(1 for r in results if r["status"] == "exists" or r["status"] == "too_small")
        
        # Save final results
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
        
        try:
            with open("parallel_results_final.json", "w") as f:
                json.dump(report, f, indent=2)
            print("\nResults saved to parallel_results_final.json")
        except Exception as e:
            print(f"Could not save final results: {e}")
        
        print("\n" + "="*70)
        print("PARALLEL CONVERSION COMPLETE")
        print("="*70)
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Initial: {current_count} files")
        print(f"Final: {final_count} files")
        print(f"Converted: {success_count} files")
        print(f"Failed: {failed_count} files")
        print(f"Skipped: {skipped_count} files")
        print(f"\nFinal Progress: {final_count}/262 ({final_count/262*100:.1f}%)")
        
        if final_count >= 262:
            print("\n*** COMPLETE! All 262 files converted! ***")
        elif final_count >= 200:
            print(f"\nAlmost there! Only {262 - final_count} files remaining!")
        
        # Estimate remaining time
        if success_count > 0:
            rate = success_count / (duration / 60)  # files per minute
            remaining_files = 262 - final_count
            estimated_minutes = remaining_files / rate
            print(f"\nEstimated time for remaining {remaining_files} files: {estimated_minutes:.1f} minutes")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)