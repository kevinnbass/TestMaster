#!/usr/bin/env python3
"""
Self-Healing Test Converter
Automatically fixes syntax errors by passing them back to the LLM
Can iterate up to 5 times to get working code
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
    """Get all modules without tests."""
    test_dir = Path("tests_new")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("test_*.py"):
            module_name = test_file.stem.replace("test_", "")
            # Remove all known suffixes
            for suffix in ["_coverage", "_gemini", "_100coverage", "_api_test", 
                          "_correct_import", "_simple", "_working", "_cov_",
                          "_edge_cases", "_branches", "_mega", "_healed"]:
                if suffix in module_name:
                    module_name = module_name.split(suffix)[0]
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} modules with tests")
    
    remaining = []
    base_dir = Path("src_new")
    
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            remaining.append(py_file)
    
    remaining.sort()
    return remaining

def fix_syntax_error(test_code, error_msg, iteration=1):
    """Use LLM to fix syntax errors in generated code."""
    print(f"    Attempting to fix syntax error (iteration {iteration}/5)...")
    
    fix_prompt = f"""Fix the syntax error in this Python test code.

ERROR MESSAGE:
{error_msg}

CODE WITH ERROR:
```python
{test_code}
```

Requirements:
1. Fix the syntax error described in the error message
2. Return ONLY the fixed Python code
3. Ensure proper indentation
4. Complete any unterminated strings or brackets
5. Fix any invalid syntax

Output the complete fixed Python code."""
    
    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=6000
        )
        
        response = model.generate_content(
            fix_prompt,
            generation_config=generation_config
        )
        
        fixed_code = response.text
        
        # Clean markdown
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0]
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0]
        
        return fixed_code
        
    except Exception as e:
        print(f"    Error fixing syntax: {e}")
        return None

def generate_test_with_healing(module_path, max_iterations=5):
    """Generate test with automatic syntax error fixing."""
    module_name = module_path.stem
    
    # Check if test exists
    test_file = Path(f"tests_new/test_{module_name}_healed.py")
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
    
    # Generate initial prompt
    prompt = f"""Generate comprehensive pytest test code for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content}
```

Requirements:
1. Start with: import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))
2. Import using: from {import_path} import *
3. Test ALL public functions and classes
4. Include edge cases
5. Use pytest
6. Use mocks for external dependencies (Mock, AsyncMock, patch)
7. Ensure syntactically correct Python code

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
        
        # Try to validate and fix syntax iteratively
        for iteration in range(1, max_iterations + 1):
            try:
                ast.parse(test_code)
                # Syntax is valid, save and return
                test_file.parent.mkdir(parents=True, exist_ok=True)
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write(test_code)
                
                if iteration > 1:
                    print(f"    Fixed after {iteration} iterations")
                
                return module_path, "success", str(test_file)
                
            except SyntaxError as e:
                if iteration < max_iterations:
                    # Get detailed error info
                    error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
                    if e.text:
                        error_msg += f"\nProblem line: {e.text}"
                    
                    # Try to fix the error
                    fixed_code = fix_syntax_error(test_code, error_msg, iteration)
                    
                    if fixed_code:
                        test_code = fixed_code
                    else:
                        # Couldn't fix, give up
                        return module_path, f"syntax_error_unfixable", None
                else:
                    # Max iterations reached
                    return module_path, f"syntax_error_after_{max_iterations}_tries", None
        
    except Exception as e:
        return module_path, f"api_error: {str(e)[:50]}", None

def process_modules_with_healing(modules, max_workers=5):
    """Process modules in parallel with self-healing."""
    results = []
    total = len(modules)
    completed = 0
    success = 0
    failed = 0
    
    print(f"\nProcessing {total} modules with {max_workers} parallel workers")
    print(f"Self-healing enabled: up to 5 iterations to fix syntax errors")
    print(f"Rate limit: 30 RPM (2 seconds between requests)")
    print("="*70)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_module = {
            executor.submit(generate_test_with_healing, module): module 
            for module in modules
        }
        
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
                
                # Progress update every 10 files
                if completed % 10 == 0:
                    current = len(list(Path("tests_new").glob("test_*.py")))
                    print(f"\n>>> PROGRESS: {current} test files | Success: {success} | Failed: {failed} <<<\n")
                    
            except Exception as e:
                print(f"Error processing module: {e}")
                failed += 1
    
    return results

def main():
    """Main self-healing conversion process."""
    print("="*70)
    print("SELF-HEALING TEST CONVERTER")
    print("Automatically fixes syntax errors through iteration")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status
    current_count = len(list(Path("tests_new").glob("test_*.py")))
    print(f"\nCurrent: {current_count} test files")
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"Found {len(remaining)} modules without tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Process first batch with self-healing
    batch_size = min(20, len(remaining))  # Process 20 at a time
    print(f"\nProcessing batch of {batch_size} modules with self-healing...")
    
    start_time = datetime.now()
    
    # Use 3 parallel workers (lower to account for potential retries)
    results = process_modules_with_healing(remaining[:batch_size], max_workers=3)
    
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
    
    with open("self_healing_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("SELF-HEALING CONVERSION COMPLETE")
    print("="*70)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Initial: {current_count} files")
    print(f"Final: {final_count} files")
    print(f"Converted: {success_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Skipped: {skipped_count} files")
    
    if success_count > 0:
        print(f"\nSuccess rate: {success_count / len(results) * 100:.1f}%")
    
    print(f"\nTo continue, run this script again")
    print(f"To measure coverage: python -m pytest tests_new --cov=src_new --cov-report=term")

if __name__ == "__main__":
    main()