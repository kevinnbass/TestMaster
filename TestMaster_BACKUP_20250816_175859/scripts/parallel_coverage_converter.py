#!/usr/bin/env python3
"""
Parallel Coverage Converter - Optimized for 30 RPM with proper parallelism.
Uses Gemini-2.5-pro with concurrent processing to achieve 100% test coverage.
Based on proven parallel_converter.py pattern.
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
from queue import Queue
import subprocess

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
    """Get all modules without test coverage, focusing on uncovered code."""
    test_dir = Path("tests_new")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("test_*_coverage.py"):
            module_name = test_file.stem.replace("test_", "").replace("_coverage", "")
            existing_tests.add(module_name)
        for test_file in test_dir.glob("test_*_gemini.py"):
            module_name = test_file.stem.replace("test_", "").replace("_gemini", "")
            existing_tests.add(module_name)
        for test_file in test_dir.glob("test_*_100coverage.py"):
            module_name = test_file.stem.replace("test_", "").replace("_100coverage", "")
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} existing test files")
    
    # Get coverage data to prioritize uncovered modules
    coverage_data = get_coverage_data()
    
    remaining = []
    base_dir = Path("src_new")
    
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            
            # Get coverage for this module
            module_coverage = get_module_coverage(str(py_file), coverage_data)
            if module_coverage < 100:  # Only include if not fully covered
                remaining.append((py_file, module_coverage))
    
    # Sort by coverage (lowest first) to prioritize uncovered modules
    remaining.sort(key=lambda x: x[1])
    
    # Return just the paths
    return [path for path, _ in remaining]

def get_coverage_data():
    """Get current coverage data."""
    try:
        # Run coverage analysis
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-report=json',
             '--tb=no', '-q', '--disable-warnings'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                return json.load(f)
    except:
        pass
    
    return {}

def get_module_coverage(module_path, coverage_data):
    """Get coverage percentage for a specific module."""
    if not coverage_data:
        return 0
    
    for file_path, file_data in coverage_data.get('files', {}).items():
        if Path(module_path).name in file_path:
            executed = len(file_data.get('executed_lines', []))
            missing = len(file_data.get('missing_lines', []))
            total = executed + missing
            if total > 0:
                return (executed / total) * 100
    
    return 0

def generate_test(module_path):
    """Generate test for a module with rate limiting."""
    module_name = module_path.stem
    
    # Check if test exists
    test_files = [
        Path(f"tests_new/test_{module_name}_coverage.py"),
        Path(f"tests_new/test_{module_name}_gemini.py"),
        Path(f"tests_new/test_{module_name}_100coverage.py")
    ]
    
    for test_file in test_files:
        if test_file.exists():
            return module_path, "exists", None
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if len(content) < 50:
            return module_path, "too_small", None
        
        if len(content) > 5000:
            content = content[:5000] + "\n# ... truncated ..."
    except Exception as e:
        return module_path, f"read_error: {e}", None
    
    # Get missing lines for targeted testing
    coverage_data = get_coverage_data()
    missing_lines = []
    for file_path, file_data in coverage_data.get('files', {}).items():
        if module_name in file_path:
            missing_lines = file_data.get('missing_lines', [])
            break
    
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
    
    # Generate prompt focused on coverage
    prompt = f"""Generate comprehensive pytest test code to achieve 100% coverage for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}
MISSING LINES: {missing_lines[:50] if missing_lines else 'All lines need coverage'}

```python
{content}
```

CRITICAL Requirements for 100% Coverage:
1. Test EVERY function and class method
2. Test ALL branches (if/else, try/except, for/while loops)
3. Test edge cases and error conditions
4. Use mocks for external dependencies (Mock, AsyncMock, patch)
5. Test both success and failure paths
6. Include tests for lines: {missing_lines[:30] if missing_lines else 'all'}
7. NO SKIP - tests must run and achieve coverage

Start EXACTLY with:
```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Import the module
from {import_path} import *
```

Generate ONLY Python test code that will achieve 100% coverage. Do NOT use try/except for imports. The imports MUST work."""

    # Apply rate limiting
    rate_limiter.wait_if_needed()
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,  # Low temperature for consistent output
            max_output_tokens=8000  # More tokens for comprehensive tests
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
        
        # Add header
        final_code = f'''#!/usr/bin/env python3
"""
100% Coverage Tests for {module_name}
Generated by Gemini-2.5-pro for complete test coverage
Target lines: {missing_lines[:30] if missing_lines else 'all'}
"""

{test_code}
'''
        
        # Save test
        test_file = Path(f"tests_new/test_{module_name}_coverage.py")
        test_file.parent.mkdir(parents=True, exist_ok=True)
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(final_code)
        
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
                status_str = "[OK]"
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
                coverage = measure_current_coverage()
                print(f"\n>>> PROGRESS: Coverage: {coverage:.2f}% | Success: {success} | Failed: {failed} <<<\n")
    
    return results

def measure_current_coverage():
    """Measure current test coverage."""
    try:
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new',
             '--cov=src_new', '--cov-report=term',
             '--tb=no', '-q', '--disable-warnings'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line:
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        return float(parts[-1].rstrip('%'))
                    except:
                        pass
    except:
        pass
    
    return 0.0

def main():
    """Main parallel conversion process for 100% coverage."""
    print("="*70)
    print("PARALLEL COVERAGE CONVERTER - Optimized for 30 RPM")
    print("Target: 100% Test Coverage")
    print("Model: Gemini-2.5-pro")
    print("="*70)
    
    # Current status
    current_coverage = measure_current_coverage()
    target_coverage = 100.0
    gap = target_coverage - current_coverage
    
    print(f"\nCurrent Coverage: {current_coverage:.2f}%")
    print(f"Target Coverage: {target_coverage:.1f}%")
    print(f"Coverage Gap: {gap:.2f}%")
    
    if current_coverage >= 100.0:
        print("\n*** TARGET ACHIEVED! 100% Coverage! ***")
        return
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"\nFound {len(remaining)} modules needing tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Process with parallelism
    start_time = datetime.now()
    
    # Use 5 parallel workers for 30 RPM (each can make 6 requests per minute)
    results = process_modules_parallel(remaining, max_workers=5)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final coverage measurement
    final_coverage = measure_current_coverage()
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if "fail" in r["status"].lower() or "error" in r["status"])
    skipped_count = sum(1 for r in results if r["status"] == "exists" or r["status"] == "too_small")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "initial_coverage": current_coverage,
        "final_coverage": final_coverage,
        "coverage_improvement": final_coverage - current_coverage,
        "duration_seconds": duration,
        "results": results,
        "summary": {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_processed": len(results)
        }
    }
    
    with open("parallel_coverage_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*70)
    print("PARALLEL COVERAGE CONVERSION COMPLETE")
    print("="*70)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Initial Coverage: {current_coverage:.2f}%")
    print(f"Final Coverage: {final_coverage:.2f}%")
    print(f"Improvement: +{final_coverage - current_coverage:.2f}%")
    print(f"Tests Generated: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    
    if final_coverage >= 100.0:
        print("\n*** SUCCESS! 100% COVERAGE ACHIEVED! ***")
    elif final_coverage >= 90.0:
        print(f"\nAlmost there! Only {100.0 - final_coverage:.2f}% to go!")
    elif final_coverage > current_coverage:
        print(f"\nGood progress! Coverage improved by {final_coverage - current_coverage:.2f}%")
    
    # Estimate remaining time
    if success_count > 0 and final_coverage < 100:
        rate = (final_coverage - current_coverage) / (duration / 60)  # % per minute
        remaining_coverage = 100.0 - final_coverage
        estimated_minutes = remaining_coverage / rate if rate > 0 else 0
        print(f"\nEstimated time to 100%: {estimated_minutes:.1f} minutes")
        print(f"Run again to continue generating tests for remaining modules")

if __name__ == "__main__":
    main()