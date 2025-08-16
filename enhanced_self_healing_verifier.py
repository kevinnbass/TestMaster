#!/usr/bin/env python3
"""
Enhanced Self-Healing Test Converter with Iterative Verification
Combines syntax healing with exhaustiveness verification in multiple passes.

Process:
1. Generate initial test
2. Syntax check → Self-heal if needed (up to 5 iterations)
3. Verifier Pass 1: Check exhaustiveness → Refine if needed
4. Syntax check → Self-heal if needed  
5. Verifier Pass 2: Further refinement → Refine if needed
6. Syntax check → Self-heal if needed
7. Verifier Pass 3: Final quality check

This ensures both syntactic correctness AND comprehensive test coverage.
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
    """Get all modules without enhanced tests."""
    test_dir = Path("tests/unit")
    existing_tests = set()
    
    if test_dir.exists():
        for test_file in test_dir.glob("*_enhanced.py"):
            module_name = test_file.stem.replace("test_", "").replace("_enhanced", "")
            existing_tests.add(module_name)
    
    print(f"Found {len(existing_tests)} modules with enhanced tests")
    
    remaining = []
    base_dir = Path("multi_coder_analysis")
    
    for py_file in base_dir.rglob("*.py"):
        if (not py_file.name.startswith("_") and 
            not py_file.name.startswith("test") and
            "__pycache__" not in str(py_file) and
            py_file.stem not in existing_tests and
            py_file.stem != "__init__"):
            remaining.append(py_file)
    
    remaining.sort()
    return remaining

def make_llm_call(prompt, purpose="generation"):
    """Make rate-limited LLM call."""
    rate_limiter.wait_if_needed()
    
    try:
        model = genai.GenerativeModel('gemini-2.5-pro')
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8000 if purpose == "verification" else 6000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        print(f"    LLM call failed: {e}")
        return None

def fix_syntax_error(test_code, error_msg, iteration=1):
    """Use LLM to fix syntax errors in generated code."""
    print(f"    [HEAL] Fixing syntax error (iteration {iteration}/5)...")
    
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
6. Maintain all existing test logic and coverage

Output the complete fixed Python code."""
    
    response = make_llm_call(fix_prompt, "healing")
    if not response:
        return None
    
    # Clean markdown
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]
    
    return response

def verify_test_exhaustiveness(test_code, module_code, module_name, pass_number=1):
    """Verify if test is exhaustive and suggest improvements."""
    print(f"    [VERIFY] Pass {pass_number}/3...")
    
    verify_prompt = f"""Analyze this test code for completeness and suggest improvements.

MODULE CODE:
```python
{module_code}
```

CURRENT TEST CODE:
```python
{test_code}
```

Analysis requirements:
1. Check if ALL public functions/classes are tested
2. Verify edge cases are covered
3. Check for error handling tests
4. Identify missing test scenarios
5. Suggest additional test cases

Respond in this format:
COMPLETENESS_SCORE: [0-100]
MISSING_COVERAGE: [list what's missing]
IMPROVEMENTS: [specific suggestions]
ENHANCED_TEST: [improved test code if score < 85, otherwise "SUFFICIENT"]

Be thorough - this is pass {pass_number} of 3."""

    response = make_llm_call(verify_prompt, "verification")
    if not response:
        return None, "verification_failed"
    
    # Parse response
    lines = response.split('\n')
    score = 0
    missing = []
    improvements = []
    enhanced_test = None
    
    current_section = None
    enhanced_code_lines = []
    in_code_block = False
    
    for line in lines:
        if line.startswith("COMPLETENESS_SCORE:"):
            try:
                score = int(''.join(filter(str.isdigit, line.split(':')[1])))
            except:
                score = 50
        elif line.startswith("MISSING_COVERAGE:"):
            missing.append(line.split(':', 1)[1].strip())
        elif line.startswith("IMPROVEMENTS:"):
            improvements.append(line.split(':', 1)[1].strip())
        elif line.startswith("ENHANCED_TEST:"):
            rest = line.split(':', 1)[1].strip()
            if rest == "SUFFICIENT":
                enhanced_test = "SUFFICIENT"
            else:
                current_section = "enhanced_test"
        elif current_section == "enhanced_test":
            if "```python" in line:
                in_code_block = True
            elif "```" in line and in_code_block:
                in_code_block = False
                enhanced_test = '\n'.join(enhanced_code_lines)
            elif in_code_block:
                enhanced_code_lines.append(line)
            else:
                enhanced_code_lines.append(line)
    
    # If we didn't get enhanced test from code block, try to extract it
    if enhanced_test is None and enhanced_code_lines:
        enhanced_test = '\n'.join(enhanced_code_lines)
    
    return {
        "score": score,
        "missing": missing,
        "improvements": improvements,
        "enhanced_test": enhanced_test
    }, "success"

def generate_enhanced_test(module_path, max_healing_iterations=5, max_verifier_passes=3):
    """Generate test with self-healing and iterative verification."""
    module_name = module_path.stem
    
    # Check if enhanced test exists
    test_file = Path(f"tests/unit/test_{module_name}_enhanced.py")
    if test_file.exists():
        return module_path, "exists", None
    
    print(f"\n[PROCESS] Processing {module_name}...")
    
    # Read module
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            module_content = f.read()
        
        if len(module_content) < 50:
            return module_path, "too_small", None
        
        if len(module_content) > 4000:
            module_content = module_content[:4000] + "\n# ... truncated ..."
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
    
    # Generate initial test
    print("    [GEN] Generating initial test...")
    initial_prompt = f"""Generate comprehensive pytest test code for this module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{module_content}
```

Requirements:
1. Import using: from {import_path.rsplit('.', 1)[0]} import {module_name}
2. NO MOCKS - test real functionality
3. Test ALL public functions and classes
4. Include edge cases and error handling
5. Use pytest with proper fixtures
6. Handle import errors with try/except
7. Ensure syntactically correct Python code

Generate ONLY Python test code."""

    response = make_llm_call(initial_prompt)
    if not response:
        return module_path, "initial_generation_failed", None
    
    # Clean markdown
    if "```python" in response:
        test_code = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        test_code = response.split("```")[1].split("```")[0]
    else:
        test_code = response
    
    # Phase 1: Self-healing for syntax
    print("    [PHASE1] Syntax healing...")
    for healing_iteration in range(1, max_healing_iterations + 1):
        try:
            ast.parse(test_code)
            print(f"    [OK] Syntax valid after {healing_iteration-1} healing iterations")
            break
        except SyntaxError as e:
            if healing_iteration < max_healing_iterations:
                error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
                if e.text:
                    error_msg += f"\nProblem line: {e.text}"
                
                fixed_code = fix_syntax_error(test_code, error_msg, healing_iteration)
                if fixed_code:
                    test_code = fixed_code
                else:
                    return module_path, f"healing_failed_iteration_{healing_iteration}", None
            else:
                return module_path, f"syntax_error_after_{max_healing_iterations}_healing_attempts", None
    
    # Phase 2: Iterative verification and refinement
    print("    [PHASE2] Iterative verification...")
    for verifier_pass in range(1, max_verifier_passes + 1):
        verification_result, status = verify_test_exhaustiveness(
            test_code, module_content, module_name, verifier_pass
        )
        
        if status != "success":
            print(f"    [FAIL] Verification pass {verifier_pass} failed")
            break
        
        score = verification_result["score"]
        enhanced_test = verification_result["enhanced_test"]
        
        print(f"    [SCORE] Completeness score: {score}/100")
        
        if score >= 85 or enhanced_test == "SUFFICIENT":
            print(f"    [OK] Test quality sufficient (score: {score})")
            break
        
        if enhanced_test and enhanced_test != "SUFFICIENT":
            print(f"    [IMPROVE] Applying improvements from verifier pass {verifier_pass}...")
            test_code = enhanced_test
            
            # Heal syntax after refinement
            for healing_iteration in range(1, max_healing_iterations + 1):
                try:
                    ast.parse(test_code)
                    break
                except SyntaxError as e:
                    if healing_iteration < max_healing_iterations:
                        error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
                        if e.text:
                            error_msg += f"\nProblem line: {e.text}"
                        
                        fixed_code = fix_syntax_error(test_code, error_msg, healing_iteration)
                        if fixed_code:
                            test_code = fixed_code
                        else:
                            return module_path, f"post_verification_healing_failed_pass_{verifier_pass}", None
                    else:
                        return module_path, f"syntax_error_after_verification_pass_{verifier_pass}", None
    
    # Final syntax check
    try:
        ast.parse(test_code)
    except SyntaxError:
        return module_path, "final_syntax_check_failed", None
    
    # Save enhanced test
    test_file.parent.mkdir(parents=True, exist_ok=True)
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_code)
    
    print(f"    [SAVED] Enhanced test saved: {test_file}")
    return module_path, "success", str(test_file)

def process_modules_enhanced(modules, max_workers=3):
    """Process modules with enhanced healing and verification."""
    results = []
    total = len(modules)
    completed = 0
    success = 0
    failed = 0
    
    print(f"\nProcessing {total} modules with enhanced healing + verification")
    print(f"Workers: {max_workers} | Rate limit: 30 RPM")
    print("="*70)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_module = {
            executor.submit(generate_enhanced_test, module): module 
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
                    status_str = f"FAIL ({status[:30]})"
                
                # Print progress
                print(f"[{completed}/{total}] {module_path.stem:<40} {status_str}")
                
                # Save result
                results.append({
                    "module": str(module_path),
                    "status": status,
                    "test_file": test_file
                })
                
                # Progress update every 5 files
                if completed % 5 == 0:
                    current = len(list(Path("tests/unit").glob("*_enhanced.py")))
                    print(f"\n>>> ENHANCED TESTS: {current} | Success: {success} | Failed: {failed} <<<\n")
                    
            except Exception as e:
                print(f"Error processing module: {e}")
                failed += 1
    
    return results

def main():
    """Main enhanced conversion process."""
    print("="*80)
    print("ENHANCED SELF-HEALING + VERIFIER TEST CONVERTER")
    print("Combines syntax healing with iterative exhaustiveness verification")
    print("Model: Gemini-2.5-pro | Healing: 5 iterations | Verification: 3 passes")
    print("="*80)
    
    # Current status
    current_enhanced = len(list(Path("tests/unit").glob("*_enhanced.py")))
    current_total = len(list(Path("tests/unit").glob("*_intelligent.py")))
    print(f"\nCurrent intelligent tests: {current_total}")
    print(f"Current enhanced tests: {current_enhanced}")
    
    # Get remaining modules
    remaining = get_remaining_modules()
    print(f"Found {len(remaining)} modules without enhanced tests")
    
    if not remaining:
        print("No modules to convert!")
        return
    
    # Process batch with enhanced system
    batch_size = min(15, len(remaining))  # Smaller batches due to intensive processing
    print(f"\nProcessing batch of {batch_size} modules with healing + verification...")
    
    start_time = datetime.now()
    
    # Use 2 workers (lower due to intensive verification)
    results = process_modules_enhanced(remaining[:batch_size], max_workers=2)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Final summary
    final_enhanced = len(list(Path("tests/unit").glob("*_enhanced.py")))
    success_count = sum(1 for r in results if r["status"] == "success")
    failed_count = sum(1 for r in results if "fail" in r["status"].lower() or "error" in r["status"])
    skipped_count = sum(1 for r in results if r["status"] == "exists")
    
    # Save results
    report = {
        "timestamp": datetime.now().isoformat(),
        "initial_enhanced": current_enhanced,
        "final_enhanced": final_enhanced,
        "duration_seconds": duration,
        "results": results,
        "summary": {
            "success": success_count,
            "failed": failed_count,
            "skipped": skipped_count,
            "total_processed": len(results)
        }
    }
    
    with open("enhanced_healing_results.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print("\n" + "="*80)
    print("ENHANCED CONVERSION COMPLETE")
    print("="*80)
    print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"Enhanced tests created: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Total enhanced tests: {final_enhanced}")
    
    if success_count > 0:
        print(f"\nSuccess rate: {success_count / len(results) * 100:.1f}%")
        print(f"Average time per test: {duration / success_count:.1f} seconds")
    
    print(f"\nTo continue: python enhanced_self_healing_verifier.py")
    print(f"To test: pytest tests/unit/*_enhanced.py -v")

if __name__ == "__main__":
    main()