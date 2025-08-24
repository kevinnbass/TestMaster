#!/usr/bin/env python3
"""
Week 5-8 Batch Converter - Convert remaining 223 modules to intelligent tests.
Uses Gemini-2.5-pro with intelligent test generation.
"""

import os
import time
import ast
from pathlib import Path
from datetime import datetime
import json

# Load environment variables from .env file
from pathlib import Path
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Google Gen AI SDK
try:
    import google.generativeai as genai
except ImportError:
    print("Installing Google Gen AI SDK...")
    os.system("pip install google-generativeai")
    import google.generativeai as genai

# Configure API
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in environment or .env file")
    exit(1)

# Configure genai with API key
genai.configure(api_key=API_KEY)

def read_module_content(module_path):
    """Read module content for test generation."""
    try:
        with open(module_path, "r", encoding="utf-8") as f:
            content = f.read()
        # Truncate if too long
        if len(content) > 8000:
            content = content[:8000] + "\n# ... truncated for length ..."
        return content
    except Exception as e:
        print(f"Error reading {module_path}: {e}")
        return None

def generate_intelligent_test(module_path, content):
    """Generate intelligent test using Gemini-2.5-pro."""
    module_name = module_path.stem
    
    # Build intelligent prompt
    prompt = f"""Generate comprehensive Python test code for this module.

MODULE: {module_name}.py
PATH: {module_path}

```python
{content}
```

Requirements:
1. Import the real module from multi_coder_analysis (use the actual import path based on the module location)
2. NO MOCKS for internal code - test real functionality
3. Test ALL public functions and classes exhaustively
4. Include edge cases and error conditions
5. Use pytest for testing
6. Include docstrings for test methods
7. Handle import errors gracefully with try/except
8. Test actual behavior, not mock behavior

Generate ONLY the Python test code, no explanations. Start with imports."""

    try:
        # Call Gemini API
        model = genai.GenerativeModel('gemini-2.5-pro')  # Using the specified model
        
        generation_config = genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=8000
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Extract code from response
        test_code = response.text
        
        # Clean markdown if present
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0]
        elif "```" in test_code:
            test_code = test_code.split("```")[1].split("```")[0]
        
        # Validate Python syntax
        try:
            ast.parse(test_code)
            return test_code
        except SyntaxError as e:
            print(f"  Syntax error in generated test: {e}")
            return None
            
    except Exception as e:
        print(f"  API error: {e}")
        return None

def fix_import_paths(test_code, module_path):
    """Fix import paths to use multi_coder_analysis."""
    # Get relative path from multi_coder_analysis
    rel_path = module_path.relative_to("multi_coder_analysis")
    import_path = str(rel_path.parent).replace("\\", ".").replace("/", ".")
    
    if import_path:
        import_path = f"multi_coder_analysis.{import_path}"
    else:
        import_path = "multi_coder_analysis"
    
    module_name = module_path.stem
    
    # Fix common import patterns
    replacements = [
        (f"from {module_name} import", f"from {import_path}.{module_name} import"),
        (f"import {module_name}", f"from {import_path} import {module_name}"),
    ]
    
    for old, new in replacements:
        if old in test_code and "multi_coder_analysis" not in old:
            test_code = test_code.replace(old, new)
    
    return test_code

def save_test_file(test_code, module_name):
    """Save the generated test file."""
    test_dir = Path("tests/unit")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / f"test_{module_name}_intelligent.py"
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_code)
    
    return test_file

def convert_batch(modules, batch_name="batch"):
    """Convert a batch of modules to intelligent tests."""
    results = {
        "batch": batch_name,
        "timestamp": datetime.now().isoformat(),
        "total": len(modules),
        "success": 0,
        "failed": 0,
        "modules": []
    }
    
    print(f"\n{'='*70}")
    print(f"Converting {len(modules)} modules for {batch_name}")
    print(f"{'='*70}")
    
    for i, module_path in enumerate(modules, 1):
        module_name = module_path.stem
        print(f"\n[{i}/{len(modules)}] Converting: {module_name}")
        
        # Skip if already has intelligent test
        test_file = Path(f"tests/unit/test_{module_name}_intelligent.py")
        if test_file.exists():
            print(f"  SKIP: Test already exists")
            continue
        
        # Read module content
        content = read_module_content(module_path)
        if not content:
            print(f"  ERROR: Could not read module")
            results["failed"] += 1
            results["modules"].append({
                "module": str(module_path),
                "status": "read_error"
            })
            continue
        
        # Generate test
        print(f"  Generating intelligent test...")
        test_code = generate_intelligent_test(module_path, content)
        
        if test_code:
            # Fix import paths
            test_code = fix_import_paths(test_code, module_path)
            
            # Save test file
            test_file = save_test_file(test_code, module_name)
            print(f"  SUCCESS: Saved to {test_file.name}")
            results["success"] += 1
            results["modules"].append({
                "module": str(module_path),
                "test_file": str(test_file),
                "status": "success"
            })
        else:
            print(f"  FAILED: Could not generate test")
            results["failed"] += 1
            results["modules"].append({
                "module": str(module_path),
                "status": "generation_failed"
            })
        
        # Rate limiting - be conservative
        time.sleep(2)  # 30 RPM = 2 seconds between requests
    
    # Save results
    results_file = Path(f"{batch_name}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Batch Complete: {results['success']}/{results['total']} successful")
    print(f"Results saved to: {results_file}")
    print(f"{'='*70}")
    
    return results

def get_priority_modules():
    """Get priority modules for conversion."""
    modules = []
    
    # Core pipeline modules
    core_dirs = [
        "multi_coder_analysis/core/pipeline",
        "multi_coder_analysis/core/regex",
        "multi_coder_analysis/consensus",
        "multi_coder_analysis/runtime",
        "multi_coder_analysis/runtime/execution",
        "multi_coder_analysis/regex",
        "multi_coder_analysis/prompt_assembly"
    ]
    
    for dir_path in core_dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                if not py_file.name.startswith("_") and "test" not in py_file.name:
                    modules.append(py_file)
    
    # Important improvement system modules
    improvement_dirs = [
        "multi_coder_analysis/improvement_system/optimization",
        "multi_coder_analysis/improvement_system/core/interfaces",
        "multi_coder_analysis/improvement_system/strategies",
        "multi_coder_analysis/improvement_system/validators",
        "multi_coder_analysis/improvement_system/verifiers"
    ]
    
    for dir_path in improvement_dirs:
        dir_path = Path(dir_path)
        if dir_path.exists():
            for py_file in dir_path.glob("*.py"):
                if not py_file.name.startswith("_") and "test" not in py_file.name:
                    modules.append(py_file)
    
    return modules

def main():
    """Main conversion process for weeks 5-8."""
    
    print("="*70)
    print("WEEK 5-8 INTELLIGENT TEST CONVERTER")
    print("="*70)
    print(f"Using Gemini-2.0-flash-exp for speed")
    print(f"Target: Convert remaining modules to intelligent tests")
    
    # Get priority modules for Week 5-6
    priority_modules = get_priority_modules()
    
    print(f"\nFound {len(priority_modules)} priority modules for Week 5-6")
    
    # Week 5-6: Convert priority modules (target 60)
    week_5_6_modules = priority_modules[:60]
    if week_5_6_modules:
        print("\n" + "="*70)
        print("WEEK 5-6: Advanced Components")
        print("="*70)
        week_5_6_results = convert_batch(week_5_6_modules, "week_5_6")
        
        # Save progress
        with open("week_5_6_complete.txt", "w") as f:
            f.write(f"Week 5-6 Complete: {week_5_6_results['success']} modules converted\n")
            f.write(f"Timestamp: {datetime.now()}\n")
    
    # Get all remaining modules
    from find_unconverted_modules import find_unconverted_modules
    all_unconverted = find_unconverted_modules()
    
    # Filter out what we already converted
    remaining = [m for m in all_unconverted if m not in week_5_6_modules]
    
    print(f"\n{len(remaining)} modules remaining for Week 7-8")
    
    # Week 7-8: Convert remaining modules
    if remaining and len(remaining) > 0:
        print("\n" + "="*70)
        print("WEEK 7-8: Final Conversion")
        print("="*70)
        
        # Process in smaller batches to avoid overwhelming the API
        batch_size = 50
        for batch_num, start_idx in enumerate(range(0, len(remaining), batch_size), 1):
            end_idx = min(start_idx + batch_size, len(remaining))
            batch_modules = remaining[start_idx:end_idx]
            
            print(f"\nBatch {batch_num}: Modules {start_idx+1}-{end_idx} of {len(remaining)}")
            batch_results = convert_batch(batch_modules, f"week_7_8_batch_{batch_num}")
            
            # Take a break between batches
            if end_idx < len(remaining):
                print("\nTaking 30 second break between batches...")
                time.sleep(30)
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE")
    print("="*70)
    print("All modules have been processed!")
    print("Run the test suite to validate: pytest tests/unit/*_intelligent.py")

if __name__ == "__main__":
    main()