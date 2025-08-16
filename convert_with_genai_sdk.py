#!/usr/bin/env python3
"""
Conversion using google.genai SDK (proper implementation)
"""

import os
import sys
import ast
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def build_test_for_module(module_path: Path, client: genai.Client):
    """Build test for a single module."""
    
    print(f"\nConverting: {module_path.name}")
    
    if not module_path.exists():
        print(f"  ERROR: Module not found")
        return False
    
    # Read module content
    content = module_path.read_text(encoding='utf-8')
    if len(content) > 8000:
        content = content[:8000] + "\n# ... (truncated)"
    
    # Generate test using proper SDK syntax
    prompt = f"""Generate comprehensive Python test code for this module.

MODULE: {module_path.name}

```python
{content}
```

Requirements:
1. Import the real module (use the actual import path based on the module location)
2. NO MOCKS for internal code - test real functionality
3. Test ALL public functions and classes exhaustively
4. Include edge cases and error conditions
5. Use pytest for testing
6. Include docstrings for test methods

Generate ONLY the Python test code, no explanations."""

    try:
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8000  # Increased for comprehensive tests
            )
        )
        
        if response.text:
            test_code = response.text.strip()
            # Clean up markdown if present
            if test_code.startswith("```python"):
                test_code = test_code[9:]
            if test_code.startswith("```"):
                test_code = test_code[3:]
            if test_code.endswith("```"):
                test_code = test_code[:-3]
            
            # Save test
            output_dir = Path("tests/unit")
            output_dir.mkdir(parents=True, exist_ok=True)
            test_filename = f"test_{module_path.stem}_intelligent.py"
            test_path = output_dir / test_filename
            
            test_path.write_text(test_code.strip(), encoding='utf-8')
            
            # Validate syntax
            try:
                ast.parse(test_code)
                test_count = test_code.count("def test_")
                print(f"  SUCCESS: Generated {test_count} test methods")
                return True
            except SyntaxError as e:
                print(f"  ERROR: Syntax error in generated test: {e}")
                return False
        else:
            print(f"  ERROR: No response from model")
            return False
            
    except Exception as e:
        print(f"  ERROR: {str(e)[:200]}")
        return False

def main():
    """Run conversions for Weeks 3-8 modules."""
    
    print("="*70)
    print("MOCK-TO-REAL CONVERSION WITH GEMINI-2.5-PRO (Google Gen AI SDK)")
    print("="*70)
    
    # Initialize client with proper SDK syntax
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Get list of modules from SIMPLE_MOCK_ANALYSIS.md
    modules_to_convert = []
    
    # Read the analysis file
    analysis_file = Path("SIMPLE_MOCK_ANALYSIS.md")
    if analysis_file.exists():
        content = analysis_file.read_text()
        
        # Extract SIMPLE tests (easier to convert)
        if "## SIMPLE:" in content:
            simple_section = content.split("## SIMPLE:")[1].split("##")[0]
            for line in simple_section.split("\n"):
                if "**test_" in line and ".py" in line:
                    test_name = line.split("**")[1]
                    # Convert test name to module name
                    module_name = test_name.replace("test_", "").replace(".py", "")
                    
                    # Try to find the module
                    search_paths = [
                        f"multi_coder_analysis/{module_name}.py",
                        f"multi_coder_analysis/runtime/{module_name}.py",
                        f"multi_coder_analysis/improvement_system/{module_name}.py",
                        f"multi_coder_analysis/utils/{module_name}.py",
                        f"multi_coder_analysis/llm_providers/{module_name}.py",
                    ]
                    
                    for path_str in search_paths:
                        if Path(path_str).exists():
                            modules_to_convert.append(path_str)
                            break
                    
                    if len(modules_to_convert) >= 30:  # Limit to 30 for Week 3-4
                        break
    
    print(f"Found {len(modules_to_convert)} modules to convert")
    
    successful = 0
    failed = 0
    
    for i, module_str in enumerate(modules_to_convert, 1):
        print(f"\n[{i}/{len(modules_to_convert)}]", end="")
        module_path = Path(module_str)
        if build_test_for_module(module_path, client):
            successful += 1
        else:
            failed += 1
    
    print("\n" + "="*70)
    print(f"CONVERSION COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful + failed > 0:
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    print("="*70)
    
    # Generate summary report
    report = f"""# Weeks 3-8 Conversion Report

## Conversion Statistics
- Total modules attempted: {len(modules_to_convert)}
- Successful conversions: {successful}
- Failed conversions: {failed}
- Success rate: {successful/(successful+failed)*100:.1f}% if successful + failed > 0 else 0%

## Files Generated
All tests saved to `tests/unit/` with `_intelligent.py` suffix.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    report_path = Path("WEEKS_3_8_CONVERSION_REPORT.md")
    report_path.write_text(report)
    print(f"\nReport saved to: {report_path}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())