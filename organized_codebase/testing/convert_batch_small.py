#!/usr/bin/env python3
"""
Small batch conversion script for Week 3-4 modules
"""

import os
import sys
import ast
import time
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.intelligence.types
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
                max_output_tokens=8000
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
    """Run small batch conversions."""
    
    print("="*70)
    print("SMALL BATCH MOCK-TO-REAL CONVERSION (WEEK 3-4 CONTINUATION)")
    print("="*70)
    
    # Initialize client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Get specific modules for Week 3-4 - SMALL BATCH (3 modules)
    modules_to_convert = [
        "multi_coder_analysis/improvement_system/config_manager.py",
        "multi_coder_analysis/improvement_system/gepa_optimizer.py",
        "multi_coder_analysis/improvement_system/integrated_optimizer_orchestrator.py",
    ]
    
    # Filter to only existing modules
    existing_modules = []
    for module_str in modules_to_convert:
        module_path = Path(module_str)
        if module_path.exists():
            existing_modules.append(module_str)
    
    print(f"Found {len(existing_modules)} modules to convert")
    
    successful = 0
    failed = 0
    
    for i, module_str in enumerate(existing_modules, 1):
        print(f"\n[{i}/{len(existing_modules)}]", end="")
        module_path = Path(module_str)
        
        if build_test_for_module(module_path, client):
            successful += 1
        else:
            failed += 1
        
        # Rate limiting - wait between requests
        if i < len(existing_modules):
            print("  Waiting 20 seconds for rate limit...")
            time.sleep(20)
    
    print("\n" + "="*70)
    print(f"SMALL BATCH CONVERSION COMPLETE")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if successful + failed > 0:
        print(f"Success rate: {successful/(successful+failed)*100:.1f}%")
    print("="*70)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())