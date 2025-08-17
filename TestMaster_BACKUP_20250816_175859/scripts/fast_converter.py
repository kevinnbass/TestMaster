#!/usr/bin/env python3
"""
Fast Test Converter - Quick Path to High Coverage
=================================================

Focused on generating tests for the most important modules first.
"""

import os
import sys
import ast
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generate_test_for_module(module_path: Path, client: genai.Client) -> bool:
    """Generate test for a single module quickly."""
    
    print(f"  {module_path.name}...", end=" ")
    
    # Read module
    content = module_path.read_text(encoding='utf-8')
    if len(content) > 5000:
        content = content[:5000] + "\n# ... (truncated)"
    
    # Build import path
    rel_path = module_path.relative_to(Path("src_new"))
    import_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
    
    # Compact prompt
    prompt = f"""Generate Python tests for this module.

```python
{content}
```

Requirements:
1. Add: import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_new"))
2. Import from: {import_path}
3. Test all public functions/classes
4. Include edge cases
5. Use pytest

Generate ONLY code:"""
    
    try:
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4000
            )
        )
        
        if response.text:
            test_code = response.text.strip()
            
            # Clean markdown
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0]
            elif "```" in test_code:
                test_code = test_code.split("```")[1].split("```")[0]
            
            # Save test
            output_dir = Path("tests_new/fast_generated")
            output_dir.mkdir(parents=True, exist_ok=True)
            test_path = output_dir / f"test_{module_path.stem}.py"
            test_path.write_text(test_code.strip(), encoding='utf-8')
            
            test_count = test_code.count("def test_")
            print(f"[{test_count} tests]")
            return True
            
    except Exception as e:
        print(f"[ERROR: {str(e)[:30]}]")
        return False
    
    return False


def main():
    """Generate tests for critical modules."""
    
    print("="*60)
    print("FAST TEST CONVERTER - Path to High Coverage")
    print("="*60)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found")
        return 1
    
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Define critical modules to test first
    critical_modules = [
        "core/application.py",
        "core/domain.py",
        "core/container.py",
        "bootstrap.py",
        "interfaces/core.py",
        "interfaces/providers.py",
        "providers/enhanced_llm_providers.py",
        "monitoring/unified_monitor.py",
        "analytics/specialized_tools.py",
        "config/config_validator.py",
    ]
    
    print(f"\nProcessing {len(critical_modules)} critical modules:\n")
    
    success_count = 0
    for module_path in critical_modules:
        full_path = Path("src_new") / module_path
        if full_path.exists():
            if generate_test_for_module(full_path, client):
                success_count += 1
        else:
            print(f"  {module_path}... [NOT FOUND]")
    
    print(f"\n{'='*60}")
    print(f"Generated tests for {success_count}/{len(critical_modules)} modules")
    
    # Quick coverage check
    print("\nTo measure coverage, run:")
    print("python -m pytest tests_new/fast_generated --cov=src_new --cov-report=term")
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())