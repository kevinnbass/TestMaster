#!/usr/bin/env python3
"""
Quick Test Generator for Key Modules
====================================

Generate tests for the most important modules first.
"""

import os
import sys
from pathlib import Path
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.test_coverage.gemini_test_generator import GeminiTestGenerator


def main():
    """Generate tests for key modules quickly."""
    print("="*60)
    print("Quick Test Generator for Key Modules")
    print("="*60)
    
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found")
        return 1
    
    # Initialize generator
    try:
        generator = GeminiTestGenerator()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return 1
    
    # Define key modules to test first
    key_modules = [
        Path("src_new/core/application.py"),
        Path("src_new/core/interfaces.py"),
        Path("src_new/core/models.py"),
        Path("src_new/providers/base.py"),
        Path("src_new/providers/openrouter.py"),
    ]
    
    # Filter existing modules
    existing_modules = [m for m in key_modules if m.exists()]
    print(f"Found {len(existing_modules)} key modules to test")
    
    success_count = 0
    
    for i, module in enumerate(existing_modules, 1):
        print(f"\n[{i}/{len(existing_modules)}] Processing {module.name}...")
        
        try:
            # Analyze
            print("  Analyzing with Gemini...", end=" ")
            analysis = generator.analyze_module(module)
            
            if "error" in analysis:
                print(f"[SKIP: {analysis['error']}]")
                continue
            
            print("[OK]")
            
            # Generate test
            print("  Generating tests...", end=" ")
            test_code = generator.generate_comprehensive_test(module, analysis)
            
            # Write test
            test_name = f"test_{module.stem}_gemini.py"
            test_path = Path(f"tests_new/gemini_generated/{test_name}")
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text(test_code, encoding='utf-8')
            
            test_count = test_code.count("def test_")
            print(f"[OK: {test_count} tests]")
            success_count += 1
            
        except Exception as e:
            print(f"[ERROR: {str(e)[:50]}]")
            continue
    
    print(f"\n{'='*60}")
    print(f"Generated tests for {success_count}/{len(existing_modules)} modules")
    
    # Measure coverage
    print("\nMeasuring coverage...")
    coverage = generator.measure_coverage()
    
    print(f"Current Coverage: {coverage:.1f}%")
    
    if coverage >= 100:
        print("*** ACHIEVED 100% COVERAGE! ***")
    elif coverage >= 90:
        print("[EXCELLENT] 90%+ coverage achieved!")
    elif coverage >= 80:
        print("[GOOD] 80%+ coverage achieved!")
    else:
        print(f"[PROGRESS] Coverage at {coverage:.1f}%")
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())