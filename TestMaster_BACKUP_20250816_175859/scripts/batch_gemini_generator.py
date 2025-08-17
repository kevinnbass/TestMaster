#!/usr/bin/env python3
"""
Batch Gemini Test Generator for 100% Coverage
==============================================

Processes files in batches to achieve 100% coverage efficiently.
"""

import os
import sys
import time
from pathlib import Path
from typing import List
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the main generator
from scripts.test_coverage.gemini_test_generator import GeminiTestGenerator


def process_batch(generator: GeminiTestGenerator, files: List[Path], batch_num: int) -> int:
    """Process a batch of files."""
    print(f"\n{'='*60}")
    print(f"Processing Batch {batch_num} ({len(files)} files)")
    print('='*60)
    
    success_count = 0
    
    for i, py_file in enumerate(files, 1):
        try:
            print(f"[{i}/{len(files)}] {py_file.name}...", end=" ")
            
            # Analyze
            analysis = generator.analyze_module(py_file)
            
            if "error" in analysis:
                print("[SKIP]")
                continue
            
            # Generate test
            test_code = generator.generate_comprehensive_test(py_file, analysis)
            
            # Write test
            test_name = f"test_{py_file.stem}_gemini.py"
            test_path = Path(f"tests_new/gemini_generated/{test_name}")
            test_path.parent.mkdir(parents=True, exist_ok=True)
            test_path.write_text(test_code, encoding='utf-8')
            
            test_count = test_code.count("def test_")
            print(f"[OK: {test_count} tests]")
            success_count += 1
            
            # No delay needed with 1000 RPM
            # time.sleep(0.1)
            
        except Exception as e:
            print(f"[ERROR: {str(e)[:50]}]")
            continue
    
    return success_count


def main():
    """Process all files in batches."""
    print("="*60)
    print("Batch Gemini Test Generator for 100% Coverage")
    print("="*60)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found")
        return 1
    
    # Initialize generator
    try:
        generator = GeminiTestGenerator()
    except Exception as e:
        print(f"Failed to initialize: {e}")
        return 1
    
    # Get all Python files
    source_dir = Path("src_new")
    all_files = list(source_dir.rglob("*.py"))
    all_files = [f for f in all_files if "__pycache__" not in str(f) and "__init__" not in f.name]
    
    print(f"\nFound {len(all_files)} Python files to process")
    
    # Process in batches of 5 for more reliable processing
    batch_size = 5
    total_success = 0
    
    for batch_num, i in enumerate(range(0, len(all_files), batch_size), 1):
        batch = all_files[i:i+batch_size]
        success = process_batch(generator, batch, batch_num)
        total_success += success
        
        print(f"Batch {batch_num} complete: {success}/{len(batch)} successful")
        
        # Measure coverage periodically
        if batch_num % 5 == 0:
            print("\nMeasuring current coverage...")
            coverage = generator.measure_coverage()
            print(f"Current coverage: {coverage:.1f}%")
            
            if coverage >= 100:
                print("\n*** ACHIEVED 100% COVERAGE! ***")
                break
    
    # Final coverage measurement
    print("\n" + "="*60)
    print("Final Coverage Measurement")
    print("="*60)
    
    coverage = generator.measure_coverage()
    
    print(f"Total files processed: {total_success}/{len(all_files)}")
    print(f"Final Coverage: {coverage:.1f}%")
    
    if coverage >= 100:
        print("\n*** SUCCESS: ACHIEVED 100% COVERAGE! ***")
    elif coverage >= 90:
        print("\n[EXCELLENT] 90%+ coverage achieved!")
    elif coverage >= 80:
        print("\n[GOOD] 80%+ coverage achieved!")
    else:
        print(f"\n[PROGRESS] Coverage at {coverage:.1f}%")
    
    # Generate HTML report
    print("\nGenerating HTML coverage report...")
    subprocess.run(
        ['python', '-m', 'pytest', 'tests_new', 
         '--cov=src_new', '--cov-report=html', '--tb=no', '-q'],
        capture_output=True
    )
    print("HTML report available at: htmlcov/index.html")
    
    print("="*60)
    return 0


if __name__ == "__main__":
    sys.exit(main())