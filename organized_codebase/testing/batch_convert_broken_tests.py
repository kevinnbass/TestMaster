#!/usr/bin/env python3
"""
Batch Convert Broken Tests

Converts multiple broken test files to real tests in batch.
Perfect for the 204 broken tests identified in our analysis.
"""

import sys
from pathlib import Path
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.convert_broken_test import BrokenTestConverter
from scripts.simple_mock_analyzer import categorize_test

def get_broken_test_files() -> List[Path]:
    """Get all test files categorized as 'broken' (no real imports)."""
    
    tests_dir = project_root / "tests" / "unit"
    test_files = [f for f in tests_dir.glob("*.py") if f.name != "__init__.py"]
    
    broken_files = []
    
    print("Scanning for broken test files...")
    for test_file in test_files:
        analysis = categorize_test(test_file)
        if analysis['category'] == 'broken':
            broken_files.append(test_file)
    
    return broken_files

def batch_convert(test_files: List[Path], max_count: int = None) -> Dict:
    """Convert multiple test files in batch."""
    
    converter = BrokenTestConverter()
    results = {
        'converted_real': [],
        'converted_todo': [],
        'errors': [],
        'skipped': []
    }
    
    files_to_process = test_files[:max_count] if max_count else test_files
    
    print(f"Converting {len(files_to_process)} broken test files...")
    print("=" * 60)
    
    for i, test_file in enumerate(files_to_process, 1):
        print(f"[{i}/{len(files_to_process)}] Converting {test_file.name}...")
        
        try:
            result = converter.convert_broken_test(test_file, backup=True)
            status = result['status']
            
            if status == 'converted_real':
                results['converted_real'].append(result)
                print(f"  OK: REAL: {result['target_module']}")
            elif status == 'converted_todo':
                results['converted_todo'].append(result)
                print(f"  ??: TODO: {result['target_module']} (not importable)")
            else:
                results['errors'].append(result)
                print(f"  XX: ERROR: {result['message']}")
                
        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e),
                'file': test_file.name
            }
            results['errors'].append(error_result)
            print(f"  XX: EXCEPTION: {e}")
    
    return results

def print_summary(results: Dict):
    """Print conversion summary."""
    
    total = len(results['converted_real']) + len(results['converted_todo']) + len(results['errors'])
    
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"Total processed: {total}")
    print(f"Converted to REAL tests: {len(results['converted_real'])}")
    print(f"Converted to TODO tests: {len(results['converted_todo'])}")
    print(f"Errors: {len(results['errors'])}")
    
    if results['converted_real']:
        print(f"\nOK: REAL TESTS ({len(results['converted_real'])}):")
        for result in results['converted_real'][:10]:  # Show first 10
            print(f"  - {result['file']} -> {result['target_module']}")
        if len(results['converted_real']) > 10:
            print(f"  ... and {len(results['converted_real']) - 10} more")
    
    if results['converted_todo']:
        print(f"\n??: TODO TESTS ({len(results['converted_todo'])}):")
        for result in results['converted_todo'][:10]:  # Show first 10
            print(f"  - {result['file']} -> {result['target_module']} (not importable)")
        if len(results['converted_todo']) > 10:
            print(f"  ... and {len(results['converted_todo']) - 10} more")
    
    if results['errors']:
        print(f"\nXX: ERRORS ({len(results['errors'])}):")
        for result in results['errors'][:5]:  # Show first 5
            print(f"  - {result['file']}: {result['message']}")
        if len(results['errors']) > 5:
            print(f"  ... and {len(results['errors']) - 5} more")

def validate_conversions(results: Dict):
    """Run pytest on converted files to validate they work."""
    
    print("\n" + "=" * 60)
    print("VALIDATION: Testing converted files...")
    print("=" * 60)
    
    real_files = [r['file'] for r in results['converted_real']]
    
    if not real_files:
        print("No real test files to validate.")
        return
    
    # Test first 5 converted files
    test_files = real_files[:5]
    
    import subprocess
    
    for test_file in test_files:
        test_path = f"tests/unit/{test_file}"
        print(f"Testing {test_file}...")
        
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pytest', test_path, '-v', '--tb=short'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  OK: PASSED")
            else:
                print(f"  XX: FAILED")
                print(f"    {result.stdout.split('FAILED')[0] if 'FAILED' in result.stdout else 'Unknown error'}")
                
        except subprocess.TimeoutExpired:
            print(f"  TT: TIMEOUT")
        except Exception as e:
            print(f"  XX: ERROR: {e}")

def main():
    """Main execution function."""
    
    # Get command line arguments
    if len(sys.argv) > 1:
        try:
            max_count = int(sys.argv[1])
        except ValueError:
            print("Usage: python batch_convert_broken_tests.py [max_count]")
            print("Example: python batch_convert_broken_tests.py 10")
            return
    else:
        max_count = 10  # Default pilot size
    
    print(f"PILOT CONVERSION: Converting up to {max_count} broken tests")
    print("=" * 60)
    
    # Get broken test files
    broken_files = get_broken_test_files()
    print(f"Found {len(broken_files)} broken test files total")
    print(f"Will convert {min(max_count, len(broken_files))} as pilot")
    print()
    
    # Convert files
    results = batch_convert(broken_files, max_count)
    
    # Print summary
    print_summary(results)
    
    # Validate conversions
    if results['converted_real']:
        validate_conversions(results)
    
    print(f"\nPilot conversion complete! Converted {len(results['converted_real']) + len(results['converted_todo'])} files")

if __name__ == "__main__":
    main()