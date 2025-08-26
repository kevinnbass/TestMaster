#!/usr/bin/env python3
"""
Run Limited Coverage Test Generation
=====================================

Test the coverage script with a small limit first.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from achieve_100_percent_coverage import Gemini25ProTestGenerator

async def main():
    """Run limited test generation."""
    print("Starting LIMITED test generation (5 modules only)...")
    print("This is a test run to verify everything works.")
    print("=" * 70)
    
    # Initialize generator
    generator = Gemini25ProTestGenerator()
    
    # Generate tests for only 5 modules first
    generated = await generator.generate_all_tests(limit=5)
    
    if generated > 0:
        print(f"\n{generated} test files generated successfully!")
        print("\nNow measuring coverage improvement...")
        coverage = generator.measure_final_coverage()
        
        print(f"\nCoverage after limited generation: {coverage:.2f}%")
        
        if coverage > 13.33:
            print("SUCCESS: Coverage improved!")
            print("\nYou can now run the full generation:")
            print("  python scripts/test_coverage/achieve_100_percent_coverage.py")
        else:
            print("WARNING: Coverage did not improve. Check generated tests.")
    else:
        print("ERROR: No tests generated. Check configuration.")
    
    return 0

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))