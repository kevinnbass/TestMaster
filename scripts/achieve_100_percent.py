#!/usr/bin/env python3
"""
Achieve 100% Test Coverage
==========================

Iteratively generates and runs tests until 100% coverage is achieved.
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class CoverageMaximizer:
    """Maximize test coverage to 100%."""
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        self.client = genai.Client(api_key=self.api_key)
        self.test_dir = Path("tests_new/coverage_100")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def get_uncovered_lines(self) -> Dict[str, List[int]]:
        """Get uncovered lines from coverage report."""
        
        print("Analyzing coverage gaps...")
        
        # Run coverage with JSON output
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new/gemini_generated', 
             '--cov=src_new', '--cov-report=json', '--tb=no', '-q'],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        uncovered = {}
        
        # Parse coverage.json
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                data = json.load(f)
                
                for file_path, file_data in data.get('files', {}).items():
                    if 'src_new' in file_path:
                        missing_lines = file_data.get('missing_lines', [])
                        if missing_lines:
                            uncovered[file_path] = missing_lines
        
        return uncovered
    
    def generate_targeted_test(self, module_path: str, missing_lines: List[int]) -> str:
        """Generate test specifically for uncovered lines."""
        
        # Read the module
        full_path = Path(module_path)
        if not full_path.exists():
            return None
            
        content = full_path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # Extract uncovered code
        uncovered_sections = []
        for line_num in missing_lines[:20]:  # Limit to first 20 lines
            if 0 < line_num <= len(lines):
                start = max(0, line_num - 3)
                end = min(len(lines), line_num + 3)
                section = '\n'.join(f"{i+1}: {lines[i]}" for i in range(start, end))
                uncovered_sections.append(f"Line {line_num}:\n{section}")
        
        # Build prompt
        prompt = f"""Generate Python tests to cover these SPECIFIC uncovered lines.

MODULE: {Path(module_path).name}
UNCOVERED LINES: {missing_lines[:20]}

UNCOVERED CODE SECTIONS:
{chr(10).join(uncovered_sections[:5])}

Requirements:
1. Add import fix: import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_new"))
2. Import from the correct module path
3. Write tests that SPECIFICALLY execute the uncovered lines
4. Focus on edge cases and error conditions
5. Use pytest

Generate ONLY test code that covers these specific lines:"""
        
        try:
            response = self.client.models.generate_content(
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
                
                return test_code.strip()
                
        except Exception as e:
            print(f"    Error generating test: {str(e)[:50]}")
            
        return None
    
    def measure_coverage(self) -> float:
        """Measure current coverage percentage."""
        
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', 
             '--cov=src_new', '--cov-report=json', '--tb=no', '-q'],
            capture_output=True,
            text=True,
            timeout=180
        )
        
        coverage_file = Path('coverage.json')
        if coverage_file.exists():
            with open(coverage_file, 'r') as f:
                data = json.load(f)
                return data.get('totals', {}).get('percent_covered', 0)
        
        return 0
    
    def achieve_100_percent(self, max_iterations: int = 10):
        """Keep generating tests until 100% coverage is achieved."""
        
        print("="*70)
        print("ACHIEVING 100% TEST COVERAGE")
        print("="*70)
        
        for iteration in range(1, max_iterations + 1):
            print(f"\n{'='*50}")
            print(f"ITERATION {iteration}")
            print('='*50)
            
            # Get current coverage
            current_coverage = self.measure_coverage()
            print(f"\nCurrent Coverage: {current_coverage:.2f}%")
            
            if current_coverage >= 100:
                print("\n[SUCCESS] ACHIEVED 100% COVERAGE!")
                return True
            
            # Get uncovered lines
            uncovered = self.get_uncovered_lines()
            
            if not uncovered:
                print("No uncovered lines found!")
                break
            
            print(f"Found {len(uncovered)} files with uncovered lines")
            
            # Generate tests for top 5 files with most uncovered lines
            sorted_files = sorted(uncovered.items(), key=lambda x: len(x[1]), reverse=True)[:5]
            
            for file_path, missing_lines in sorted_files:
                module_name = Path(file_path).stem
                print(f"\n  Generating test for {module_name} ({len(missing_lines)} uncovered lines)...")
                
                test_code = self.generate_targeted_test(file_path, missing_lines)
                
                if test_code:
                    # Save test
                    test_name = f"test_{module_name}_iter{iteration}.py"
                    test_path = self.test_dir / test_name
                    test_path.write_text(test_code, encoding='utf-8')
                    
                    test_count = test_code.count("def test_")
                    print(f"    Generated {test_count} targeted tests")
                else:
                    print(f"    Failed to generate test")
            
            # Measure new coverage
            new_coverage = self.measure_coverage()
            improvement = new_coverage - current_coverage
            
            print(f"\nNew Coverage: {new_coverage:.2f}% (+{improvement:.2f}%)")
            
            if improvement < 0.1:
                print("Coverage not improving, stopping...")
                break
        
        return False
    
    def fix_all_tests(self):
        """Fix any failing tests to ensure they all pass."""
        
        print("\nFixing failing tests...")
        
        # Run tests and capture failures
        result = subprocess.run(
            ['python', '-m', 'pytest', 'tests_new', '-v', '--tb=short'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Parse failures
        failures = []
        for line in result.stdout.split('\n'):
            if 'FAILED' in line:
                failures.append(line)
        
        if failures:
            print(f"Found {len(failures)} failing tests")
            # Here we could generate fixes for failing tests
        else:
            print("All tests passing!")
        
        return len(failures) == 0


def main():
    """Main entry point."""
    
    maximizer = CoverageMaximizer()
    
    # Try to achieve 100% coverage
    success = maximizer.achieve_100_percent(max_iterations=10)
    
    if success:
        print("\n" + "="*70)
        print("[SUCCESS] 100% TEST COVERAGE ACHIEVED!")
        print("="*70)
        
        # Ensure all tests pass
        if maximizer.fix_all_tests():
            print("\n[OK] All tests passing!")
        else:
            print("\n[WARNING] Some tests still failing - manual fixes needed")
    else:
        coverage = maximizer.measure_coverage()
        print(f"\n[STATS] Final coverage: {coverage:.2f}%")
        print("Additional manual work may be needed to reach 100%")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())