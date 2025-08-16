#!/usr/bin/env python3
"""
Intelligent Test Converter for 100% Coverage
============================================

Adapted from tot_branch_minimal approach to achieve complete test coverage
for the regex_gen codebase using Google GenAI SDK.
"""

import os
import sys
import ast
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from dotenv import load_dotenv
import subprocess

load_dotenv()

class IntelligentTestConverter:
    """Convert and generate tests for 100% coverage using GenAI SDK."""
    
    def __init__(self):
        """Initialize with Google API key."""
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = 'gemini-2.0-flash-exp'  # Fast and effective
        self.src_dir = Path("src_new")
        self.test_dir = Path("tests_new/intelligent_converted")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_module(self, module_path: Path) -> Dict:
        """Analyze a module to understand its structure."""
        content = module_path.read_text(encoding='utf-8')
        
        # Parse with AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {"error": "Syntax error", "path": str(module_path)}
        
        # Extract key information
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                classes.append({
                    "name": node.name,
                    "methods": methods,
                    "is_abstract": any(
                        isinstance(base, ast.Name) and base.id == "ABC" 
                        for base in node.bases
                    )
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                functions.append({
                    "name": node.name,
                    "is_async": False,
                    "params": [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, ast.AsyncFunctionDef) and node.col_offset == 0:
                functions.append({
                    "name": node.name,
                    "is_async": True,
                    "params": [arg.arg for arg in node.args.args]
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    imports.append(node.module or "")
        
        return {
            "path": str(module_path),
            "name": module_path.stem,
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "lines": len(content.splitlines()),
            "has_tests_needed": len(classes) > 0 or len(functions) > 0
        }
    
    def generate_comprehensive_test(self, module_path: Path, analysis: Dict) -> Optional[str]:
        """Generate comprehensive test using Gemini API."""
        
        # Read module content
        content = module_path.read_text(encoding='utf-8')
        if len(content) > 10000:
            content = content[:10000] + "\n# ... (truncated)"
        
        # Build import path
        rel_path = module_path.relative_to(self.src_dir)
        import_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        # Create prompt
        prompt = f"""Generate EXHAUSTIVE Python test code for 100% coverage.

MODULE: {module_path.name}
IMPORT PATH: {import_path}

MODULE CODE:
```python
{content}
```

ANALYSIS:
- Classes: {len(analysis['classes'])} ({', '.join(c['name'] for c in analysis['classes'])})
- Functions: {len(analysis['functions'])} ({', '.join(f['name'] for f in analysis['functions'])})
- Lines: {analysis['lines']}

REQUIREMENTS:
1. Add this at the top to fix imports:
   ```python
   import sys
   from pathlib import Path
   sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src_new"))
   ```
2. Import from: {import_path}
3. Test EVERY public class, method, and function
4. Test ALL branches and conditions for 100% coverage
5. Include edge cases: None, empty, invalid inputs, exceptions
6. Test async functions with @pytest.mark.asyncio
7. Mock ONLY external dependencies (not internal code)
8. Each test must have a clear docstring
9. Include at least 3 tests per function/method
10. Test error paths with pytest.raises

Generate ONLY Python test code, no explanations."""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8000
                )
            )
            
            if response.text:
                test_code = response.text.strip()
                
                # Clean markdown
                if test_code.startswith("```python"):
                    test_code = test_code[9:]
                if test_code.startswith("```"):
                    test_code = test_code[3:]
                if test_code.endswith("```"):
                    test_code = test_code[:-3]
                
                # Validate syntax
                try:
                    ast.parse(test_code)
                    return test_code.strip()
                except SyntaxError:
                    # Try to fix common issues
                    test_code = test_code.replace("...", "pass")
                    return test_code.strip()
            
        except Exception as e:
            print(f"  ERROR generating test: {str(e)[:100]}")
            return None
    
    def process_module(self, module_path: Path) -> bool:
        """Process a single module."""
        print(f"\nProcessing: {module_path.name}")
        
        # Analyze
        print("  Analyzing...", end=" ")
        analysis = self.analyze_module(module_path)
        
        if "error" in analysis:
            print(f"[SKIP: {analysis['error']}]")
            return False
        
        if not analysis['has_tests_needed']:
            print("[SKIP: No testable code]")
            return False
        
        print(f"[{len(analysis['classes'])} classes, {len(analysis['functions'])} functions]")
        
        # Generate test
        print("  Generating...", end=" ")
        test_code = self.generate_comprehensive_test(module_path, analysis)
        
        if not test_code:
            print("[FAILED]")
            return False
        
        # Write test
        test_name = f"test_{module_path.stem}.py"
        test_path = self.test_dir / test_name
        test_path.write_text(test_code, encoding='utf-8')
        
        test_count = test_code.count("def test_")
        print(f"[OK: {test_count} tests]")
        
        return True
    
    def convert_batch(self, modules: List[Path]) -> Dict:
        """Convert a batch of modules."""
        results = {
            "success": [],
            "failed": [],
            "skipped": []
        }
        
        for module in modules:
            try:
                if self.process_module(module):
                    results["success"].append(str(module))
                else:
                    results["skipped"].append(str(module))
            except Exception as e:
                print(f"  ERROR: {str(e)[:100]}")
                results["failed"].append(str(module))
        
        return results
    
    def measure_coverage(self) -> float:
        """Measure current test coverage."""
        print("\nMeasuring coverage...")
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', 
                 'tests_new/intelligent_converted', 
                 '--cov=src_new', 
                 '--cov-report=json', 
                 '--tb=no', '-q'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse coverage
            coverage_file = Path('coverage.json')
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    data = json.load(f)
                    return data.get('totals', {}).get('percent_covered', 0)
        except Exception as e:
            print(f"  Coverage measurement failed: {e}")
        
        return 0
    
    def run_to_100_percent(self):
        """Run conversion until 100% coverage is achieved."""
        print("="*70)
        print("INTELLIGENT TEST CONVERTER FOR 100% COVERAGE")
        print("="*70)
        
        # Get all Python modules
        all_modules = list(self.src_dir.rglob("*.py"))
        all_modules = [
            m for m in all_modules 
            if "__pycache__" not in str(m) 
            and "__init__" not in m.name
        ]
        
        print(f"\nFound {len(all_modules)} modules to process")
        
        # Process in batches
        batch_size = 10
        total_processed = 0
        
        for i in range(0, len(all_modules), batch_size):
            batch = all_modules[i:i+batch_size]
            
            print(f"\n{'='*50}")
            print(f"BATCH {i//batch_size + 1} ({len(batch)} modules)")
            print('='*50)
            
            results = self.convert_batch(batch)
            total_processed += len(results["success"])
            
            print(f"\nBatch results:")
            print(f"  Success: {len(results['success'])}")
            print(f"  Skipped: {len(results['skipped'])}")
            print(f"  Failed: {len(results['failed'])}")
            
            # Measure coverage
            if total_processed > 0:
                coverage = self.measure_coverage()
                print(f"\nCurrent coverage: {coverage:.1f}%")
                
                if coverage >= 100:
                    print("\n" + "="*70)
                    print("*** ACHIEVED 100% COVERAGE! ***")
                    print("="*70)
                    return coverage
                elif coverage >= 90:
                    print("[EXCELLENT] 90%+ coverage!")
                elif coverage >= 80:
                    print("[GOOD] 80%+ coverage!")
        
        # Final measurement
        final_coverage = self.measure_coverage()
        print("\n" + "="*70)
        print(f"FINAL COVERAGE: {final_coverage:.1f}%")
        print("="*70)
        
        return final_coverage


def main():
    """Main entry point."""
    converter = IntelligentTestConverter()
    coverage = converter.run_to_100_percent()
    
    if coverage >= 100:
        print("\n🎉 SUCCESS: Achieved 100% test coverage!")
        return 0
    else:
        print(f"\n📊 Final coverage: {coverage:.1f}%")
        print("Run again to continue improving coverage.")
        return 1


if __name__ == "__main__":
    sys.exit(main())