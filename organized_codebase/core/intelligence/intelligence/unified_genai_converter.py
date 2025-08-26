#!/usr/bin/env python3
"""
Unified GenAI Converter - Consolidated from intelligent_converter.py + fast_converter.py
====================================================================================

Combines the best features from both converters with mode selection:
- INTELLIGENT MODE: Comprehensive AST analysis, coverage tracking, detailed prompts
- FAST MODE: Quick critical modules processing, streamlined approach

CONSOLIDATION LOG: 2025-08-21
- Merged intelligent_converter.py (335 lines) + fast_converter.py (136 lines)
- Enhanced with unified architecture supporting both operation modes
- Zero functionality loss with feature enhancement
- Archived original files in archives/redundancy_analysis/genai_converters/
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

class UnifiedGenAIConverter:
    """Unified test converter supporting both intelligent and fast modes."""
    
    # Mode configurations
    MODES = {
        "intelligent": {
            "analysis": True,
            "batch_size": 10,
            "max_tokens": 8000,
            "content_limit": 10000,
            "coverage_tracking": True,
            "detailed_prompts": True,
            "output_dir": "tests_new/intelligent_converted",
            "target_coverage": 100.0
        },
        "fast": {
            "analysis": False,
            "batch_size": None,
            "max_tokens": 4000,
            "content_limit": 5000,
            "coverage_tracking": False,
            "detailed_prompts": False,
            "output_dir": "tests_new/fast_generated",
            "critical_modules": [
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
        },
        "healing": {
            "analysis": True,
            "batch_size": 20,
            "max_tokens": 6000,
            "content_limit": 4000,
            "coverage_tracking": False,
            "detailed_prompts": True,
            "output_dir": "tests_new/healing_generated",
            "max_healing_iterations": 5,
            "ast_validation": True,
            "error_recovery": True,
            "max_workers": 3  # Lower for healing to account for retries
        },
        "targeted": {
            "analysis": False,
            "batch_size": None,
            "max_tokens": 3000,
            "content_limit": 3000,
            "coverage_tracking": False,
            "detailed_prompts": False,
            "output_dir": "tests_new/targeted_generated",
            "target_modules": [
                ("async_generator_fixed", "src_new/pipeline/core/core/async_generator_fixed.py"),
                ("automated_test_generation", "src_new/testing/automated_test_generation.py"),
                ("comprehensive_test_framework", "src_new/testing/comprehensive_test_framework.py"),
                ("integration_test_matrix", "src_new/testing/integration_test_matrix.py"),
                ("prompt_ab_test", "src_new/pipeline/core/utils/prompt_ab_test.py")
            ]
        }
    }
    
    def __init__(self, mode: str = "intelligent"):
        """Initialize with specified mode configuration."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Available: {list(self.MODES.keys())}")
        
        self.mode = mode
        self.config = self.MODES[mode]
        
        # Initialize API
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model = 'gemini-2.0-flash-exp'
        self.src_dir = Path("src_new")
        self.test_dir = Path(self.config["output_dir"])
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initialized UnifiedGenAIConverter in {mode.upper()} mode")
        
    def analyze_module(self, module_path: Path) -> Dict:
        """Analyze module structure (intelligent mode only)."""
        if not self.config["analysis"]:
            return {"path": str(module_path), "name": module_path.stem, "has_tests_needed": True}
        
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
    
    def generate_test(self, module_path: Path, analysis: Optional[Dict] = None) -> Optional[str]:
        """Generate test using appropriate prompt for mode."""
        
        # Read module content
        content = module_path.read_text(encoding='utf-8')
        if len(content) > self.config["content_limit"]:
            content = content[:self.config["content_limit"]] + "\n# ... (truncated)"
        
        # Build import path
        rel_path = module_path.relative_to(self.src_dir)
        import_path = str(rel_path.with_suffix('')).replace(os.sep, '.')
        
        # Generate prompt based on mode
        if self.config["detailed_prompts"] and analysis:
            prompt = self._generate_detailed_prompt(module_path, content, import_path, analysis)
        else:
            prompt = self._generate_fast_prompt(module_path, content, import_path)
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=self.config["max_tokens"]
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
                
                # Validate syntax (intelligent mode only)
                if self.config["analysis"]:
                    try:
                        ast.parse(test_code)
                        return test_code.strip()
                    except SyntaxError:
                        # Try to fix common issues
                        test_code = test_code.replace("...", "pass")
                        return test_code.strip()
                else:
                    return test_code.strip()
            
        except Exception as e:
            print(f"  ERROR generating test: {str(e)[:100]}")
            return None
    
    def _generate_detailed_prompt(self, module_path: Path, content: str, import_path: str, analysis: Dict) -> str:
        """Generate detailed prompt for intelligent mode."""
        return f"""Generate EXHAUSTIVE Python test code for 100% coverage.

MODULE: {module_path.name}
IMPORT PATH: {import_path}

MODULE CODE:
```python
{content}
```

ANALYSIS:
- Classes: {len(analysis.get('classes', []))} ({', '.join(c['name'] for c in analysis.get('classes', []))})
- Functions: {len(analysis.get('functions', []))} ({', '.join(f['name'] for f in analysis.get('functions', []))})
- Lines: {analysis.get('lines', 0)}

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
    
    def _generate_fast_prompt(self, module_path: Path, content: str, import_path: str) -> str:
        """Generate compact prompt for fast mode."""
        return f"""Generate Python tests for this module.

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
    
    def fix_syntax_error(self, test_code: str, error_msg: str, iteration: int) -> Optional[str]:
        """Use LLM to fix syntax errors in generated code (healing mode only)."""
        if not self.config.get("error_recovery", False):
            return None
            
        print(f"    Attempting to fix syntax error (iteration {iteration}/{self.config['max_healing_iterations']})...")
        
        fix_prompt = f"""Fix the syntax error in this Python test code.

ERROR MESSAGE:
{error_msg}

CODE WITH ERROR:
```python
{test_code}
```

Requirements:
1. Fix the syntax error described in the error message
2. Return ONLY the fixed Python code
3. Ensure proper indentation
4. Complete any unterminated strings or brackets
5. Fix any invalid syntax

Output the complete fixed Python code."""
        
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=fix_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=self.config["max_tokens"]
                )
            )
            
            if response.text:
                fixed_code = response.text.strip()
                
                # Clean markdown
                if "```python" in fixed_code:
                    fixed_code = fixed_code.split("```python")[1].split("```")[0]
                elif "```" in fixed_code:
                    fixed_code = fixed_code.split("```")[1].split("```")[0]
                
                return fixed_code
                
        except Exception as e:
            print(f"    Error fixing syntax: {e}")
            return None
    
    def generate_test_with_healing(self, module_path: Path, analysis: Optional[Dict] = None) -> Optional[str]:
        """Generate test with automatic syntax error fixing (healing mode only)."""
        if not self.config.get("error_recovery", False):
            return self.generate_test(module_path, analysis)
        
        # Generate initial test
        test_code = self.generate_test(module_path, analysis)
        if not test_code:
            return None
        
        # Try to validate and fix syntax iteratively
        max_iterations = self.config.get("max_healing_iterations", 5)
        
        for iteration in range(1, max_iterations + 1):
            try:
                if self.config.get("ast_validation", True):
                    ast.parse(test_code)
                # Syntax is valid
                if iteration > 1:
                    print(f"    Fixed after {iteration-1} iterations")
                return test_code
                
            except SyntaxError as e:
                if iteration < max_iterations:
                    # Get detailed error info
                    error_msg = f"SyntaxError: {e.msg} at line {e.lineno}"
                    if e.text:
                        error_msg += f"\nProblem line: {e.text}"
                    
                    # Try to fix the error
                    fixed_code = self.fix_syntax_error(test_code, error_msg, iteration)
                    
                    if fixed_code:
                        test_code = fixed_code
                    else:
                        # Couldn't fix, give up
                        return None
                else:
                    # Max iterations reached
                    print(f"    Could not fix syntax after {max_iterations} attempts")
                    return None
        
        return test_code
    
    def process_module(self, module_path: Path) -> bool:
        """Process a single module."""
        print(f"  {module_path.name}...", end=" ")
        
        # Analyze (intelligent mode only)
        analysis = None
        if self.config["analysis"]:
            analysis = self.analyze_module(module_path)
            
            if "error" in analysis:
                print(f"[SKIP: {analysis['error']}]")
                return False
            
            if not analysis['has_tests_needed']:
                print("[SKIP: No testable code]")
                return False
        
        # Generate test (with healing if enabled)
        if self.config.get("error_recovery", False):
            test_code = self.generate_test_with_healing(module_path, analysis)
        else:
            test_code = self.generate_test(module_path, analysis)
        
        if not test_code:
            print("[FAILED]")
            return False
        
        # Write test
        test_name = f"test_{module_path.stem}.py"
        test_path = self.test_dir / test_name
        test_path.write_text(test_code, encoding='utf-8')
        
        test_count = test_code.count("def test_")
        print(f"[{test_count} tests]")
        
        return True
    
    def get_modules_to_process(self) -> List[Path]:
        """Get modules to process based on mode."""
        if self.mode == "targeted" and "target_modules" in self.config:
            # Targeted mode: use specific target module list
            modules = []
            for module_name, module_path in self.config["target_modules"]:
                full_path = Path(module_path)
                if full_path.exists():
                    modules.append(full_path)
                    print(f"  Targeting: {module_name} -> {module_path}")
                else:
                    print(f"  {module_name} ({module_path})... [NOT FOUND]")
            return modules
        elif self.mode == "fast" and "critical_modules" in self.config:
            # Fast mode: use critical module list
            modules = []
            for module_path in self.config["critical_modules"]:
                full_path = self.src_dir / module_path
                if full_path.exists():
                    modules.append(full_path)
                else:
                    print(f"  {module_path}... [NOT FOUND]")
            return modules
        else:
            # Intelligent/healing mode: discover all modules
            all_modules = list(self.src_dir.rglob("*.py"))
            return [
                m for m in all_modules 
                if "__pycache__" not in str(m) 
                and "__init__" not in m.name
            ]
    
    def measure_coverage(self) -> float:
        """Measure test coverage (intelligent mode only)."""
        if not self.config["coverage_tracking"]:
            return 0.0
        
        print("\nMeasuring coverage...")
        
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', 
                 str(self.test_dir),
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
    
    def run_conversion(self):
        """Run the conversion process."""
        print("="*70)
        print(f"UNIFIED GENAI CONVERTER - {self.mode.upper()} MODE")
        print("="*70)
        
        # Get modules to process
        modules = self.get_modules_to_process()
        print(f"\nFound {len(modules)} modules to process")
        
        if not modules:
            print("No modules to process!")
            return 0
        
        # Process modules
        start_time = datetime.now()
        success_count = 0
        
        if self.config["batch_size"]:
            # Batch processing (intelligent mode)
            for i in range(0, len(modules), self.config["batch_size"]):
                batch = modules[i:i+self.config["batch_size"]]
                
                print(f"\n{'='*50}")
                print(f"BATCH {i//self.config['batch_size'] + 1} ({len(batch)} modules)")
                print('='*50)
                
                for module in batch:
                    if self.process_module(module):
                        success_count += 1
                
                # Measure coverage
                if self.config["coverage_tracking"] and success_count > 0:
                    coverage = self.measure_coverage()
                    print(f"\nCurrent coverage: {coverage:.1f}%")
                    
                    if coverage >= self.config["target_coverage"]:
                        print(f"\n*** ACHIEVED {self.config['target_coverage']}% COVERAGE! ***")
                        break
        else:
            # Sequential processing (fast mode)
            print()
            for module in modules:
                if self.process_module(module):
                    success_count += 1
        
        # Final results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"\n{'='*70}")
        print(f"CONVERSION COMPLETE - {self.mode.upper()} MODE")
        print("="*70)
        print(f"Duration: {duration:.1f} seconds")
        print(f"Processed: {success_count}/{len(modules)} modules")
        
        if self.config["coverage_tracking"]:
            final_coverage = self.measure_coverage()
            print(f"Final coverage: {final_coverage:.1f}%")
            return 1 if final_coverage >= self.config["target_coverage"] else 0
        else:
            print("\nTo measure coverage, run:")
            print(f"python -m pytest {self.test_dir} --cov=src_new --cov-report=term")
            return 0


def main():
    """Main entry point with mode selection."""
    
    # Parse command line arguments
    mode = "intelligent"  # default
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in UnifiedGenAIConverter.MODES:
            print(f"Error: Unknown mode '{mode}'")
            print(f"Available modes: {list(UnifiedGenAIConverter.MODES.keys())}")
            return 1
    
    try:
        converter = UnifiedGenAIConverter(mode)
        return converter.run_conversion()
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())