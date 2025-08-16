#!/usr/bin/env python3
"""
Gemini-Powered Test Generator
==============================

Uses Gemini 2.5 Pro to generate comprehensive tests for 100% coverage.
"""

import ast
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import asyncio

# Google GenAI SDK
from google import genai
from google.genai import types

@dataclass
class CodeAnalysis:
    """Analysis of code to test."""
    file_path: str
    classes: List[str]
    functions: List[str]
    async_functions: List[str]
    uncovered_lines: Set[int]
    imports: List[str]
    complexity: int

class GeminiTestGenerator:
    """Generate tests using Gemini AI."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key required. Set GOOGLE_API_KEY env var.")
        
        # Initialize Gemini client with 2.5 Pro
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-pro"  # Most advanced model for code generation
        
        self.src_dir = Path("src_new")
        self.test_dir = Path("tests_new")
        self.generated_count = 0
        
    def analyze_code_file(self, file_path: Path) -> CodeAnalysis:
        """Analyze a Python file to understand what needs testing."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
        except:
            return None
        
        analysis = CodeAnalysis(
            file_path=str(file_path),
            classes=[],
            functions=[],
            async_functions=[],
            uncovered_lines=set(),
            imports=[],
            complexity=0
        )
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                analysis.classes.append(node.name)
                analysis.complexity += 1
                
            elif isinstance(node, ast.FunctionDef):
                if isinstance(node, ast.AsyncFunctionDef):
                    analysis.async_functions.append(node.name)
                else:
                    analysis.functions.append(node.name)
                analysis.complexity += 1
                
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis.imports.append(alias.name)
                    
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    analysis.imports.append(node.module)
        
        return analysis
    
    def get_uncovered_lines(self, file_path: Path) -> Set[int]:
        """Get uncovered lines from coverage data."""
        coverage_file = Path('coverage.json')
        if not coverage_file.exists():
            return set()
        
        with open(coverage_file, 'r') as f:
            data = json.load(f)
        
        for covered_file, file_data in data.get('files', {}).items():
            if file_path.name in covered_file:
                return set(file_data.get('missing_lines', []))
        
        return set()
    
    async def generate_test_with_gemini(self, analysis: CodeAnalysis) -> str:
        """Use Gemini to generate comprehensive test code."""
        
        # Read the source code
        with open(analysis.file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Create a focused prompt for test generation
        prompt = f"""You are an expert Python test engineer. Generate comprehensive pytest tests to achieve 100% code coverage.

SOURCE CODE TO TEST:
```python
{source_code[:8000]}  # Limit to avoid token limits
```

CODE ANALYSIS:
- Classes: {', '.join(analysis.classes[:10])}
- Functions: {', '.join(analysis.functions[:10])}
- Async Functions: {', '.join(analysis.async_functions[:10])}
- Uncovered Lines: {sorted(list(analysis.uncovered_lines))[:50] if analysis.uncovered_lines else 'Unknown'}

REQUIREMENTS:
1. Generate pytest tests that cover ALL functions and classes
2. Test both success and failure paths
3. Use mocks for external dependencies
4. Include edge cases and error conditions
5. For async functions, use pytest.mark.asyncio
6. Mock any database, API, or file system calls
7. Ensure tests are isolated and fast
8. Include docstrings explaining what each test does
9. Focus on the uncovered lines if provided

Generate ONLY the Python test code. Start with imports and then test functions.
Use this structure:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src_new"))

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
# Other imports as needed

# Import the module being tested
from ... import ...

# Test classes and functions here
```

Focus on practical, working tests that will actually execute and increase coverage.
"""

        try:
            # Call Gemini API with proper configuration
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,  # Lower temperature for more focused code
                    top_p=0.95,
                    top_k=40,
                    max_output_tokens=8000,
                    candidate_count=1
                )
            )
            
            # Extract code from response
            if response and response.text:
                code = self.extract_code_from_response(response.text)
                return code
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            return None
        
        return None
    
    def extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from Gemini response."""
        # Look for code blocks
        lines = response_text.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip() == '```' and in_code_block:
                in_code_block = False
                continue
            elif in_code_block:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # If no code blocks found, assume entire response is code
        # but filter out obvious non-code lines
        filtered = []
        for line in lines:
            if not line.strip().startswith('#') or 'import' in line or 'def ' in line or 'class ' in line:
                if not any(skip in line.lower() for skip in ['explanation:', 'note:', 'here', 'this']):
                    filtered.append(line)
        
        return '\n'.join(filtered)
    
    async def generate_tests_for_module(self, module_path: Path) -> bool:
        """Generate tests for a single module."""
        print(f"Analyzing {module_path.name}...")
        
        # Analyze the module
        analysis = self.analyze_code_file(module_path)
        if not analysis:
            return False
        
        # Get uncovered lines
        analysis.uncovered_lines = self.get_uncovered_lines(module_path)
        
        # Skip if already well-covered
        if len(analysis.uncovered_lines) == 0 and len(analysis.classes) == 0:
            return False
        
        print(f"  Generating tests with Gemini 2.5 Pro...")
        
        # Generate test code with Gemini
        test_code = await self.generate_test_with_gemini(analysis)
        
        if test_code:
            # Save the test file
            test_file = self.test_dir / f"test_{module_path.stem}_gemini.py"
            
            # Add header
            final_code = f'''#!/usr/bin/env python3
"""
AI-Generated tests for {module_path.stem} module.
Generated by Gemini 2.5 Pro for 100% coverage.
"""

{test_code}
'''
            
            test_file.write_text(final_code, encoding='utf-8')
            print(f"  [OK] Generated {test_file.name}")
            self.generated_count += 1
            return True
        
        return False
    
    async def generate_all_tests(self, limit: int = 20):
        """Generate tests for modules needing coverage."""
        print("=" * 70)
        print("GEMINI-POWERED TEST GENERATION")
        print(f"Using model: {self.model_name}")
        print("=" * 70)
        
        # Find Python modules that need tests
        modules_to_test = []
        
        for py_file in self.src_dir.rglob("*.py"):
            if '__pycache__' in str(py_file) or '__init__' in py_file.name:
                continue
            
            # Check if test already exists
            test_name = f"test_{py_file.stem}_gemini.py"
            if not (self.test_dir / test_name).exists():
                modules_to_test.append(py_file)
        
        # Sort by priority (core modules first)
        priority_modules = ['application', 'domain', 'container', 'bootstrap']
        modules_to_test.sort(key=lambda x: (
            0 if x.stem in priority_modules else 1,
            x.stem
        ))
        
        print(f"\nFound {len(modules_to_test)} modules needing tests")
        print(f"Generating tests for up to {limit} modules...\n")
        
        # Generate tests with rate limiting
        for i, module_path in enumerate(modules_to_test[:limit]):
            if i > 0 and i % 5 == 0:
                print(f"\nPausing to avoid rate limits... ({i}/{limit})")
                await asyncio.sleep(2)  # Brief pause every 5 requests
            
            try:
                await self.generate_tests_for_module(module_path)
            except Exception as e:
                print(f"  Error generating test for {module_path.name}: {e}")
        
        print(f"\n{self.generated_count} test files generated with Gemini")
        return self.generated_count
    
    def validate_generated_tests(self):
        """Validate that generated tests are syntactically correct."""
        print("\n" + "=" * 70)
        print("VALIDATING GENERATED TESTS")
        print("=" * 70)
        
        valid_count = 0
        invalid_files = []
        
        for test_file in self.test_dir.glob("test_*_gemini.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                    ast.parse(code)
                valid_count += 1
            except SyntaxError as e:
                invalid_files.append((test_file.name, str(e)))
        
        print(f"\nValid tests: {valid_count}")
        if invalid_files:
            print(f"Invalid tests: {len(invalid_files)}")
            for name, error in invalid_files[:5]:
                print(f"  {name}: {error[:100]}")
        
        return valid_count, invalid_files


async def main():
    """Run Gemini-powered test generation."""
    
    # Check for API key
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        print("Please set: export GOOGLE_API_KEY='your-api-key'")
        return 1
    
    generator = GeminiTestGenerator(api_key)
    
    # Generate tests
    generated = await generator.generate_all_tests(limit=20)
    
    if generated > 0:
        # Validate generated tests
        valid, invalid = generator.validate_generated_tests()
        
        print("\n" + "=" * 70)
        print("GEMINI TEST GENERATION COMPLETE")
        print("=" * 70)
        print(f"\nGenerated: {generated} test files")
        print(f"Valid: {valid} test files")
        if invalid:
            print(f"Invalid: {len(invalid)} test files (may need manual fixes)")
        
        print("\nNext steps:")
        print("1. Run tests to measure coverage improvement")
        print("2. Fix any syntax errors in generated tests")
        print("3. Continue generation for remaining modules")
    else:
        print("\nNo tests generated. Check API key and connectivity.")
    
    return 0


if __name__ == "__main__":
    # Run async main
    sys.exit(asyncio.run(main()))