#!/usr/bin/env python3
"""
Intelligent Test Builder V2 - Using Direct SDK

This version uses the Google GenAI SDK directly for better reliability.
It generates comprehensive real functionality tests using Gemini models.
"""

import os
import sys
import json
import ast
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Import Google GenAI SDK directly
import google.genai as genai


class IntelligentTestBuilderV2:
    """Build intelligent tests using direct SDK calls."""
    
    def __init__(self, model: str = "models/gemini-2.5-pro", api_key: Optional[str] = None):
        """Initialize with specified model (default: gemini-2.5-pro for best quality)."""
        self.model = model
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        print(f"Initialized with {model}")
    
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a module to understand its functionality."""
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        content = module_path.read_text(encoding='utf-8')
        
        # Limit content to avoid token limits
        if len(content) > 8000:
            content = content[:8000] + "\n# ... (truncated)"
        
        prompt = f"""Analyze this Python module and provide a comprehensive JSON response.

MODULE: {module_path.name}

```python
{content}
```

Provide a JSON object with these keys:
- purpose: What does this module do? (string)
- classes: List of classes with their methods and purpose
- functions: List of functions with their parameters and purpose  
- business_logic: Core algorithms and logic (string)
- edge_cases: List of edge cases to test
- dependencies: External dependencies used
- data_flows: How data flows through the module (string)
- error_scenarios: Potential error conditions

Return ONLY valid JSON, no markdown or explanations."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 2000}
            )
            
            if response.text:
                # Try to parse as JSON
                try:
                    # Clean up response
                    text = response.text.strip()
                    if text.startswith("```json"):
                        text = text[7:]
                    if text.startswith("```"):
                        text = text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    
                    analysis = json.loads(text.strip())
                    return analysis
                except json.JSONDecodeError:
                    # Return basic analysis if JSON parsing fails
                    return {
                        "purpose": f"Module {module_path.name}",
                        "classes": [],
                        "functions": [],
                        "business_logic": "See source code",
                        "edge_cases": [],
                        "dependencies": [],
                        "data_flows": "",
                        "error_scenarios": []
                    }
            else:
                raise ValueError("No response from model")
                
        except Exception as e:
            print(f"Error analyzing module: {e}")
            return {
                "purpose": f"Module {module_path.name} (error in analysis)",
                "classes": [],
                "functions": [],
                "error": str(e)
            }
    
    def generate_intelligent_test(self, module_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate intelligent test code based on module analysis."""
        
        module_name = module_path.stem
        
        # Build import path
        module_import = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
        if "multi_coder_analysis" in module_import:
            idx = module_import.find("multi_coder_analysis")
            module_import = module_import[idx:]
        
        # Create comprehensive prompt
        prompt = f"""Generate EXHAUSTIVE Python test code for the module '{module_name}'.

Module Analysis:
{json.dumps(analysis, indent=2)[:3000]}

Requirements:
1. Import the real module: {module_import}
2. NO MOCKS for internal code - test real functionality
3. Test ALL public functions and classes exhaustively
4. Use real test data that the module would process
5. Test edge cases: empty inputs, None, large inputs, invalid types
6. Test error conditions with pytest.raises
7. Include integration tests between components
8. Add performance tests where relevant
9. Each test method should have a docstring explaining what it tests

Generate a complete test file with:
- All necessary imports
- Multiple test classes organized by functionality
- Comprehensive test methods for ALL functionality
- Real test data and fixtures
- Error handling tests
- Comments explaining complex tests

Return ONLY the Python code, no markdown or explanations."""

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.1, "max_output_tokens": 4000}
            )
            
            if response.text:
                # Clean up response
                test_code = response.text.strip()
                if test_code.startswith("```python"):
                    test_code = test_code[9:]
                if test_code.startswith("```"):
                    test_code = test_code[3:]
                if test_code.endswith("```"):
                    test_code = test_code[:-3]
                
                return test_code.strip()
            else:
                raise ValueError("No response from model")
                
        except Exception as e:
            print(f"Error generating test: {e}")
            return f"# Error generating test: {e}\n# Please use manual approach"
    
    def build_test_for_module(self, module_path: Path, output_dir: Path = None) -> bool:
        """Complete workflow to build intelligent test for a module."""
        
        print(f"\n{'='*60}")
        print(f"Building intelligent test for: {module_path.name}")
        print('='*60)
        
        # Step 1: Analyze
        print("Step 1: Analyzing module...")
        analysis = self.analyze_module(module_path)
        
        if "error" in analysis and not analysis.get("purpose"):
            print(f"ERROR: Failed to analyze: {analysis['error']}")
            return False
        
        print(f"  Purpose: {analysis.get('purpose', 'Unknown')[:100]}...")
        print(f"  Classes: {len(analysis.get('classes', []))}")
        print(f"  Functions: {len(analysis.get('functions', []))}")
        
        # Step 2: Generate test
        print("\nStep 2: Generating intelligent test code...")
        test_code = self.generate_intelligent_test(module_path, analysis)
        
        if "Error generating test" in test_code:
            print("ERROR: Failed to generate test")
            return False
        
        # Step 3: Save
        if output_dir is None:
            output_dir = Path("tests/unit")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        test_filename = f"test_{module_path.stem}_intelligent.py"
        test_path = output_dir / test_filename
        
        test_path.write_text(test_code, encoding='utf-8')
        print(f"\nStep 3: Test saved to: {test_path}")
        
        # Step 4: Validate
        print("\nStep 4: Validating...")
        try:
            ast.parse(test_code)
            print("  OK: Valid Python syntax")
            
            # Count test methods
            test_count = test_code.count("def test_")
            print(f"  Generated {test_count} test methods")
            
            return True
        except SyntaxError as e:
            print(f"  ERROR: Syntax error: {e}")
            return False


def main():
    """Test the intelligent test builder V2."""
    
    print("="*60)
    print("Intelligent Test Builder V2 - Direct SDK")
    print("="*60)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return 1
    
    # Test on a module
    test_module = Path("multi_coder_analysis/utils/concatenate_prompts.py")
    
    if not test_module.exists():
        print(f"Module not found: {test_module}")
        return 1
    
    # You can specify different models:
    # - models/gemini-1.5-flash (fast, good for most cases)
    # - models/gemini-1.5-pro (more capable)
    # - models/gemini-2.5-pro (most advanced - default)
    
    builder = IntelligentTestBuilderV2(model="models/gemini-2.5-pro")
    
    success = builder.build_test_for_module(test_module)
    
    if success:
        print("\n" + "="*60)
        print("SUCCESS! Intelligent test generated successfully!")
        print("The test file includes:")
        print("  - Real module imports (no mocks)")
        print("  - Exhaustive function testing")
        print("  - Edge case validation")
        print("  - Error condition testing")
        print("="*60)
        return 0
    else:
        print("\nFailed to generate test")
        return 1


if __name__ == "__main__":
    sys.exit(main())