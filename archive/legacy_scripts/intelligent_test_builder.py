#!/usr/bin/env python3
"""
Intelligent Test Builder using Gemini-2.5-pro

This script uses Gemini-2.5-pro to intelligently analyze modules and generate
comprehensive real functionality tests (not mocks).
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("Note: python-dotenv not installed, using system environment variables")

# Import the Gemini provider from the codebase
from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider


class IntelligentTestBuilder:
    """Build intelligent, exhaustive tests using Gemini models."""
    
    def __init__(self, model: str = None, api_key: Optional[str] = None):
        """Initialize with best available Gemini model."""
        # Try models in order of preference
        if model is None:
            models_to_try = [
                "models/gemini-2.5-pro",
                "models/gemini-2.0-flash", 
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro",
            ]
            print("Auto-detecting best available Gemini model...")
            for test_model in models_to_try:
                try:
                    print(f"  Trying {test_model}...")
                    provider = GeminiProvider(
                        model=test_model,
                        api_key=api_key or os.getenv("GOOGLE_API_KEY")
                    )
                    # Quick test
                    response = provider("test", temperature=0.0, max_output_tokens=1)
                    if response:
                        model = test_model
                        print(f"  SUCCESS: Using {model}")
                        break
                except Exception as e:
                    print(f"    Failed: {str(e)[:50]}")
                    continue
            
            if model is None:
                raise ValueError("No Gemini models available. Check API key and quota.")
        
        self.model = model
        self.provider = GeminiProvider(
            model=model,
            api_key=api_key or os.getenv("GOOGLE_API_KEY")
        )
        print(f"Initialized IntelligentTestBuilder with {model}")
    
    def analyze_module(self, module_path: Path) -> Dict[str, Any]:
        """Analyze a module to understand its functionality."""
        if not module_path.exists():
            raise FileNotFoundError(f"Module not found: {module_path}")
        
        content = module_path.read_text(encoding='utf-8')
        
        prompt = f"""Analyze this Python module and provide a comprehensive understanding:

MODULE: {module_path.name}

{content[:8000]}  # Limit to 8000 chars for initial analysis

Provide:
1. Module Purpose: What does this module do?
2. Key Classes: List all classes and their responsibilities
3. Key Functions: List all functions and what they do
4. Business Logic: What are the core algorithms/logic?
5. Edge Cases: What edge cases should be tested?
6. Dependencies: What external dependencies does it have?
7. Data Flows: How does data flow through the module?
8. Error Scenarios: What errors could occur?

Format as JSON with these keys: purpose, classes, functions, business_logic, edge_cases, dependencies, data_flows, error_scenarios"""

        try:
            # Call the Gemini provider
            response = self.provider(prompt, temperature=0.1, max_output_tokens=2000)
            
            # Handle list response
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = str(response)
            
            # Try to parse as JSON
            try:
                # Clean up the response - remove markdown code blocks if present
                clean_response = response_text.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.startswith("```"):
                    clean_response = clean_response[3:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                
                analysis = json.loads(clean_response.strip())
            except json.JSONDecodeError:
                # If not JSON, return structured dict from text
                analysis = {
                    "purpose": "Analysis completed but not in JSON format",
                    "raw_analysis": response_text,
                    "classes": [],
                    "functions": [],
                    "business_logic": "",
                    "edge_cases": [],
                    "dependencies": [],
                    "data_flows": "",
                    "error_scenarios": []
                }
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing module: {e}")
            # If we got a list response instead of proper analysis
            if isinstance(analysis, list):
                return {
                    "error": "Got list response instead of analysis",
                    "purpose": "Module analysis",
                    "raw_response": str(analysis),
                    "classes": [],
                    "functions": [],
                    "business_logic": "",
                    "edge_cases": [],
                    "dependencies": [],
                    "data_flows": "",
                    "error_scenarios": []
                }
            return {
                "error": str(e),
                "purpose": "Error during analysis",
                "classes": [],
                "functions": []
            }
    
    def generate_intelligent_test(self, module_path: Path, analysis: Dict[str, Any]) -> str:
        """Generate intelligent test code based on module analysis."""
        
        module_name = module_path.stem
        module_import_path = str(module_path).replace("\\", "/").replace(".py", "")
        module_import_path = module_import_path.replace("/", ".")
        if "multi_coder_analysis" in module_import_path:
            module_import_path = module_import_path.split("multi_coder_analysis")[-1]
            module_import_path = "multi_coder_analysis" + module_import_path
        
        prompt = f"""Generate INTELLIGENT, EXHAUSTIVE real functionality tests for this module.

MODULE: {module_name}
IMPORT PATH: {module_import_path}

ANALYSIS:
{json.dumps(analysis, indent=2)[:4000]}

REQUIREMENTS:
1. NO MOCKS for internal code - only mock external APIs
2. Test ALL public functions, classes, methods exhaustively
3. Test with REAL data that the module would actually process
4. Test REAL business logic and algorithms
5. Test ALL edge cases and error conditions
6. Tests must FAIL if the module is broken
7. Include integration tests between components
8. Test performance characteristics where relevant

Generate a complete test file with:
- Proper imports (using sys.path.insert if needed)
- Test class for each major component
- Multiple test methods covering ALL functionality
- Real data fixtures
- Error condition testing
- Integration testing
- Comments explaining what each test validates

Return ONLY the Python code, no markdown or explanations."""

        try:
            response = self.provider(prompt, temperature=0.1, max_output_tokens=4000)
            
            # Handle list response
            if isinstance(response, list):
                response_text = response[0] if response else ""
            else:
                response_text = str(response)
            
            # Clean up response
            test_code = response_text.strip()
            if test_code.startswith("```python"):
                test_code = test_code[9:]
            if test_code.startswith("```"):
                test_code = test_code[3:]
            if test_code.endswith("```"):
                test_code = test_code[:-3]
            
            return test_code.strip()
            
        except Exception as e:
            print(f"Error generating test: {e}")
            return f"# Error generating test: {e}\n# Manual intervention required"
    
    def build_test_for_module(self, module_path: Path, output_dir: Path = None) -> bool:
        """Complete workflow to build intelligent test for a module."""
        
        print(f"\n{'='*60}")
        print(f"Building intelligent test for: {module_path.name}")
        print('='*60)
        
        # Step 1: Analyze the module
        print("Step 1: Analyzing module structure and functionality...")
        analysis = self.analyze_module(module_path)
        
        # Handle case where analysis returns a list instead of dict
        if isinstance(analysis, list):
            print("WARNING: Got list response, using fallback analysis")
            analysis = {
                "purpose": f"Analysis of {module_path.name}",
                "classes": [],
                "functions": ["Module functions"],
                "business_logic": "See module source",
                "edge_cases": [],
                "dependencies": [],
                "data_flows": "",
                "error_scenarios": []
            }
        
        if "error" in analysis:
            print(f"ERROR: Failed to analyze module: {analysis['error']}")
            return False
        
        print(f"  Purpose: {analysis.get('purpose', 'Unknown')[:100]}...")
        print(f"  Classes found: {len(analysis.get('classes', []))}")
        print(f"  Functions found: {len(analysis.get('functions', []))}")
        
        # Step 2: Generate intelligent test
        print("\nStep 2: Generating intelligent test code...")
        test_code = self.generate_intelligent_test(module_path, analysis)
        
        if "Error generating test" in test_code:
            print("ERROR: Failed to generate test code")
            return False
        
        # Step 3: Save the test
        if output_dir is None:
            output_dir = Path("tests/unit")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        test_filename = f"test_{module_path.stem}_intelligent.py"
        test_path = output_dir / test_filename
        
        test_path.write_text(test_code, encoding='utf-8')
        print(f"\nStep 3: Test saved to: {test_path}")
        
        # Step 4: Validate the test (try to import it)
        print("\nStep 4: Validating test file...")
        try:
            import ast
            ast.parse(test_code)
            print("  OK: Test file has valid Python syntax")
            return True
        except SyntaxError as e:
            print(f"  ERROR: Syntax error in generated test: {e}")
            return False


def test_gemini_connection():
    """Test that Gemini API is working."""
    print("Testing Gemini connection...")
    
    try:
        # Let it auto-detect the best model
        builder = IntelligentTestBuilder()
        
        # Simple test prompt - be very explicit
        test_prompt = "Please respond with exactly this text and nothing else: Gemini is working!"
        response = builder.provider(test_prompt, temperature=0.0, max_output_tokens=50)
        
        # Handle list response from the provider
        if isinstance(response, list):
            response_text = response[0] if response else ""
        else:
            response_text = str(response)
        
        print(f"Response: {response_text}")
        
        if "working" in response_text.lower() or "gemini" in response_text.lower():
            print("OK: Gemini-2.5-pro connection successful!")
            return True
        else:
            print("ERROR: Unexpected response from Gemini")
            return False
            
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            print(f"ERROR: Quota exceeded. The API key has hit its usage limits.")
            print("       You need to either:")
            print("       1. Wait for quota to reset (usually daily for free tier)")
            print("       2. Upgrade to a paid API key")
            print("       3. Use a different API key")
        else:
            print(f"ERROR: Error connecting to Gemini: {e}")
        return False


def main():
    """Main function to test the intelligent test builder."""
    
    # First test the connection
    if not test_gemini_connection():
        print("\nFailed to connect to Gemini-2.5-pro. Please check:")
        print("1. GOOGLE_API_KEY environment variable is set")
        print("2. You have access to gemini-2.5-pro model")
        return 1
    
    print("\n" + "="*60)
    print("Gemini-2.5-pro Test Builder Ready!")
    print("="*60)
    
    # Test on a simple module
    test_module = Path("multi_coder_analysis/utils/concatenate_prompts.py")
    
    if test_module.exists():
        print(f"\nTesting on module: {test_module}")
        builder = IntelligentTestBuilder(model="models/gemini-2.5-pro")
        success = builder.build_test_for_module(test_module)
        
        if success:
            print("\nOK: Successfully generated intelligent test!")
            return 0
        else:
            print("\nERROR: Failed to generate test")
            return 1
    else:
        print(f"\nTest module not found: {test_module}")
        return 1


if __name__ == "__main__":
    sys.exit(main())