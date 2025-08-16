#!/usr/bin/env python3
"""
Intelligent Test Generator for TestMaster

Consolidates functionality from:
- intelligent_test_builder.py (Gemini provider based)
- intelligent_test_builder_v2.py (Direct SDK based) 
- intelligent_test_builder_offline.py (Template based)

Provides intelligent test generation using multiple approaches with fallback support.
"""

import os
import sys
import json
import ast
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import BaseGenerator, AnalysisBasedGenerator, ModuleAnalysis, GenerationConfig

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded .env file")
except ImportError:
    print("Note: python-dotenv not installed, using system environment variables")


class IntelligentTestGenerator(AnalysisBasedGenerator):
    """
    Intelligent test generator with multiple provider support.
    
    Supports three generation modes:
    1. Gemini Provider (via multi_coder_analysis)
    2. Direct SDK (via google.genai)
    3. Template-based (offline, no API calls)
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None, 
                 api_key: Optional[str] = None,
                 config: Optional[GenerationConfig] = None):
        """
        Initialize intelligent test generator.
        
        Args:
            mode: "provider", "sdk", "template", or "auto" (try in order)
            model: Gemini model to use (defaults to best available)
            api_key: API key (defaults to GOOGLE_API_KEY env var)
            config: Generation configuration
        """
        super().__init__(config)
        self.mode = mode
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.provider = None
        self.client = None
        
        # Initialize based on mode
        if mode == "auto":
            self._auto_initialize()
        elif mode == "provider":
            self._init_provider_mode()
        elif mode == "sdk":
            self._init_sdk_mode()
        elif mode == "template":
            self._init_template_mode()
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'provider', 'sdk', 'template', or 'auto'")
    
    def _auto_initialize(self):
        """Auto-detect best available mode."""
        print("Auto-detecting best available generation mode...")
        
        # Try provider mode first
        try:
            self._init_provider_mode()
            print("SUCCESS: Using provider mode (via multi_coder_analysis)")
            self.mode = "provider"
            return
        except Exception as e:
            print(f"Provider mode failed: {str(e)[:50]}")
        
        # Try SDK mode
        try:
            self._init_sdk_mode()
            print("SUCCESS: Using SDK mode (via google.genai)")
            self.mode = "sdk"
            return
        except Exception as e:
            print(f"SDK mode failed: {str(e)[:50]}")
        
        # Fallback to template mode
        print("Using template mode (offline)")
        self.mode = "template"
        self._init_template_mode()
    
    def _init_provider_mode(self):
        """Initialize using GeminiProvider from multi_coder_analysis."""
        try:
            from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider
            
            # Auto-detect best model if not specified
            if self.model is None:
                models_to_try = [
                    "models/gemini-2.5-pro",
                    "models/gemini-2.0-flash", 
                    "models/gemini-1.5-flash",
                    "models/gemini-1.5-pro",
                ]
                
                for test_model in models_to_try:
                    try:
                        provider = GeminiProvider(model=test_model, api_key=self.api_key)
                        # Quick test
                        response = provider("test", temperature=0.0, max_output_tokens=1)
                        if response:
                            self.model = test_model
                            break
                    except:
                        continue
                
                if self.model is None:
                    raise ValueError("No working Gemini models found")
            
            self.provider = GeminiProvider(model=self.model, api_key=self.api_key)
            print(f"Initialized provider mode with {self.model}")
            
        except ImportError:
            raise ImportError("GeminiProvider not available")
    
    def _init_sdk_mode(self):
        """Initialize using direct Google GenAI SDK."""
        try:
            import google.genai as genai
            
            if self.model is None:
                self.model = "models/gemini-2.5-pro"
            
            self.client = genai.Client(api_key=self.api_key)
            print(f"Initialized SDK mode with {self.model}")
            
        except ImportError:
            raise ImportError("google.genai not available")
    
    def _init_template_mode(self):
        """Initialize template-based mode (no API required)."""
        print("Initialized template mode (offline)")
    
    def analyze_module(self, module_path: Path) -> ModuleAnalysis:
        """Analyze module using appropriate method based on mode."""
        if self.mode in ["provider", "sdk"]:
            return self._analyze_module_with_ai(module_path)
        else:
            return self.analyze_module_ast(module_path)
    
    def _analyze_module_with_ai(self, module_path: Path) -> ModuleAnalysis:
        """Analyze module using AI (provider or SDK mode)."""
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
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.1, max_output_tokens=2000)
                
                # Handle list response
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
                    
            else:  # SDK mode
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 2000}
                )
                response_text = response.text if response.text else ""
            
            # Parse JSON response
            try:
                # Clean up response
                text = response_text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                
                analysis_dict = json.loads(text.strip())
                
                # Convert to ModuleAnalysis
                return ModuleAnalysis(
                    purpose=analysis_dict.get("purpose", f"Module {module_path.name}"),
                    classes=analysis_dict.get("classes", []),
                    functions=analysis_dict.get("functions", []),
                    business_logic=analysis_dict.get("business_logic", ""),
                    edge_cases=analysis_dict.get("edge_cases", []),
                    dependencies=analysis_dict.get("dependencies", []),
                    data_flows=analysis_dict.get("data_flows", ""),
                    error_scenarios=analysis_dict.get("error_scenarios", [])
                )
                
            except json.JSONDecodeError:
                # Fallback to AST analysis
                print("Warning: AI analysis failed, falling back to AST analysis")
                return self.analyze_module_ast(module_path)
                
        except Exception as e:
            print(f"Error in AI analysis: {e}")
            # Fallback to AST analysis
            return self.analyze_module_ast(module_path)
    
    def generate_test_code(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """Generate test code using appropriate method based on mode."""
        if self.mode in ["provider", "sdk"]:
            return self._generate_test_with_ai(module_path, analysis)
        else:
            return self._generate_test_template(module_path, analysis)
    
    def _generate_test_with_ai(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """Generate test using AI (provider or SDK mode)."""
        module_name = module_path.stem
        module_import_path = self.build_import_path(module_path)
        
        # Convert analysis to dict for JSON serialization
        analysis_dict = {
            "purpose": analysis.purpose,
            "classes": analysis.classes,
            "functions": analysis.functions,
            "business_logic": analysis.business_logic,
            "edge_cases": analysis.edge_cases,
            "dependencies": analysis.dependencies
        }
        
        prompt = f"""Generate EXHAUSTIVE Python test code for the module '{module_name}'.

Module Analysis:
{json.dumps(analysis_dict, indent=2)[:3000]}

Requirements:
1. Import the real module: {module_import_path}
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
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.1, max_output_tokens=4000)
                
                # Handle list response
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
                    
            else:  # SDK mode
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config={"temperature": 0.1, "max_output_tokens": 4000}
                )
                response_text = response.text if response.text else ""
            
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
            print(f"Error generating test with AI: {e}")
            # Fallback to template
            return self._generate_test_template(module_path, analysis)
    
    def _generate_test_template(self, module_path: Path, analysis: ModuleAnalysis) -> str:
        """Generate comprehensive test template."""
        module_name = module_path.stem
        module_import_path = self.build_import_path(module_path)
        
        test_code = f'''"""
INTELLIGENT Real Functionality Tests for {module_name}

This test file provides exhaustive testing of ALL public APIs.
Tests focus on real functionality, not mocks.

Auto-generated by TestMaster IntelligentTestGenerator
Mode: {self.mode}
"""

import pytest
import sys
from pathlib import Path
from typing import Any

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module under test
try:
    import {module_import_path} as test_module
except ImportError as e:
    pytest.skip(f"Could not import {module_import_path}: {{e}}", allow_module_level=True)


class TestModuleImports:
    """Test that the module imports correctly and has expected structure."""
    
    def test_module_imports_successfully(self):
        """Test that the module can be imported."""
        assert test_module is not None
        
    def test_module_has_expected_attributes(self):
        """Test that module has all expected classes and functions."""
        # Classes
'''
        
        # Add class existence tests
        for cls in analysis.classes:
            test_code += f'''        assert hasattr(test_module, "{cls['name']}"), "Missing class: {cls['name']}"\n'''
        
        test_code += '''        
        # Functions
'''
        for func in analysis.functions:
            if not func.get("is_private", False):
                test_code += f'''        assert hasattr(test_module, "{func['name']}"), "Missing function: {func['name']}"\n'''
        
        # Add test classes for each class in the module
        for cls in analysis.classes:
            test_code += f'''


class Test{cls['name']}:
    """Exhaustive tests for {cls['name']} class."""
    
    def test_class_exists(self):
        """Test that {cls['name']} class exists."""
        assert hasattr(test_module, "{cls['name']}")
        assert isinstance(getattr(test_module, "{cls['name']}"), type)
    
    def test_class_instantiation(self):
        """Test that {cls['name']} can be instantiated."""
        cls_obj = getattr(test_module, "{cls['name']}")
        # TODO: Add appropriate constructor arguments
        # instance = cls_obj()
        # assert instance is not None
    '''
            
            # Add method tests
            for method in cls.get("methods", []):
                if not method.get("is_private", False) or method["name"] == "__init__":
                    safe_method_name = method['name'].replace('__', '_')
                    test_code += f'''
    def test_method_{safe_method_name}(self):
        """Test {cls['name']}.{method['name']} method."""
        cls_obj = getattr(test_module, "{cls['name']}")
        assert hasattr(cls_obj, "{method['name']}"), "Missing method: {method['name']}"
        
        # TODO: Test actual functionality
        # instance = cls_obj()
        # result = instance.{method['name']}(...)
        # assert result == expected_value
    '''
        
        # Add test functions for module-level functions
        if analysis.functions:
            test_code += '''


class TestModuleFunctions:
    """Exhaustive tests for module-level functions."""
'''
            
            for func in analysis.functions:
                if not func.get("is_private", False):
                    test_code += f'''
    
    def test_{func['name']}(self):
        """Test {func['name']} function with real data."""
        func_obj = getattr(test_module, "{func['name']}")
        
        # Test with valid inputs
        # TODO: Add real test data
        # result = func_obj(test_input)
        # assert result is not None
        
        # Test edge cases
        # TODO: Add edge case tests
        
        # Test error conditions
        # TODO: Add error condition tests
'''
        
        # Add integration tests section
        test_code += '''


class TestIntegration:
    """Integration tests between components."""
    
    def test_component_integration(self):
        """Test that components work together correctly."""
        # TODO: Add integration tests
        pass
    
    def test_data_flow(self):
        """Test data flow through the module."""
        # TODO: Add data flow tests
        pass
    
    def test_error_propagation(self):
        """Test that errors propagate correctly."""
        # TODO: Add error propagation tests
        pass


class TestPerformance:
    """Performance and efficiency tests."""
    
    def test_performance_characteristics(self):
        """Test performance meets requirements."""
        # TODO: Add performance tests
        pass


class TestEdgeCases:
    """Comprehensive edge case testing."""
    
    def test_boundary_values(self):
        """Test boundary value conditions."""
        # TODO: Add boundary tests
        pass
    
    def test_null_empty_inputs(self):
        """Test handling of null/empty inputs."""
        # TODO: Add null/empty tests
        pass
    
    def test_large_inputs(self):
        """Test handling of large inputs."""
        # TODO: Add large input tests
        pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        
        return test_code
    
    def test_connection(self) -> bool:
        """Test that the current mode is working."""
        if self.mode == "template":
            return True  # Template mode always works
        
        try:
            if self.mode == "provider":
                response = self.provider("test", temperature=0.0, max_output_tokens=1)
                return bool(response)
            elif self.mode == "sdk":
                response = self.client.models.generate_content(
                    model=self.model,
                    contents="test",
                    config={"temperature": 0.0, "max_output_tokens": 1}
                )
                return bool(response.text)
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
        
        return False


def test_gemini_connection():
    """Test that Gemini API is working."""
    print("Testing Gemini connection...")
    
    try:
        generator = IntelligentTestGenerator(mode="auto")
        
        if generator.mode == "template":
            print("Using template mode (no API connection needed)")
            return True
        
        success = generator.test_connection()
        
        if success:
            print(f"OK: {generator.mode} mode connection successful!")
            return True
        else:
            print("ERROR: Connection test failed")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    """Main function to test the intelligent test generator."""
    
    print("="*60)
    print("TestMaster Intelligent Test Generator")
    print("="*60)
    
    # Test connection
    if not test_gemini_connection():
        print("\nWARNING: API connection failed, using template mode")
    
    # Test on a module
    test_module = Path("multi_coder_analysis/utils/concatenate_prompts.py")
    
    if not test_module.exists():
        print(f"Test module not found: {test_module}")
        print("Using placeholder test...")
        test_module = Path(__file__)  # Test on self
    
    print(f"\nTesting on module: {test_module}")
    generator = IntelligentTestGenerator(mode="auto")
    success = generator.build_test_for_module(test_module)
    
    # Print stats
    generator.print_stats()
    
    if success:
        print("\nSUCCESS! Intelligent test generated!")
        return 0
    else:
        print("\nFailed to generate test")
        return 1


if __name__ == "__main__":
    sys.exit(main())