#!/usr/bin/env python3
"""
Intelligent Test Converter for TestMaster

Consolidates functionality from multiple converter scripts:
- accelerated_converter.py
- fast_converter.py  
- intelligent_converter.py
- self_healing_converter.py

Provides intelligent test conversion with AI assistance and self-healing capabilities.
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

from .base import BaseConverter, ConversionResult, ConversionConfig

# Try to load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class IntelligentConverter(BaseConverter):
    """
    Intelligent test converter with AI assistance and self-healing.
    
    Supports multiple AI providers with fallback to template generation.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None,
                 api_key: Optional[str] = None,
                 config: Optional[ConversionConfig] = None):
        """
        Initialize intelligent converter.
        
        Args:
            mode: "provider", "sdk", "template", or "auto"
            model: AI model to use
            api_key: API key for AI services
            config: Conversion configuration
        """
        super().__init__(config)
        self.mode = mode
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.provider = None
        self.client = None
        
        # Initialize AI provider
        if mode == "auto":
            self._auto_initialize()
        elif mode == "provider":
            self._init_provider_mode()
        elif mode == "sdk":
            self._init_sdk_mode()
        elif mode == "template":
            self._init_template_mode()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _auto_initialize(self):
        """Auto-detect best available AI provider."""
        print("Auto-detecting best available conversion mode...")
        
        # Try provider mode first
        try:
            self._init_provider_mode()
            print("SUCCESS: Using provider mode")
            self.mode = "provider"
            return
        except Exception as e:
            print(f"Provider mode failed: {str(e)[:50]}")
        
        # Try SDK mode
        try:
            self._init_sdk_mode()
            print("SUCCESS: Using SDK mode")
            self.mode = "sdk"
            return
        except Exception as e:
            print(f"SDK mode failed: {str(e)[:50]}")
        
        # Fallback to template mode
        print("Using template mode (offline)")
        self.mode = "template"
        self._init_template_mode()
    
    def _init_provider_mode(self):
        """Initialize using GeminiProvider."""
        try:
            from multi_coder_analysis.llm_providers.gemini_provider import GeminiProvider
            
            if self.model is None:
                self.model = "models/gemini-2.5-pro"
            
            self.provider = GeminiProvider(model=self.model, api_key=self.api_key)
            print(f"Initialized provider mode with {self.model}")
            
        except ImportError:
            raise ImportError("GeminiProvider not available")
    
    def _init_sdk_mode(self):
        """Initialize using direct Google GenAI SDK."""
        try:
            import google.generativeai as genai
            
            if self.model is None:
                self.model = "models/gemini-2.5-pro"
            
            genai.configure(api_key=self.api_key)
            self.client = genai
            print(f"Initialized SDK mode with {self.model}")
            
        except ImportError:
            raise ImportError("google.generativeai not available")
    
    def _init_template_mode(self):
        """Initialize template-based mode."""
        print("Initialized template mode (offline)")
    
    def convert_module(self, module_path: Path) -> ConversionResult:
        """Convert a single module to test."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.get_cached_result(module_path)
            if cached_result:
                test_path = self.save_test_file(cached_result, module_path)
                execution_time = time.time() - start_time
                
                return ConversionResult(
                    success=True,
                    module_path=str(module_path),
                    test_path=str(test_path) if test_path else None,
                    execution_time=execution_time,
                    test_count=self.count_test_methods(cached_result)
                )
            
            # Generate test code
            test_code = self._generate_test_code(module_path)
            
            if not test_code:
                return ConversionResult(
                    success=False,
                    module_path=str(module_path),
                    error_message="Failed to generate test code",
                    execution_time=time.time() - start_time
                )
            
            # Validate syntax
            if not self.validate_test_syntax(test_code):
                # Try self-healing
                healed_code = self._attempt_self_healing(test_code, module_path)
                if healed_code and self.validate_test_syntax(healed_code):
                    test_code = healed_code
                else:
                    return ConversionResult(
                        success=False,
                        module_path=str(module_path),
                        error_message="Generated test has syntax errors",
                        execution_time=time.time() - start_time
                    )
            
            # Save test file
            test_path = self.save_test_file(test_code, module_path)
            if not test_path:
                return ConversionResult(
                    success=False,
                    module_path=str(module_path),
                    error_message="Failed to save test file",
                    execution_time=time.time() - start_time
                )
            
            # Cache successful result
            self.cache_result(module_path, test_code)
            
            execution_time = time.time() - start_time
            test_count = self.count_test_methods(test_code)
            
            return ConversionResult(
                success=True,
                module_path=str(module_path),
                test_path=str(test_path),
                execution_time=execution_time,
                test_count=test_count,
                quality_score=self._estimate_quality_score(test_code)
            )
            
        except Exception as e:
            return ConversionResult(
                success=False,
                module_path=str(module_path),
                error_message=f"Conversion error: {e}",
                execution_time=time.time() - start_time
            )
    
    def _generate_test_code(self, module_path: Path) -> Optional[str]:
        """Generate test code using appropriate method."""
        if self.mode in ["provider", "sdk"]:
            return self._generate_with_ai(module_path)
        else:
            return self._generate_template(module_path)
    
    def _generate_with_ai(self, module_path: Path) -> Optional[str]:
        """Generate test code using AI."""
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Read module content
            content = module_path.read_text(encoding='utf-8')
            if len(content) > 8000:
                content = content[:8000] + "\n# ... (truncated)"
            
            # Build import path
            module_import = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
            if "multi_coder_analysis" in module_import:
                idx = module_import.find("multi_coder_analysis")
                module_import = module_import[idx:]
            
            prompt = f"""Generate comprehensive Python test code for this module.

MODULE: {module_path.name}
IMPORT: {module_import}

SOURCE CODE:
```python
{content}
```

Requirements:
1. Import the real module: {module_import}
2. NO MOCKS for internal code - test real functionality
3. Test ALL public functions and classes
4. Use real test data
5. Test edge cases and error conditions
6. Use pytest framework
7. Include comprehensive docstrings

Generate complete test file with all necessary imports and test classes.
Return ONLY the Python code, no markdown."""

            # Make AI call
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.1, max_output_tokens=4000)
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
            else:  # SDK mode
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=4000
                    )
                )
                response_text = response.text if response.text else ""
            
            # Clean response
            test_code = response_text.strip()
            if test_code.startswith("```python"):
                test_code = test_code[9:]
            if test_code.startswith("```"):
                test_code = test_code[3:]
            if test_code.endswith("```"):
                test_code = test_code[:-3]
            
            return test_code.strip()
            
        except Exception as e:
            print(f"AI generation failed: {e}")
            # Fallback to template
            return self._generate_template(module_path)
    
    def _generate_template(self, module_path: Path) -> str:
        """Generate template test code."""
        module_name = module_path.stem
        
        # Build import path
        module_import = str(module_path).replace("\\", "/").replace(".py", "").replace("/", ".")
        if "multi_coder_analysis" in module_import:
            idx = module_import.find("multi_coder_analysis")
            module_import = module_import[idx:]
        
        # Analyze module structure
        try:
            content = module_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            classes = []
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    methods = [item.name for item in node.body 
                              if isinstance(item, ast.FunctionDef)]
                    classes.append({"name": node.name, "methods": methods})
                elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                    if not node.name.startswith('_'):
                        functions.append(node.name)
        except:
            classes = []
            functions = []
        
        # Generate template
        template = f'''"""
Test suite for {module_name}

Auto-generated test coverage using TestMaster IntelligentConverter.
"""

import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import the module under test
try:
    import {module_import} as test_module
except ImportError as e:
    pytest.skip(f"Could not import {module_import}: {{e}}", allow_module_level=True)


class TestModuleStructure:
    """Test module structure and imports."""
    
    def test_module_imports(self):
        """Test that module imports successfully."""
        assert test_module is not None
'''

        # Add class tests
        for cls in classes:
            template += f'''


class Test{cls['name']}:
    """Test {cls['name']} class."""
    
    def test_class_exists(self):
        """Test that {cls['name']} class exists."""
        assert hasattr(test_module, "{cls['name']}")
        cls_obj = getattr(test_module, "{cls['name']}")
        assert isinstance(cls_obj, type)
'''
            
            # Add method tests
            for method in cls.get("methods", []):
                if not method.startswith('_') or method == "__init__":
                    safe_method = method.replace('__', '_')
                    template += f'''
    def test_{safe_method}(self):
        """Test {cls['name']}.{method} method."""
        cls_obj = getattr(test_module, "{cls['name']}")
        assert hasattr(cls_obj, "{method}")
        # TODO: Add actual test implementation
'''

        # Add function tests
        if functions:
            template += '''


class TestModuleFunctions:
    """Test module-level functions."""
'''
            
            for func in functions:
                template += f'''
    def test_{func}(self):
        """Test {func} function."""
        func_obj = getattr(test_module, "{func}")
        assert callable(func_obj)
        # TODO: Add actual test implementation
'''

        template += '''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
'''
        
        return template
    
    def _attempt_self_healing(self, test_code: str, module_path: Path) -> Optional[str]:
        """Attempt to self-heal syntax errors in test code."""
        if self.mode not in ["provider", "sdk"]:
            return None
        
        try:
            # Rate limiting
            self.rate_limiter.wait_if_needed()
            
            prompt = f"""Fix the syntax errors in this Python test code:

```python
{test_code}
```

Requirements:
1. Fix all syntax errors
2. Maintain the original test logic
3. Ensure all imports are correct
4. Return only the corrected Python code
5. No explanations or markdown

Return the fixed code:"""

            # Make healing call
            if self.mode == "provider":
                response = self.provider(prompt, temperature=0.0, max_output_tokens=4000)
                if isinstance(response, list):
                    response_text = response[0] if response else ""
                else:
                    response_text = str(response)
            else:  # SDK mode
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=self.client.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=4000
                    )
                )
                response_text = response.text if response.text else ""
            
            # Clean response
            healed_code = response_text.strip()
            if healed_code.startswith("```python"):
                healed_code = healed_code[9:]
            if healed_code.startswith("```"):
                healed_code = healed_code[3:]
            if healed_code.endswith("```"):
                healed_code = healed_code[:-3]
            
            return healed_code.strip()
            
        except Exception as e:
            print(f"Self-healing failed: {e}")
            return None
    
    def _estimate_quality_score(self, test_code: str) -> float:
        """Estimate quality score of generated test."""
        try:
            tree = ast.parse(test_code)
            
            # Count different types of elements
            test_methods = 0
            assertions = 0
            docstrings = 0
            imports = 0
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    test_methods += 1
                    if ast.get_docstring(node):
                        docstrings += 1
                elif isinstance(node, ast.Assert):
                    assertions += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports += 1
            
            # Simple quality scoring
            base_score = min(test_methods * 10, 50)  # Up to 50 for test methods
            assertion_score = min(assertions * 2, 30)  # Up to 30 for assertions
            doc_score = min(docstrings * 3, 15)  # Up to 15 for docstrings
            import_score = min(imports, 5)  # Up to 5 for imports
            
            total_score = base_score + assertion_score + doc_score + import_score
            return min(total_score, 100.0)
            
        except:
            return 50.0  # Default score if analysis fails


def main():
    """Test the intelligent converter."""
    print("="*60)
    print("TestMaster Intelligent Converter")
    print("="*60)
    
    # Test conversion
    converter = IntelligentConverter(mode="auto")
    
    # Get test module
    test_module = Path("multi_coder_analysis/utils/concatenate_prompts.py")
    if not test_module.exists():
        print("Test module not found, using converter itself")
        test_module = Path(__file__)
    
    print(f"Converting module: {test_module}")
    result = converter.convert_module(test_module)
    
    if result.success:
        print(f"✅ Conversion successful!")
        print(f"Test file: {result.test_path}")
        print(f"Test methods: {result.test_count}")
        print(f"Quality score: {result.quality_score:.1f}")
        print(f"Execution time: {result.execution_time:.1f}s")
    else:
        print(f"❌ Conversion failed: {result.error_message}")
    
    converter.print_stats()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())