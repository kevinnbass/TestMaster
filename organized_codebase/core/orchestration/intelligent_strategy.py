"""
Intelligent Converter Strategy

This module implements the intelligent conversion strategy using
AI-powered test generation with advanced context understanding.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple

from .converter_base import (
    ConverterStrategy, ConversionResult, ConversionStatus,
    ConversionConfig
)


class IntelligentConverterStrategy(ConverterStrategy):
    """
    Intelligent conversion strategy that uses AI to generate
    comprehensive tests with deep understanding of code context.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.ai_client = None
        self._initialize_ai_client()
    
    def _initialize_ai_client(self):
        """Initialize AI client for test generation"""
        try:
            if self.config.api_key:
                # Initialize Gemini or other AI client
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                
                # Configure model
                generation_config = {
                    "temperature": self.config.temperature,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192
                }
                
                self.ai_client = genai.GenerativeModel(
                    self.config.model_name,
                    generation_config=generation_config
                )
                
                self.logger.info(f"AI client initialized with model: {self.config.model_name}")
        except Exception as e:
            self.logger.warning(f"AI client initialization failed: {e}")
            self.ai_client = None
    
    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a single file using intelligent analysis"""
        start_time = time.time()
        
        try:
            # Read file content
            content = await self.read_file_content(file_path)
            if not content:
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.FAILED,
                    strategy_used=self.config.strategy,
                    error_message="Failed to read file",
                    duration=time.time() - start_time
                )
            
            # Generate test using AI
            test_content = await self._generate_intelligent_test(file_path, content)
            if not test_content:
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.FAILED,
                    strategy_used=self.config.strategy,
                    error_message="Failed to generate test",
                    duration=time.time() - start_time
                )
            
            # Write output
            output_path = self.get_output_path(file_path)
            success = await self.write_output(output_path, test_content)
            
            if success:
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.COMPLETED,
                    strategy_used=self.config.strategy,
                    output_path=output_path,
                    duration=time.time() - start_time,
                    metadata={"lines_generated": len(test_content.split("\n"))}
                )
            else:
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.FAILED,
                    strategy_used=self.config.strategy,
                    error_message="Failed to write output",
                    duration=time.time() - start_time
                )
                
        except Exception as e:
            self.logger.error(f"Intelligent conversion failed for {file_path}: {e}")
            return ConversionResult(
                file_path=file_path,
                status=ConversionStatus.FAILED,
                strategy_used=self.config.strategy,
                error_message=str(e),
                duration=time.time() - start_time
            )
    
    async def batch_convert(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Convert multiple files in batch"""
        results = []
        
        # Process in batches for efficiency
        batch_size = self.config.batch_size
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            # Convert batch concurrently
            batch_tasks = [self.convert(file_path) for file_path in batch]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            
            # Small delay between batches to avoid rate limiting
            if i + batch_size < len(file_paths):
                await asyncio.sleep(1)
        
        return results
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate if file can be converted"""
        if not file_path.exists():
            return False, "File does not exist"
        
        if not file_path.suffix == ".py":
            return False, "Not a Python file"
        
        if file_path.stat().st_size < 50:
            return False, "File too small"
        
        if file_path.stat().st_size > 100000:
            return False, "File too large"
        
        return True, None
    
    async def _generate_intelligent_test(self, file_path: Path, content: str) -> Optional[str]:
        """Generate intelligent test using AI"""
        if not self.ai_client:
            # Fallback to template-based generation
            return self._generate_template_test(file_path, content)
        
        try:
            # Build import path
            import_path = self.build_import_path(file_path)
            module_name = file_path.stem
            
            # Create comprehensive prompt
            prompt = self._build_intelligent_prompt(module_name, import_path, content)
            
            # Generate test using AI
            response = self.ai_client.generate_content(prompt)
            
            if response and response.text:
                # Extract code from response
                test_code = self._extract_code_from_response(response.text)
                return test_code
            
            return None
            
        except Exception as e:
            self.logger.error(f"AI generation failed: {e}")
            # Fallback to template
            return self._generate_template_test(file_path, content)
    
    def _build_intelligent_prompt(self, module_name: str, import_path: str, content: str) -> str:
        """Build comprehensive prompt for AI test generation"""
        return f"""Generate comprehensive pytest test code for this Python module.

MODULE: {module_name}.py
IMPORT PATH: {import_path}

```python
{content}
```

Requirements:
1. Create complete pytest test file with proper imports
2. Test ALL public functions and classes
3. Include edge cases and error conditions
4. Use appropriate mocks for external dependencies
5. Follow pytest best practices
6. Add descriptive test names and docstrings
7. Include both positive and negative test cases
8. Test async functions with pytest-asyncio if present
9. Add parametrized tests where appropriate
10. Ensure 100% code coverage goal

Return ONLY the complete Python test code, no explanations."""
    
    def _extract_code_from_response(self, response_text: str) -> str:
        """Extract Python code from AI response"""
        # Remove markdown code blocks if present
        if "```python" in response_text:
            start = response_text.find("```python") + 9
            end = response_text.find("```", start)
            if end > start:
                return response_text[start:end].strip()
        
        # Return as-is if no markdown
        return response_text.strip()
    
    def _generate_template_test(self, file_path: Path, content: str) -> str:
        """Generate template-based test as fallback"""
        module_name = file_path.stem
        import_path = self.build_import_path(file_path)
        
        # Parse content to find testable items
        functions = self._extract_functions(content)
        classes = self._extract_classes(content)
        
        # Build test template
        test_code = f'''"""
Intelligent tests for {module_name} module.
Generated by IntelligentConverterStrategy.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from {import_path} import *


class Test{module_name.replace("_", " ").title().replace(" ", "")}:
    """Test suite for {module_name} module."""
    
    def setup_method(self):
        """Setup test fixtures."""
        pass
    
    def teardown_method(self):
        """Cleanup after tests."""
        pass
'''
        
        # Add test methods for functions
        for func_name in functions:
            test_code += f'''
    
    def test_{func_name}(self):
        """Test {func_name} function."""
        # TODO: Implement test for {func_name}
        pass
    
    def test_{func_name}_edge_cases(self):
        """Test {func_name} with edge cases."""
        # TODO: Test edge cases
        pass
'''
        
        # Add test methods for classes
        for class_name in classes:
            test_code += f'''
    
    def test_{class_name.lower()}_initialization(self):
        """Test {class_name} initialization."""
        # TODO: Test class initialization
        pass
    
    def test_{class_name.lower()}_methods(self):
        """Test {class_name} methods."""
        # TODO: Test class methods
        pass
'''
        
        return test_code
    
    def _extract_functions(self, content: str) -> List[str]:
        """Extract function names from content"""
        functions = []
        for line in content.split("\n"):
            if line.strip().startswith("def ") and not line.strip().startswith("def _"):
                func_name = line.split("def ")[1].split("(")[0]
                functions.append(func_name)
        return functions
    
    def _extract_classes(self, content: str) -> List[str]:
        """Extract class names from content"""
        classes = []
        for line in content.split("\n"):
            if line.strip().startswith("class "):
                class_name = line.split("class ")[1].split("(")[0].split(":")[0]
                classes.append(class_name)
        return classes