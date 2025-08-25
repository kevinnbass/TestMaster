"""
Batch Converter Strategy

This module implements the batch conversion strategy for efficient
processing of multiple files in organized batches.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .converter_base import (
    ConverterStrategy, ConversionResult, ConversionStatus,
    ConversionConfig
)


class BatchConverterStrategy(ConverterStrategy):
    """
    Batch conversion strategy that processes files in optimized batches
    for improved efficiency and resource utilization.
    """
    
    def __init__(self, config: ConversionConfig):
        super().__init__(config)
        self.batch_queue = asyncio.Queue()
        self.results_cache = {}
    
    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a single file"""
        # For single file, just process directly
        return await self._process_file(file_path)
    
    async def batch_convert(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Convert multiple files in optimized batches"""
        results = []
        total_files = len(file_paths)
        
        self.logger.info(f"Starting batch conversion of {total_files} files")
        
        # Process files in batches
        batch_size = self.config.batch_size
        for batch_start in range(0, total_files, batch_size):
            batch_end = min(batch_start + batch_size, total_files)
            batch = file_paths[batch_start:batch_end]
            
            self.logger.debug(f"Processing batch {batch_start//batch_size + 1}: files {batch_start+1}-{batch_end}")
            
            # Process batch
            batch_results = await self._process_batch(batch)
            results.extend(batch_results)
            
            # Progress update
            if self.config.verbose:
                progress = len(results) / total_files * 100
                print(f"Batch progress: {len(results)}/{total_files} ({progress:.1f}%)")
        
        return results
    
    def validate_file(self, file_path: Path) -> Tuple[bool, Optional[str]]:
        """Validate if file can be converted"""
        if not file_path.exists():
            return False, "File does not exist"
        
        if not file_path.suffix == ".py":
            return False, "Not a Python file"
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size < 50:
            return False, "File too small"
        if file_size > 50000:
            return False, "File too large for batch processing"
        
        return True, None
    
    async def _process_batch(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Process a batch of files"""
        # Validate all files first
        validated_files = []
        results = []
        
        for file_path in file_paths:
            is_valid, error_msg = self.validate_file(file_path)
            if is_valid:
                validated_files.append(file_path)
            else:
                results.append(ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.SKIPPED,
                    strategy_used=self.config.strategy,
                    error_message=error_msg,
                    duration=0.0
                ))
        
        # Process validated files concurrently
        if validated_files:
            tasks = [self._process_file(fp) for fp in validated_files]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for file_path, result in zip(validated_files, batch_results):
                if isinstance(result, Exception):
                    results.append(ConversionResult(
                        file_path=file_path,
                        status=ConversionStatus.FAILED,
                        strategy_used=self.config.strategy,
                        error_message=str(result),
                        duration=0.0
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _process_file(self, file_path: Path) -> ConversionResult:
        """Process a single file"""
        start_time = time.time()
        
        try:
            # Check if already exists
            if self.check_exists(file_path):
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.EXISTS,
                    strategy_used=self.config.strategy,
                    output_path=self.get_output_path(file_path),
                    duration=time.time() - start_time
                )
            
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
            
            # Generate test content
            test_content = await self._generate_batch_test(file_path, content)
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
                    metadata={"batch_processed": True}
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
            self.logger.error(f"Batch processing failed for {file_path}: {e}")
            return ConversionResult(
                file_path=file_path,
                status=ConversionStatus.FAILED,
                strategy_used=self.config.strategy,
                error_message=str(e),
                duration=time.time() - start_time
            )
    
    async def _generate_batch_test(self, file_path: Path, content: str) -> Optional[str]:
        """Generate test content for batch processing"""
        try:
            module_name = file_path.stem
            import_path = self.build_import_path(file_path)
            
            # Analyze module structure
            module_info = self._analyze_module(content)
            
            # Generate optimized test for batch
            test_code = self._generate_test_template(
                module_name,
                import_path,
                module_info
            )
            
            return test_code
            
        except Exception as e:
            self.logger.error(f"Test generation failed: {e}")
            return None
    
    def _analyze_module(self, content: str) -> Dict[str, Any]:
        """Analyze module structure for test generation"""
        info = {
            "has_classes": False,
            "has_functions": False,
            "has_async": False,
            "imports": [],
            "classes": [],
            "functions": [],
            "constants": []
        }
        
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            
            # Check for classes
            if stripped.startswith("class "):
                info["has_classes"] = True
                class_name = stripped.split("class ")[1].split("(")[0].split(":")[0]
                info["classes"].append(class_name)
            
            # Check for functions
            elif stripped.startswith("def "):
                info["has_functions"] = True
                func_name = stripped.split("def ")[1].split("(")[0]
                if not func_name.startswith("_"):
                    info["functions"].append(func_name)
            
            # Check for async
            elif stripped.startswith("async def "):
                info["has_async"] = True
                func_name = stripped.split("async def ")[1].split("(")[0]
                if not func_name.startswith("_"):
                    info["functions"].append(f"async_{func_name}")
            
            # Check for imports
            elif stripped.startswith("import ") or stripped.startswith("from "):
                info["imports"].append(stripped)
            
            # Check for constants
            elif "=" in stripped and stripped[0].isupper():
                const_name = stripped.split("=")[0].strip()
                if const_name.isupper():
                    info["constants"].append(const_name)
        
        return info
    
    def _generate_test_template(self, module_name: str, import_path: str,
                               module_info: Dict[str, Any]) -> str:
        """Generate test template based on module analysis"""
        test_code = f'''"""
Batch-generated tests for {module_name} module.
Generated by BatchConverterStrategy.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
'''
        
        if module_info["has_async"]:
            test_code += "import asyncio\nfrom unittest.mock import AsyncMock\n"
        
        test_code += f"from {import_path} import *\n\n\n"
        
        # Generate test class
        class_name = module_name.replace("_", " ").title().replace(" ", "")
        test_code += f'''class Test{class_name}:
    """Batch test suite for {module_name} module."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        return {{
            "test_value": 123,
            "test_string": "test",
            "test_list": [1, 2, 3]
        }}
'''
        
        # Add tests for constants
        if module_info["constants"]:
            test_code += '''
    
    def test_constants_defined(self):
        """Test that module constants are defined."""'''
            for const in module_info["constants"][:5]:  # Limit to 5
                test_code += f'''
        assert {const} is not None'''
        
        # Add tests for classes
        for class_name in module_info["classes"][:3]:  # Limit to 3
            test_code += f'''
    
    def test_{class_name.lower()}_class(self):
        """Test {class_name} class."""
        # Test class is importable
        assert {class_name} is not None
        # TODO: Add specific tests for {class_name}
'''
        
        # Add tests for functions
        for func_name in module_info["functions"][:5]:  # Limit to 5
            if func_name.startswith("async_"):
                test_code += f'''
    
    @pytest.mark.asyncio
    async def test_{func_name}(self):
        """Test {func_name.replace('async_', '')} async function."""
        # TODO: Implement async test
        pass
'''
            else:
                test_code += f'''
    
    def test_{func_name}(self):
        """Test {func_name} function."""
        # TODO: Implement test for {func_name}
        pass
'''
        
        # Add integration test
        test_code += '''
    
    def test_module_integration(self, setup_data):
        """Test module integration."""
        # TODO: Add integration tests
        assert setup_data is not None
'''
        
        return test_code