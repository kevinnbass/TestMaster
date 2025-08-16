#!/usr/bin/env python3
"""
Base Converter Classes for TestMaster

Provides foundation classes for all test converters in the system.
"""

import os
import sys
import ast
import time
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import logging

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)

@dataclass
class ConversionResult:
    """Result of a test conversion operation."""
    success: bool
    module_path: str
    test_path: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    test_count: int = 0
    quality_score: float = 0.0

@dataclass
class ConversionConfig:
    """Configuration for test conversion."""
    max_workers: int = 4
    rate_limit_rpm: int = 30
    timeout_seconds: int = 120
    retry_attempts: int = 3
    quality_threshold: float = 70.0
    use_caching: bool = True
    output_directory: str = "tests/unit"
    batch_size: int = 10

class RateLimiter:
    """Thread-safe rate limiter for API calls."""
    
    def __init__(self, calls_per_minute: int = 30):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = time.time()
            time_since_last = now - self.last_call
            if time_since_last < self.min_interval:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)
            self.last_call = time.time()

class BaseConverter(ABC):
    """Base class for all test converters."""
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        """Initialize converter with configuration."""
        self.config = config or ConversionConfig()
        self.rate_limiter = RateLimiter(self.config.rate_limit_rpm)
        self.stats = {
            "conversions_attempted": 0,
            "conversions_successful": 0,
            "conversions_failed": 0,
            "total_execution_time": 0.0,
            "start_time": time.time()
        }
        self.cache = {} if self.config.use_caching else None
    
    @abstractmethod
    def convert_module(self, module_path: Path) -> ConversionResult:
        """Convert a single module to test."""
        pass
    
    def get_remaining_modules(self, base_directory: str = "multi_coder_analysis") -> List[Path]:
        """Get all modules that don't have tests yet."""
        test_dir = Path(self.config.output_directory)
        existing_tests = set()
        
        if test_dir.exists():
            for test_file in test_dir.glob("test_*.py"):
                # Extract module name from test file
                module_name = test_file.stem.replace("test_", "")
                if module_name.endswith("_intelligent"):
                    module_name = module_name.replace("_intelligent", "")
                existing_tests.add(module_name)
        
        remaining = []
        base_dir = Path(base_directory)
        
        if not base_dir.exists():
            logger.warning(f"Base directory {base_directory} does not exist")
            return remaining
        
        for py_file in base_dir.rglob("*.py"):
            if (py_file.name != "__init__.py" and 
                "__pycache__" not in str(py_file) and
                py_file.stem not in existing_tests):
                remaining.append(py_file)
        
        return sorted(remaining)
    
    def validate_test_syntax(self, test_code: str) -> bool:
        """Validate that test code has correct syntax."""
        try:
            ast.parse(test_code)
            return True
        except SyntaxError as e:
            logger.error(f"Syntax error in test code: {e}")
            return False
    
    def count_test_methods(self, test_code: str) -> int:
        """Count the number of test methods in generated code."""
        try:
            tree = ast.parse(test_code)
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                    count += 1
            return count
        except:
            return 0
    
    def save_test_file(self, test_code: str, module_path: Path) -> Optional[Path]:
        """Save test code to appropriate file."""
        output_dir = Path(self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        test_filename = f"test_{module_path.stem}.py"
        test_path = output_dir / test_filename
        
        try:
            test_path.write_text(test_code, encoding='utf-8')
            return test_path
        except Exception as e:
            logger.error(f"Failed to save test file {test_path}: {e}")
            return None
    
    def update_stats(self, result: ConversionResult):
        """Update conversion statistics."""
        self.stats["conversions_attempted"] += 1
        if result.success:
            self.stats["conversions_successful"] += 1
        else:
            self.stats["conversions_failed"] += 1
        self.stats["total_execution_time"] += result.execution_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversion statistics."""
        elapsed = time.time() - self.stats["start_time"]
        attempted = self.stats["conversions_attempted"]
        successful = self.stats["conversions_successful"]
        
        return {
            **self.stats,
            "elapsed_time": elapsed,
            "success_rate": (successful / max(1, attempted)) * 100,
            "average_time_per_conversion": (
                self.stats["total_execution_time"] / max(1, attempted)
            ),
            "conversions_per_minute": (attempted / max(1, elapsed / 60))
        }
    
    def print_stats(self):
        """Print conversion statistics."""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print("CONVERSION STATISTICS")
        print('='*60)
        print(f"Attempted: {stats['conversions_attempted']}")
        print(f"Successful: {stats['conversions_successful']}")
        print(f"Failed: {stats['conversions_failed']}")
        print(f"Success rate: {stats['success_rate']:.1f}%")
        print(f"Total time: {stats['elapsed_time']:.1f}s")
        print(f"Avg time per conversion: {stats['average_time_per_conversion']:.1f}s")
        print(f"Conversions per minute: {stats['conversions_per_minute']:.1f}")

class ParallelConverter(BaseConverter):
    """Base class for parallel test conversion."""
    
    def convert_modules_parallel(self, modules: List[Path]) -> List[ConversionResult]:
        """Convert multiple modules in parallel."""
        results = []
        
        print(f"Starting parallel conversion of {len(modules)} modules...")
        print(f"Using {self.config.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_module = {
                executor.submit(self.convert_module, module): module 
                for module in modules
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    result = future.result(timeout=self.config.timeout_seconds)
                    results.append(result)
                    self.update_stats(result)
                    
                    status = "✅" if result.success else "❌"
                    print(f"{status} {module.name} - {result.execution_time:.1f}s")
                    
                except Exception as e:
                    error_result = ConversionResult(
                        success=False,
                        module_path=str(module),
                        error_message=f"Execution error: {e}"
                    )
                    results.append(error_result)
                    self.update_stats(error_result)
                    print(f"❌ {module.name} - Error: {e}")
        
        return results

class BatchConverter(BaseConverter):
    """Base class for batch test conversion."""
    
    def convert_modules_batch(self, modules: List[Path]) -> List[ConversionResult]:
        """Convert modules in batches."""
        results = []
        total_batches = (len(modules) + self.config.batch_size - 1) // self.config.batch_size
        
        print(f"Starting batch conversion of {len(modules)} modules...")
        print(f"Using batch size: {self.config.batch_size}")
        print(f"Total batches: {total_batches}")
        
        for i in range(0, len(modules), self.config.batch_size):
            batch = modules[i:i + self.config.batch_size]
            batch_num = (i // self.config.batch_size) + 1
            
            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} modules)")
            
            batch_results = []
            for module in batch:
                try:
                    result = self.convert_module(module)
                    batch_results.append(result)
                    self.update_stats(result)
                    
                    status = "✅" if result.success else "❌"
                    print(f"  {status} {module.name}")
                    
                except Exception as e:
                    error_result = ConversionResult(
                        success=False,
                        module_path=str(module),
                        error_message=f"Batch error: {e}"
                    )
                    batch_results.append(error_result)
                    self.update_stats(error_result)
                    print(f"  ❌ {module.name} - Error: {e}")
            
            results.extend(batch_results)
            
            # Brief pause between batches
            if batch_num < total_batches:
                time.sleep(1)
        
        return results

class CachedConverter(BaseConverter):
    """Base class for converters with caching support."""
    
    def __init__(self, config: Optional[ConversionConfig] = None, cache_file: str = "conversion_cache.json"):
        super().__init__(config)
        self.cache_file = Path(cache_file)
        self.load_cache()
    
    def load_cache(self):
        """Load conversion cache from file."""
        if self.config.use_caching and self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    def save_cache(self):
        """Save conversion cache to file."""
        if self.config.use_caching and self.cache:
            try:
                with open(self.cache_file, 'w') as f:
                    json.dump(self.cache, f, indent=2)
                logger.info(f"Saved cache with {len(self.cache)} entries")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")
    
    def get_cache_key(self, module_path: Path) -> str:
        """Generate cache key for module."""
        return f"{module_path}:{module_path.stat().st_mtime}"
    
    def get_cached_result(self, module_path: Path) -> Optional[str]:
        """Get cached test code for module."""
        if not self.config.use_caching:
            return None
        
        cache_key = self.get_cache_key(module_path)
        return self.cache.get(cache_key)
    
    def cache_result(self, module_path: Path, test_code: str):
        """Cache test code for module."""
        if self.config.use_caching:
            cache_key = self.get_cache_key(module_path)
            self.cache[cache_key] = test_code