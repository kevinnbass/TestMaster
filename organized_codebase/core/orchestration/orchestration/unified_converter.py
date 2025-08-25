"""
Unified Converter Framework

This module provides a unified converter that consolidates all conversion
strategies into a single, powerful framework using the strategy pattern.

Author: Agent B - Orchestration & Workflow Specialist
Created: 2025-01-22
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .converter_base import (
    ConversionConfig, ConversionResult, ConversionMetrics,
    ConversionStatus, ConversionStrategy, ConverterStrategy
)
from .intelligent_strategy import IntelligentConverterStrategy
from .parallel_strategy import ParallelConverterStrategy
from .batch_strategy import BatchConverterStrategy


class UnifiedConverter:
    """
    Unified converter that manages all conversion strategies and provides
    a single interface for test generation and conversion operations.
    """
    
    def __init__(self, config: Optional[ConversionConfig] = None):
        self.config = config or ConversionConfig()
        self.logger = logging.getLogger(__name__)
        
        # Strategy registry
        self.strategies: Dict[ConversionStrategy, ConverterStrategy] = {}
        
        # Metrics tracking
        self.metrics = ConversionMetrics()
        
        # Execution state
        self.is_running = False
        self.current_tasks = []
        
        # Initialize strategies
        self._initialize_strategies()
        
        self.logger.info(f"UnifiedConverter initialized with strategy: {self.config.strategy.value}")
    
    def _initialize_strategies(self):
        """Initialize available conversion strategies"""
        # Register core strategies
        self.strategies[ConversionStrategy.INTELLIGENT] = IntelligentConverterStrategy(self.config)
        self.strategies[ConversionStrategy.PARALLEL] = ParallelConverterStrategy(self.config)
        self.strategies[ConversionStrategy.BATCH] = BatchConverterStrategy(self.config)
        
        # Additional strategies can be registered here
        self.logger.debug(f"Initialized {len(self.strategies)} conversion strategies")
    
    def register_strategy(self, strategy_type: ConversionStrategy, strategy: ConverterStrategy):
        """Register a custom conversion strategy"""
        self.strategies[strategy_type] = strategy
        self.logger.info(f"Registered strategy: {strategy_type.value}")
    
    def set_strategy(self, strategy: ConversionStrategy):
        """Change the active conversion strategy"""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        self.config.strategy = strategy
        self.logger.info(f"Switched to strategy: {strategy.value}")
    
    async def convert_file(self, file_path: Path) -> ConversionResult:
        """Convert a single file using the configured strategy"""
        start_time = time.time()
        
        try:
            # Get active strategy
            strategy = self._get_active_strategy()
            
            # Validate file
            is_valid, error_msg = strategy.validate_file(file_path)
            if not is_valid:
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.SKIPPED,
                    strategy_used=self.config.strategy,
                    error_message=error_msg,
                    duration=time.time() - start_time
                )
            
            # Check if already exists
            if strategy.check_exists(file_path):
                return ConversionResult(
                    file_path=file_path,
                    status=ConversionStatus.EXISTS,
                    strategy_used=self.config.strategy,
                    output_path=strategy.get_output_path(file_path),
                    duration=time.time() - start_time
                )
            
            # Perform conversion
            result = await strategy.convert(file_path)
            
            # Update metrics
            self._update_metrics(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Conversion failed for {file_path}: {e}")
            return ConversionResult(
                file_path=file_path,
                status=ConversionStatus.FAILED,
                strategy_used=self.config.strategy,
                error_message=str(e),
                duration=time.time() - start_time
            )
    
    async def convert_directory(self, directory: Path, 
                              recursive: bool = True) -> List[ConversionResult]:
        """Convert all eligible files in a directory"""
        self.logger.info(f"Converting directory: {directory} (recursive={recursive})")
        
        # Find files to convert
        files = self._find_convertible_files(directory, recursive)
        self.logger.info(f"Found {len(files)} files to convert")
        
        if not files:
            return []
        
        # Reset metrics
        self.metrics = ConversionMetrics(total_files=len(files))
        
        # Convert based on strategy
        strategy = self._get_active_strategy()
        
        if self.config.strategy in [ConversionStrategy.BATCH, ConversionStrategy.PARALLEL]:
            # Use batch conversion
            results = await strategy.batch_convert(files)
        else:
            # Convert one by one
            results = []
            for file_path in files:
                result = await self.convert_file(file_path)
                results.append(result)
                
                # Progress update
                if self.config.verbose and len(results) % 10 == 0:
                    self._print_progress(len(results), len(files))
        
        # Final metrics
        self._calculate_final_metrics()
        
        return results
    
    async def convert_with_coverage(self, file_paths: List[Path]) -> List[ConversionResult]:
        """Convert files with coverage analysis"""
        self.logger.info(f"Converting {len(file_paths)} files with coverage analysis")
        
        # Temporarily switch to coverage strategy
        original_strategy = self.config.strategy
        self.set_strategy(ConversionStrategy.COVERAGE)
        
        results = []
        for file_path in file_paths:
            result = await self.convert_file(file_path)
            results.append(result)
        
        # Restore original strategy
        self.set_strategy(original_strategy)
        
        return results
    
    def _get_active_strategy(self) -> ConverterStrategy:
        """Get the currently active conversion strategy"""
        strategy = self.strategies.get(self.config.strategy)
        if not strategy:
            raise ValueError(f"Strategy not initialized: {self.config.strategy}")
        return strategy
    
    def _find_convertible_files(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all files eligible for conversion"""
        files = []
        
        # Get file iterator
        if recursive:
            file_iterator = directory.rglob("*.py")
        else:
            file_iterator = directory.glob("*.py")
        
        # Filter files
        for file_path in file_iterator:
            # Skip if matches exclude pattern
            if any(file_path.match(pattern) for pattern in self.config.exclude_patterns):
                continue
            
            # Skip test files
            if file_path.name.startswith("test_"):
                continue
            
            # Skip __init__ files
            if file_path.name == "__init__.py":
                continue
            
            # Add to list
            files.append(file_path)
        
        return sorted(files)
    
    def _update_metrics(self, result: ConversionResult):
        """Update metrics based on conversion result"""
        if result.status == ConversionStatus.COMPLETED:
            self.metrics.completed += 1
        elif result.status == ConversionStatus.FAILED:
            self.metrics.failed += 1
            if result.error_message:
                self.metrics.errors.append(result.error_message)
        elif result.status in [ConversionStatus.SKIPPED, ConversionStatus.EXISTS]:
            self.metrics.skipped += 1
        
        self.metrics.total_duration += result.duration
    
    def _calculate_final_metrics(self):
        """Calculate final metrics after conversion"""
        if self.metrics.completed > 0:
            self.metrics.average_duration = self.metrics.total_duration / self.metrics.completed
        
        total_processed = self.metrics.completed + self.metrics.failed
        if total_processed > 0:
            self.metrics.success_rate = self.metrics.completed / total_processed
    
    def _print_progress(self, current: int, total: int):
        """Print conversion progress"""
        percentage = (current / total) * 100
        print(f"Progress: {current}/{total} ({percentage:.1f}%)")
    
    def get_metrics(self) -> ConversionMetrics:
        """Get current conversion metrics"""
        return self.metrics
    
    def print_summary(self):
        """Print conversion summary"""
        print("\n" + "=" * 60)
        print("CONVERSION SUMMARY")
        print("=" * 60)
        print(f"Strategy Used: {self.config.strategy.value}")
        print(f"Total Files: {self.metrics.total_files}")
        print(f"Completed: {self.metrics.completed}")
        print(f"Failed: {self.metrics.failed}")
        print(f"Skipped: {self.metrics.skipped}")
        print(f"Success Rate: {self.metrics.success_rate:.1%}")
        print(f"Average Duration: {self.metrics.average_duration:.2f}s")
        print(f"Total Duration: {self.metrics.total_duration:.2f}s")
        
        if self.metrics.errors:
            print(f"\nErrors ({len(self.metrics.errors)}):")
            for error in self.metrics.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.metrics.errors) > 5:
                print(f"  ... and {len(self.metrics.errors) - 5} more")


async def main():
    """Example usage of UnifiedConverter"""
    
    # Configure converter
    config = ConversionConfig(
        strategy=ConversionStrategy.INTELLIGENT,
        max_workers=4,
        batch_size=10,
        verbose=True
    )
    
    # Create converter
    converter = UnifiedConverter(config)
    
    # Convert a directory
    test_dir = Path("multi_coder_analysis")
    if test_dir.exists():
        results = await converter.convert_directory(test_dir, recursive=True)
        
        # Print summary
        converter.print_summary()
        
        # Show successful conversions
        successful = [r for r in results if r.status == ConversionStatus.COMPLETED]
        if successful:
            print(f"\nSuccessful Conversions ({len(successful)}):")
            for result in successful[:10]:
                print(f"  ✓ {result.file_path.name} -> {result.output_path.name}")
    else:
        print(f"Directory not found: {test_dir}")


if __name__ == "__main__":
    asyncio.run(main())