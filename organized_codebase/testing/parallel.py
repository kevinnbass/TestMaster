#!/usr/bin/env python3
"""
Parallel Test Converter for TestMaster

Consolidates functionality from parallel converter scripts:
- parallel_converter.py
- parallel_converter_fixed.py
- parallel_converter_working.py
- accelerated_converter.py
- turbo_converter.py

Provides high-performance parallel test conversion with rate limiting and optimization.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import ParallelConverter, ConversionResult, ConversionConfig
from .intelligent import IntelligentConverter

class ParallelTestConverter(ParallelConverter):
    """
    High-performance parallel test converter.
    
    Optimized for batch processing with rate limiting and intelligent fallbacks.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None,
                 api_key: Optional[str] = None,
                 config: Optional[ConversionConfig] = None):
        """
        Initialize parallel converter.
        
        Args:
            mode: AI mode ("provider", "sdk", "template", "auto")
            model: AI model to use
            api_key: API key for AI services
            config: Conversion configuration
        """
        super().__init__(config)
        
        # Create intelligent converter for actual conversion work
        self.converter = IntelligentConverter(
            mode=mode,
            model=model,
            api_key=api_key,
            config=config
        )
        
        print(f"Initialized ParallelTestConverter")
        print(f"  Mode: {self.converter.mode}")
        print(f"  Workers: {self.config.max_workers}")
        print(f"  Rate limit: {self.config.rate_limit_rpm} RPM")
    
    def convert_module(self, module_path: Path) -> ConversionResult:
        """Convert a single module (delegates to intelligent converter)."""
        return self.converter.convert_module(module_path)
    
    def convert_all_remaining(self, base_directory: str = "multi_coder_analysis") -> List[ConversionResult]:
        """Convert all remaining modules in parallel."""
        print(f"\n{'='*60}")
        print("PARALLEL CONVERSION - ALL REMAINING MODULES")
        print('='*60)
        
        # Get remaining modules
        modules = self.get_remaining_modules(base_directory)
        
        if not modules:
            print("No modules need conversion!")
            return []
        
        print(f"Found {len(modules)} modules to convert")
        print(f"Estimated time: {len(modules) * 10 / self.config.max_workers / 60:.1f} minutes")
        
        # Convert in parallel
        results = self.convert_modules_parallel(modules)
        
        # Print summary
        self._print_conversion_summary(results)
        
        return results
    
    def convert_priority_modules(self, priority_patterns: List[str] = None) -> List[ConversionResult]:
        """Convert priority modules first."""
        if priority_patterns is None:
            priority_patterns = [
                "**/core/**",
                "**/utils/**", 
                "**/generators/**",
                "**/llm_providers/**"
            ]
        
        print(f"\n{'='*60}")
        print("PARALLEL CONVERSION - PRIORITY MODULES")
        print('='*60)
        
        all_modules = self.get_remaining_modules()
        priority_modules = []
        
        # Filter for priority modules
        for pattern in priority_patterns:
            for module in all_modules:
                if module.match(pattern) and module not in priority_modules:
                    priority_modules.append(module)
        
        if not priority_modules:
            print("No priority modules found!")
            return []
        
        print(f"Found {len(priority_modules)} priority modules")
        
        # Convert priority modules
        results = self.convert_modules_parallel(priority_modules)
        
        # Print summary
        self._print_conversion_summary(results)
        
        return results
    
    def convert_batch_with_progress(self, modules: List[Path], batch_size: int = None) -> List[ConversionResult]:
        """Convert modules in batches with detailed progress reporting."""
        if batch_size is None:
            batch_size = self.config.max_workers * 2
        
        print(f"\n{'='*60}")
        print("PARALLEL BATCH CONVERSION WITH PROGRESS")
        print('='*60)
        
        total_modules = len(modules)
        total_batches = (total_modules + batch_size - 1) // batch_size
        all_results = []
        
        print(f"Total modules: {total_modules}")
        print(f"Batch size: {batch_size}")
        print(f"Total batches: {total_batches}")
        print(f"Workers per batch: {self.config.max_workers}")
        
        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_modules)
            batch_modules = modules[start_idx:end_idx]
            
            print(f"\n--- Batch {batch_num + 1}/{total_batches} ---")
            print(f"Modules {start_idx + 1}-{end_idx} of {total_modules}")
            print(f"Processing {len(batch_modules)} modules...")
            
            batch_start = time.time()
            batch_results = self.convert_modules_parallel(batch_modules)
            batch_time = time.time() - batch_start
            
            # Batch statistics
            successful = sum(1 for r in batch_results if r.success)
            failed = len(batch_results) - successful
            
            print(f"Batch {batch_num + 1} completed in {batch_time:.1f}s")
            print(f"  Successful: {successful}")
            print(f"  Failed: {failed}")
            print(f"  Success rate: {successful/len(batch_results)*100:.1f}%")
            
            all_results.extend(batch_results)
            
            # Overall progress
            completed = len(all_results)
            overall_success = sum(1 for r in all_results if r.success)
            progress = completed / total_modules * 100
            
            print(f"Overall progress: {completed}/{total_modules} ({progress:.1f}%)")
            print(f"Overall success rate: {overall_success/completed*100:.1f}%")
            
            # Brief pause between batches (except for last batch)
            if batch_num < total_batches - 1:
                print("Pausing between batches...")
                time.sleep(2)
        
        # Final summary
        self._print_conversion_summary(all_results)
        
        return all_results
    
    def convert_with_retry(self, modules: List[Path], max_retries: int = 2) -> List[ConversionResult]:
        """Convert modules with automatic retry for failures."""
        print(f"\n{'='*60}")
        print("PARALLEL CONVERSION WITH RETRY")
        print('='*60)
        
        print(f"Initial conversion of {len(modules)} modules...")
        
        # First attempt
        results = self.convert_modules_parallel(modules)
        
        # Find failures
        failed_modules = [
            Path(r.module_path) for r in results 
            if not r.success
        ]
        
        # Retry failed modules
        retry_count = 0
        while failed_modules and retry_count < max_retries:
            retry_count += 1
            print(f"\nRetry {retry_count}/{max_retries} for {len(failed_modules)} failed modules...")
            
            # Brief delay before retry
            time.sleep(5)
            
            # Retry with reduced parallelism
            original_workers = self.config.max_workers
            self.config.max_workers = max(1, original_workers // 2)
            
            retry_results = self.convert_modules_parallel(failed_modules)
            
            # Restore original worker count
            self.config.max_workers = original_workers
            
            # Update results
            for i, result in enumerate(results):
                if not result.success:
                    # Find corresponding retry result
                    module_path = result.module_path
                    for retry_result in retry_results:
                        if retry_result.module_path == module_path:
                            results[i] = retry_result
                            break
            
            # Update failed modules list
            failed_modules = [
                Path(r.module_path) for r in results 
                if not r.success
            ]
            
            if failed_modules:
                print(f"Still {len(failed_modules)} failures after retry {retry_count}")
            else:
                print(f"All modules successful after retry {retry_count}!")
                break
        
        # Final summary
        self._print_conversion_summary(results)
        
        return results
    
    def _print_conversion_summary(self, results: List[ConversionResult]):
        """Print detailed conversion summary."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_time = sum(r.execution_time for r in results)
        total_tests = sum(r.test_count for r in successful)
        avg_quality = sum(r.quality_score for r in successful) / max(1, len(successful))
        
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print('='*60)
        print(f"Total modules: {len(results)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
        print(f"Total tests generated: {total_tests}")
        print(f"Average quality score: {avg_quality:.1f}")
        print(f"Total execution time: {total_time:.1f}s")
        print(f"Average time per module: {total_time/len(results):.1f}s")
        
        if failed:
            print(f"\nFailed modules:")
            for result in failed[:10]:  # Show first 10 failures
                module_name = Path(result.module_path).name
                print(f"  ❌ {module_name}: {result.error_message}")
            if len(failed) > 10:
                print(f"  ... and {len(failed) - 10} more")


def main():
    """Test the parallel converter."""
    print("="*60)
    print("TestMaster Parallel Converter")
    print("="*60)
    
    # Create converter with optimized settings
    config = ConversionConfig(
        max_workers=4,
        rate_limit_rpm=30,
        timeout_seconds=120,
        batch_size=8
    )
    
    converter = ParallelTestConverter(mode="auto", config=config)
    
    # Test on priority modules
    results = converter.convert_priority_modules()
    
    # Print final stats
    converter.print_stats()
    
    success_count = sum(1 for r in results if r.success)
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())