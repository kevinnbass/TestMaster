#!/usr/bin/env python3
"""
Batch Test Converter for TestMaster

Consolidates functionality from batch converter scripts:
- batch_convert_broken_tests.py
- week_5_8_batch_converter.py
- week_7_8_converter.py

Provides batch processing for test conversion with special handling for broken tests.
"""

import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .base import BatchConverter, ConversionResult, ConversionConfig
from .intelligent import IntelligentConverter

class BatchTestConverter(BatchConverter):
    """
    Batch test converter with special handling for broken tests and large datasets.
    """
    
    def __init__(self, 
                 mode: str = "auto",
                 model: str = None,
                 api_key: Optional[str] = None,
                 config: Optional[ConversionConfig] = None):
        """Initialize batch converter."""
        super().__init__(config)
        
        # Create intelligent converter for actual conversion work
        self.converter = IntelligentConverter(
            mode=mode,
            model=model,
            api_key=api_key,
            config=config
        )
        
        print(f"Initialized BatchTestConverter")
        print(f"  Mode: {self.converter.mode}")
        print(f"  Batch size: {self.config.batch_size}")
    
    def convert_module(self, module_path: Path) -> ConversionResult:
        """Convert a single module (delegates to intelligent converter)."""
        return self.converter.convert_module(module_path)
    
    def convert_broken_tests(self, test_directory: str = "tests") -> List[ConversionResult]:
        """Find and convert broken test files."""
        print(f"\n{'='*60}")
        print("BATCH CONVERSION - BROKEN TESTS")
        print('='*60)
        
        # Find broken test files
        broken_tests = self._find_broken_test_files(test_directory)
        
        if not broken_tests:
            print("No broken test files found!")
            return []
        
        print(f"Found {len(broken_tests)} broken test files")
        
        # Convert to modules for processing
        modules_to_fix = []
        for test_file in broken_tests:
            # Try to find corresponding module
            module_path = self._find_module_for_test(test_file)
            if module_path:
                modules_to_fix.append(module_path)
        
        if not modules_to_fix:
            print("No corresponding modules found for broken tests!")
            return []
        
        print(f"Converting {len(modules_to_fix)} modules to fix broken tests")
        
        # Convert in batches
        results = self.convert_modules_batch(modules_to_fix)
        
        # Print summary
        self._print_broken_test_summary(results, broken_tests)
        
        return results
    
    def convert_by_priority(self, priority_list: List[str]) -> List[ConversionResult]:
        """Convert modules based on priority list."""
        print(f"\n{'='*60}")
        print("BATCH CONVERSION - PRIORITY LIST")
        print('='*60)
        
        modules = []
        for priority_pattern in priority_list:
            if "/" in priority_pattern:
                # Specific file path
                module_path = Path(priority_pattern)
                if module_path.exists():
                    modules.append(module_path)
            else:
                # Pattern search
                base_dir = Path("multi_coder_analysis")
                if base_dir.exists():
                    for py_file in base_dir.rglob(f"*{priority_pattern}*.py"):
                        if py_file not in modules:
                            modules.append(py_file)
        
        if not modules:
            print("No modules found matching priority patterns!")
            return []
        
        print(f"Processing {len(modules)} priority modules")
        
        # Convert in batches
        results = self.convert_modules_batch(modules)
        
        return results
    
    def convert_week_range(self, start_week: int, end_week: int) -> List[ConversionResult]:
        """Convert modules for specific week range (legacy compatibility)."""
        print(f"\n{'='*60}")
        print(f"BATCH CONVERSION - WEEKS {start_week}-{end_week}")
        print('='*60)
        
        # Get all remaining modules
        all_modules = self.get_remaining_modules()
        
        if not all_modules:
            print("No modules need conversion!")
            return []
        
        # Calculate slice for week range
        total_modules = len(all_modules)
        modules_per_week = max(1, total_modules // 10)  # Assume 10 week project
        
        start_idx = (start_week - 1) * modules_per_week
        end_idx = min(end_week * modules_per_week, total_modules)
        
        week_modules = all_modules[start_idx:end_idx]
        
        print(f"Week {start_week}-{end_week} modules: {len(week_modules)}")
        print(f"Module range: {start_idx + 1}-{end_idx} of {total_modules}")
        
        # Convert in batches
        results = self.convert_modules_batch(week_modules)
        
        return results
    
    def convert_incremental_save(self, modules: List[Path], save_interval: int = 5) -> List[ConversionResult]:
        """Convert with incremental saving of results."""
        print(f"\n{'='*60}")
        print("BATCH CONVERSION - INCREMENTAL SAVE")
        print('='*60)
        
        all_results = []
        results_file = Path("conversion_results.json")
        
        # Load existing results if available
        if results_file.exists():
            try:
                import json
                with open(results_file, 'r') as f:
                    saved_data = json.load(f)
                    print(f"Loaded {len(saved_data)} previous results")
            except:
                saved_data = {}
        else:
            saved_data = {}
        
        processed_count = 0
        
        for i, module in enumerate(modules):
            # Check if already processed
            module_key = str(module)
            if module_key in saved_data:
                print(f"Skipping {module.name} (already processed)")
                continue
            
            # Convert module
            result = self.convert_module(module)
            all_results.append(result)
            processed_count += 1
            
            # Save result
            saved_data[module_key] = {
                "success": result.success,
                "test_path": result.test_path,
                "error_message": result.error_message,
                "execution_time": result.execution_time,
                "test_count": result.test_count,
                "timestamp": time.time()
            }
            
            status = "✅" if result.success else "❌"
            print(f"{status} {module.name} ({i+1}/{len(modules)})")
            
            # Incremental save
            if processed_count % save_interval == 0:
                self._save_results(results_file, saved_data)
                print(f"Saved results ({processed_count} processed)")
        
        # Final save
        self._save_results(results_file, saved_data)
        print(f"Final save completed ({processed_count} total processed)")
        
        return all_results
    
    def _find_broken_test_files(self, test_directory: str) -> List[Path]:
        """Find test files with syntax errors or import issues."""
        broken_tests = []
        test_dir = Path(test_directory)
        
        if not test_dir.exists():
            return broken_tests
        
        for test_file in test_dir.rglob("test_*.py"):
            try:
                # Try to parse the file
                content = test_file.read_text(encoding='utf-8')
                import ast
                ast.parse(content)
                
                # Try to import (basic check)
                # This is a simple heuristic - could be more sophisticated
                if "import" not in content or len(content) < 100:
                    broken_tests.append(test_file)
                    
            except (SyntaxError, UnicodeDecodeError):
                broken_tests.append(test_file)
            except Exception:
                # Other issues - might be broken
                broken_tests.append(test_file)
        
        return broken_tests
    
    def _find_module_for_test(self, test_file: Path) -> Optional[Path]:
        """Find the source module corresponding to a test file."""
        # Extract module name from test file
        test_name = test_file.stem
        if test_name.startswith("test_"):
            module_name = test_name[5:]  # Remove "test_" prefix
        else:
            module_name = test_name
        
        # Remove common suffixes
        module_name = module_name.replace("_intelligent", "")
        module_name = module_name.replace("_test", "")
        
        # Search for module
        base_dirs = ["multi_coder_analysis", "src", "."]
        
        for base_dir in base_dirs:
            base_path = Path(base_dir)
            if base_path.exists():
                # Direct match
                module_file = base_path / f"{module_name}.py"
                if module_file.exists():
                    return module_file
                
                # Recursive search
                for py_file in base_path.rglob(f"{module_name}.py"):
                    return py_file
        
        return None
    
    def _save_results(self, results_file: Path, data: Dict[str, Any]):
        """Save results to JSON file."""
        try:
            import json
            with open(results_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")
    
    def _print_broken_test_summary(self, results: List[ConversionResult], broken_tests: List[Path]):
        """Print summary for broken test conversion."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        print(f"\n{'='*60}")
        print("BROKEN TEST CONVERSION SUMMARY")
        print('='*60)
        print(f"Broken test files found: {len(broken_tests)}")
        print(f"Modules converted: {len(results)}")
        print(f"Successful conversions: {len(successful)}")
        print(f"Failed conversions: {len(failed)}")
        print(f"Success rate: {len(successful)/max(1,len(results))*100:.1f}%")
        
        if successful:
            print(f"\nFixed test files:")
            for result in successful[:5]:  # Show first 5
                module_name = Path(result.module_path).name
                print(f"  ✅ {module_name} → {result.test_path}")
            if len(successful) > 5:
                print(f"  ... and {len(successful) - 5} more")


def main():
    """Test the batch converter."""
    print("="*60)
    print("TestMaster Batch Converter")
    print("="*60)
    
    # Create converter
    config = ConversionConfig(
        batch_size=5,
        rate_limit_rpm=20,  # Slower for batch processing
        timeout_seconds=180
    )
    
    converter = BatchTestConverter(mode="auto", config=config)
    
    # Test broken test conversion
    results = converter.convert_broken_tests()
    
    if not results:
        # Try converting a few remaining modules
        modules = converter.get_remaining_modules()[:10]  # First 10
        if modules:
            print("\nNo broken tests found, converting sample modules...")
            results = converter.convert_modules_batch(modules)
    
    # Print final stats
    converter.print_stats()
    
    success_count = sum(1 for r in results if r.success)
    return 0 if success_count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())