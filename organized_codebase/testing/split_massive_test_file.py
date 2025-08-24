#!/usr/bin/env python3
"""
TestMaster Test File Modularization Tool
Splits the massive test_tot_output.py (18,164 lines) into organized modules
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

class TestFileModularizer:
    """Splits massive test files into organized, maintainable modules"""
    
    def __init__(self, source_file: str, output_dir: str):
        self.source_file = source_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create category mappings for logical organization
        self.category_mappings = {
            'integration': ['layer2_integration', 'async_executor', 'async_monitor', 'async_state_manager'],
            'quality': ['benchmarking_suite', 'quality_inspector', 'quality_monitor', 'scoring_system', 'validation_engine'],
            'analysis': ['coverage_analyzer', 'dependency_resolver', 'flow_analyzer', 'structure_mapper'],
            'execution': ['concurrent_scheduler', 'thread_pool_manager', 'parallel_executor', 'execution_router'],
            'messaging': ['claude_messenger', 'dynamic_handoff', 'message_queue', 'failure_detector'],
            'core': ['context_manager', 'feature_flags', 'shared_state', 'tracking_manager', 'orchestrator'],
            'language': ['language_parsers', 'universal_ast', 'framework_adapters', 'universal_detector'],
            'generation': ['test_generator', 'universal_test', 'intelligent', 'test_thought_generator'],
            'reasoning': ['tot_reasoning', 'universal_tot_integration'],
            'monitoring': ['dependency_tracker', 'file_watcher', 'idle_detector', 'test_monitor', 'test_scheduler'],
            'tagging': ['tag_reader', 'file_tagger'],
            'workflow': ['workflow_graph', 'handoff_manager', 'work_distributor'],
            'intelligence': ['coverage_intelligence', 'investigator'],
            'performance': ['performance_dashboard', 'performance_monitor', 'system_profiler'],
            'reporting': ['regression_tracker', 'dashboard_builder', 'data_collector', 'metrics_analyzer', 'report_generator'],
            'streaming': ['collaborative_generator', 'incremental_enhancer', 'live_feedback', 'stream_generator', 'stream_monitor'],
            'telemetry': ['telemetry_collector', 'telemetry_dashboard', 'alert_system'],
            'dashboard': ['dashboard', 'metrics_display'],
            'base': ['base', '__init__'],
            'self_healing': ['self_healing']
        }
        
        # Reverse mapping for quick lookup
        self.test_to_category = {}
        for category, tests in self.category_mappings.items():
            for test in tests:
                self.test_to_category[test] = category
    
    def extract_test_classes(self) -> List[Tuple[str, int, int]]:
        """Extract all test classes with their line ranges"""
        test_classes = []
        
        with open(self.source_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Find all test class start lines
        class_starts = []
        for i, line in enumerate(lines):
            if line.strip().startswith('class Test') and line.strip().endswith('Tests:'):
                class_name = line.strip().replace('class ', '').replace(':', '')
                class_starts.append((class_name, i + 1))  # 1-based line numbers
        
        # Calculate end lines for each class
        for i, (class_name, start_line) in enumerate(class_starts):
            if i < len(class_starts) - 1:
                end_line = class_starts[i + 1][1] - 1
            else:
                end_line = len(lines)
            
            test_classes.append((class_name, start_line, end_line))
        
        return test_classes
    
    def categorize_test(self, class_name: str) -> str:
        """Categorize a test class based on its name"""
        # Extract the core name (remove Test prefix and _Tests suffix)
        core_name = class_name.replace('Test', '').replace('_Tests', '').lower()
        
        # Handle special cases
        if core_name == '' or core_name == '__init__':
            return 'base'
        
        # Look for exact matches first
        if core_name in self.test_to_category:
            return self.test_to_category[core_name]
        
        # Look for partial matches
        for test_pattern, category in self.test_to_category.items():
            if test_pattern in core_name or core_name in test_pattern:
                return category
        
        # Default categorization based on keywords
        if any(keyword in core_name for keyword in ['async', 'thread', 'parallel', 'concurrent']):
            return 'execution'
        elif any(keyword in core_name for keyword in ['monitor', 'watch', 'detect']):
            return 'monitoring'
        elif any(keyword in core_name for keyword in ['dashboard', 'report', 'metrics', 'display']):
            return 'reporting'
        elif any(keyword in core_name for keyword in ['generate', 'create', 'build']):
            return 'generation'
        elif any(keyword in core_name for keyword in ['intelligence', 'smart', 'ai']):
            return 'intelligence'
        else:
            return 'misc'
    
    def create_module_file(self, category: str, test_classes: List[Tuple[str, int, int]], source_lines: List[str]):
        """Create a modular test file for a specific category"""
        output_file = self.output_dir / f"test_{category}.py"
        
        # Create module header
        header = f'''"""
TestMaster {category.title()} Tests
Generated from test_tot_output.py modularization
Category: {category}
"""

import pytest

'''
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(header)
            
            for class_name, start_line, end_line in test_classes:
                # Write the class (convert to 0-based indexing for slicing)
                class_lines = source_lines[start_line-1:end_line]
                f.writelines(class_lines)
                f.write('\n')  # Add spacing between classes
        
        print(f"Created {output_file} with {len(test_classes)} test classes")
        return len(test_classes)
    
    def split_file(self):
        """Split the massive test file into organized modules"""
        print(f"Modularizing {self.source_file}...")
        
        # Read all lines
        with open(self.source_file, 'r', encoding='utf-8') as f:
            source_lines = f.readlines()
        
        print(f"Total lines: {len(source_lines)}")
        
        # Extract all test classes
        test_classes = self.extract_test_classes()
        print(f"Found {len(test_classes)} test classes")
        
        # Group by category
        categories = {}
        for class_name, start_line, end_line in test_classes:
            category = self.categorize_test(class_name)
            if category not in categories:
                categories[category] = []
            categories[category].append((class_name, start_line, end_line))
        
        print(f"Organized into {len(categories)} categories:")
        
        total_classes_written = 0
        files_created = 0
        
        # Create module files
        for category, classes in categories.items():
            print(f"  {category}: {len(classes)} classes")
            classes_written = self.create_module_file(category, classes, source_lines)
            total_classes_written += classes_written
            files_created += 1
        
        # Create an __init__.py file for the test modules
        init_file = self.output_dir / "__init__.py"
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write('"""TestMaster Modularized Tests"""\n')
        
        # Create a summary report
        summary_file = self.output_dir / "MODULARIZATION_SUMMARY.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"""# TestMaster Test Modularization Summary

## Original File
- File: {self.source_file}
- Lines: {len(source_lines):,}
- Test Classes: {len(test_classes)}

## Modularization Results
- Files Created: {files_created}
- Categories: {len(categories)}
- Total Classes Processed: {total_classes_written}

## Category Breakdown
""")
            for category, classes in sorted(categories.items()):
                f.write(f"- **{category}**: {len(classes)} classes\n")
                for class_name, _, _ in classes[:5]:  # Show first 5 classes
                    f.write(f"  - {class_name}\n")
                if len(classes) > 5:
                    f.write(f"  - ... and {len(classes) - 5} more\n")
                f.write("\n")
        
        print(f"\nModularization Complete!")
        print(f"- Created {files_created} modular test files")
        print(f"- Processed {total_classes_written} test classes")
        print(f"- Average: {len(source_lines) // files_created:.0f} lines per module")
        print(f"- Summary: {summary_file}")
        
        return files_created, total_classes_written


def main():
    """Main execution"""
    source_file = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/test_tot_output.py"
    output_dir = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/tests/modularized"
    
    if not os.path.exists(source_file):
        print(f"Error: Source file not found: {source_file}")
        return
    
    modularizer = TestFileModularizer(source_file, output_dir)
    files_created, classes_processed = modularizer.split_file()
    
    print(f"\n🎉 Successfully modularized massive test file!")
    print(f"Original: 1 file with 18,164 lines")
    print(f"Result: {files_created} files with manageable sizes")


if __name__ == "__main__":
    main()