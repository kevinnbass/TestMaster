"""
Agent C Test Splitter - Splits the 18,164-line test file into organized modules
"""

import re
import os
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

class TestFileSplitter:
    """Splits massive test file into organized, modular test files"""
    
    def __init__(self, input_file: str, output_dir: str = "tests"):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.test_inventory = defaultdict(list)
        self.module_tests = defaultdict(list)
        self.class_tests = defaultdict(list)
        self.stats = {
            'total_tests': 0,
            'valid_tests': 0,
            'stub_tests': 0,
            'duplicate_tests': 0,
            'modules_found': set(),
            'classes_found': set(),
            'files_created': 0
        }
        
    def analyze_file(self) -> Dict:
        """Analyze the test file and create inventory"""
        print(f"Analyzing {self.input_file}...")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_module = None
        current_class = None
        current_test = []
        current_test_name = None
        in_test = False
        line_num = 0
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            # Detect module test class
            module_match = re.match(r'^class Test(.+?)_Tests:', line)
            if module_match:
                if current_test and current_test_name:
                    self._save_test(current_module, current_class, current_test_name, current_test)
                
                current_module = module_match.group(1).lower()
                current_class = None
                current_test = []
                current_test_name = None
                in_test = False
                self.stats['modules_found'].add(current_module)
                continue
            
            # Detect nested class
            nested_class_match = re.match(r'^\s+class Test(.+?)_Tests:', line)
            if nested_class_match:
                if current_test and current_test_name:
                    self._save_test(current_module, current_class, current_test_name, current_test)
                
                current_class = nested_class_match.group(1)
                current_test = []
                current_test_name = None
                in_test = False
                self.stats['classes_found'].add(current_class)
                continue
            
            # Detect test method
            test_match = re.match(r'\s+def (test_.+?)\(', line)
            if test_match:
                if current_test and current_test_name:
                    self._save_test(current_module, current_class, current_test_name, current_test)
                
                current_test_name = test_match.group(1)
                current_test = [line]
                in_test = True
                self.stats['total_tests'] += 1
                continue
            
            # Collect test lines
            if in_test:
                # Check if we've reached the next method or class
                if re.match(r'^\s+def ', line) or re.match(r'^class ', line):
                    if current_test and current_test_name:
                        self._save_test(current_module, current_class, current_test_name, current_test)
                    current_test = []
                    current_test_name = None
                    in_test = False
                else:
                    current_test.append(line)
        
        # Save last test if exists
        if current_test and current_test_name:
            self._save_test(current_module, current_class, current_test_name, current_test)
        
        return self.stats
    
    def _save_test(self, module: str, class_name: str, test_name: str, test_lines: List[str]):
        """Save a test to the appropriate collection"""
        test_content = ''.join(test_lines)
        
        # Determine test quality
        if 'assert' in test_content and '# Unknown assertion type' in test_content:
            self.stats['stub_tests'] += 1
            quality = 'stub'
        elif 'assert result == None' in test_content:
            quality = 'basic'
        else:
            self.stats['valid_tests'] += 1
            quality = 'valid'
        
        # Create test entry
        test_entry = {
            'name': test_name,
            'lines': test_lines,
            'quality': quality,
            'class': class_name,
            'module': module
        }
        
        # Store by module
        if module:
            self.module_tests[module].append(test_entry)
        
        # Store by class if exists
        if class_name:
            self.class_tests[class_name].append(test_entry)
    
    def split_into_files(self, max_lines: int = 300):
        """Split tests into organized files under 300 lines each"""
        print(f"\nSplitting tests into files (max {max_lines} lines each)...")
        
        # Create output directory structure
        self._create_directory_structure()
        
        # Process each module
        for module_name, tests in self.module_tests.items():
            self._create_module_test_files(module_name, tests, max_lines)
        
        print(f"\nCreated {self.stats['files_created']} test files")
        
    def _create_directory_structure(self):
        """Create the test directory structure"""
        dirs = [
            self.output_dir / 'unit' / 'layer2',
            self.output_dir / 'unit' / 'quality',
            self.output_dir / 'unit' / 'async',
            self.output_dir / 'unit' / 'communication',
            self.output_dir / 'unit' / 'converters',
            self.output_dir / 'unit' / 'test_generation',
            self.output_dir / 'integration',
            self.output_dir / 'fixtures',
            self.output_dir / 'utils'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _categorize_module(self, module_name: str) -> str:
        """Determine which category a module belongs to"""
        module_lower = module_name.lower()
        
        if 'layer2' in module_lower or 'integration' in module_lower:
            return 'layer2'
        elif any(x in module_lower for x in ['quality', 'benchmark', 'inspector', 'monitor', 'scoring', 'validation']):
            return 'quality'
        elif any(x in module_lower for x in ['async', 'concurrent', 'thread', 'executor', 'scheduler']):
            return 'async'
        elif any(x in module_lower for x in ['message', 'claude', 'handoff', 'queue', 'tag', 'failure']):
            return 'communication'
        elif any(x in module_lower for x in ['convert', 'cache', 'batch', 'parallel']):
            return 'converters'
        elif any(x in module_lower for x in ['thought', 'generation', 'healing', 'intelligence']):
            return 'test_generation'
        else:
            return 'misc'
    
    def _create_module_test_files(self, module_name: str, tests: List[Dict], max_lines: int):
        """Create test files for a module, splitting if necessary"""
        category = self._categorize_module(module_name)
        
        # Group tests by quality
        valid_tests = [t for t in tests if t['quality'] == 'valid']
        basic_tests = [t for t in tests if t['quality'] == 'basic']
        stub_tests = [t for t in tests if t['quality'] == 'stub']
        
        # Create files
        if valid_tests or basic_tests:
            self._write_test_file(category, module_name, valid_tests + basic_tests, max_lines, 'main')
        
        if stub_tests and len(stub_tests) > 20:  # Only create stub file if significant number
            self._write_test_file(category, module_name, stub_tests, max_lines, 'stubs')
    
    def _write_test_file(self, category: str, module_name: str, tests: List[Dict], 
                         max_lines: int, suffix: str = 'main'):
        """Write tests to a file, splitting if necessary"""
        if not tests:
            return
        
        # Calculate directory path
        if category == 'misc':
            dir_path = self.output_dir / 'unit' / 'misc'
        else:
            dir_path = self.output_dir / 'unit' / category
        
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split tests into chunks if needed
        chunks = self._split_into_chunks(tests, max_lines)
        
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                filename = f"test_{module_name}_{suffix}_part{i+1}.py"
            else:
                filename = f"test_{module_name}_{suffix}.py" if suffix != 'main' else f"test_{module_name}.py"
            
            file_path = dir_path / filename
            
            # Write file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('"""Auto-generated test file - needs review and improvement"""\n\n')
                f.write('import pytest\n')
                f.write('# TODO: Add proper imports\n\n')
                
                # Group by class
                class_groups = defaultdict(list)
                for test in chunk:
                    class_groups[test.get('class', '')].append(test)
                
                for class_name, class_tests in class_groups.items():
                    if class_name:
                        f.write(f'\nclass Test{class_name}:\n')
                        f.write(f'    """Tests for {class_name} class"""\n\n')
                        indent = '    '
                    else:
                        f.write(f'\nclass Test{module_name.title()}:\n')
                        f.write(f'    """Tests for {module_name} module"""\n\n')
                        indent = '    '
                    
                    for test in class_tests:
                        for line in test['lines']:
                            if not line.startswith('    '):
                                f.write(indent + line)
                            else:
                                f.write(line)
                        f.write('\n')
            
            self.stats['files_created'] += 1
            print(f"  Created: {file_path}")
    
    def _split_into_chunks(self, tests: List[Dict], max_lines: int) -> List[List[Dict]]:
        """Split tests into chunks under max_lines"""
        chunks = []
        current_chunk = []
        current_lines = 0
        
        for test in tests:
            test_lines = len(test['lines'])
            
            if current_lines + test_lines > max_lines and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_lines = 0
            
            current_chunk.append(test)
            current_lines += test_lines
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def generate_report(self):
        """Generate a detailed report of the split operation"""
        report_path = self.output_dir / 'SPLIT_REPORT.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('# Test File Split Report\n\n')
            f.write('## Statistics\n')
            f.write(f'- Total tests analyzed: {self.stats["total_tests"]}\n')
            f.write(f'- Valid tests: {self.stats["valid_tests"]}\n')
            f.write(f'- Stub tests: {self.stats["stub_tests"]}\n')
            f.write(f'- Modules found: {len(self.stats["modules_found"])}\n')
            f.write(f'- Classes found: {len(self.stats["classes_found"])}\n')
            f.write(f'- Files created: {self.stats["files_created"]}\n\n')
            
            f.write('## Modules Processed\n')
            for module in sorted(self.stats['modules_found']):
                test_count = len(self.module_tests.get(module, []))
                f.write(f'- {module}: {test_count} tests\n')
            
            f.write('\n## Next Steps\n')
            f.write('1. Review and fix stub tests\n')
            f.write('2. Add proper imports\n')
            f.write('3. Fix invalid assertions\n')
            f.write('4. Add fixtures and mocks\n')
            f.write('5. Run tests and fix failures\n')
        
        print(f"\nReport saved to: {report_path}")

def main():
    """Main execution"""
    import sys
    
    # Allow passing input file as argument
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = r"C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster\archive\test_tot_output_original_18164_lines_20250821_040109.py"
    
    # Choose output dir based on input file
    if 'test_misc' in input_file:
        output_dir = r"C:\Users\kbass\OneDrive\Documents\testmaster\tests_misc"
    elif 'test_uncategorized' in input_file:
        output_dir = r"C:\Users\kbass\OneDrive\Documents\testmaster\tests_uncategorized"
    else:
        output_dir = r"C:\Users\kbass\OneDrive\Documents\testmaster\tests"
    
    splitter = TestFileSplitter(input_file, output_dir)
    
    # Analyze the file
    stats = splitter.analyze_file()
    
    print("\nAnalysis Complete:")
    print(f"  Total tests: {stats['total_tests']}")
    print(f"  Valid tests: {stats['valid_tests']}")
    print(f"  Stub tests: {stats['stub_tests']}")
    print(f"  Modules: {len(stats['modules_found'])}")
    
    # Split into files
    splitter.split_into_files(max_lines=250)
    
    # Generate report
    splitter.generate_report()
    
    print("\nTest file splitting complete!")

if __name__ == "__main__":
    main()