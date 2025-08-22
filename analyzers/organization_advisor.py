#!/usr/bin/env python3
"""
Code Organization Advisor
=========================

Analyzes your codebase and suggests how to better organize files
into directories and modules for improved maintainability.
"""

import ast
import json
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple
from collections import defaultdict, Counter

class CodeOrganizationAdvisor:
    """Suggests better organization for your codebase"""
    
    def __init__(self):
        self.files = {}
        self.file_purposes = {}
        self.suggested_structure = {}
    
    def analyze_organization(self):
        """Analyze current organization and suggest improvements"""
        print("Code Organization Analysis")
        print("=" * 30)
        
        current_dir = Path.cwd()
        python_files = [f for f in current_dir.glob("*.py") if f.is_file()]
        
        print(f"Analyzing {len(python_files)} Python files...")
        print()
        
        # Analyze each file to understand its purpose
        for file_path in python_files:
            purpose = self.analyze_file_purpose(file_path)
            if purpose:
                self.file_purposes[file_path.name] = purpose
                self.files[file_path.name] = purpose
        
        # Generate organization suggestions
        self.suggest_directory_structure()
        self.suggest_module_groupings()
        self.identify_utility_files()
        self.suggest_config_organization()
        
        return self.suggested_structure
    
    def analyze_file_purpose(self, file_path: Path) -> Dict[str, Any]:
        """Analyze what a file's main purpose is"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return None
        
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {'type': 'syntax_error', 'confidence': 0}
        
        purpose = {
            'type': 'unknown',
            'confidence': 0,
            'indicators': [],
            'classes': [],
            'functions': [],
            'imports': [],
            'has_main': False,
            'line_count': len(content.split('\n'))
        }
        
        # Analyze content
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                purpose['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                purpose['functions'].append(node.name)
                if node.name == 'main':
                    purpose['has_main'] = True
            elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                import_info = self.extract_import_info(node)
                purpose['imports'].extend(import_info)
        
        # Determine file type based on patterns
        purpose = self.classify_file_purpose(file_path.name, purpose, content)
        
        return purpose
    
    def extract_import_info(self, node) -> List[str]:
        """Extract import information"""
        imports = []
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        return imports
    
    def classify_file_purpose(self, filename: str, purpose: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Classify the file's main purpose"""
        indicators = []
        
        # Check filename patterns
        if 'test' in filename.lower():
            purpose['type'] = 'test'
            indicators.append('filename contains "test"')
            purpose['confidence'] = 90
        elif 'config' in filename.lower() or 'settings' in filename.lower():
            purpose['type'] = 'config'
            indicators.append('filename suggests configuration')
            purpose['confidence'] = 85
        elif filename.startswith('__'):
            purpose['type'] = 'special'
            indicators.append('special Python file')
            purpose['confidence'] = 95
        elif 'util' in filename.lower() or 'helper' in filename.lower():
            purpose['type'] = 'utility'
            indicators.append('filename suggests utility functions')
            purpose['confidence'] = 80
        
        # Check imports for clues
        test_imports = ['unittest', 'pytest', 'mock', 'test']
        web_imports = ['flask', 'django', 'fastapi', 'tornado', 'bottle']
        data_imports = ['pandas', 'numpy', 'scipy', 'matplotlib']
        ml_imports = ['sklearn', 'tensorflow', 'torch', 'keras']
        
        import_strs = [imp.lower() for imp in purpose['imports']]
        
        if any(test_imp in ' '.join(import_strs) for test_imp in test_imports):
            purpose['type'] = 'test'
            indicators.append('imports testing frameworks')
            purpose['confidence'] = 95
        elif any(web_imp in ' '.join(import_strs) for web_imp in web_imports):
            purpose['type'] = 'web'
            indicators.append('imports web frameworks')
            purpose['confidence'] = 85
        elif any(data_imp in ' '.join(import_strs) for data_imp in data_imports):
            purpose['type'] = 'data_analysis'
            indicators.append('imports data analysis libraries')
            purpose['confidence'] = 80
        elif any(ml_imp in ' '.join(import_strs) for ml_imp in ml_imports):
            purpose['type'] = 'machine_learning'
            indicators.append('imports ML libraries')
            purpose['confidence'] = 80
        
        # Check content patterns
        if purpose['has_main'] and purpose['confidence'] < 50:
            purpose['type'] = 'script'
            indicators.append('has main function')
            purpose['confidence'] = 70
        
        # Check for class-heavy files
        if len(purpose['classes']) > 3 and purpose['confidence'] < 50:
            purpose['type'] = 'model'
            indicators.append(f"contains {len(purpose['classes'])} classes")
            purpose['confidence'] = 60
        
        # Check for function-heavy utility files
        if len(purpose['functions']) > 8 and len(purpose['classes']) == 0:
            purpose['type'] = 'utility'
            indicators.append(f"contains {len(purpose['functions'])} utility functions")
            purpose['confidence'] = 70
        
        # Check for analyzer/processor patterns
        if 'analy' in filename.lower() or 'process' in filename.lower():
            purpose['type'] = 'analyzer'
            indicators.append('filename suggests analysis/processing')
            purpose['confidence'] = 75
        
        purpose['indicators'] = indicators
        return purpose
    
    def suggest_directory_structure(self):
        """Suggest directory structure based on file purposes"""
        print("Suggested Directory Structure:")
        print("-" * 30)
        
        structure = defaultdict(list)
        
        for filename, purpose in self.file_purposes.items():
            file_type = purpose['type']
            
            if file_type == 'test':
                structure['tests/'].append(filename)
            elif file_type == 'config':
                structure['config/'].append(filename)
            elif file_type == 'utility':
                structure['utils/'].append(filename)
            elif file_type == 'web':
                structure['web/'].append(filename)
            elif file_type == 'data_analysis':
                structure['analysis/'].append(filename)
            elif file_type == 'machine_learning':
                structure['ml/'].append(filename)
            elif file_type == 'model':
                structure['models/'].append(filename)
            elif file_type == 'analyzer':
                structure['analyzers/'].append(filename)
            elif file_type == 'script':
                structure['scripts/'].append(filename)
            elif file_type == 'special':
                structure['./'].append(filename)  # Keep in root
            else:
                structure['core/'].append(filename)  # Default location
        
        # Print suggested structure
        for directory, files in sorted(structure.items()):
            if files:
                print(f"{directory}")
                for file in sorted(files):
                    confidence = self.file_purposes[file]['confidence']
                    print(f"  {file} (confidence: {confidence}%)")
                print()
        
        self.suggested_structure['directories'] = dict(structure)
        
        # Suggest further subdivisions for large directories
        self.suggest_subdivisions(structure)
    
    def suggest_subdivisions(self, structure: Dict[str, List[str]]):
        """Suggest subdivisions for directories with many files"""
        print("Subdivision suggestions for large directories:")
        print("-" * 45)
        
        for directory, files in structure.items():
            if len(files) > 6:  # More than 6 files might benefit from subdivision
                print(f"{directory} ({len(files)} files):")
                
                # Group by similar naming patterns
                groups = self.group_by_patterns(files)
                if len(groups) > 1:
                    print("  Could be organized as:")
                    for group_name, group_files in groups.items():
                        subdir = f"{directory}{group_name}/"
                        print(f"    {subdir}")
                        for file in group_files:
                            print(f"      {file}")
                    print()
                else:
                    print("  Files seem related - subdivision may not be needed")
                    print()
    
    def group_by_patterns(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by common naming patterns"""
        groups = defaultdict(list)
        
        for file in files:
            # Remove .py extension for pattern matching
            name = file.replace('.py', '')
            
            # Group by common prefixes (if underscore separated)
            if '_' in name:
                prefix = name.split('_')[0]
                if len(prefix) > 2:  # Meaningful prefix
                    groups[prefix].append(file)
                    continue
            
            # Group by common suffixes
            common_suffixes = ['analyzer', 'processor', 'handler', 'manager', 'service', 'util', 'helper']
            grouped = False
            for suffix in common_suffixes:
                if suffix in name.lower():
                    groups[suffix].append(file)
                    grouped = True
                    break
            
            if not grouped:
                groups['misc'].append(file)
        
        # Only return groups with meaningful groupings
        meaningful_groups = {k: v for k, v in groups.items() if len(v) > 1 or k != 'misc'}
        if not meaningful_groups and groups.get('misc'):
            meaningful_groups['misc'] = groups['misc']
        
        return meaningful_groups
    
    def suggest_module_groupings(self):
        """Suggest how files could be grouped into modules"""
        print("Module Grouping Suggestions:")
        print("-" * 28)
        
        # Find files that might work well together
        related_files = self.find_related_files()
        
        for group_name, files in related_files.items():
            if len(files) > 1:
                print(f"{group_name} module:")
                for file in files:
                    purpose = self.file_purposes[file]
                    print(f"  {file} - {purpose['type']} ({purpose['confidence']}% confidence)")
                print()
        
        if not related_files:
            print("No obvious module groupings found.")
            print("Consider grouping files by functionality as your codebase grows.")
            print()
    
    def find_related_files(self) -> Dict[str, List[str]]:
        """Find files that might be related and could form modules"""
        groups = defaultdict(list)
        
        # Group by similar purposes and naming
        purpose_groups = defaultdict(list)
        for filename, purpose in self.file_purposes.items():
            if purpose['type'] not in ['special', 'test']:
                purpose_groups[purpose['type']].append(filename)
        
        # Only keep groups with multiple files
        for purpose, files in purpose_groups.items():
            if len(files) > 1:
                groups[purpose] = files
        
        return dict(groups)
    
    def identify_utility_files(self):
        """Identify files that are pure utilities"""
        print("Utility File Analysis:")
        print("-" * 22)
        
        utility_files = []
        for filename, purpose in self.file_purposes.items():
            if (purpose['type'] == 'utility' or 
                (len(purpose['functions']) > 5 and len(purpose['classes']) == 0)):
                utility_files.append((filename, purpose))
        
        if utility_files:
            print("Files that appear to be utilities:")
            for filename, purpose in utility_files:
                print(f"  {filename}: {len(purpose['functions'])} functions")
            print("\nConsider:")
            print("  - Moving these to a utils/ directory")
            print("  - Breaking large utility files into focused modules")
            print("  - Creating clear utility categories (file_utils, string_utils, etc.)")
        else:
            print("No obvious utility files found.")
        print()
    
    def suggest_config_organization(self):
        """Suggest how to organize configuration"""
        print("Configuration Organization:")
        print("-" * 26)
        
        config_files = [f for f, p in self.file_purposes.items() if p['type'] == 'config']
        
        if config_files:
            print("Configuration files found:")
            for file in config_files:
                print(f"  {file}")
            print("\nConsider:")
            print("  - Moving to config/ directory")
            print("  - Using consistent config format (JSON, YAML, or .py)")
            print("  - Separating dev/prod configurations")
        else:
            print("No obvious configuration files found.")
            print("Consider creating a config/ directory for settings as needed.")
        print()
    
    def save_results(self, filename: str = "organization_analysis.json"):
        """Save analysis results"""
        results = {
            'file_purposes': self.file_purposes,
            'suggested_structure': self.suggested_structure
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Organization analysis saved to {filename}")

def main():
    """Main analysis function"""
    print("Personal Code Organization Advisor")
    print("Suggestions for better file organization")
    print()
    
    advisor = CodeOrganizationAdvisor()
    advisor.analyze_organization()
    advisor.save_results()

if __name__ == "__main__":
    main()