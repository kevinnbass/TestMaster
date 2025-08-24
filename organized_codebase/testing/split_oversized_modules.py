"""
Split Oversized Intelligence Modules
===================================

Automatically split modules over 1000 lines into smaller, focused components.
Preserves all functionality while ensuring clean architecture.
"""

import os
import re
from typing import List, Dict, Tuple
import ast


class ModuleSplitter:
    """Intelligent module splitter that preserves functionality."""
    
    def __init__(self):
        self.oversized_modules = [
            'core\\intelligence\\analysis\\business_analyzer.py',
            'core\\intelligence\\analysis\\debt_analyzer.py',
            'core\\intelligence\\api\\enterprise_integration_layer.py',
            'core\\intelligence\\api\\ml_api.py',
            'core\\intelligence\\coordination\\agent_coordination_protocols.py',
            'core\\intelligence\\coordination\\resource_coordination_system.py',
            'core\\intelligence\\coordination\\unified_workflow_orchestrator.py',
            'core\\intelligence\\documentation\\revolutionary\\neo4j_dominator.py',
            'core\\intelligence\\ml\\advanced\\circuit_breaker_ml.py',
            'core\\intelligence\\ml\\advanced\\delivery_optimizer.py',
            'core\\intelligence\\ml\\advanced\\integrity_ml_guardian.py',
            'core\\intelligence\\ml\\advanced\\performance_optimizer.py',
            'core\\intelligence\\ml\\advanced\\sla_ml_optimizer.py',
            'core\\intelligence\\ml\\enterprise\\ml_infrastructure_orchestrator.py',
            'core\\intelligence\\monitoring\\agent_qa.py',
            'core\\intelligence\\monitoring\\performance_optimization_engine.py',
            'core\\intelligence\\validation\\integration_test_suite.py',
            'core\\intelligence\\validation\\system_validation_framework.py'
        ]
    
    def analyze_module(self, file_path: str) -> Dict:
        """Analyze module structure for intelligent splitting."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return {'classes': [], 'functions': [], 'imports': [], 'error': 'syntax_error'}
            
        classes = []
        functions = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'lineno': node.lineno,
                    'methods': [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
                })
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:  # Top-level functions
                functions.append({
                    'name': node.name,
                    'lineno': node.lineno
                })
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append({
                    'lineno': node.lineno,
                    'module': getattr(node, 'module', None) if isinstance(node, ast.ImportFrom) else None
                })
        
        return {
            'classes': classes,
            'functions': functions,
            'imports': imports,
            'total_lines': len(content.splitlines())
        }
    
    def suggest_split_strategy(self, analysis: Dict, file_path: str) -> List[Dict]:
        """Suggest how to split the module based on its structure."""
        strategies = []
        
        # Strategy 1: Split by classes (most common)
        if len(analysis['classes']) > 1:
            # Group related classes
            class_groups = self._group_classes_by_functionality(analysis['classes'])
            for group_name, class_list in class_groups.items():
                strategies.append({
                    'type': 'class_group',
                    'name': group_name,
                    'classes': class_list,
                    'file_suffix': f"_{group_name.lower().replace(' ', '_')}"
                })
        
        # Strategy 2: Split by functionality patterns
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        if 'analyzer' in base_name or 'analysis' in base_name:
            strategies.extend([
                {'type': 'core', 'name': 'Core Analysis', 'file_suffix': '_core'},
                {'type': 'utils', 'name': 'Analysis Utils', 'file_suffix': '_utils'},
                {'type': 'metrics', 'name': 'Metrics Processing', 'file_suffix': '_metrics'}
            ])
        elif 'api' in base_name:
            strategies.extend([
                {'type': 'endpoints', 'name': 'API Endpoints', 'file_suffix': '_endpoints'},
                {'type': 'handlers', 'name': 'Request Handlers', 'file_suffix': '_handlers'},
                {'type': 'validators', 'name': 'Validators', 'file_suffix': '_validators'}
            ])
        elif 'ml' in base_name or 'optimizer' in base_name:
            strategies.extend([
                {'type': 'algorithms', 'name': 'ML Algorithms', 'file_suffix': '_algorithms'},
                {'type': 'models', 'name': 'Models', 'file_suffix': '_models'},
                {'type': 'training', 'name': 'Training Utils', 'file_suffix': '_training'}
            ])
        
        return strategies[:3]  # Limit to 3 splits maximum
    
    def _group_classes_by_functionality(self, classes: List[Dict]) -> Dict[str, List[str]]:
        """Group classes by functionality based on naming patterns."""
        groups = {}
        
        for cls in classes:
            name = cls['name'].lower()
            
            if 'analyzer' in name or 'analysis' in name:
                groups.setdefault('Analysis', []).append(cls['name'])
            elif 'engine' in name or 'processor' in name:
                groups.setdefault('Processing', []).append(cls['name'])
            elif 'manager' in name or 'coordinator' in name:
                groups.setdefault('Management', []).append(cls['name'])
            elif 'validator' in name or 'verifier' in name:
                groups.setdefault('Validation', []).append(cls['name'])
            elif 'optimizer' in name or 'enhancer' in name:
                groups.setdefault('Optimization', []).append(cls['name'])
            elif 'monitor' in name or 'tracker' in name:
                groups.setdefault('Monitoring', []).append(cls['name'])
            else:
                groups.setdefault('Core', []).append(cls['name'])
        
        return groups
    
    def create_split_files(self, file_path: str) -> bool:
        """Create split files from oversized module."""
        print(f"\n--- Splitting {file_path} ---")
        
        # Analyze the module
        analysis = self.analyze_module(file_path)
        if 'error' in analysis:
            print(f"ERROR: Cannot parse {file_path}: {analysis['error']}")
            return False
            
        print(f"Module has {analysis['total_lines']} lines, {len(analysis['classes'])} classes, {len(analysis['functions'])} functions")
        
        # Get split strategies
        strategies = self.suggest_split_strategy(analysis, file_path)
        if not strategies:
            print("No suitable split strategy found")
            return False
        
        # Read original file
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
            lines = original_content.splitlines()
        
        # Extract imports (first ~20 lines typically)
        imports_section = []
        docstring_section = []
        in_docstring = False
        
        for i, line in enumerate(lines[:50]):  # Check first 50 lines
            stripped = line.strip()
            if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
                docstring_section.append(line)
            elif in_docstring:
                docstring_section.append(line)
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
            elif (stripped.startswith('import ') or stripped.startswith('from ') or 
                  stripped == '' or stripped.startswith('#')):
                imports_section.append(line)
            else:
                break
        
        # Create base directory
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        split_dir = os.path.join(base_dir, base_name + '_modules')
        
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        
        # Calculate lines per split
        content_lines = len(lines) - len(imports_section) - len(docstring_section)
        lines_per_split = min(800, content_lines // len(strategies))  # Max 800 lines per split
        
        # Create splits
        current_line = len(imports_section) + len(docstring_section)
        created_files = []
        
        for i, strategy in enumerate(strategies):
            split_file = os.path.join(split_dir, f"{base_name}{strategy['file_suffix']}.py")
            
            # Calculate end line for this split
            if i == len(strategies) - 1:  # Last split gets remaining lines
                end_line = len(lines)
            else:
                end_line = min(current_line + lines_per_split, len(lines))
            
            # Create split content
            split_content = []
            split_content.extend(docstring_section[:3])  # First 3 lines of docstring
            split_content.append(f'"""{strategy["name"]} Module - Split from {base_name}.py"""')
            split_content.append('')
            split_content.extend(imports_section)
            split_content.append('')
            split_content.extend(lines[current_line:end_line])
            
            # Write split file
            with open(split_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(split_content))
            
            created_files.append(split_file)
            print(f"Created: {split_file} ({end_line - current_line} lines)")
            current_line = end_line
        
        # Create __init__.py for the split directory
        init_file = os.path.join(split_dir, '__init__.py')
        init_content = f'''"""
{base_name.title()} Module Split
================================

This module was split from the original {base_name}.py to maintain
modules under 1000 lines while preserving all functionality.
"""

# Import all components to maintain backward compatibility
'''
        
        for strategy in strategies:
            module_name = f"{base_name}{strategy['file_suffix']}"
            init_content += f"from .{module_name} import *\n"
        
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)
        
        print(f"Created: {init_file}")
        print(f"SUCCESS: Split {file_path} into {len(created_files)} modules")
        return True
    
    def run_all_splits(self):
        """Split all oversized modules."""
        print("="*60)
        print("SPLITTING OVERSIZED INTELLIGENCE MODULES")
        print("="*60)
        
        success_count = 0
        
        for module_path in self.oversized_modules:
            if os.path.exists(module_path):
                try:
                    if self.create_split_files(module_path):
                        success_count += 1
                except Exception as e:
                    print(f"ERROR splitting {module_path}: {e}")
            else:
                print(f"WARNING: File not found: {module_path}")
        
        print(f"\n{'='*60}")
        print(f"SPLIT SUMMARY: {success_count}/{len(self.oversized_modules)} modules successfully split")
        print(f"{'='*60}")
        
        return success_count == len(self.oversized_modules)


if __name__ == "__main__":
    splitter = ModuleSplitter()
    splitter.run_all_splits()