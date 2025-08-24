#!/usr/bin/env python3
"""
Coverage Analyzer Module Modularization Tool
Splits the large coverage_analyzer.py into organized, maintainable modules
"""

import os
import re
from pathlib import Path

def modularize_coverage_analyzer():
    """Modularize the large coverage analyzer file"""
    source_file = "C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/testmaster/analysis/coverage_analyzer.py"
    output_dir = Path("C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/testmaster/analysis/coverage")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(source_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    print(f"Modularizing {source_file} ({len(lines)} lines)")
    
    # 1. Create base types module
    base_content = '''"""
Coverage Analysis Base Types and Data Classes
"""

import ast
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter


@dataclass
class FunctionCoverage:
    """Coverage information for a function."""
    name: str
    line_start: int
    line_end: int
    covered_lines: Set[int]
    total_lines: int
    coverage_percentage: float
    complexity: int
    missing_lines: Set[int]
    is_tested: bool = False
    test_quality_score: float = 0.0


@dataclass
class ModuleCoverage:
    """Coverage information for a module."""
    name: str
    file_path: str
    functions: List[FunctionCoverage]
    total_lines: int
    covered_lines: int
    coverage_percentage: float
    missing_lines: Set[int]
    complexity_score: float = 0.0


@dataclass
class CoverageReport:
    """Complete coverage report."""
    timestamp: datetime
    total_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    modules: List[ModuleCoverage]
    summary: Dict[str, Any]
    recommendations: List[str] = field(default_factory=list)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
'''
    
    base_file = output_dir / "base.py"
    with open(base_file, 'w', encoding='utf-8') as f:
        f.write(base_content)
    print(f"Created {base_file}")
    
    # Find class boundaries
    class_sections = {}
    current_class = None
    indent_level = 0
    
    for i, line in enumerate(lines):
        if line.strip().startswith('class ') and line.strip().endswith(':'):
            if current_class:
                class_sections[current_class]['end'] = i - 1
            
            class_name = line.strip().replace('class ', '').replace(':', '')
            current_class = class_name
            class_sections[current_class] = {'start': i, 'end': None, 'content': []}
            indent_level = len(line) - len(line.lstrip())
        
        if current_class:
            class_sections[current_class]['content'].append(line)
    
    # Set end for last class
    if current_class:
        class_sections[current_class]['end'] = len(lines) - 1
    
    # Create separate modules for each major class
    modules_info = {
        'CoverageAnalyzer': 'analyzer.py',
        'ComprehensiveCodebaseAnalyzer': 'codebase_analyzer.py', 
        'AdvancedDependencyMapper': 'dependency_mapper.py',
        'CodebaseHealthAssessment': 'health_assessment.py'
    }
    
    modules_created = []
    
    # Extract imports from original file (first 40 lines should cover most imports)
    import_lines = []
    for line in lines[:40]:
        if (line.strip().startswith('import ') or 
            line.strip().startswith('from ') or
            line.strip().startswith('#') or
            line.strip() == '' or
            '"""' in line):
            import_lines.append(line)
        else:
            break
    
    imports_text = '\n'.join(import_lines)
    
    for class_name, filename in modules_info.items():
        if class_name not in class_sections:
            print(f"Warning: Class {class_name} not found in source file")
            continue
            
        module_file = output_dir / filename
        class_info = class_sections[class_name]
        
        with open(module_file, 'w', encoding='utf-8') as f:
            # Write imports
            f.write(imports_text)
            f.write('\n\n')
            
            # Import base types if needed
            if class_name != 'CoverageAnalyzer':  # Main analyzer might not need base import
                f.write('from .base import FunctionCoverage, ModuleCoverage, CoverageReport\n\n')
            
            # Write the class content
            f.write('\n'.join(class_info['content']))
            
        modules_created.append(module_file)
        print(f"Created {module_file} ({len(class_info['content'])} lines)")
    
    # Create a unified interface module
    interface_content = f'''"""
Unified Coverage Analysis Interface
"""

from .base import FunctionCoverage, ModuleCoverage, CoverageReport
from .analyzer import CoverageAnalyzer
from .codebase_analyzer import ComprehensiveCodebaseAnalyzer
from .dependency_mapper import AdvancedDependencyMapper
from .health_assessment import CodebaseHealthAssessment


class UnifiedCoverageAnalyzer:
    """
    Unified interface for all coverage analysis functionality.
    """
    
    def __init__(self):
        """Initialize the unified coverage analyzer."""
        self.analyzer = CoverageAnalyzer()
        self.codebase_analyzer = ComprehensiveCodebaseAnalyzer()
        self.dependency_mapper = AdvancedDependencyMapper()
        self.health_assessment = CodebaseHealthAssessment()
    
    def analyze_coverage(self, project_path: str = ".") -> CoverageReport:
        """Run comprehensive coverage analysis."""
        return self.analyzer.analyze_project_coverage(project_path)
    
    def analyze_codebase(self, project_path: str = "."):
        """Run comprehensive codebase analysis."""
        return self.codebase_analyzer.analyze_codebase(project_path)
    
    def map_dependencies(self, project_path: str = "."):
        """Map project dependencies."""
        return self.dependency_mapper.analyze_dependencies(project_path)
    
    def assess_health(self, project_path: str = "."):
        """Assess codebase health."""
        return self.health_assessment.assess_project_health(project_path)
    
    def generate_comprehensive_report(self, project_path: str = "."):
        """Generate a comprehensive analysis report."""
        coverage = self.analyze_coverage(project_path)
        codebase = self.analyze_codebase(project_path)
        dependencies = self.map_dependencies(project_path)
        health = self.assess_health(project_path)
        
        return {{
            'coverage': coverage,
            'codebase': codebase, 
            'dependencies': dependencies,
            'health': health,
            'timestamp': coverage.timestamp
        }}
'''
    
    interface_file = output_dir / "interface.py"
    with open(interface_file, 'w', encoding='utf-8') as f:
        f.write(interface_content)
    print(f"Created {interface_file}")
    modules_created.append(interface_file)
    
    # Create __init__.py
    init_content = '''"""
Coverage Analysis Module
"""

from .interface import UnifiedCoverageAnalyzer
from .base import FunctionCoverage, ModuleCoverage, CoverageReport
from .analyzer import CoverageAnalyzer
from .codebase_analyzer import ComprehensiveCodebaseAnalyzer
from .dependency_mapper import AdvancedDependencyMapper
from .health_assessment import CodebaseHealthAssessment

__all__ = [
    'UnifiedCoverageAnalyzer',
    'FunctionCoverage',
    'ModuleCoverage', 
    'CoverageReport',
    'CoverageAnalyzer',
    'ComprehensiveCodebaseAnalyzer',
    'AdvancedDependencyMapper',
    'CodebaseHealthAssessment'
]
'''
    
    init_file = output_dir / "__init__.py"
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    print(f"Created {init_file}")
    
    # Archive the original file
    archive_dir = Path("C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster/archive")
    archive_file = archive_dir / "coverage_analyzer_original_2697_lines.py"
    os.rename(source_file, str(archive_file))
    print(f"Archived original file to: {archive_file}")
    
    print(f"\\nModularization Complete!")
    print(f"- Created {len(modules_created) + 2} modular files")
    print(f"- Original: 1 file with 2,697 lines")
    print(f"- Result: {len(modules_created) + 2} focused modules")
    
    return len(modules_created) + 2

if __name__ == "__main__":
    modularize_coverage_analyzer()