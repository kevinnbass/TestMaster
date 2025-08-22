#!/usr/bin/env python3
"""
Agent B Comprehensive Analysis Script
=====================================
Performs functional comments, module overviews, and modularization analysis.

Author: Agent B - Intelligence Specialist
"""

import ast
import os
import glob
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

@dataclass
class FunctionAnalysis:
    """Analysis result for a function."""
    name: str
    line_count: int
    has_docstring: bool
    parameters: List[str]
    complexity_score: int
    should_extract: bool = False

@dataclass  
class ClassAnalysis:
    """Analysis result for a class."""
    name: str
    line_count: int
    has_docstring: bool
    method_count: int
    should_split: bool = False

@dataclass
class ModuleAnalysis:
    """Analysis result for a module."""
    filepath: str
    line_count: int
    has_module_docstring: bool
    functions: List[FunctionAnalysis]
    classes: List[ClassAnalysis]
    imports: Dict[str, List[str]]
    should_divide: bool = False
    modularization_suggestions: List[str] = field(default_factory=list)

class AgentBAnalyzer:
    """Main analyzer for Agent B mission."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.modules = {}
        self.dependency_graph = defaultdict(list)
        
    def analyze_file(self, filepath: str) -> ModuleAnalysis:
        """Analyze a single Python file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Check for module-level docstring
            has_module_docstring = (
                len(tree.body) > 0 and 
                isinstance(tree.body[0], ast.Expr) and
                isinstance(tree.body[0].value, ast.Constant) and
                isinstance(tree.body[0].value.value, str)
            )
            
            functions = []
            classes = []
            imports = {'direct': [], 'from': [], 'relative': []}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_analysis = self._analyze_function(node, lines)
                    functions.append(func_analysis)
                elif isinstance(node, ast.ClassDef):
                    class_analysis = self._analyze_class(node, lines)
                    classes.append(class_analysis)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports['direct'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    if node.level > 0:
                        imports['relative'].append(module)
                    else:
                        imports['from'].append(module)
            
            line_count = len(lines)
            should_divide = line_count > 300
            
            # Generate modularization suggestions
            suggestions = self._generate_modularization_suggestions(
                line_count, functions, classes
            )
            
            return ModuleAnalysis(
                filepath=filepath,
                line_count=line_count,
                has_module_docstring=has_module_docstring,
                functions=functions,
                classes=classes,
                imports=imports,
                should_divide=should_divide,
                modularization_suggestions=suggestions
            )
            
        except Exception as e:
            print(f"Error analyzing {filepath}: {e}")
            return None
    
    def _analyze_function(self, node: ast.FunctionDef, lines: List[str]) -> FunctionAnalysis:
        """Analyze a function definition."""
        line_count = 0
        if hasattr(node, 'end_lineno'):
            line_count = node.end_lineno - node.lineno + 1
        
        # Check for docstring
        has_docstring = (
            len(node.body) > 0 and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
        
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]
        
        # Simple complexity score (count decision points)
        complexity_score = 1  # Base complexity
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                complexity_score += 1
        
        should_extract = line_count > 50
        
        return FunctionAnalysis(
            name=node.name,
            line_count=line_count,
            has_docstring=has_docstring,
            parameters=parameters,
            complexity_score=complexity_score,
            should_extract=should_extract
        )
    
    def _analyze_class(self, node: ast.ClassDef, lines: List[str]) -> ClassAnalysis:
        """Analyze a class definition."""
        line_count = 0
        if hasattr(node, 'end_lineno'):
            line_count = node.end_lineno - node.lineno + 1
        
        # Check for docstring
        has_docstring = (
            len(node.body) > 0 and
            isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)
        )
        
        # Count methods
        method_count = sum(1 for child in node.body if isinstance(child, ast.FunctionDef))
        
        should_split = line_count > 200
        
        return ClassAnalysis(
            name=node.name,
            line_count=line_count,
            has_docstring=has_docstring,
            method_count=method_count,
            should_split=should_split
        )
    
    def _generate_modularization_suggestions(
        self, 
        line_count: int, 
        functions: List[FunctionAnalysis], 
        classes: List[ClassAnalysis]
    ) -> List[str]:
        """Generate modularization suggestions."""
        suggestions = []
        
        if line_count > 300:
            suggestions.append(f"Module has {line_count} lines (>300), consider splitting")
        
        large_functions = [f for f in functions if f.should_extract]
        if large_functions:
            suggestions.append(
                f"Extract {len(large_functions)} large functions: " +
                ", ".join([f.name for f in large_functions[:3]]) +
                ("..." if len(large_functions) > 3 else "")
            )
        
        large_classes = [c for c in classes if c.should_split]
        if large_classes:
            suggestions.append(
                f"Split {len(large_classes)} large classes: " +
                ", ".join([c.name for c in large_classes[:3]]) +
                ("..." if len(large_classes) > 3 else "")
            )
        
        # Check for missing docstrings
        missing_func_docs = [f for f in functions if not f.has_docstring]
        missing_class_docs = [c for c in classes if not c.has_docstring]
        
        if missing_func_docs:
            suggestions.append(f"Add docstrings to {len(missing_func_docs)} functions")
        
        if missing_class_docs:
            suggestions.append(f"Add docstrings to {len(missing_class_docs)} classes")
        
        return suggestions
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase."""
        py_files = glob.glob(os.path.join(self.base_path, "**/*.py"), recursive=True)
        
        total_modules = 0
        total_lines = 0
        modules_needing_split = 0
        functions_needing_extraction = 0
        classes_needing_split = 0
        modules_missing_docs = 0
        
        for py_file in py_files:
            if "__pycache__" in py_file or ".backup" in py_file:
                continue
                
            analysis = self.analyze_file(py_file)
            if analysis:
                self.modules[py_file] = analysis
                
                total_modules += 1
                total_lines += analysis.line_count
                
                if analysis.should_divide:
                    modules_needing_split += 1
                
                if not analysis.has_module_docstring:
                    modules_missing_docs += 1
                
                functions_needing_extraction += len([f for f in analysis.functions if f.should_extract])
                classes_needing_split += len([c for c in analysis.classes if c.should_split])
        
        return {
            "total_modules": total_modules,
            "total_lines": total_lines,
            "modules_needing_split": modules_needing_split,
            "functions_needing_extraction": functions_needing_extraction,
            "classes_needing_split": classes_needing_split,
            "modules_missing_docs": modules_missing_docs,
            "modules": {filepath: {
                "line_count": analysis.line_count,
                "has_module_docstring": analysis.has_module_docstring,
                "function_count": len(analysis.functions),
                "class_count": len(analysis.classes),
                "should_divide": analysis.should_divide,
                "suggestions": analysis.modularization_suggestions,
                "functions": [{
                    "name": f.name,
                    "line_count": f.line_count,
                    "has_docstring": f.has_docstring,
                    "should_extract": f.should_extract,
                    "complexity_score": f.complexity_score
                } for f in analysis.functions],
                "classes": [{
                    "name": c.name,
                    "line_count": c.line_count,
                    "has_docstring": c.has_docstring,
                    "should_split": c.should_split,
                    "method_count": c.method_count
                } for c in analysis.classes],
                "imports": analysis.imports
            } for filepath, analysis in self.modules.items()}
        }
    
    def generate_neo4j_data(self) -> Dict[str, Any]:
        """Generate data for Neo4j import."""
        nodes = []
        relationships = []
        
        for filepath, analysis in self.modules.items():
            module_name = os.path.basename(filepath).replace('.py', '')
            
            # Module node
            nodes.append({
                "id": f"module_{module_name}",
                "type": "Module",
                "name": module_name,
                "filepath": filepath,
                "line_count": analysis.line_count,
                "has_docstring": analysis.has_module_docstring,
                "should_divide": analysis.should_divide
            })
            
            # Function nodes
            for func in analysis.functions:
                func_id = f"func_{module_name}_{func.name}"
                nodes.append({
                    "id": func_id,
                    "type": "Function",
                    "name": func.name,
                    "line_count": func.line_count,
                    "has_docstring": func.has_docstring,
                    "should_extract": func.should_extract,
                    "complexity_score": func.complexity_score
                })
                
                # Relationship: Module contains Function
                relationships.append({
                    "from": f"module_{module_name}",
                    "to": func_id,
                    "type": "CONTAINS"
                })
            
            # Class nodes
            for cls in analysis.classes:
                cls_id = f"class_{module_name}_{cls.name}"
                nodes.append({
                    "id": cls_id,
                    "type": "Class",
                    "name": cls.name,
                    "line_count": cls.line_count,
                    "has_docstring": cls.has_docstring,
                    "should_split": cls.should_split,
                    "method_count": cls.method_count
                })
                
                # Relationship: Module contains Class
                relationships.append({
                    "from": f"module_{module_name}",
                    "to": cls_id,
                    "type": "CONTAINS"
                })
        
        return {
            "nodes": nodes,
            "relationships": relationships,
            "metadata": {
                "generated_by": "Agent B Analysis",
                "timestamp": datetime.now().isoformat(),
                "total_nodes": len(nodes),
                "total_relationships": len(relationships)
            }
        }

def main():
    """Main analysis execution."""
    base_path = r"C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster\core"
    
    print("Agent B Analysis Starting...")
    print("=" * 50)
    
    analyzer = AgentBAnalyzer(base_path)
    results = analyzer.analyze_codebase()
    
    # Print summary
    print(f"Total modules analyzed: {results['total_modules']}")
    print(f"Total lines of code: {results['total_lines']:,}")
    print(f"Modules needing split (>300 lines): {results['modules_needing_split']}")
    print(f"Functions needing extraction (>50 lines): {results['functions_needing_extraction']}")
    print(f"Classes needing split (>200 lines): {results['classes_needing_split']}")
    print(f"Modules missing docstrings: {results['modules_missing_docs']}")
    
    # Save detailed results
    output_file = "agent_b_comprehensive_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed analysis saved to: {output_file}")
    
    # Generate Neo4j data
    neo4j_data = analyzer.generate_neo4j_data()
    neo4j_file = "agent_b_neo4j_export.json"
    with open(neo4j_file, 'w') as f:
        json.dump(neo4j_data, f, indent=2)
    print(f"Neo4j export saved to: {neo4j_file}")
    
    print("\nAgent B Analysis Complete!")

if __name__ == "__main__":
    main()