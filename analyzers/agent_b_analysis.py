#!/usr/bin/env python3
"""
🏗️ MODULE: Agent B Enhanced Analysis - Code Pattern Detection & Personal Insights
==================================================================

📋 PURPOSE:
    Enhanced comprehensive analysis script for code pattern detection, personal coding
    habit tracking, and development metrics collection. Builds upon existing analysis
    capabilities to provide personalized development insights.

🎯 CORE FUNCTIONALITY:
    • Code pattern detection and personal habit analysis
    • Development metrics collection and tracking
    • Code quality assessment and improvement suggestions
    • Modularization analysis and recommendations
    • Personal coding insights generation

🔄 EDIT HISTORY (Last 5 Changes):
==================================================================
📝 [2025-08-23 04:15:00] | Agent B | 🆕 FEATURE
   └─ Goal: Enhance analyzer with pattern detection capabilities
   └─ Changes: Added pattern analysis classes and personal insights
   └─ Impact: Enables personal coding pattern tracking

🏷️ METADATA:
==================================================================
📅 Created: 2025-08-22 by Agent B
🔧 Language: Python
📦 Dependencies: ast, os, glob, json, re, dataclasses, datetime, collections
🎯 Integration Points: Streaming intelligence platform, existing analyzers
⚡ Performance Notes: Optimized for personal project analysis
🔒 Security Notes: No sensitive data handling

Author: Agent B - Intelligence & Pattern Analysis Specialist
"""

import ast
import os
import glob
import json
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
import hashlib

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

@dataclass
class CodePattern:
    """Represents a detected code pattern."""
    pattern_type: str  # e.g., "error_handling", "naming_convention", "design_pattern"
    pattern_name: str  # e.g., "try_except_else", "snake_case", "singleton"
    occurrences: int
    locations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    description: str = ""

@dataclass
class CodingHabit:
    """Represents a personal coding habit."""
    habit_type: str  # e.g., "function_length", "comment_style", "import_organization"
    frequency: float  # How often this habit appears
    consistency: float  # How consistent the developer is with this habit
    examples: List[str] = field(default_factory=list)
    recommendation: str = ""

@dataclass
class PersonalInsight:
    """Personal development insight based on code analysis."""
    insight_type: str  # e.g., "strength", "improvement_area", "trend"
    title: str
    description: str
    evidence: List[str] = field(default_factory=list)
    priority: str = "medium"  # low, medium, high
    action_items: List[str] = field(default_factory=list)

class AgentBAnalyzer:
    """Enhanced analyzer with pattern detection and personal insights."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.modules = {}
        self.dependency_graph = defaultdict(list)
        self.detected_patterns = []
        self.coding_habits = []
        self.personal_insights = []
        self.pattern_cache = {}  # Cache for pattern detection results
        
    def analyze_file(self, filepath: str) -> ModuleAnalysis:
        """Analyze a single Python file with pattern detection."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            lines = content.split('\n')
            
            # Detect patterns in this file
            patterns = self.detect_code_patterns(filepath, tree)
            self.detected_patterns.extend(patterns)
            
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
    
    def detect_code_patterns(self, filepath: str, tree: ast.AST) -> List[CodePattern]:
        """Detect code patterns in a Python file."""
        patterns = []
        
        # Detect error handling patterns
        error_patterns = self._detect_error_handling_patterns(tree, filepath)
        patterns.extend(error_patterns)
        
        # Detect naming convention patterns
        naming_patterns = self._detect_naming_patterns(tree, filepath)
        patterns.extend(naming_patterns)
        
        # Detect design patterns
        design_patterns = self._detect_design_patterns(tree, filepath)
        patterns.extend(design_patterns)
        
        return patterns
    
    def _detect_error_handling_patterns(self, tree: ast.AST, filepath: str) -> List[CodePattern]:
        """Detect error handling patterns in code."""
        patterns = []
        try_except_count = 0
        try_except_else_count = 0
        try_except_finally_count = 0
        bare_except_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                try_except_count += 1
                if node.orelse:
                    try_except_else_count += 1
                if node.finalbody:
                    try_except_finally_count += 1
                
                # Check for bare except
                for handler in node.handlers:
                    if handler.type is None:
                        bare_except_count += 1
        
        if try_except_count > 0:
            patterns.append(CodePattern(
                pattern_type="error_handling",
                pattern_name="try_except",
                occurrences=try_except_count,
                locations=[filepath],
                confidence=1.0,
                description="Standard try-except error handling"
            ))
        
        if bare_except_count > 0:
            patterns.append(CodePattern(
                pattern_type="error_handling",
                pattern_name="bare_except",
                occurrences=bare_except_count,
                locations=[filepath],
                confidence=1.0,
                description="Bare except clauses (not recommended)"
            ))
        
        return patterns
    
    def _detect_naming_patterns(self, tree: ast.AST, filepath: str) -> List[CodePattern]:
        """Detect naming convention patterns."""
        patterns = []
        snake_case_count = 0
        camel_case_count = 0
        pascal_case_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    snake_case_count += 1
                elif re.match(r'^[a-z][a-zA-Z0-9]*$', node.name):
                    camel_case_count += 1
            elif isinstance(node, ast.ClassDef):
                if re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    pascal_case_count += 1
        
        if snake_case_count > 0:
            patterns.append(CodePattern(
                pattern_type="naming_convention",
                pattern_name="snake_case_functions",
                occurrences=snake_case_count,
                locations=[filepath],
                confidence=0.9,
                description="Functions using snake_case naming"
            ))
        
        if pascal_case_count > 0:
            patterns.append(CodePattern(
                pattern_type="naming_convention",
                pattern_name="pascal_case_classes",
                occurrences=pascal_case_count,
                locations=[filepath],
                confidence=0.9,
                description="Classes using PascalCase naming"
            ))
        
        return patterns
    
    def _detect_design_patterns(self, tree: ast.AST, filepath: str) -> List[CodePattern]:
        """Detect common design patterns."""
        patterns = []
        
        # Detect singleton pattern
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                has_instance = False
                has_new = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name == "__new__":
                            has_new = True
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name) and "_instance" in target.id:
                                has_instance = True
                
                if has_instance or has_new:
                    patterns.append(CodePattern(
                        pattern_type="design_pattern",
                        pattern_name="singleton_candidate",
                        occurrences=1,
                        locations=[f"{filepath}:{node.name}"],
                        confidence=0.7,
                        description="Possible singleton pattern implementation"
                    ))
        
        return patterns
    
    def analyze_coding_habits(self) -> List[CodingHabit]:
        """Analyze personal coding habits across the codebase."""
        habits = []
        
        # Analyze function length preferences
        function_lengths = []
        for analysis in self.modules.values():
            for func in analysis.functions:
                function_lengths.append(func.line_count)
        
        if function_lengths:
            avg_length = sum(function_lengths) / len(function_lengths)
            consistency = 1.0 - (max(function_lengths) - min(function_lengths)) / (max(function_lengths) + 1)
            
            habits.append(CodingHabit(
                habit_type="function_length",
                frequency=len(function_lengths),
                consistency=consistency,
                examples=[f"Average: {avg_length:.1f} lines"],
                recommendation=self._get_function_length_recommendation(avg_length)
            ))
        
        # Analyze docstring usage
        docstring_count = 0
        total_count = 0
        for analysis in self.modules.values():
            for func in analysis.functions:
                total_count += 1
                if func.has_docstring:
                    docstring_count += 1
        
        if total_count > 0:
            docstring_ratio = docstring_count / total_count
            habits.append(CodingHabit(
                habit_type="documentation",
                frequency=docstring_ratio,
                consistency=docstring_ratio,  # Consistency is same as frequency for binary habits
                examples=[f"{docstring_count}/{total_count} functions documented"],
                recommendation=self._get_documentation_recommendation(docstring_ratio)
            ))
        
        return habits
    
    def _get_function_length_recommendation(self, avg_length: float) -> str:
        """Get recommendation for function length."""
        if avg_length < 20:
            return "Your functions are concise. Good practice!"
        elif avg_length < 50:
            return "Function length is reasonable. Consider splitting functions over 50 lines."
        else:
            return "Consider breaking down longer functions for better maintainability."
    
    def _get_documentation_recommendation(self, ratio: float) -> str:
        """Get recommendation for documentation habits."""
        if ratio > 0.8:
            return "Excellent documentation coverage!"
        elif ratio > 0.5:
            return "Good documentation, aim for >80% coverage."
        else:
            return "Consider adding more docstrings to improve code documentation."
    
    def generate_personal_insights(self) -> List[PersonalInsight]:
        """Generate personal development insights based on analysis."""
        insights = []
        
        # Analyze code complexity trends
        high_complexity_functions = []
        for filepath, analysis in self.modules.items():
            for func in analysis.functions:
                if func.complexity_score > 10:
                    high_complexity_functions.append((filepath, func.name, func.complexity_score))
        
        if high_complexity_functions:
            insights.append(PersonalInsight(
                insight_type="improvement_area",
                title="Complex Functions Detected",
                description=f"Found {len(high_complexity_functions)} functions with high complexity",
                evidence=[f"{name}: complexity {score}" for _, name, score in high_complexity_functions[:3]],
                priority="high",
                action_items=[
                    "Refactor complex functions into smaller, focused functions",
                    "Consider extracting complex logic into separate methods",
                    "Add unit tests before refactoring"
                ]
            ))
        
        # Analyze modularization opportunities
        large_modules = [m for m in self.modules.values() if m.should_divide]
        if large_modules:
            insights.append(PersonalInsight(
                insight_type="improvement_area",
                title="Large Modules Need Splitting",
                description=f"{len(large_modules)} modules exceed 300 lines",
                evidence=[m.filepath for m in large_modules[:3]],
                priority="medium",
                action_items=[
                    "Split large modules into focused, single-responsibility modules",
                    "Group related functionality together",
                    "Create clear module interfaces"
                ]
            ))
        
        # Positive insights
        well_documented = sum(1 for m in self.modules.values() if m.has_module_docstring)
        total = len(self.modules)
        if well_documented / total > 0.7:
            insights.append(PersonalInsight(
                insight_type="strength",
                title="Strong Documentation Practice",
                description=f"{well_documented}/{total} modules have documentation",
                evidence=[],
                priority="low",
                action_items=["Keep up the good documentation practices!"]
            ))
        
        return insights

def main():
    """Enhanced main analysis execution with pattern detection."""
    base_path = r"C:\Users\kbass\OneDrive\Documents\testmaster"
    
    print("=" * 60)
    print("Agent B Enhanced Analysis - Pattern Detection & Insights")
    print("=" * 60)
    
    analyzer = AgentBAnalyzer(base_path)
    results = analyzer.analyze_codebase()
    
    # Print basic summary
    print("\nCODEBASE ANALYSIS SUMMARY")
    print("-" * 40)
    print(f"Total modules analyzed: {results['total_modules']}")
    print(f"Total lines of code: {results['total_lines']:,}")
    print(f"Modules needing split (>300 lines): {results['modules_needing_split']}")
    print(f"Functions needing extraction (>50 lines): {results['functions_needing_extraction']}")
    print(f"Classes needing split (>200 lines): {results['classes_needing_split']}")
    print(f"Modules missing docstrings: {results['modules_missing_docs']}")
    
    # Analyze coding habits
    print("\nPERSONAL CODING HABITS")
    print("-" * 40)
    habits = analyzer.analyze_coding_habits()
    for habit in habits:
        print(f"{habit.habit_type}: {habit.recommendation}")
        if habit.examples:
            print(f"  Examples: {', '.join(habit.examples)}")
    
    # Generate personal insights
    print("\nPERSONAL DEVELOPMENT INSIGHTS")
    print("-" * 40)
    insights = analyzer.generate_personal_insights()
    for insight in insights:
        print(f"[{insight.priority.upper()}] {insight.title}")
        print(f"  {insight.description}")
        if insight.action_items:
            print(f"  Actions: {insight.action_items[0]}")
    
    # Save enhanced results
    enhanced_results = {
        **results,
        "coding_habits": [
            {
                "habit_type": h.habit_type,
                "frequency": h.frequency,
                "consistency": h.consistency,
                "examples": h.examples,
                "recommendation": h.recommendation
            } for h in habits
        ],
        "personal_insights": [
            {
                "insight_type": i.insight_type,
                "title": i.title,
                "description": i.description,
                "evidence": i.evidence,
                "priority": i.priority,
                "action_items": i.action_items
            } for i in insights
        ],
        "analysis_metadata": {
            "analyzer_version": "2.0",
            "pattern_detection_enabled": True,
            "timestamp": datetime.now().isoformat()
        }
    }
    
    output_file = "agent_b_pattern_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    print(f"\nEnhanced analysis saved to: {output_file}")
    
    # Generate Neo4j data
    neo4j_data = analyzer.generate_neo4j_data()
    neo4j_file = "agent_b_neo4j_export.json"
    with open(neo4j_file, 'w') as f:
        json.dump(neo4j_data, f, indent=2)
    print(f"Neo4j export saved to: {neo4j_file}")
    
    print("\nAgent B Enhanced Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()