from SECURITY_PATCHES.fix_eval_exec_vulnerabilities import SafeCodeExecutor
"""
TestMaster Pattern Analysis Utilities

Common pattern detection and analysis utilities that were stubbed
in the analysis subdirectories. Now fully implemented.
"""

import ast
import re
from typing import Dict, Any, List, Optional
from .shared_utils import COMMON_PATTERNS


def extract_common_patterns(tree: ast.AST, content: str) -> List[Dict[str, Any]]:
    """Extract common patterns from AST and content"""
    patterns_found = []
    
    # Check all pattern categories
    for category, pattern_groups in COMMON_PATTERNS.items():
        for pattern_type, patterns in pattern_groups.items():
            for pattern in patterns:
                # Search in content
                if pattern in content:
                    patterns_found.append({
                        "category": category,
                        "type": pattern_type,
                        "pattern": pattern,
                        "found_in": "content",
                        "line": content.find(pattern)
                    })
                
                # Search in AST node names
                for node in ast.walk(tree):
                    if hasattr(node, 'name') and pattern in str(node.name):
                        patterns_found.append({
                            "category": category,
                            "type": pattern_type,
                            "pattern": pattern,
                            "found_in": "ast_node",
                            "line": getattr(node, 'lineno', 0),
                            "node_type": type(node).__name__
                        })
    
    return patterns_found


def calculate_complexity_score(node: ast.AST) -> int:
    """Calculate complexity score for a node"""
    complexity = 0
    
    # Cyclomatic complexity factors
    complexity_nodes = {
        ast.If: 1,
        ast.While: 1,
        ast.For: 1,
        ast.Try: 1,
        ast.With: 1,
        ast.AsyncWith: 1,
        ast.AsyncFor: 1,
        ast.comprehension: 1,
    }
    
    # Boolean operators add complexity
    boolean_ops = {
        ast.And: 1,
        ast.Or: 1,
    }
    
    for child in ast.walk(node):
        # Count control flow structures
        node_type = type(child)
        if node_type in complexity_nodes:
            complexity += complexity_nodes[node_type]
        
        # Count boolean operations
        if isinstance(child, ast.BoolOp):
            op_type = type(child.op)
            if op_type in boolean_ops:
                complexity += boolean_ops[op_type] * (len(child.values) - 1)
        
        # Count exception handlers
        if isinstance(child, ast.ExceptHandler):
            complexity += 1
    
    return max(1, complexity)  # Minimum complexity of 1


def detect_code_patterns(tree: ast.AST, content: str) -> Dict[str, Any]:
    """Detect various code patterns and anti-patterns"""
    pattern_results = {
        "design_patterns": [],
        "anti_patterns": [],
        "complexity_hotspots": [],
        "naming_issues": [],
        "structure_issues": []
    }
    
    # Detect design patterns
    design_patterns = _detect_design_patterns(tree)
    pattern_results["design_patterns"] = design_patterns
    
    # Detect anti-patterns
    anti_patterns = _detect_anti_patterns(tree, content)
    pattern_results["anti_patterns"] = anti_patterns
    
    # Find complexity hotspots
    complexity_hotspots = _find_complexity_hotspots(tree)
    pattern_results["complexity_hotspots"] = complexity_hotspots
    
    # Check naming conventions
    naming_issues = _check_naming_patterns(tree)
    pattern_results["naming_issues"] = naming_issues
    
    # Analyze structure
    structure_issues = _analyze_structure_patterns(tree)
    pattern_results["structure_issues"] = structure_issues
    
    return pattern_results


def _detect_design_patterns(tree: ast.AST) -> List[Dict[str, Any]]:
    """Detect common design patterns"""
    patterns = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Singleton pattern
            if any("singleton" in method.name.lower() for method in node.body if isinstance(method, ast.FunctionDef)):
                patterns.append({
                    "pattern": "Singleton",
                    "class": node.name,
                    "line": node.lineno,
                    "confidence": "medium"
                })
            
            # Factory pattern
            if any("create" in method.name.lower() or "factory" in method.name.lower() 
                   for method in node.body if isinstance(method, ast.FunctionDef)):
                patterns.append({
                    "pattern": "Factory",
                    "class": node.name,
                    "line": node.lineno,
                    "confidence": "low"
                })
            
            # Observer pattern
            if any("notify" in method.name.lower() or "observer" in method.name.lower()
                   for method in node.body if isinstance(method, ast.FunctionDef)):
                patterns.append({
                    "pattern": "Observer",
                    "class": node.name,
                    "line": node.lineno,
                    "confidence": "medium"
                })
    
    return patterns


def _detect_anti_patterns(tree: ast.AST, content: str) -> List[Dict[str, Any]]:
    """Detect anti-patterns and code smells"""
    anti_patterns = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # God object (large class)
            method_count = len([n for n in node.body if isinstance(n, ast.FunctionDef)])
            if method_count > 20:
                anti_patterns.append({
                    "pattern": "God Object",
                    "class": node.name,
                    "line": node.lineno,
                    "severity": "high",
                    "details": f"{method_count} methods"
                })
        
        elif isinstance(node, ast.FunctionDef):
            # Long method
            method_length = len(node.body)
            if method_length > 30:
                anti_patterns.append({
                    "pattern": "Long Method",
                    "function": node.name,
                    "line": node.lineno,
                    "severity": "medium",
                    "details": f"{method_length} statements"
                })
            
            # Too many parameters
            param_count = len(node.args.args)
            if param_count > 7:
                anti_patterns.append({
                    "pattern": "Long Parameter List",
                    "function": node.name,
                    "line": node.lineno,
                    "severity": "medium",
                    "details": f"{param_count} parameters"
                })
    
    # Duplicate code detection (simplified)
    lines = content.split('\n')
    line_hashes = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if len(stripped) > 10:  # Ignore short lines
            if stripped in line_hashes:
                anti_patterns.append({
                    "pattern": "Duplicate Code",
                    "line": i + 1,
                    "severity": "low",
                    "details": f"Duplicate of line {line_hashes[stripped] + 1}"
                })
            else:
                line_hashes[stripped] = i
    
    return anti_patterns


def _find_complexity_hotspots(tree: ast.AST) -> List[Dict[str, Any]]:
    """Find functions with high complexity"""
    hotspots = []
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            complexity = calculate_complexity_score(node)
            if complexity > 10:  # High complexity threshold
                hotspots.append({
                    "function": node.name,
                    "line": node.lineno,
                    "complexity": complexity,
                    "severity": "high" if complexity > 20 else "medium"
                })
    
    return sorted(hotspots, key=lambda x: x["complexity"], reverse=True)


def _check_naming_patterns(tree: ast.AST) -> List[Dict[str, Any]]:
    """Check naming convention compliance"""
    issues = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Class names should be PascalCase
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                issues.append({
                    "issue": "Class naming convention",
                    "name": node.name,
                    "line": node.lineno,
                    "recommendation": "Use PascalCase for class names"
                })
        
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function names should be snake_case
            if not re.match(r'^[a-z][a-z0-9_]*$', node.name) and not node.name.startswith('__'):
                issues.append({
                    "issue": "Function naming convention",
                    "name": node.name,
                    "line": node.lineno,
                    "recommendation": "Use snake_case for function names"
                })
        
        elif isinstance(node, ast.Name):
            # Variable names should be snake_case
            if isinstance(node.ctx, ast.Store):
                if not re.match(r'^[a-z][a-z0-9_]*$', node.id) and not node.id.isupper():
                    issues.append({
                        "issue": "Variable naming convention",
                        "name": node.id,
                        "line": node.lineno,
                        "recommendation": "Use snake_case for variables or UPPER_CASE for constants"
                    })
    
    return issues


def _analyze_structure_patterns(tree: ast.AST) -> List[Dict[str, Any]]:
    """Analyze structural patterns and issues"""
    issues = []
    
    # Count nested levels
    def count_nesting(node, current_depth=0):
        max_depth = current_depth
        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
            current_depth += 1
        
        for child in ast.iter_child_nodes(node):
            child_depth = count_nesting(child, current_depth)
            max_depth = max(max_depth, child_depth)
        
        return max_depth
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nesting = count_nesting(node)
            if nesting > 5:
                issues.append({
                    "issue": "Deep nesting",
                    "function": node.name,
                    "line": node.lineno,
                    "depth": nesting,
                    "recommendation": "Consider refactoring to reduce nesting"
                })
    
    return issues