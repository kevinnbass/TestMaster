"""
Software Metrics Analyzer
=========================

Implements comprehensive software metrics:
- Halstead Metrics (Volume, Difficulty, Effort, etc.)
- McCabe Cyclomatic Complexity
- Source Lines of Code (SLOC) Analysis
- Maintainability Index
"""

import ast
import re
import math
import statistics
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

from .base_analyzer import BaseAnalyzer


class SoftwareMetricsAnalyzer(BaseAnalyzer):
    """Analyzer for software metrics."""
    
    def analyze(self) -> Dict[str, Any]:
        """Perform comprehensive software metrics analysis."""
        print("[INFO] Analyzing Software Metrics...")
        
        metrics = {
            "halstead_metrics": self._calculate_halstead_metrics(),
            "mccabe_complexity": self._calculate_mccabe_complexity(),
            "sloc_metrics": self._calculate_sloc_metrics(),
            "maintainability_index": self._calculate_maintainability_index(),
        }
        
        print(f"  [OK] Analyzed {len(metrics)} metric categories")
        return metrics
    
    def _calculate_halstead_metrics(self) -> Dict[str, Any]:
        """Calculate Halstead software science metrics."""
        operators = set()
        operands = set()
        total_operators = 0
        total_operands = 0
        
        operator_patterns = [
            r'[+\-*/=%<>!&|^~]', r'\b(and|or|not|in|is)\b',
            r'[(){}\[\];:,.]', r'\b(if|else|elif|while|for|try|except|finally|with|def|class|import|from|return|yield|lambda)\b'
        ]
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                tree = self._get_ast(py_file)
                
                # Count operators
                for pattern in operator_patterns:
                    matches = re.findall(pattern, content)
                    for match in matches:
                        operators.add(match)
                        total_operators += 1
                
                # Count operands (identifiers, literals)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        operands.add(node.id)
                        total_operands += 1
                    elif isinstance(node, ast.Constant):
                        operands.add(str(node.value))
                        total_operands += 1
                        
            except Exception:
                continue
        
        n1, n2 = len(operators), len(operands)
        N1, N2 = total_operators, total_operands
        
        if n1 == 0 or n2 == 0:
            return {"unique_operators": n1, "unique_operands": n2, "total_operators": N1, "total_operands": N2, "volume": 0}
        
        vocabulary = n1 + n2
        length = N1 + N2
        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume
        
        return {
            "vocabulary": vocabulary,
            "length": length,
            "volume": volume,
            "difficulty": difficulty,
            "effort": effort,
            "time_seconds": effort / 18,
            "estimated_bugs": volume / 3000,
            "unique_operators": n1,
            "unique_operands": n2,
            "total_operators": N1,
            "total_operands": N2
        }
    
    def _calculate_mccabe_complexity(self) -> Dict[str, Any]:
        """Calculate McCabe cyclomatic complexity."""
        complexity_data = {}
        total_complexity = 0
        function_count = 0
        
        for py_file in self._get_python_files():
            try:
                tree = self._get_ast(py_file)
                file_complexity = {}
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        complexity = self._calculate_function_complexity(node)
                        file_complexity[node.name] = complexity
                        total_complexity += complexity
                        function_count += 1
                
                if file_complexity:
                    complexity_data[str(py_file.relative_to(self.base_path))] = file_complexity
                    
            except Exception:
                continue
        
        return {
            "per_file": complexity_data,
            "total_complexity": total_complexity,
            "average_complexity": total_complexity / max(function_count, 1),
            "function_count": function_count,
            "high_complexity_functions": len([f for file_funcs in complexity_data.values() 
                                            for f, complexity in file_funcs.items() if complexity > 10])
        }
    
    def _calculate_sloc_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive Source Lines of Code metrics."""
        metrics = {
            "total_lines": 0,
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "mixed_lines": 0,
            "docstring_lines": 0,
            "per_file": {}
        }
        
        for py_file in self._get_python_files():
            try:
                content = self._get_file_content(py_file)
                lines = content.split('\n')
                
                # Parse AST to identify docstrings
                tree = self._get_ast(py_file)
                docstring_lines = self._find_docstring_lines(tree, lines)
                
                file_metrics = {
                    "total": len(lines),
                    "code": 0,
                    "comments": 0,
                    "blank": 0,
                    "mixed": 0,
                    "docstring": len(docstring_lines)
                }
                
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if not stripped:
                        file_metrics["blank"] += 1
                    elif i in docstring_lines:
                        continue  # Already counted in docstring
                    elif stripped.startswith('#'):
                        file_metrics["comments"] += 1
                    elif '#' in stripped and not stripped.startswith('#'):
                        file_metrics["mixed"] += 1
                        file_metrics["code"] += 1
                    else:
                        file_metrics["code"] += 1
                
                metrics["per_file"][str(py_file.relative_to(self.base_path))] = file_metrics
                for key in ["total", "code", "comments", "blank", "mixed", "docstring"]:
                    metrics[f"{key}_lines"] += file_metrics[key]
                
            except Exception:
                continue
        
        # Calculate ratios
        if metrics["total_lines"] > 0:
            metrics["comment_ratio"] = metrics["comment_lines"] / metrics["total_lines"]
            metrics["code_ratio"] = metrics["code_lines"] / metrics["total_lines"]
            metrics["docstring_ratio"] = metrics["docstring_lines"] / metrics["total_lines"]
        
        return metrics
    
    def _find_docstring_lines(self, tree: ast.AST, lines: List[str]) -> set:
        """Find line numbers that contain docstrings."""
        docstring_lines = set()
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                docstring = ast.get_docstring(node)
                if docstring:
                    # Find docstring lines (simplified approach)
                    for i, line in enumerate(lines):
                        if '"""' in line or "'''" in line:
                            start = i
                            while i < len(lines) and ('"""' not in lines[i][lines[i].find('"""')+3:] if '"""' in lines[i] else "'''" not in lines[i][lines[i].find("'''")+3:] if "'''" in lines[i] else True):
                                docstring_lines.add(i)
                                i += 1
                                if i >= len(lines):
                                    break
                            if i < len(lines):
                                docstring_lines.add(i)
                            break
        
        return docstring_lines
    
    def _calculate_maintainability_index(self) -> Dict[str, float]:
        """Calculate maintainability index."""
        halstead = self._calculate_halstead_metrics()
        sloc = self._calculate_sloc_metrics()
        
        volume = halstead.get("volume", 1)
        loc = sloc.get("code_lines", 1)
        comment_ratio = sloc.get("comment_ratio", 0) * 100
        
        if volume <= 0 or loc <= 0:
            return {"maintainability_index": 0.0}
        
        # Simplified MI calculation
        try:
            mi = max(0, 171 - 5.2 * math.log(volume) - 16.2 * math.log(loc) + 50 * comment_ratio / 100)
            return {"maintainability_index": mi, "volume": volume, "loc": loc}
        except:
            return {"maintainability_index": 0.0}