"""
Base Analyzer Class
==================

Common functionality for all analysis modules.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import statistics
from difflib import SequenceMatcher


class BaseAnalyzer:
    """Base class for all analysis modules."""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self._file_cache = {}
        self._ast_cache = {}
    
    def _get_file_content(self, file_path: Path) -> str:
        """Get file content with caching."""
        if file_path not in self._file_cache:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    self._file_cache[file_path] = f.read()
            except Exception:
                self._file_cache[file_path] = ""
        return self._file_cache[file_path]
    
    def _get_ast(self, file_path: Path) -> ast.AST:
        """Get AST with caching."""
        if file_path not in self._ast_cache:
            try:
                content = self._get_file_content(file_path)
                self._ast_cache[file_path] = ast.parse(content)
            except Exception:
                self._ast_cache[file_path] = ast.parse("")  # Empty AST
        return self._ast_cache[file_path]
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""
        return (file_path.suffix == '.py' and 
                '__pycache__' not in str(file_path) and
                not file_path.name.startswith('.') and
                'test_' not in file_path.name)
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files to analyze."""
        return [f for f in self.base_path.rglob("*.py") if self._should_analyze_file(f)]
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """Calculate distribution of values into ranges."""
        if not values:
            return {}
        
        ranges = {
            "0.0": 0,        # Zero values
            "0.1-0.5": 0,    # Low values  
            "0.6-1.0": 0,    # Moderate values
            ">1.0": 0        # High values
        }
        
        for value in values:
            if value == 0:
                ranges["0.0"] += 1
            elif 0 < value <= 0.5:
                ranges["0.1-0.5"] += 1
            elif 0.5 < value <= 1.0:
                ranges["0.6-1.0"] += 1
            else:
                ranges[">1.0"] += 1
        
        return ranges
    
    def _calculate_function_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, 
                               ast.With, ast.AsyncWith, ast.Try)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Break, ast.Continue)):
                complexity += 1
                
        return complexity
    
    def _extract_identifiers(self, tree: ast.AST) -> List[str]:
        """Extract all identifiers from AST."""
        identifiers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                identifiers.append(node.id)
        return identifiers
    
    def _calculate_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 or not seq2:
            return 0.0
        matcher = SequenceMatcher(None, seq1, seq2)
        return matcher.ratio()