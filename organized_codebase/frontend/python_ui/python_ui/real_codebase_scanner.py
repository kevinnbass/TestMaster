"""
Real Codebase Scanner API
========================
Comprehensive codebase analysis and scanning with full implementation.
"""

import ast
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

class CodebaseScanner:
    """Comprehensive codebase scanner with full implementation."""
    
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.go', '.rs', '.php'}
        self.ignore_patterns = {
            '__pycache__', '.git', '.svn', 'node_modules', 
            '.venv', 'venv', 'env', '.env', 'dist', 'build'
        }
        self.analysis_cache = {}
    
    def scan_directory(self, directory_path: str) -> Dict[str, Any]:
        """Scan a directory for code files and analyze structure."""
        
        directory = Path(directory_path)
        if not directory.exists():
            return {"error": "Directory not found", "path": directory_path}
        
        result = {
            "path": str(directory),
            "total_files": 0,
            "code_files": 0,
            "languages": defaultdict(int),
            "file_sizes": {"total": 0, "average": 0},
            "complexity_metrics": {},
            "structure": {},
            "analysis_time": time.time()
        }
        
        code_files = []
        total_size = 0
        
        # Walk through directory
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            for file in files:
                file_path = Path(root) / file
                result["total_files"] += 1
                
                # Check if it's a code file
                if file_path.suffix in self.supported_extensions:
                    result["code_files"] += 1
                    result["languages"][file_path.suffix] += 1
                    
                    try:
                        file_size = file_path.stat().st_size
                        total_size += file_size
                        code_files.append({
                            "path": str(file_path),
                            "size": file_size,
                            "extension": file_path.suffix
                        })
                    except OSError:
                        continue
        
        # Calculate metrics
        result["file_sizes"]["total"] = total_size
        result["file_sizes"]["average"] = total_size / max(result["code_files"], 1)
        
        # Analyze code complexity
        result["complexity_metrics"] = self._analyze_complexity(code_files)
        
        # Generate structure map
        result["structure"] = self._generate_structure_map(directory)
        
        result["analysis_time"] = time.time() - result["analysis_time"]
        
        return result
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single code file in detail."""
        
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": "File not found", "path": str(file_path)}
        
        # Check cache
        file_key = f"{file_path}_{file_path.stat().st_mtime}"
        if file_key in self.analysis_cache:
            return self.analysis_cache[file_key]
        
        result = {
            "path": str(file_path),
            "size": file_path.stat().st_size,
            "extension": file_path.suffix,
            "lines_of_code": 0,
            "blank_lines": 0,
            "comment_lines": 0,
            "functions": [],
            "classes": [],
            "imports": [],
            "complexity_score": 0,
            "issues": [],
            "metadata": {}
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            lines = content.split('\n')
            result["lines_of_code"] = len(lines)
            
            # Count blank and comment lines
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    result["blank_lines"] += 1
                elif stripped.startswith('#') or stripped.startswith('//'):
                    result["comment_lines"] += 1
            
            # Language-specific analysis
            if file_path.suffix == '.py':
                result.update(self._analyze_python_file(content))
            elif file_path.suffix in {'.js', '.ts'}:
                result.update(self._analyze_javascript_file(content))
            
            # Calculate complexity score
            result["complexity_score"] = self._calculate_complexity_score(result)
            
        except Exception as e:
            result["error"] = str(e)
        
        # Cache result
        self.analysis_cache[file_key] = result
        return result
    
    def _analyze_python_file(self, content: str) -> Dict[str, Any]:
        """Analyze Python-specific elements."""
        
        result = {
            "functions": [],
            "classes": [],
            "imports": [],
            "docstrings": 0
        }
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "args": len(node.args.args),
                        "decorators": len(node.decorator_list),
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "has_docstring": ast.get_docstring(node) is not None
                    }
                    result["functions"].append(func_info)
                    
                    if func_info["has_docstring"]:
                        result["docstrings"] += 1
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        "name": node.name,
                        "line": node.lineno,
                        "methods": sum(1 for n in node.body if isinstance(n, ast.FunctionDef)),
                        "bases": len(node.bases),
                        "decorators": len(node.decorator_list),
                        "has_docstring": ast.get_docstring(node) is not None
                    }
                    result["classes"].append(class_info)
                    
                    if class_info["has_docstring"]:
                        result["docstrings"] += 1
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            result["imports"].append({
                                "module": alias.name,
                                "alias": alias.asname,
                                "type": "import"
                            })
                    else:
                        for alias in node.names:
                            result["imports"].append({
                                "module": node.module,
                                "name": alias.name,
                                "alias": alias.asname,
                                "type": "from_import"
                            })
        
        except SyntaxError as e:
            result["syntax_error"] = str(e)
        
        return result
    
    def _analyze_javascript_file(self, content: str) -> Dict[str, Any]:
        """Analyze JavaScript/TypeScript-specific elements."""
        
        result = {
            "functions": [],
            "classes": [],
            "imports": [],
            "exports": []
        }
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Function detection (simplified)
            if 'function ' in stripped or '=>' in stripped:
                result["functions"].append({
                    "line": i,
                    "type": "function",
                    "content": stripped[:50]
                })
            
            # Class detection
            if stripped.startswith('class '):
                result["classes"].append({
                    "line": i,
                    "content": stripped[:50]
                })
            
            # Import/Export detection
            if stripped.startswith('import ') or stripped.startswith('export '):
                result["imports" if "import" in stripped else "exports"].append({
                    "line": i,
                    "content": stripped[:50]
                })
        
        return result
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> int:
        """Calculate complexity score based on various metrics."""
        
        score = 0
        
        # Base score from size
        score += min(analysis.get("lines_of_code", 0) // 10, 50)
        
        # Function complexity
        functions = analysis.get("functions", [])
        score += len(functions) * 2
        
        # Class complexity
        classes = analysis.get("classes", [])
        score += len(classes) * 3
        
        # Import complexity
        imports = analysis.get("imports", [])
        score += len(imports)
        
        # Penalize files with no docstrings
        if analysis.get("docstrings", 0) == 0 and (functions or classes):
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    def _analyze_complexity(self, code_files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze overall codebase complexity."""
        
        metrics = {
            "total_complexity": 0,
            "average_complexity": 0,
            "high_complexity_files": [],
            "language_breakdown": defaultdict(int)
        }
        
        complexities = []
        
        for file_info in code_files:
            if file_info["extension"] == ".py":
                # Simplified complexity based on file size
                complexity = min(file_info["size"] // 100, 100)
                complexities.append(complexity)
                metrics["language_breakdown"]["python"] += complexity
                
                if complexity > 70:
                    metrics["high_complexity_files"].append({
                        "path": file_info["path"],
                        "complexity": complexity
                    })
        
        if complexities:
            metrics["total_complexity"] = sum(complexities)
            metrics["average_complexity"] = sum(complexities) / len(complexities)
        
        return metrics
    
    def _generate_structure_map(self, directory: Path) -> Dict[str, Any]:
        """Generate a structural map of the codebase."""
        
        structure = {
            "directories": {},
            "max_depth": 0,
            "file_distribution": defaultdict(int)
        }
        
        for root, dirs, files in os.walk(directory):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_patterns]
            
            rel_path = Path(root).relative_to(directory)
            depth = len(rel_path.parts) if str(rel_path) != '.' else 0
            structure["max_depth"] = max(structure["max_depth"], depth)
            
            # Count files by type
            for file in files:
                ext = Path(file).suffix
                if ext in self.supported_extensions:
                    structure["file_distribution"][ext] += 1
        
        return structure
    
    def get_project_statistics(self, directory_path: str) -> Dict[str, Any]:
        """Get comprehensive project statistics."""
        
        scan_result = self.scan_directory(directory_path)
        
        stats = {
            "overview": {
                "total_files": scan_result.get("total_files", 0),
                "code_files": scan_result.get("code_files", 0),
                "total_size_mb": scan_result.get("file_sizes", {}).get("total", 0) / (1024 * 1024),
                "languages": dict(scan_result.get("languages", {}))
            },
            "complexity": scan_result.get("complexity_metrics", {}),
            "structure": scan_result.get("structure", {}),
            "recommendations": self._generate_recommendations(scan_result)
        }
        
        return stats
    
    def _generate_recommendations(self, scan_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on scan results."""
        
        recommendations = []
        
        complexity = scan_result.get("complexity_metrics", {})
        avg_complexity = complexity.get("average_complexity", 0)
        
        if avg_complexity > 70:
            recommendations.append("Consider refactoring high-complexity files")
        
        if scan_result.get("code_files", 0) > 1000:
            recommendations.append("Large codebase - consider modularization")
        
        languages = scan_result.get("languages", {})
        if len(languages) > 5:
            recommendations.append("Multiple languages detected - ensure consistent standards")
        
        return recommendations

# Global scanner instance
scanner = CodebaseScanner()

def scan_codebase(directory: str) -> Dict[str, Any]:
    """Scan codebase and return analysis."""
    return scanner.scan_directory(directory)

def analyze_file_detail(file_path: str) -> Dict[str, Any]:
    """Analyze single file in detail."""
    return scanner.analyze_file(file_path)

def get_project_stats(directory: str) -> Dict[str, Any]:
    """Get project statistics."""
    return scanner.get_project_statistics(directory)
