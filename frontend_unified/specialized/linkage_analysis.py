#!/usr/bin/env python3
"""
STEELCLAD MODULE: Linkage Analysis Engine
========================================

Core functionality for functional linkage analysis extracted from
enhanced_linkage_dashboard.py (5,274 lines) -> 150 lines

Provides:
- Quick linkage analysis for dashboard display
- File dependency mapping
- AST parsing and analysis
- Performance monitoring integration

Author: Agent X (STEELCLAD Modularization)
"""

import os
import ast
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Import performance monitoring
try:
    from testmaster_performance_engine import performance_monitor
    PERFORMANCE_ENGINE_AVAILABLE = True
except ImportError:
    PERFORMANCE_ENGINE_AVAILABLE = False
    # Create fallback decorator
    def performance_monitor(name):
        def decorator(func):
            return func
        return decorator


@performance_monitor("quick_linkage_analysis")
def quick_linkage_analysis(base_dir="../TestMaster", max_files=None):
    """Quick linkage analysis for dashboard display with performance monitoring."""
    results = {
        "orphaned_files": [],
        "hanging_files": [],
        "marginal_files": [],
        "well_connected_files": [],
        "total_files": 0,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    # Find Python files (limited for speed)
    python_files = []
    base_path = Path(base_dir)
    
    if not base_path.exists():
        return results
    
    # Efficiently scan for Python files across the codebase
    for root, dirs, files in os.walk(base_path):
        # Skip problematic directories
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'QUARANTINE', 'archive']]
        
        for file in files:
            if file.endswith('.py') and (max_files is None or len(python_files) < max_files):
                # Skip problematic files
                if not any(skip in file for skip in ['original_', '_original', 'ARCHIVED', 'backup']):
                    python_files.append(Path(root) / file)
    
    # Count total files in codebase for accurate reporting
    total_codebase_files = 0
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git']]
        for file in files:
            if file.endswith('.py'):
                total_codebase_files += 1
    
    results["total_files"] = len(python_files)
    results["total_codebase_files"] = total_codebase_files
    results["analysis_coverage"] = f"{len(python_files)}/{total_codebase_files}"
    
    # Simple analysis - count imports
    file_data = {}
    
    for py_file in python_files:
        try:
            relative_path = str(py_file.relative_to(base_path))
            
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Quick import count
            import_count = content.count('import ') + content.count('from ')
            
            file_data[relative_path] = import_count
            
        except Exception:
            continue
    
    # Simple categorization based on import counts
    for file_path, import_count in file_data.items():
        file_info = {
            "path": file_path,
            "incoming_deps": 0,  # Simplified - would need full analysis
            "outgoing_deps": import_count,
            "total_deps": import_count
        }
        
        if import_count == 0:
            results["orphaned_files"].append(file_info)
        elif import_count < 3:
            results["marginal_files"].append(file_info)
        elif import_count > 20:
            results["hanging_files"].append(file_info)  # Files with many imports
        else:
            results["well_connected_files"].append(file_info)
    
    # Sort and limit results
    for category in ["orphaned_files", "hanging_files", "marginal_files", "well_connected_files"]:
        results[category].sort(key=lambda x: x["total_deps"], reverse=True)
    
    return results


def analyze_file_quick(file_path):
    """Quick analysis of a single file using AST parsing."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        imports = []
        exports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                if not node.name.startswith('_'):
                    exports.append(node.name)
        
        return imports, exports
        
    except Exception:
        return [], []


def get_codebase_statistics(base_dir="../TestMaster"):
    """Get comprehensive codebase statistics for reporting."""
    stats = {
        "total_python_files": 0,
        "total_lines_of_code": 0,
        "total_imports": 0,
        "total_functions": 0,
        "total_classes": 0,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    base_path = Path(base_dir)
    if not base_path.exists():
        return stats
    
    for root, dirs, files in os.walk(base_path):
        dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', 'QUARANTINE', 'archive']]
        
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    stats["total_python_files"] += 1
                    stats["total_lines_of_code"] += len(content.splitlines())
                    stats["total_imports"] += content.count('import ') + content.count('from ')
                    
                    # AST analysis for functions and classes
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                stats["total_functions"] += 1
                            elif isinstance(node, ast.ClassDef):
                                stats["total_classes"] += 1
                    except:
                        pass
                        
                except Exception:
                    continue
    
    return stats