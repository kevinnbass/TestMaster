#!/usr/bin/env python3
"""
Simple linkage test to debug the issue.
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def simple_linkage_test():
    """Simple linkage analysis test."""
    print("Starting simple linkage analysis...")
    
    results = {
        "orphaned_files": [],
        "hanging_files": [],
        "marginal_files": [],
        "well_connected_files": [],
        "total_files": 0,
        "analysis_timestamp": datetime.now().isoformat()
    }
    
    # Find Python files (very limited for testing)
    python_files = []
    base_path = Path("TestMaster")
    
    if not base_path.exists():
        print("TestMaster directory not found")
        return results
    
    # Only look at top-level files for testing
    for file in base_path.glob("*.py"):
        if len(python_files) < 50:  # Very small limit for testing
            python_files.append(file)
    
    print(f"Found {len(python_files)} Python files to analyze")
    results["total_files"] = len(python_files)
    
    # Simple analysis - just check file existence and basic structure
    file_imports = {}
    
    for py_file in python_files:
        try:
            relative_path = str(py_file.relative_to(base_path))
            print(f"Analyzing {relative_path}")
            
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Count imports quickly
            import_count = content.count('import ')
            
            file_imports[relative_path] = import_count
            
        except Exception as e:
            print(f"Error analyzing {py_file}: {e}")
            continue
    
    print(f"Successfully analyzed {len(file_imports)} files")
    
    # Simple categorization
    for file_path, import_count in file_imports.items():
        file_info = {
            "path": file_path,
            "incoming_deps": 0,  # Simplified for testing
            "outgoing_deps": import_count,
            "total_deps": import_count
        }
        
        if import_count == 0:
            results["orphaned_files"].append(file_info)
        elif import_count < 3:
            results["marginal_files"].append(file_info)
        else:
            results["well_connected_files"].append(file_info)
    
    print("Analysis complete!")
    return results

if __name__ == "__main__":
    result = simple_linkage_test()
    print(f"Results: {result['total_files']} files, {len(result['orphaned_files'])} orphaned")