#!/usr/bin/env python3
"""
Fix Dashboard Core Import Paths
================================
Updates all import statements from 'core.' to 'dashboard.dashboard_core.'
in dashboard API files after the directory rename.
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern 1: from core.X import Y
        content = re.sub(
            r'from core\.([a-zA-Z_][a-zA-Z0-9_]*)',
            r'from dashboard.dashboard_core.\1',
            content
        )
        
        # Pattern 2: import core.X
        content = re.sub(
            r'import core\.([a-zA-Z_][a-zA-Z0-9_]*)',
            r'import dashboard.dashboard_core.\1',
            content
        )
        
        # Pattern 3: from dashboard.core.X import Y (if any)
        content = re.sub(
            r'from dashboard\.core\.([a-zA-Z_][a-zA-Z0-9_]*)',
            r'from dashboard.dashboard_core.\1',
            content
        )
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False

def main():
    """Main function to fix all dashboard imports."""
    dashboard_dir = Path('dashboard')
    
    # Files that need fixing (from grep results)
    files_to_fix = [
        'api/orchestration_flask.py',
        'api/llm.py', 
        'api/crew_orchestration.py',
        'api/swarm_orchestration.py',
        'api/quality_assurance.py',
        'api/flow_optimization.py',
        'api/workflow.py',
        'api/intelligence.py',
        'api/coverage.py',
        'api/health.py',
        'api/monitoring.py',
        'api/analytics.py',
        'api/comprehensive_monitoring.py',
        'test_ultra_reliability.py',
        'test_final_enhancements.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        full_path = dashboard_dir / file_path
        if full_path.exists():
            print(f"Processing: {full_path}")
            if fix_imports_in_file(full_path):
                print(f"  [FIXED] imports in {file_path}")
                fixed_count += 1
            else:
                print(f"  - No changes needed in {file_path}")
        else:
            print(f"  [NOT FOUND] File: {full_path}")
    
    print(f"\n[COMPLETE] Fixed imports in {fixed_count} files")
    print("Dashboard core import paths have been updated!")

if __name__ == "__main__":
    main()