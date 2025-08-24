#!/usr/bin/env python3
"""
Fix TestMaster Import Paths
============================
Fixes import paths in testmaster package to use correct paths.
"""

import os
import re
from pathlib import Path

def fix_imports():
    """Fix all import issues in testmaster package."""
    
    testmaster_dir = Path('testmaster')
    
    # Find all Python files
    python_files = list(testmaster_dir.rglob('*.py'))
    
    print(f"Found {len(python_files)} Python files to check")
    
    fixed_count = 0
    for file_path in python_files:
        try:
            content = file_path.read_text(encoding='utf-8')
            original_content = content
            
            # Fix various import patterns
            # Pattern 1: from testmaster.core.X -> from core.X
            content = re.sub(
                r'from testmaster\.core\.([a-zA-Z_][a-zA-Z0-9_]*)',
                r'from core.\1',
                content
            )
            
            # Pattern 2: from ..core.X -> from core.X (for nested modules)
            content = re.sub(
                r'from \.\.core\.([a-zA-Z_][a-zA-Z0-9_]*)',
                r'from core.\1',
                content
            )
            
            # Pattern 3: import testmaster.core.X -> import core.X
            content = re.sub(
                r'import testmaster\.core\.([a-zA-Z_][a-zA-Z0-9_]*)',
                r'import core.\1',
                content
            )
            
            # Pattern 4: Fix SharedState specific imports
            content = content.replace(
                'from testmaster.core.shared_state import',
                'from core.shared_state import'
            )
            content = content.replace(
                'from ..core.shared_state import',
                'from core.shared_state import'
            )
            
            # Pattern 5: Fix observability imports
            content = content.replace(
                'from testmaster.core.observability import',
                'from core.observability import'
            )
            content = content.replace(
                'from ..core.observability import',
                'from core.observability import'
            )
            
            # Pattern 6: Fix state imports
            content = content.replace(
                'from testmaster.state.',
                'from state.'
            )
            content = content.replace(
                'from ..state.',
                'from state.'
            )
            
            # Pattern 7: Fix monitoring imports
            content = content.replace(
                'from testmaster.monitoring.',
                'from monitoring.'
            )
            
            # Pattern 8: Fix observability unified imports
            content = content.replace(
                'from testmaster.observability.',
                'from observability.'
            )
            
            if content != original_content:
                file_path.write_text(content, encoding='utf-8')
                fixed_count += 1
                print(f"  Fixed: {file_path}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
    
    print(f"\nFixed {fixed_count} files")
    
    # Also need to add TestMaster to Python path in key files
    init_files = [
        'testmaster/__init__.py',
        'testmaster/generators/__init__.py',
        'testmaster/intelligence/__init__.py'
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            
            # Add path setup at the beginning if not present
            if 'sys.path.insert' not in content:
                lines = content.split('\n')
                
                # Find the first non-comment, non-docstring line
                insert_pos = 0
                in_docstring = False
                for i, line in enumerate(lines):
                    if '"""' in line or "'''" in line:
                        in_docstring = not in_docstring
                    elif not in_docstring and not line.startswith('#') and line.strip():
                        insert_pos = i
                        break
                
                # Insert path setup
                lines.insert(insert_pos, '''import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
''')
                
                path.write_text('\n'.join(lines), encoding='utf-8')
                print(f"  Added path setup to: {init_file}")

def main():
    """Main function."""
    print("Fixing TestMaster import paths...")
    print("="*60)
    
    fix_imports()
    
    print("\n" + "="*60)
    print("[COMPLETE] Import paths fixed!")

if __name__ == "__main__":
    main()