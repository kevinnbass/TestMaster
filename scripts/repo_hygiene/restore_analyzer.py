#!/usr/bin/env python3
"""
Restore analyzer.py from backup if missing
"""

import os
import shutil
from pathlib import Path

def restore_analyzer():
    root = Path(__file__).parent.parent.parent
    analyzer_path = root / "tools" / "codebase_monitor" / "analyzer.py"
    backup_path = root / "tools" / "codebase_monitor" / "backup" / "analyzer.py"
    
    if not analyzer_path.exists() and backup_path.exists():
        print(f"Restoring analyzer.py from backup...")
        shutil.copy2(backup_path, analyzer_path)
        print(f"Restored: {analyzer_path}")
        return True
    elif not analyzer_path.exists():
        print(f"ERROR: analyzer.py missing and no backup found at {backup_path}")
        return False
    else:
        print(f"analyzer.py exists at {analyzer_path}")
        return True

if __name__ == "__main__":
    import sys
    success = restore_analyzer()
    sys.exit(0 if success else 1)