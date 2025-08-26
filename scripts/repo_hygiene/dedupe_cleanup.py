#!/usr/bin/env python3
"""
Dedupe cleanup script - Python version for cross-platform compatibility
Conservatively removes duplicates from generated/temp directories only
"""

import json
import os
import sys
import argparse
from pathlib import Path


def is_safe_to_remove(file_path: str) -> bool:
    """Check if file is safe to remove (in generated/temp directories)."""
    path_lower = file_path.lower().replace('\\', '/')
    
    safe_patterns = [
        '/telemetry/',
        '/logs/',
        '/production_packages/',
        '/__pycache__/',
        '/.pytest_cache/',
        '/node_modules/',
        '/dist/',
        '/build/',
        '/coverage/',
        '.log',
        '.tmp',
        '.bak',
        '.pyc',
        '.pyo'
    ]
    
    return any(pattern in path_lower for pattern in safe_patterns)


def main():
    parser = argparse.ArgumentParser(description='Conservative duplicate file cleanup')
    parser.add_argument('--repo-root', default="C:\\Users\\kbass\\OneDrive\\Documents\\testmaster",
                        help='Repository root directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root)
    
    # First save top duplicate groups
    scan_path = repo_root / "tools" / "codebase_monitor" / "reports" / "latest" / "scan.json"
    if not scan_path.exists():
        print(f"Error: Scan report not found at {scan_path}")
        return 1
    
    with open(scan_path, 'r', encoding='utf-8') as f:
        scan_data = json.load(f)
    
    duplicates = scan_data.get('duplicates', [])
    
    # Save top 200 duplicate groups
    top_groups = duplicates[:200]
    groups_path = repo_root / "tools" / "codebase_monitor" / "outputs" / "duplicate_groups_topN.json"
    with open(groups_path, 'w', encoding='utf-8') as f:
        json.dump(top_groups, f, indent=2)
    
    print(f"Saved top {len(top_groups)} duplicate groups to {groups_path}")
    
    # Process cleanup
    actions_taken = []
    log_path = repo_root / "tools" / "codebase_monitor" / "outputs" / "dedupe_actions.log"
    
    for group in top_groups:
        if len(group) < 2:
            continue
        
        # Keep the first file, consider removing others
        keep_file = group[0]
        
        for i in range(1, len(group)):
            candidate = group[i]
            candidate_path = repo_root / candidate
            
            if is_safe_to_remove(candidate):
                if args.dry_run:
                    action = f"[DRY-RUN] Would remove {candidate}"
                    print(action)
                    actions_taken.append(action)
                else:
                    try:
                        if candidate_path.exists():
                            candidate_path.unlink()  # Remove file
                            action = f"Removed {candidate}"
                            print(action)
                            actions_taken.append(action)
                    except Exception as e:
                        error = f"Failed to remove {candidate}: {e}"
                        print(error)
                        actions_taken.append(error)
            else:
                # Not safe to remove
                action = f"Skipped (not safe): {candidate}"
                actions_taken.append(action)
    
    # Save actions log
    with open(log_path, 'w', encoding='utf-8') as f:
        for action in actions_taken:
            f.write(action + '\n')
    
    print(f"\nCompleted. Log saved to {log_path}")
    print(f"Actions taken: {len(actions_taken)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())