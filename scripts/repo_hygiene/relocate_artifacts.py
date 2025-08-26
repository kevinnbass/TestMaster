#!/usr/bin/env python3
"""
Relocate artifacts script - Python version for cross-platform compatibility
"""

import json
import os
import shutil
import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description='Relocate artifacts to external directory')
    parser.add_argument('--repo-root', default="C:\\Users\\kbass\\OneDrive\\Documents\\testmaster",
                        help='Repository root directory')
    parser.add_argument('--artifact-root', default="C:\\Users\\kbass\\OneDrive\\Documents\\testmaster_artifacts",
                        help='Artifacts destination directory')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root)
    artifact_root = Path(args.artifact_root)
    manifest_path = repo_root / "tools" / "codebase_monitor" / "outputs" / "artifact_manifest.json"
    
    if not manifest_path.exists():
        print(f"Error: Manifest not found at {manifest_path}")
        return 1
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    patterns = manifest.get('ignore_patterns', [])
    actions_taken = []
    
    for pattern in patterns:
        # Convert pattern to actual paths
        src_path = repo_root / pattern.rstrip('/')
        
        if src_path.exists():
            # Calculate destination
            rel_path = src_path.relative_to(repo_root)
            dst_path = artifact_root / rel_path
            
            if args.dry_run:
                action = f"[DRY-RUN] Would move {src_path} -> {dst_path}"
                print(action)
                actions_taken.append(action)
            else:
                try:
                    # Create parent directory
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move the file/directory
                    shutil.move(str(src_path), str(dst_path))
                    action = f"Moved {src_path} -> {dst_path}"
                    print(action)
                    actions_taken.append(action)
                except Exception as e:
                    error = f"Failed to move {src_path}: {e}"
                    print(error)
                    actions_taken.append(error)
    
    # Save actions log
    log_path = repo_root / "tools" / "codebase_monitor" / "outputs" / "relocate_actions.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        for action in actions_taken:
            f.write(action + '\n')
    
    print(f"\nCompleted. Log saved to {log_path}")
    print(f"Actions taken: {len(actions_taken)}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())