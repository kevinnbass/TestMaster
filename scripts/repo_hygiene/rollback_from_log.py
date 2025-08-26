#!/usr/bin/env python3
"""
Rollback mechanism for cleanup operations
"""

import os
import shutil
import argparse
from pathlib import Path

def rollback_relocations(repo_root: str, artifact_root: str, dry_run: bool = True):
    """
    Rollback file relocations using the relocation log
    
    Args:
        repo_root: Repository root path
        artifact_root: Artifacts root path  
        dry_run: If True, only show what would be restored
    """
    
    repo_path = Path(repo_root).resolve()
    artifact_path = Path(artifact_root).resolve()
    log_path = repo_path / "tools" / "codebase_monitor" / "outputs" / "relocate_actions.log"
    
    if not log_path.exists():
        print(f"ERROR: Relocation log not found at {log_path}")
        return False
    
    print(f"{'[DRY-RUN] ' if dry_run else ''}Rolling back relocations from log...")
    
    actions = []
    restored_count = 0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and headers
            if not line or line.startswith('=') or '[DRY-RUN]' in line:
                continue
            
            # Parse "Moved X -> Y" lines
            if line.startswith('Moved ') and ' -> ' in line:
                try:
                    parts = line.replace('Moved ', '').split(' -> ')
                    if len(parts) == 2:
                        original_path = Path(parts[0].strip())
                        moved_path = Path(parts[1].strip())
                        
                        # Check if the moved file exists in artifacts
                        if moved_path.exists():
                            if dry_run:
                                print(f"[WOULD RESTORE] {moved_path} -> {original_path}")
                                actions.append(f"[DRY-RUN] Would restore {moved_path} -> {original_path}")
                            else:
                                try:
                                    # Create parent directory if needed
                                    original_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    # Move back to original location
                                    shutil.move(str(moved_path), str(original_path))
                                    print(f"[RESTORED] {moved_path} -> {original_path}")
                                    actions.append(f"Restored {moved_path} -> {original_path}")
                                    restored_count += 1
                                except Exception as e:
                                    error_msg = f"Failed to restore {moved_path}: {e}"
                                    print(f"ERROR: {error_msg}")
                                    actions.append(f"Error: {error_msg}")
                        else:
                            print(f"SKIP: File not found in artifacts: {moved_path}")
                            actions.append(f"Skipped (not found): {moved_path}")
                            
                except Exception as e:
                    print(f"ERROR parsing line {line_num}: {line} - {e}")
                    actions.append(f"Parse error line {line_num}: {e}")
    
    # Write rollback log
    rollback_log_path = repo_path / "tools" / "codebase_monitor" / "outputs" / "rollback_actions.log"
    with open(rollback_log_path, 'w', encoding='utf-8') as f:
        f.write(f"=== Rollback {'(DRY-RUN) ' if dry_run else ''}Log ===\n")
        f.write(f"Actions processed: {len(actions)}\n")
        if not dry_run:
            f.write(f"Files restored: {restored_count}\n")
        f.write("\nDetailed actions:\n")
        for action in actions:
            f.write(f"{action}\n")
    
    if dry_run:
        would_restore = sum(1 for a in actions if '[WOULD RESTORE]' in a)
        print(f"\n[DRY-RUN] Would restore {would_restore} items")
        print(f"Run with --execute to perform actual rollback")
    else:
        print(f"\nRestored {restored_count} items to repository")
    
    print(f"Rollback log written to: {rollback_log_path}")
    return True

def rollback_deletions(repo_root: str, artifact_root: str, dry_run: bool = True):
    """
    Attempt to rollback deletions (limited recovery from artifacts)
    
    Args:
        repo_root: Repository root path
        artifact_root: Artifacts root path
        dry_run: If True, only show what could be recovered
    """
    
    repo_path = Path(repo_root).resolve()
    artifact_path = Path(artifact_root).resolve()
    log_path = repo_path / "tools" / "codebase_monitor" / "outputs" / "dedupe_actions.log"
    
    if not log_path.exists():
        print(f"WARNING: Deletion log not found at {log_path}")
        return False
    
    print(f"{'[DRY-RUN] ' if dry_run else ''}Attempting recovery of deleted files...")
    
    actions = []
    recovered_count = 0
    
    with open(log_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Parse "Deleted: X" lines
            if line.startswith('Deleted: '):
                deleted_file = line.replace('Deleted: ', '').strip()
                
                # Look for the file in artifacts directory
                potential_backup = artifact_path / deleted_file
                original_location = repo_path / deleted_file
                
                if potential_backup.exists():
                    if dry_run:
                        print(f"[COULD RECOVER] {potential_backup} -> {original_location}")
                        actions.append(f"[DRY-RUN] Could recover {deleted_file}")
                    else:
                        try:
                            original_location.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(potential_backup), str(original_location))
                            print(f"[RECOVERED] {deleted_file}")
                            actions.append(f"Recovered {deleted_file}")
                            recovered_count += 1
                        except Exception as e:
                            error_msg = f"Failed to recover {deleted_file}: {e}"
                            print(f"ERROR: {error_msg}")
                            actions.append(f"Error: {error_msg}")
                else:
                    actions.append(f"No backup found for {deleted_file}")
    
    if dry_run:
        could_recover = sum(1 for a in actions if '[COULD RECOVER]' in a)
        print(f"\n[DRY-RUN] Could potentially recover {could_recover} deleted files")
    else:
        print(f"\nRecovered {recovered_count} deleted files from artifacts")
    
    return len(actions) > 0

def main():
    parser = argparse.ArgumentParser(description="Rollback cleanup operations")
    parser.add_argument("--repo-root", default=".", help="Repository root path")
    parser.add_argument("--artifact-root", 
                       default="../testmaster_artifacts", 
                       help="Artifacts directory path")
    parser.add_argument("--execute", action="store_true", 
                       help="Execute rollback (default is dry-run)")
    parser.add_argument("--type", choices=["relocations", "deletions", "all"], 
                       default="all", help="Type of rollback to perform")
    
    args = parser.parse_args()
    
    repo_root = os.path.abspath(args.repo_root)
    artifact_root = os.path.abspath(args.artifact_root)
    
    print(f"Repository root: {repo_root}")
    print(f"Artifact root: {artifact_root}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Type: {args.type}")
    print()
    
    success = True
    
    if args.type in ["relocations", "all"]:
        success &= rollback_relocations(repo_root, artifact_root, dry_run=not args.execute)
    
    if args.type in ["deletions", "all"]:
        success &= rollback_deletions(repo_root, artifact_root, dry_run=not args.execute)
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())