#!/usr/bin/env python3
"""
Tripwire integrity check and auto-restore script - Python version for cross-platform compatibility
"""

import json
import os
import sys
import hashlib
import shutil
import argparse
from pathlib import Path


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description='Verify integrity of critical files and auto-restore if needed')
    parser.add_argument('--repo-root', default="C:\\Users\\kbass\\OneDrive\\Documents\\testmaster",
                        help='Repository root directory')
    parser.add_argument('--update-hashes', action='store_true', 
                        help='Update stored hashes for existing files')
    
    args = parser.parse_args()
    
    repo_root = Path(args.repo_root)
    tripwire_path = repo_root / "tools" / "codebase_monitor" / "tripwire.json"
    
    if not tripwire_path.exists():
        print(f"Error: Tripwire file missing at {tripwire_path}")
        return 1
    
    # Load tripwire config
    with open(tripwire_path, 'r', encoding='utf-8') as f:
        tripwire_config = json.load(f)
    
    critical_files = tripwire_config.get('critical_files', [])
    stored_hashes = tripwire_config.get('hashes', {})
    
    actions_taken = []
    integrity_violations = []
    
    for rel_file_path in critical_files:
        file_path = repo_root / rel_file_path.replace('\\', os.sep)
        backup_name = file_path.name
        backup_path = repo_root / "tools" / "codebase_monitor" / "backup" / backup_name
        
        if not file_path.exists():
            # File is missing - try to restore from backup
            if backup_path.exists():
                try:
                    # Create parent directories if needed
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(backup_path, file_path)
                    action = f"Restored missing {rel_file_path} from backup"
                    print(action)
                    actions_taken.append(action)
                except Exception as e:
                    error = f"Failed to restore {rel_file_path}: {e}"
                    print(f"ERROR: {error}")
                    actions_taken.append(error)
            else:
                error = f"Missing critical file: {rel_file_path} and no backup available"
                print(f"ERROR: {error}")
                integrity_violations.append(error)
        else:
            # File exists - check integrity if we have a stored hash
            current_hash = calculate_file_hash(file_path)
            stored_hash = stored_hashes.get(rel_file_path, "")
            
            if stored_hash and current_hash != stored_hash:
                # Hash mismatch - potential corruption or unauthorized change
                warning = f"Integrity violation: {rel_file_path} hash mismatch"
                print(f"WARNING: {warning}")
                integrity_violations.append(warning)
                
                # Optionally restore from backup if available
                if backup_path.exists():
                    backup_hash = calculate_file_hash(backup_path)
                    if backup_hash == stored_hash:
                        print(f"  Backup hash matches stored hash, restoring...")
                        try:
                            shutil.copy2(backup_path, file_path)
                            action = f"Restored corrupted {rel_file_path} from backup"
                            print(f"  {action}")
                            actions_taken.append(action)
                        except Exception as e:
                            error = f"Failed to restore {rel_file_path}: {e}"
                            print(f"  ERROR: {error}")
                            actions_taken.append(error)
            elif not stored_hash and args.update_hashes:
                # No stored hash - add current hash
                stored_hashes[rel_file_path] = current_hash
                action = f"Added hash for {rel_file_path}"
                print(action)
                actions_taken.append(action)
    
    # Update tripwire config if hashes were updated
    if args.update_hashes:
        tripwire_config['hashes'] = stored_hashes
        with open(tripwire_path, 'w', encoding='utf-8') as f:
            json.dump(tripwire_config, f, indent=2)
        print(f"Updated hashes in {tripwire_path}")
    
    # Save actions log
    if actions_taken:
        log_path = repo_root / "tools" / "codebase_monitor" / "outputs" / "integrity_actions.log"
        with open(log_path, 'w', encoding='utf-8') as f:
            for action in actions_taken:
                f.write(action + '\n')
        print(f"\nActions log saved to {log_path}")
    
    # Report summary
    print(f"\nIntegrity Check Summary:")
    print(f"  Critical files checked: {len(critical_files)}")
    print(f"  Actions taken: {len(actions_taken)}")
    print(f"  Integrity violations: {len(integrity_violations)}")
    
    # Return appropriate exit code
    if integrity_violations:
        print("\n[WARNING] Integrity violations detected!")
        return 2
    elif actions_taken:
        print("\n[SUCCESS] Integrity check completed with restorations.")
        return 0
    else:
        print("\n[SUCCESS] All critical files are intact.")
        return 0


if __name__ == "__main__":
    sys.exit(main())