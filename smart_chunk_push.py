#!/usr/bin/env python3
"""
Smart Chunk Push - No History Rewrite!
======================================

Creates new commits that push the same files in smaller chunks,
without rewriting git history. We temporarily revert the big commit,
then re-add files in small batches.
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime

def safe_print(msg):
    print(msg)
    sys.stdout.flush()

class SmartChunkPush:
    def __init__(self):
        self.big_commit = "154482eb2ac24e8d2f92a37ba499efb7e9a28a9a"
        self.chunk_size = 100  # Files per chunk
        self.backup_dir = Path("temp_backup_for_push")
        
    def get_changed_files(self, commit_hash):
        """Get list of files changed in commit."""
        try:
            result = subprocess.run([
                'git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash
            ], capture_output=True, text=True, check=True)
            
            files = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return [f for f in files if f]
        except Exception as e:
            safe_print(f"Error: {e}")
            return []
    
    def create_backup_of_current_state(self):
        """Backup current state of files."""
        safe_print("\n[BACKUP] Creating backup of current state...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        # Get files from the big commit
        files = self.get_changed_files(self.big_commit)
        
        # Copy each file to backup
        for file_path in files:
            src = Path(file_path)
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    safe_print(f"  Warning: Could not backup {file_path}: {e}")
        
        safe_print(f"[BACKUP] Backed up {len(files)} files")
        return files
    
    def restore_from_backup(self, file_list):
        """Restore specific files from backup."""
        restored = 0
        for file_path in file_list:
            src = self.backup_dir / file_path
            dst = Path(file_path)
            
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                    restored += 1
                except Exception as e:
                    safe_print(f"  Warning: Could not restore {file_path}: {e}")
        
        return restored
    
    def push_with_retry(self, max_retries=5):
        """Try to push with retries."""
        for attempt in range(1, max_retries + 1):
            safe_print(f"  [ATTEMPT {attempt}/{max_retries}] Pushing...")
            
            try:
                result = subprocess.run([
                    'git', 'push', 'origin', 'HEAD:master'
                ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    safe_print(f"  [SUCCESS] Push completed!")
                    return True
                else:
                    safe_print(f"  [FAILED] {result.stderr[:100]}")
                    
            except subprocess.TimeoutExpired:
                safe_print(f"  [TIMEOUT] Push timed out after 5 minutes")
            except Exception as e:
                safe_print(f"  [ERROR] {e}")
            
            if attempt < max_retries:
                safe_print(f"  [RETRY] Waiting 10 seconds...")
                import time
                time.sleep(10)
        
        return False
    
    def run_smart_chunk_push(self):
        """Main execution - smart chunking without history rewrite."""
        safe_print("SMART CHUNK PUSH SYSTEM")
        safe_print("=" * 60)
        safe_print("This will NOT rewrite history - just create new commits")
        safe_print("")
        
        # Step 1: Analyze the big commit
        safe_print(f"[ANALYZE] Checking commit {self.big_commit[:8]}...")
        files = self.get_changed_files(self.big_commit)
        safe_print(f"[ANALYZE] Found {len(files)} files to push")
        
        if not files:
            safe_print("[ERROR] No files found")
            return False
        
        # Group files by directory
        file_groups = {}
        for file_path in files:
            parts = file_path.split('/')
            top_dir = parts[0] if len(parts) > 1 else "root"
            
            if top_dir not in file_groups:
                file_groups[top_dir] = []
            file_groups[top_dir].append(file_path)
        
        safe_print(f"\n[GROUPS] Files by directory:")
        for dir_name, dir_files in sorted(file_groups.items(), key=lambda x: len(x[1])):
            safe_print(f"  {dir_name}: {len(dir_files)} files")
        
        # Step 2: Create backup
        safe_print("\n" + "=" * 60)
        response = input("Create backup and proceed with chunked push? (yes/no): ")
        if response.lower() != 'yes':
            safe_print("Aborted.")
            return False
        
        self.create_backup_of_current_state()
        
        # Step 3: Temporarily revert the big commit's files
        safe_print("\n[REVERT] Temporarily reverting large commit files...")
        subprocess.run(['git', 'rm', '-r', '--cached', '.'], capture_output=True)
        subprocess.run(['git', 'add', '.'], capture_output=True)
        
        # Step 4: Re-add and push in chunks
        safe_print("\n[CHUNK] Starting chunked push process...")
        
        batch_num = 1
        total_batches = sum(
            (len(files) + self.chunk_size - 1) // self.chunk_size 
            for files in file_groups.values()
        )
        
        for dir_name, dir_files in sorted(file_groups.items(), key=lambda x: len(x[1])):
            # Process directory in chunks
            for i in range(0, len(dir_files), self.chunk_size):
                chunk = dir_files[i:i + self.chunk_size]
                
                safe_print(f"\n[BATCH {batch_num}/{total_batches}] {dir_name} ({len(chunk)} files)")
                
                # Restore files from backup
                restored = self.restore_from_backup(chunk)
                safe_print(f"  [RESTORE] Restored {restored} files")
                
                # Add files to git
                for file_path in chunk:
                    if Path(file_path).exists():
                        subprocess.run(['git', 'add', file_path], capture_output=True)
                
                # Commit
                commit_msg = f"Phase 1 Chunk {batch_num}/{total_batches}: {dir_name} ({len(chunk)} files)"
                result = subprocess.run([
                    'git', 'commit', '-m', commit_msg
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    safe_print(f"  [COMMIT] Created commit")
                    
                    # Try to push
                    if self.push_with_retry(max_retries=3):
                        safe_print(f"  [PUSHED] Batch {batch_num} successfully pushed!")
                    else:
                        safe_print(f"  [FAILED] Could not push batch {batch_num}")
                        safe_print(f"  [INFO] Will continue adding files and try again later")
                else:
                    safe_print(f"  [SKIP] No changes to commit")
                
                batch_num += 1
        
        # Step 5: Clean up
        safe_print("\n[CLEANUP] Removing temporary backup...")
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        safe_print("\n[COMPLETE] Chunked push process finished!")
        safe_print("Run 'git status' to check current state")
        
        return True

def main():
    pusher = SmartChunkPush()
    try:
        pusher.run_smart_chunk_push()
    except KeyboardInterrupt:
        safe_print("\n[STOPPED] Process interrupted")
    except Exception as e:
        safe_print(f"\n[ERROR] {e}")

if __name__ == "__main__":
    main()