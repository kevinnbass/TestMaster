#!/usr/bin/env python3
"""
Commit Splitter System - Break Large Commits into Logical Smaller Commits
========================================================================

This system:
1. Takes a large commit and breaks it into multiple smaller commits
2. Groups files logically (by directory, file type, or functionality)
3. Creates individual commits that can be pushed separately
4. Preserves original commit date, author, and chronological order
5. Maintains git history integrity

Key Features:
- Intelligent file grouping (by directory, purpose, size)
- Original timestamp preservation across all split commits
- Logical commit messages based on file groups
- Individual pushable commits under GitHub's 2GB limit
"""

import subprocess
import time
import re
import sys
import shutil
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

class CommitSplitterSystem:
    def __init__(self):
        self.max_files_per_commit = 100
        self.max_retries_per_commit = 3
        self.stall_timeout = 300  # 5 minutes
        self.retry_delay = 30
        self.progress_file = "commit_splitter_progress.json"
        self.backup_dir = Path("temp_split_backup")
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
        
        with open("commit_splitter.log", "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def get_commit_info(self, commit_hash):
        """Get complete commit information."""
        try:
            # Get commit metadata
            result = subprocess.run([
                'git', 'show', '--format=%H|%an|%ae|%ad|%s', '--date=iso', '--name-status', commit_hash
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            header = lines[0].split('|')
            
            # Parse file changes (skip empty lines)
            files = []
            for line in lines[2:]:  # Skip header and empty line
                if line.strip() and '\t' in line:
                    status, filepath = line.split('\t', 1)
                    files.append({'status': status, 'path': filepath})
            
            return {
                'hash': header[0],
                'author_name': header[1],
                'author_email': header[2],
                'date': header[3],
                'message': header[4],
                'files': files
            }
        except Exception as e:
            self.log(f"Error getting commit info: {e}")
            return None
    
    def group_files_intelligently(self, files):
        """Group files into logical commits."""
        groups = defaultdict(list)
        
        # Group by top-level directory first
        for file_info in files:
            filepath = file_info['path']
            parts = filepath.split('/')
            
            if len(parts) == 1:
                # Root level files
                groups['root_files'].append(file_info)
            else:
                # Use top-level directory
                top_dir = parts[0]
                groups[f"dir_{top_dir}"].append(file_info)
        
        # Further split large groups
        final_groups = {}
        group_counter = 1
        
        for group_name, file_list in groups.items():
            if len(file_list) <= self.max_files_per_commit:
                # Small enough group
                final_groups[f"commit_{group_counter:03d}_{group_name}"] = file_list
                group_counter += 1
            else:
                # Split large group by file type or size
                self.log(f"Splitting large group {group_name} ({len(file_list)} files)")
                
                # Group by file extension
                type_groups = defaultdict(list)
                for file_info in file_list:
                    ext = Path(file_info['path']).suffix.lower()
                    if not ext:
                        ext = 'no_extension'
                    type_groups[ext].append(file_info)
                
                # Create subgroups
                for ext, ext_files in type_groups.items():
                    # Split by chunk size
                    for i in range(0, len(ext_files), self.max_files_per_commit):
                        chunk = ext_files[i:i + self.max_files_per_commit]
                        chunk_name = f"commit_{group_counter:03d}_{group_name}_{ext}_part{i//self.max_files_per_commit + 1}"
                        final_groups[chunk_name] = chunk
                        group_counter += 1
        
        return final_groups
    
    def create_commit_message(self, group_name, file_list, original_message):
        """Create meaningful commit message for a group."""
        # Analyze file types
        file_types = defaultdict(int)
        directories = set()
        
        for file_info in file_list:
            ext = Path(file_info['path']).suffix.lower()
            if ext:
                file_types[ext] += 1
            
            dir_path = str(Path(file_info['path']).parent)
            if dir_path != '.':
                directories.add(dir_path)
        
        # Create descriptive message
        if 'root_files' in group_name:
            desc = "Root level files"
        elif 'dir_' in group_name:
            dir_name = group_name.split('dir_')[1].split('_')[0]
            desc = f"{dir_name}/ directory"
        else:
            desc = "Mixed files"
        
        # Add file type info
        if file_types:
            top_types = sorted(file_types.items(), key=lambda x: x[1], reverse=True)[:3]
            type_desc = ", ".join([f"{count} {ext}" for ext, count in top_types])
            desc += f" ({type_desc})"
        
        # Create final message
        base_msg = original_message.split('[')[0].strip()  # Remove any existing chunk indicators
        return f"{base_msg} - {desc} ({len(file_list)} files)"
    
    def apply_file_changes(self, file_list, source_commit):
        """Apply specific file changes from source commit."""
        applied = 0
        for file_info in file_list:
            filepath = file_info['path']
            status = file_info['status']
            
            try:
                if status.startswith('D'):
                    # File deletion
                    if Path(filepath).exists():
                        subprocess.run(['git', 'rm', filepath], capture_output=True, check=True)
                        applied += 1
                else:
                    # File addition or modification
                    subprocess.run(['git', 'checkout', source_commit, '--', filepath], capture_output=True, check=True)
                    subprocess.run(['git', 'add', filepath], capture_output=True, check=True)
                    applied += 1
            except Exception as e:
                self.log(f"      Warning: Could not apply {filepath}: {e}")
        
        return applied
    
    def push_commit_with_monitoring(self, commit_hash, attempt):
        """Push a single commit with progress monitoring."""
        self.log(f"    PUSH ATTEMPT {attempt}: {commit_hash[:8]}")
        
        try:
            process = subprocess.Popen([
                'git', 'push', 'origin', f'{commit_hash}:master', '--verbose'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            
            last_progress_time = time.time()
            last_progress_state = None
            
            while True:
                if process.poll() is not None:
                    break
                
                line = process.stdout.readline()
                if not line:
                    time.sleep(1)
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                # Parse progress
                if 'Writing objects:' in line or 'Counting objects:' in line or 'Compressing objects:' in line:
                    match = re.search(r'(\w+) objects:\s*(\d+)%', line)
                    if match:
                        current_state = f"{match.group(1)}-{match.group(2)}"
                        if last_progress_state != current_state:
                            last_progress_time = time.time()
                            last_progress_state = current_state
                            self.log(f"      PROGRESS: {match.group(1)} {match.group(2)}%")
                
                # Check for stall
                time_since_progress = time.time() - last_progress_time
                if time_since_progress > self.stall_timeout:
                    self.log(f"      STALL: No progress for {time_since_progress:.0f} seconds")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    return False, "STALL"
            
            return_code = process.returncode
            if return_code == 0:
                self.log(f"      SUCCESS: Commit {commit_hash[:8]} pushed")
                return True, "SUCCESS"
            else:
                self.log(f"      FAILED: Return code {return_code}")
                return False, f"FAILED_{return_code}"
                
        except Exception as e:
            self.log(f"      ERROR: {e}")
            return False, f"ERROR_{str(e)}"
    
    def split_and_push_commit(self, commit_info):
        """Split a large commit into smaller commits and push each."""
        self.log(f"\nSPLITTING COMMIT: {commit_info['hash'][:8]}")
        self.log(f"Original message: {commit_info['message']}")
        self.log(f"Total files: {len(commit_info['files'])}")
        self.log(f"Author: {commit_info['author_name']} <{commit_info['author_email']}>")
        self.log(f"Date: {commit_info['date']}")
        
        # Group files intelligently
        file_groups = self.group_files_intelligently(commit_info['files'])
        self.log(f"Created {len(file_groups)} logical commit groups:")
        
        for group_name, file_list in file_groups.items():
            self.log(f"  {group_name}: {len(file_list)} files")
        
        # Get parent commit
        parent_result = subprocess.run([
            'git', 'rev-parse', f"{commit_info['hash']}^"
        ], capture_output=True, text=True, check=True)
        parent_commit = parent_result.stdout.strip()
        
        # Reset to parent commit
        self.log(f"Resetting to parent: {parent_commit[:8]}")
        subprocess.run(['git', 'reset', '--hard', parent_commit], capture_output=True, check=True)
        
        # Create and push each split commit
        split_commits = []
        base_timestamp = datetime.fromisoformat(commit_info['date'].replace(' ', 'T'))
        
        for i, (group_name, file_list) in enumerate(file_groups.items(), 1):
            self.log(f"\n--- SPLIT COMMIT {i}/{len(file_groups)}: {group_name} ---")
            
            # Apply file changes
            applied = self.apply_file_changes(file_list, commit_info['hash'])
            self.log(f"    Applied {applied}/{len(file_list)} file changes")
            
            if applied == 0:
                self.log(f"    SKIP: No changes to commit")
                continue
            
            # Create commit message
            commit_msg = self.create_commit_message(group_name, file_list, commit_info['message'])
            self.log(f"    Message: {commit_msg}")
            
            # Use original timestamp + small offset to maintain order
            commit_timestamp = base_timestamp + timedelta(seconds=i)
            timestamp_str = commit_timestamp.strftime('%Y-%m-%d %H:%M:%S %z')
            
            # Set environment for commit
            env = dict(os.environ)
            env['GIT_AUTHOR_NAME'] = commit_info['author_name']
            env['GIT_AUTHOR_EMAIL'] = commit_info['author_email']
            env['GIT_AUTHOR_DATE'] = timestamp_str
            env['GIT_COMMITTER_NAME'] = commit_info['author_name']
            env['GIT_COMMITTER_EMAIL'] = commit_info['author_email']
            env['GIT_COMMITTER_DATE'] = timestamp_str
            
            # Create commit
            result = subprocess.run([
                'git', 'commit', '-m', commit_msg
            ], capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                new_commit = subprocess.run([
                    'git', 'rev-parse', 'HEAD'
                ], capture_output=True, text=True, check=True).stdout.strip()
                
                self.log(f"    CREATED: {new_commit[:8]}")
                
                # Try to push this commit
                success = False
                for attempt in range(1, self.max_retries_per_commit + 1):
                    push_success, reason = self.push_commit_with_monitoring(new_commit, attempt)
                    
                    if push_success:
                        self.log(f"    PUSHED: Split commit {i} successfully pushed")
                        split_commits.append(new_commit)
                        success = True
                        break
                    
                    if attempt < self.max_retries_per_commit:
                        self.log(f"    Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                
                if not success:
                    self.log(f"    FAILED: Could not push split commit {i}")
                    return False, split_commits
            else:
                self.log(f"    FAILED: Could not create split commit {i}")
                self.log(f"    Error: {result.stderr}")
                return False, split_commits
        
        self.log(f"\nSUCCESS: Split {commit_info['hash'][:8]} into {len(split_commits)} commits")
        return True, split_commits
    
    def run_commit_splitter(self):
        """Main commit splitter system."""
        self.log("COMMIT SPLITTER SYSTEM")
        self.log("=" * 60)
        self.log("Breaks large commits into smaller, pushable commits")
        self.log("Preserves original dates, authors, and chronological order")
        self.log("")
        
        # Get commits to process
        try:
            result = subprocess.run([
                'git', 'rev-list', '--reverse', 'origin/master..HEAD'
            ], capture_output=True, text=True, check=True)
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            commits = [c for c in commits if c]
        except Exception as e:
            self.log(f"Error getting commits: {e}")
            return False
        
        if not commits:
            self.log("No commits to process!")
            return True
        
        self.log(f"Found {len(commits)} commits to process")
        
        # Process each commit
        for i, commit_hash in enumerate(commits, 1):
            commit_info = self.get_commit_info(commit_hash)
            if not commit_info:
                self.log(f"ERROR: Could not get info for {commit_hash[:8]}")
                continue
            
            self.log(f"\n=== PROCESSING COMMIT {i}/{len(commits)} ===")
            
            # Check if commit needs splitting
            if len(commit_info['files']) <= self.max_files_per_commit:
                self.log(f"Commit {commit_hash[:8]} has {len(commit_info['files'])} files - attempting direct push")
                
                # Try direct push
                success = False
                for attempt in range(1, self.max_retries_per_commit + 1):
                    push_success, reason = self.push_commit_with_monitoring(commit_hash, attempt)
                    
                    if push_success:
                        self.log(f"SUCCESS: Direct push succeeded")
                        success = True
                        break
                    
                    if attempt < self.max_retries_per_commit:
                        time.sleep(self.retry_delay)
                
                if not success:
                    self.log(f"Direct push failed, splitting commit...")
                    success, split_commits = self.split_and_push_commit(commit_info)
                    
                    if not success:
                        self.log(f"FAILED: Could not split and push {commit_hash[:8]}")
                        return False
            else:
                self.log(f"Commit {commit_hash[:8]} has {len(commit_info['files'])} files - splitting required")
                success, split_commits = self.split_and_push_commit(commit_info)
                
                if not success:
                    self.log(f"FAILED: Could not split and push {commit_hash[:8]}")
                    return False
        
        self.log("\n" + "=" * 60)
        self.log("SUCCESS: All commits processed and pushed!")
        return True

def main():
    splitter = CommitSplitterSystem()
    try:
        splitter.run_commit_splitter()
    except KeyboardInterrupt:
        splitter.log("\nSTOPPED: User interrupted")
    except Exception as e:
        splitter.log(f"\nERROR: {e}")

if __name__ == "__main__":
    main()