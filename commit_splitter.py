#!/usr/bin/env python3
"""
Commit Splitter - Break Large Commits Into Smaller Individual Commits
====================================================================

This system rewrites git history by:
1. Taking a large commit and splitting it into smaller commits
2. Preserving the original commit date and author
3. Maintaining chronological order 
4. Creating multiple smaller commits that each stay under GitHub's limits

Key: This DOES rewrite history, but preserves all dates and ordering.
"""

import subprocess
import time
import sys
import shutil
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

class CommitSplitter:
    def __init__(self):
        self.max_files_per_commit = 50  # Conservative limit
        self.backup_dir = Path("commit_split_backup")
        self.split_log = "commit_split.log"
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
        
        with open(self.split_log, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def get_commit_details(self, commit_hash):
        """Get complete commit information."""
        try:
            # Get commit metadata
            result = subprocess.run([
                'git', 'show', '--format=%H|%an|%ae|%ad|%cd|%s', '--date=iso', '--name-only', commit_hash
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            header = lines[0].split('|')
            files = [f for f in lines[2:] if f.strip()]  # Skip empty line
            
            return {
                'hash': header[0],
                'author_name': header[1],
                'author_email': header[2],
                'author_date': header[3],
                'commit_date': header[4],
                'message': header[5],
                'files': files
            }
        except Exception as e:
            self.log(f"Error getting commit details: {e}")
            return None
    
    def get_file_sizes(self, commit_hash, files):
        """Get file sizes for intelligent grouping."""
        file_info = []
        
        for file_path in files:
            try:
                # Get file size from git
                result = subprocess.run([
                    'git', 'show', f'{commit_hash}:{file_path}'
                ], capture_output=True, text=True)
                
                size = len(result.stdout.encode('utf-8'))
                file_info.append({
                    'path': file_path,
                    'size': size
                })
            except:
                # If file doesn't exist or error, assume small
                file_info.append({
                    'path': file_path,
                    'size': 1000  # Default small size
                })
        
        return file_info
    
    def group_files_intelligently(self, file_info, max_files, max_size_mb=10):
        """Group files into commits by size and count."""
        max_size_bytes = max_size_mb * 1024 * 1024
        groups = []
        current_group = []
        current_size = 0
        
        # Sort by size (largest first) for better packing
        sorted_files = sorted(file_info, key=lambda x: x['size'], reverse=True)
        
        for file_info_item in sorted_files:
            file_size = file_info_item['size']
            
            # Check if adding this file would exceed limits
            if (len(current_group) >= max_files or 
                (current_size + file_size) > max_size_bytes):
                
                # Start new group if current has files
                if current_group:
                    groups.append(current_group)
                    current_group = []
                    current_size = 0
            
            current_group.append(file_info_item['path'])
            current_size += file_size
        
        # Add final group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def split_commit_into_multiple(self, commit_hash):
        """Split a single commit into multiple smaller commits."""
        self.log(f"SPLITTING: Analyzing commit {commit_hash[:8]}")
        
        # Get commit details
        commit_info = self.get_commit_details(commit_hash)
        if not commit_info:
            self.log("ERROR: Could not get commit details")
            return False
        
        self.log(f"Original commit: {len(commit_info['files'])} files")
        self.log(f"Author: {commit_info['author_name']} <{commit_info['author_email']}>")
        self.log(f"Date: {commit_info['author_date']}")
        self.log(f"Message: {commit_info['message']}")
        
        # Get file sizes for intelligent grouping
        self.log("Analyzing file sizes...")
        file_info = self.get_file_sizes(commit_hash, commit_info['files'])
        
        # Group files intelligently
        file_groups = self.group_files_intelligently(file_info, self.max_files_per_commit)
        self.log(f"Split into {len(file_groups)} commits:")
        
        for i, group in enumerate(file_groups, 1):
            total_size = sum(next(f['size'] for f in file_info if f['path'] == path) for path in group)
            self.log(f"  Commit {i}: {len(group)} files, ~{total_size/1024/1024:.1f}MB")
        
        # Confirm with user
        response = input(f"\nSplit {commit_hash[:8]} into {len(file_groups)} commits? (yes/no): ")
        if response.lower() != 'yes':
            return False
        
        # Reset to parent of the commit we're splitting
        parent_commit = f"{commit_hash}^"
        subprocess.run(['git', 'reset', '--hard', parent_commit], check=True)
        self.log(f"Reset to parent commit: {parent_commit}")
        
        # Create each split commit
        base_date = datetime.fromisoformat(commit_info['author_date'].replace(' ', 'T'))
        
        for i, file_group in enumerate(file_groups):
            self.log(f"\nCreating split commit {i+1}/{len(file_groups)}")
            
            # Checkout files for this group
            for file_path in file_group:
                try:
                    subprocess.run([
                        'git', 'checkout', commit_hash, '--', file_path
                    ], check=True)
                    subprocess.run(['git', 'add', file_path], check=True)
                except Exception as e:
                    self.log(f"  Warning: Could not add {file_path}: {e}")
            
            # Create commit message
            if len(file_groups) == 1:
                commit_msg = commit_info['message']
            else:
                commit_msg = f"{commit_info['message']} [PART {i+1}/{len(file_groups)}]"
            
            # Add small time offset to maintain order (1 second apart)
            commit_date = base_date + timedelta(seconds=i)
            commit_date_str = commit_date.strftime('%Y-%m-%d %H:%M:%S %z')
            
            # Set environment for date/author preservation
            env = dict(os.environ)
            env['GIT_AUTHOR_NAME'] = commit_info['author_name']
            env['GIT_AUTHOR_EMAIL'] = commit_info['author_email']
            env['GIT_AUTHOR_DATE'] = commit_date_str
            env['GIT_COMMITTER_NAME'] = commit_info['author_name']
            env['GIT_COMMITTER_EMAIL'] = commit_info['author_email']
            env['GIT_COMMITTER_DATE'] = commit_date_str
            
            # Create the commit
            result = subprocess.run([
                'git', 'commit', '-m', commit_msg
            ], capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                new_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                        capture_output=True, text=True).stdout.strip()
                self.log(f"  Created: {new_hash[:8]} with {len(file_group)} files")
            else:
                self.log(f"  ERROR: Failed to create commit {i+1}")
                self.log(f"  {result.stderr}")
                return False
        
        self.log(f"\nSUCCESS: Split {commit_hash[:8]} into {len(file_groups)} commits")
        return True
    
    def analyze_commits_for_splitting(self):
        """Analyze which commits need splitting."""
        self.log("COMMIT SPLITTER - Analyzing Repository")
        self.log("=" * 60)
        
        # Get commits ahead of origin
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
            self.log("No commits to analyze")
            return True
        
        self.log(f"Analyzing {len(commits)} commits...\n")
        
        # Analyze each commit
        large_commits = []
        for commit_hash in commits:
            commit_info = self.get_commit_details(commit_hash)
            if commit_info:
                file_count = len(commit_info['files'])
                self.log(f"{commit_hash[:8]}: {file_count:4d} files - {commit_info['message'][:50]}")
                
                # Consider commits with >100 files as potentially too large
                if file_count > 100:
                    large_commits.append((commit_hash, file_count, commit_info['message']))
        
        if not large_commits:
            self.log("\nNo large commits found - all should push normally")
            return True
        
        self.log(f"\nFound {len(large_commits)} large commits that may need splitting:")
        for commit_hash, count, message in large_commits:
            self.log(f"  {commit_hash[:8]}: {count} files - {message[:50]}")
        
        self.log(f"\nRecommendation: Split commits with >50 files for reliable pushing")
        
        # Ask if user wants to split the largest one
        if large_commits:
            largest = max(large_commits, key=lambda x: x[1])
            self.log(f"\nLargest commit: {largest[0][:8]} ({largest[1]} files)")
            
            response = input(f"Split the largest commit {largest[0][:8]}? (yes/no): ")
            if response.lower() == 'yes':
                return self.split_commit_into_multiple(largest[0])
        
        return True

def main():
    splitter = CommitSplitter()
    try:
        splitter.analyze_commits_for_splitting()
    except KeyboardInterrupt:
        splitter.log("\nSTOPPED: User interrupted")
    except Exception as e:
        splitter.log(f"\nERROR: {e}")

if __name__ == "__main__":
    main()