#!/usr/bin/env python3
"""
Hybrid Push System - Incremental with Chunked Fallback
=====================================================

Strategy:
1. Try pushing commits one at a time (oldest first)
2. Monitor for stalls (5 minutes with no progress)
3. If commit fails 3 times, break it into smaller chunks
4. Preserve original commit dates and author info
5. Maintain chronological order

Key Features:
- Progress monitoring with stall detection
- Date/author preservation when chunking
- Chronological commit ordering
- Automatic fallback to chunking
"""

import subprocess
import time
import re
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path

class HybridPushSystem:
    def __init__(self):
        self.max_retries_per_commit = 3
        self.stall_timeout = 300  # 5 minutes
        self.chunk_size = 100  # files per chunk
        self.retry_delay = 30
        self.progress_file = "hybrid_push_progress.json"
        self.backup_dir = Path("temp_chunk_backup")
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
        
        # Also write to file
        with open("hybrid_push.log", "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def save_progress(self, data):
        """Save progress to JSON file."""
        with open(self.progress_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_progress(self):
        """Load progress from JSON file."""
        try:
            with open(self.progress_file, "r") as f:
                return json.load(f)
        except:
            return {"completed_commits": [], "current_position": None}
    
    def get_commits_to_push(self):
        """Get list of commits ahead of origin (oldest first)."""
        try:
            result = subprocess.run([
                'git', 'rev-list', '--reverse', 'origin/master..HEAD'
            ], capture_output=True, text=True, check=True)
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return [c for c in commits if c]
        except Exception as e:
            self.log(f"Error getting commits: {e}")
            return []
    
    def get_commit_info(self, commit_hash):
        """Get complete commit information including date and author."""
        try:
            # Get commit details
            result = subprocess.run([
                'git', 'show', '--format=%H|%an|%ae|%ad|%s', '--date=iso', '--name-only', commit_hash
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            header = lines[0].split('|')
            files = [f for f in lines[2:] if f.strip()]  # Skip empty line after header
            
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
    
    def parse_push_progress(self, line):
        """Parse git push progress from output line."""
        progress_info = {'phase': 'unknown', 'percent': 0, 'has_progress': False}
        
        # Writing objects (most important for detecting progress)
        if 'Writing objects:' in line:
            match = re.search(r'Writing objects:\s*(\d+)%', line)
            if match:
                progress_info.update({
                    'phase': 'writing',
                    'percent': int(match.group(1)),
                    'has_progress': True
                })
        
        # Other phases
        elif 'Counting objects:' in line:
            match = re.search(r'Counting objects:\s*(\d+)%', line)
            if match:
                progress_info.update({
                    'phase': 'counting',
                    'percent': int(match.group(1)),
                    'has_progress': True
                })
        
        elif 'Compressing objects:' in line:
            match = re.search(r'Compressing objects:\s*(\d+)%', line)
            if match:
                progress_info.update({
                    'phase': 'compressing', 
                    'percent': int(match.group(1)),
                    'has_progress': True
                })
        
        return progress_info
    
    def push_commit_with_monitoring(self, commit_hash, attempt):
        """Push a single commit with stall detection."""
        self.log(f"  ATTEMPT {attempt}: Pushing {commit_hash[:8]} with monitoring")
        
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
                
                self.log(f"    OUTPUT: {line}")
                
                # Parse progress
                progress = self.parse_push_progress(line)
                
                if progress['has_progress']:
                    current_state = f"{progress['phase']}-{progress['percent']}"
                    
                    # Update progress time if we have real advancement
                    if last_progress_state != current_state:
                        last_progress_time = time.time()
                        last_progress_state = current_state
                        self.log(f"    PROGRESS: {progress['phase'].upper()} {progress['percent']}%")
                
                # Check for stall
                time_since_progress = time.time() - last_progress_time
                if time_since_progress > self.stall_timeout:
                    self.log(f"    STALL: No progress for {time_since_progress:.0f} seconds")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    return False, "STALL"
            
            return_code = process.returncode
            if return_code == 0:
                self.log(f"    SUCCESS: Commit {commit_hash[:8]} pushed")
                return True, "SUCCESS"
            else:
                self.log(f"    FAILED: Return code {return_code}")
                return False, f"FAILED_{return_code}"
                
        except Exception as e:
            self.log(f"    ERROR: {e}")
            return False, f"ERROR_{str(e)}"
    
    def chunk_commit_with_date_preservation(self, commit_info):
        """Break commit into chunks while preserving date and authorship."""
        self.log(f"CHUNKING: Breaking {commit_info['hash'][:8]} into pieces")
        self.log(f"Original: {len(commit_info['files'])} files")
        self.log(f"Author: {commit_info['author_name']} <{commit_info['author_email']}>")
        self.log(f"Date: {commit_info['date']}")
        
        # Create backup
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        self.backup_dir.mkdir()
        
        # Backup files from this commit
        for file_path in commit_info['files']:
            src = Path(file_path)
            if src.exists():
                dst = self.backup_dir / file_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(src, dst)
                except Exception as e:
                    self.log(f"    Warning: Could not backup {file_path}: {e}")
        
        # Reset to before this commit
        parent_commit = f"{commit_info['hash']}^"
        subprocess.run(['git', 'reset', '--hard', parent_commit], capture_output=True)
        
        # Create chunks
        files = commit_info['files']
        chunks = [files[i:i + self.chunk_size] for i in range(0, len(files), self.chunk_size)]
        
        self.log(f"CHUNKING: Created {len(chunks)} chunks of ~{self.chunk_size} files each")
        
        # Process each chunk
        for i, chunk in enumerate(chunks, 1):
            self.log(f"CHUNK {i}/{len(chunks)}: Processing {len(chunk)} files")
            
            # Restore files for this chunk
            for file_path in chunk:
                src = self.backup_dir / file_path
                dst = Path(file_path)
                if src.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(src, dst)
                        subprocess.run(['git', 'add', file_path], capture_output=True)
                    except Exception as e:
                        self.log(f"      Warning: Could not restore {file_path}: {e}")
            
            # Create commit with original date and author
            commit_msg = f"{commit_info['message']} [CHUNK {i}/{len(chunks)}]"
            
            # Set environment variables for date/author preservation
            env = dict(os.environ)
            env['GIT_AUTHOR_NAME'] = commit_info['author_name']
            env['GIT_AUTHOR_EMAIL'] = commit_info['author_email']
            env['GIT_AUTHOR_DATE'] = commit_info['date']
            env['GIT_COMMITTER_NAME'] = commit_info['author_name']
            env['GIT_COMMITTER_EMAIL'] = commit_info['author_email']
            env['GIT_COMMITTER_DATE'] = commit_info['date']
            
            result = subprocess.run([
                'git', 'commit', '-m', commit_msg
            ], capture_output=True, text=True, env=env)
            
            if result.returncode == 0:
                self.log(f"      COMMIT: Created chunk {i} commit")
                
                # Try to push this chunk
                success = False
                for attempt in range(1, self.max_retries_per_commit + 1):
                    push_success, reason = self.push_commit_with_monitoring('HEAD', attempt)
                    if push_success:
                        self.log(f"      PUSHED: Chunk {i} successfully pushed")
                        success = True
                        break
                    
                    if attempt < self.max_retries_per_commit:
                        self.log(f"      Retrying chunk {i} in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                
                if not success:
                    self.log(f"      FAILED: Could not push chunk {i}")
                    return False
            else:
                self.log(f"      FAILED: Could not create chunk {i} commit")
                return False
        
        # Cleanup
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.log(f"CHUNKING: Successfully chunked and pushed {commit_info['hash'][:8]}")
        return True
    
    def run_hybrid_push(self):
        """Main hybrid push system."""
        self.log("HYBRID PUSH SYSTEM - Incremental with Chunked Fallback")
        self.log("=" * 70)
        self.log("Strategy:")
        self.log("1. Push commits one at a time (oldest first)")
        self.log("2. Monitor for stalls (5-minute timeout)")
        self.log("3. After 3 failures, break commit into chunks")
        self.log("4. Preserve original dates and authorship")
        self.log("")
        
        # Load previous progress
        progress = self.load_progress()
        
        # Get commits to push
        commits = self.get_commits_to_push()
        if not commits:
            self.log("No commits to push!")
            return True
        
        # Filter out already completed commits
        remaining_commits = [c for c in commits if c not in progress.get('completed_commits', [])]
        
        self.log(f"Total commits: {len(commits)}")
        self.log(f"Completed: {len(progress.get('completed_commits', []))}")
        self.log(f"Remaining: {len(remaining_commits)}")
        
        if not remaining_commits:
            self.log("All commits already pushed!")
            return True
        
        # Process each commit
        for i, commit_hash in enumerate(remaining_commits, 1):
            commit_info = self.get_commit_info(commit_hash)
            if not commit_info:
                self.log(f"ERROR: Could not get info for {commit_hash[:8]}")
                continue
            
            self.log(f"\n=== COMMIT {i}/{len(remaining_commits)} ===")
            self.log(f"Hash: {commit_hash[:8]}")
            self.log(f"Message: {commit_info['message']}")
            self.log(f"Files: {len(commit_info['files'])}")
            self.log(f"Date: {commit_info['date']}")
            
            # Try incremental push first
            success = False
            for attempt in range(1, self.max_retries_per_commit + 1):
                push_success, reason = self.push_commit_with_monitoring(commit_hash, attempt)
                
                if push_success:
                    self.log(f"SUCCESS: Incremental push succeeded on attempt {attempt}")
                    success = True
                    break
                
                self.log(f"FAILED: Attempt {attempt} failed ({reason})")
                
                if attempt < self.max_retries_per_commit:
                    self.log(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
            
            # If incremental failed, try chunking
            if not success:
                self.log(f"FALLBACK: Incremental push failed, switching to chunking")
                
                if self.chunk_commit_with_date_preservation(commit_info):
                    self.log(f"SUCCESS: Chunking succeeded for {commit_hash[:8]}")
                    success = True
                else:
                    self.log(f"FAILED: Chunking also failed for {commit_hash[:8]}")
                    break
            
            # Update progress
            if success:
                progress['completed_commits'].append(commit_hash)
                progress['current_position'] = commit_hash
                self.save_progress(progress)
                self.log(f"Progress: {len(progress['completed_commits'])}/{len(commits)} commits")
        
        # Final status
        if len(progress['completed_commits']) == len(commits):
            self.log("\n" + "=" * 70)
            self.log("SUCCESS: All commits pushed successfully!")
            return True
        else:
            self.log(f"\nPARTIAL: {len(progress['completed_commits'])}/{len(commits)} commits pushed")
            return False

def main():
    pusher = HybridPushSystem()
    try:
        pusher.run_hybrid_push()
    except KeyboardInterrupt:
        pusher.log("\nSTOPPED: User interrupted")
    except Exception as e:
        pusher.log(f"\nERROR: {e}")

if __name__ == "__main__":
    main()