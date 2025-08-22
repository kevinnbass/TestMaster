#!/usr/bin/env python3
"""
Chunked Git Push System
======================

Pushes commits in small batches to avoid GitHub HTTP timeouts.
Tracks progress and resumes from failures.
"""

import subprocess
import time
import json
import re
from datetime import datetime
from pathlib import Path

class ChunkedGitPush:
    def __init__(self):
        self.status_file = Path("push_status.json")
        self.chunk_size = 3  # Push 3 commits at a time
        
    def get_unpushed_commits(self):
        """Get list of commits ahead of origin."""
        try:
            result = subprocess.run([
                'git', 'rev-list', '--reverse', 'origin/master..HEAD'
            ], capture_output=True, text=True, check=True)
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return commits
        except subprocess.CalledProcessError:
            return []
    
    def get_commit_info(self, commit_hash):
        """Get commit message and stats."""
        try:
            # Get commit message
            msg_result = subprocess.run([
                'git', 'log', '--format=%s', '-n', '1', commit_hash
            ], capture_output=True, text=True, check=True)
            
            # Get commit stats
            stat_result = subprocess.run([
                'git', 'show', '--stat', '--format=', commit_hash
            ], capture_output=True, text=True, check=True)
            
            return {
                'hash': commit_hash[:8],
                'message': msg_result.stdout.strip(),
                'stats': stat_result.stdout.strip()
            }
        except:
            return {'hash': commit_hash[:8], 'message': 'Unknown', 'stats': ''}
    
    def push_commit_range(self, from_commit, to_commit):
        """Push a range of commits with monitoring."""
        print(f"\n[PUSH] Pushing commits {from_commit[:8]}..{to_commit[:8]}")
        print("-" * 50)
        
        try:
            # Push the specific range
            cmd = ['git', 'push', 'origin', f'{from_commit}^:{to_commit}:master']
            
            start_time = time.time()
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # Monitor the push
            output_lines = []
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    output_lines.append(line)
                    print(f"[GIT] {line}")
                    
                    # Look for progress indicators
                    if 'Writing objects:' in line:
                        match = re.search(r'(\d+)%.*?(\d+\.?\d*\s*[KMGT]?iB)', line)
                        if match:
                            percent, size = match.groups()
                            elapsed = int(time.time() - start_time)
                            print(f"[STATUS] Writing: {percent}% | {size} | {elapsed}s elapsed")
            
            elapsed = time.time() - start_time
            return_code = process.poll()
            
            if return_code == 0:
                print(f"[SUCCESS] Push completed in {elapsed:.1f}s")
                return True, output_lines
            else:
                print(f"[ERROR] Push failed with code {return_code}")
                return False, output_lines
                
        except Exception as e:
            print(f"[ERROR] Exception during push: {e}")
            return False, []
    
    def push_single_commit(self, commit_hash):
        """Push a single commit."""
        print(f"\n[PUSH] Pushing single commit {commit_hash[:8]}")
        print("-" * 30)
        
        try:
            start_time = time.time()
            result = subprocess.run([
                'git', 'push', 'origin', f'{commit_hash}:master'
            ], capture_output=True, text=True, timeout=180)  # 3 minute timeout
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                print(f"[SUCCESS] Commit pushed in {elapsed:.1f}s")
                return True
            else:
                print(f"[ERROR] Push failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("[ERROR] Push timed out after 3 minutes")
            return False
        except Exception as e:
            print(f"[ERROR] Exception: {e}")
            return False
    
    def run_chunked_push(self):
        """Main chunked push execution."""
        print("Chunked Git Push System")
        print("=" * 40)
        print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Get unpushed commits
        commits = self.get_unpushed_commits()
        if not commits:
            print("[INFO] No commits to push!")
            return True
        
        print(f"[INFO] Found {len(commits)} commits to push")
        
        # Show commit summary
        print("\n[COMMITS] Summary:")
        for i, commit in enumerate(commits):
            info = self.get_commit_info(commit)
            print(f"  {i+1:2d}. {info['hash']} {info['message'][:60]}")
        
        print(f"\n[STRATEGY] Pushing 1 commit at a time for maximum reliability")
        print()
        
        # Push commits one by one
        success_count = 0
        for i, commit in enumerate(commits):
            info = self.get_commit_info(commit)
            print(f"\n[COMMIT {i+1}/{len(commits)}] {info['hash']} - {info['message']}")
            
            if self.push_single_commit(commit):
                success_count += 1
                print(f"[PROGRESS] {success_count}/{len(commits)} commits pushed successfully")
            else:
                print(f"[FAILED] Stopping at commit {i+1}")
                break
                
            # Small delay between pushes
            if i < len(commits) - 1:
                print("[WAIT] Pausing 2 seconds...")
                time.sleep(2)
        
        print(f"\n[FINAL] Successfully pushed {success_count}/{len(commits)} commits")
        
        if success_count == len(commits):
            print("[SUCCESS] All commits pushed successfully!")
            return True
        else:
            remaining = len(commits) - success_count
            print(f"[PARTIAL] {remaining} commits still need to be pushed")
            return False

def main():
    pusher = ChunkedGitPush()
    pusher.run_chunked_push()

if __name__ == "__main__":
    main()