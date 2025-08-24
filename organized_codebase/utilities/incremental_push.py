#!/usr/bin/env python3
"""
Incremental Push System - Push One Commit at a Time
==================================================

Pushes commits individually to avoid GitHub's 2GB push limit.
Works by pushing each commit separately with retries.
"""

import subprocess
import time
import sys
from datetime import datetime

class IncrementalPush:
    def __init__(self):
        self.max_retries_per_commit = 5
        self.retry_delay = 30
        
    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
    
    def get_commits_to_push(self):
        """Get list of commits ahead of origin."""
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
                'message': msg_result.stdout.strip(),
                'stats': stat_result.stdout.strip()
            }
        except Exception as e:
            return {'message': 'Unknown', 'stats': f'Error: {e}'}
    
    def push_commit_range(self, from_commit, to_commit, attempt):
        """Push a specific commit range."""
        self.log(f"  ATTEMPT {attempt}: Pushing {from_commit[:8]}..{to_commit[:8]}")
        
        try:
            result = subprocess.run([
                'git', 'push', 'origin', f'{to_commit}:master'
            ], capture_output=True, text=True, timeout=1800)  # 30 minute timeout
            
            if result.returncode == 0:
                self.log(f"  SUCCESS: Pushed to {to_commit[:8]}")
                return True
            else:
                self.log(f"  FAILED: {result.stderr[:100]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"  TIMEOUT: Push timed out after 30 minutes")
            return False
        except Exception as e:
            self.log(f"  ERROR: {e}")
            return False
    
    def run_incremental_push(self):
        """Push commits incrementally."""
        self.log("INCREMENTAL PUSH SYSTEM")
        self.log("=" * 60)
        
        commits = self.get_commits_to_push()
        if not commits:
            self.log("No commits to push!")
            return True
        
        self.log(f"Found {len(commits)} commits to push")
        
        # Show all commits first
        self.log("\nCommits to push:")
        for i, commit in enumerate(commits, 1):
            info = self.get_commit_info(commit)
            self.log(f"  {i:2d}. {commit[:8]} - {info['message'][:60]}")
        
        self.log(f"\nStrategy: Push each commit individually")
        response = input("\nProceed with incremental push? (yes/no): ")
        if response.lower() != 'yes':
            return False
        
        # Get current position
        try:
            current_result = subprocess.run([
                'git', 'rev-parse', 'origin/master'
            ], capture_output=True, text=True, check=True)
            current_origin = current_result.stdout.strip()
        except:
            current_origin = "unknown"
        
        # Push each commit
        for i, commit in enumerate(commits, 1):
            info = self.get_commit_info(commit)
            
            self.log(f"\n=== COMMIT {i}/{len(commits)} ===")
            self.log(f"Commit: {commit[:8]}")
            self.log(f"Message: {info['message']}")
            
            # Show file count if available
            lines = info['stats'].split('\n')
            for line in lines[-3:]:  # Last few lines usually have summary
                if 'file' in line and 'changed' in line:
                    self.log(f"Changes: {line.strip()}")
                    break
            
            # Try to push this commit with retries
            success = False
            for attempt in range(1, self.max_retries_per_commit + 1):
                if self.push_commit_range(current_origin, commit, attempt):
                    success = True
                    current_origin = commit  # Update current position
                    break
                
                if attempt < self.max_retries_per_commit:
                    self.log(f"  Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
            
            if not success:
                self.log(f"FAILED: Could not push commit {commit[:8]} after {self.max_retries_per_commit} attempts")
                self.log(f"Successfully pushed {i-1}/{len(commits)} commits")
                return False
            
            self.log(f"Progress: {i}/{len(commits)} commits pushed")
        
        self.log("\n" + "=" * 60)
        self.log("SUCCESS: All commits pushed incrementally!")
        return True

def main():
    pusher = IncrementalPush()
    try:
        pusher.run_incremental_push()
    except KeyboardInterrupt:
        pusher.log("\nSTOPPED: User interrupted")
    except Exception as e:
        pusher.log(f"\nERROR: {e}")

if __name__ == "__main__":
    main()