#!/usr/bin/env python3
"""
Persistent Push System - Never Give Up!
=======================================

Pushes commits one at a time with persistent retry logic.
If a push fails or times out, it keeps retrying that same commit until it succeeds.
Then moves on to the next commit. Never gives up until everything is pushed!
"""

import subprocess
import time
import sys
import os
import json
from datetime import datetime
from pathlib import Path

def safe_print(message):
    """Print with ASCII-safe encoding."""
    try:
        print(message)
        sys.stdout.flush()  # Force immediate output
    except UnicodeEncodeError:
        ascii_message = message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)
        sys.stdout.flush()

class PersistentPushSystem:
    def __init__(self):
        self.max_retries_per_commit = 50  # Try each commit up to 50 times
        self.retry_delay_start = 10       # Start with 10 seconds between retries
        self.retry_delay_max = 300        # Max 5 minutes between retries
        self.timeout_per_attempt = 600    # 10 minutes per attempt
        self.progress_file = Path("push_progress.json")
        
    def load_progress(self):
        """Load previous progress if exists."""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"completed_commits": [], "current_commit_index": 0}
    
    def save_progress(self, progress):
        """Save current progress."""
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            safe_print(f"[WARNING] Could not save progress: {e}")
    
    def get_commits_to_push(self):
        """Get list of unpushed commits."""
        try:
            result = subprocess.run([
                'git', 'rev-list', '--reverse', 'origin/master..HEAD'
            ], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return commits
        except Exception as e:
            safe_print(f"[ERROR] Getting commits: {e}")
            return []
    
    def get_commit_info(self, commit_hash):
        """Get commit message and stats."""
        try:
            # Get commit message
            msg_result = subprocess.run([
                'git', 'log', '--format=%s', '-n', '1', commit_hash
            ], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            # Get commit stats
            stat_result = subprocess.run([
                'git', 'show', '--stat', '--format=', commit_hash
            ], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            return {
                'hash': commit_hash[:8],
                'message': msg_result.stdout.strip()[:60],
                'stats': len(stat_result.stdout.strip().split('\n'))
            }
        except:
            return {'hash': commit_hash[:8], 'message': 'Unknown', 'stats': 0}
    
    def attempt_push_commit(self, commit_hash, attempt_num):
        """Attempt to push a single commit with timeout."""
        safe_print(f"    [ATTEMPT {attempt_num}] Pushing {commit_hash[:8]}...")
        
        try:
            # Use git push with specific commit
            process = subprocess.Popen([
                'git', 'push', 'origin', f'{commit_hash}:refs/heads/master'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, 
               encoding='utf-8', errors='ignore')
            
            # Monitor with timeout
            start_time = time.time()
            output_lines = []
            
            while True:
                # Check if process finished
                return_code = process.poll()
                if return_code is not None:
                    break
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self.timeout_per_attempt:
                    safe_print(f"    [TIMEOUT] Killing process after {self.timeout_per_attempt}s")
                    process.terminate()
                    time.sleep(5)  # Give it time to terminate
                    if process.poll() is None:
                        process.kill()  # Force kill if needed
                    return False, "Timeout"
                
                # Try to read output (non-blocking)
                try:
                    import select
                    if hasattr(select, 'select'):
                        ready, _, _ = select.select([process.stdout], [], [], 1)
                        if ready:
                            line = process.stdout.readline()
                            if line:
                                clean_line = line.strip().encode('ascii', errors='ignore').decode('ascii')
                                if clean_line and 'Writing objects' in clean_line:
                                    safe_print(f"    [PROGRESS] {clean_line}")
                                output_lines.append(clean_line)
                    else:
                        # Windows fallback - just wait
                        time.sleep(1)
                except:
                    time.sleep(1)  # Fallback - just wait
            
            # Check final result
            elapsed = time.time() - start_time
            if return_code == 0:
                safe_print(f"    [SUCCESS] Pushed in {elapsed:.1f}s")
                return True, "Success"
            else:
                safe_print(f"    [FAILED] Return code: {return_code}")
                return False, f"Exit code {return_code}"
                
        except Exception as e:
            safe_print(f"    [ERROR] Exception: {e}")
            return False, str(e)
    
    def push_single_commit_with_retry(self, commit_hash, commit_info, commit_index, total_commits):
        """Push a single commit with persistent retry logic."""
        safe_print(f"\n[COMMIT {commit_index+1}/{total_commits}] {commit_info['message']}")
        safe_print(f"[HASH] {commit_hash}")
        safe_print(f"[STATS] ~{commit_info['stats']} changed files")
        safe_print("-" * 70)
        
        retry_delay = self.retry_delay_start
        
        for attempt in range(1, self.max_retries_per_commit + 1):
            success, reason = self.attempt_push_commit(commit_hash, attempt)
            
            if success:
                safe_print(f"[SUCCESS] Commit {commit_hash[:8]} pushed successfully!")
                return True
            
            # Failed - prepare for retry
            safe_print(f"    [FAILED] {reason}")
            
            if attempt < self.max_retries_per_commit:
                safe_print(f"    [RETRY] Waiting {retry_delay}s before attempt {attempt + 1}/{self.max_retries_per_commit}")
                safe_print(f"    [STRATEGY] Will keep trying this commit until it succeeds!")
                
                time.sleep(retry_delay)
                
                # Exponential backoff with max
                retry_delay = min(retry_delay * 1.5, self.retry_delay_max)
            else:
                safe_print(f"[EXHAUSTED] Failed after {self.max_retries_per_commit} attempts")
                return False
        
        return False
    
    def run_persistent_push(self):
        """Main execution - never give up until everything is pushed!"""
        safe_print("PERSISTENT PUSH SYSTEM - NEVER GIVE UP!")
        safe_print("=" * 60)
        safe_print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        safe_print("")
        
        # Load previous progress
        progress = self.load_progress()
        
        # Get commits to push
        all_commits = self.get_commits_to_push()
        if not all_commits:
            safe_print("[INFO] No commits to push!")
            return True
        
        # Filter out already completed commits
        completed = set(progress.get("completed_commits", []))
        remaining_commits = [c for c in all_commits if c not in completed]
        
        safe_print(f"[INFO] Total commits: {len(all_commits)}")
        safe_print(f"[INFO] Already completed: {len(completed)}")
        safe_print(f"[INFO] Remaining to push: {len(remaining_commits)}")
        
        if not remaining_commits:
            safe_print("[SUCCESS] All commits already pushed!")
            return True
        
        safe_print(f"\n[STRATEGY] Push 1 commit at a time")
        safe_print(f"[STRATEGY] Up to {self.max_retries_per_commit} retries per commit")
        safe_print(f"[STRATEGY] {self.timeout_per_attempt}s timeout per attempt")
        safe_print(f"[STRATEGY] Exponential backoff: {self.retry_delay_start}s to {self.retry_delay_max}s")
        safe_print("\nPress Ctrl+C to stop at any time (progress will be saved)")
        safe_print("")
        
        # Push remaining commits one by one
        total_success = 0
        
        for i, commit_hash in enumerate(remaining_commits):
            commit_info = self.get_commit_info(commit_hash)
            
            if self.push_single_commit_with_retry(commit_hash, commit_info, i, len(remaining_commits)):
                total_success += 1
                
                # Update progress
                progress["completed_commits"].append(commit_hash)
                progress["current_commit_index"] = i + 1
                self.save_progress(progress)
                
                safe_print(f"[PROGRESS] {total_success}/{len(remaining_commits)} commits pushed successfully")
            else:
                safe_print(f"[FAILED] Could not push commit {commit_hash[:8]} after {self.max_retries_per_commit} attempts")
                safe_print(f"[MANUAL] You may need to check this commit manually")
                break
        
        # Final summary
        safe_print(f"\n[FINAL RESULTS]")
        safe_print(f"Successfully pushed: {total_success}/{len(remaining_commits)} remaining commits")
        safe_print(f"Total pushed in session: {total_success}")
        
        if total_success == len(remaining_commits):
            safe_print("[SUCCESS] ALL COMMITS SUCCESSFULLY PUSHED TO GITHUB!")
            # Clean up progress file
            if self.progress_file.exists():
                self.progress_file.unlink()
            return True
        else:
            safe_print(f"[PARTIAL] {len(remaining_commits) - total_success} commits still need to be pushed")
            safe_print(f"[RESUME] Run this script again to continue from where it left off")
            return False

def main():
    pusher = PersistentPushSystem()
    try:
        pusher.run_persistent_push()
    except KeyboardInterrupt:
        safe_print("\n[STOPPED] Push interrupted by user")
        safe_print("[PROGRESS] Progress has been saved - run again to continue")
    except Exception as e:
        safe_print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()