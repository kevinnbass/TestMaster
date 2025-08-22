#!/usr/bin/env python3
"""
Simple Chunk Push - ASCII Only
==============================

Pushes commits in very small chunks (10MB target) with ASCII-only output.
No unicode characters that cause Windows encoding issues.
"""

import subprocess
import time
import sys
import os
from datetime import datetime

def safe_print(message):
    """Print with ASCII-safe encoding."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Strip unicode and print ASCII only
        ascii_message = message.encode('ascii', errors='ignore').decode('ascii')
        print(ascii_message)

class SimpleChunkPush:
    def __init__(self):
        self.chunk_delay = 5  # 5 second delay between pushes
        
    def get_commits_to_push(self):
        """Get list of unpushed commits."""
        try:
            result = subprocess.run([
                'git', 'rev-list', '--reverse', 'origin/master..HEAD'
            ], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            return commits
        except Exception as e:
            safe_print(f"Error getting commits: {e}")
            return []
    
    def get_commit_message(self, commit_hash):
        """Get commit message safely."""
        try:
            result = subprocess.run([
                'git', 'log', '--format=%s', '-n', '1', commit_hash
            ], capture_output=True, text=True, check=True, encoding='utf-8', errors='ignore')
            
            return result.stdout.strip()[:60]
        except:
            return "Unknown commit"
    
    def push_single_commit(self, commit_hash):
        """Push a single commit with basic monitoring."""
        commit_msg = self.get_commit_message(commit_hash)
        safe_print(f"\n[PUSH] {commit_hash[:8]} - {commit_msg}")
        safe_print("-" * 60)
        
        try:
            start_time = time.time()
            
            # Use basic git push with timeout
            process = subprocess.Popen([
                'git', 'push', 'origin', f'{commit_hash}:master'
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore')
            
            # Monitor with simple timeout
            timeout = 300  # 5 minutes per commit
            output_lines = []
            
            while True:
                try:
                    line = process.stdout.readline()
                    if line:
                        clean_line = line.strip().encode('ascii', errors='ignore').decode('ascii')
                        if clean_line:
                            safe_print(f"[GIT] {clean_line}")
                            output_lines.append(clean_line)
                    
                    if process.poll() is not None:
                        break
                        
                    elapsed = time.time() - start_time
                    if elapsed > timeout:
                        safe_print("[TIMEOUT] Killing process after 5 minutes")
                        process.terminate()
                        return False
                        
                except Exception as e:
                    safe_print(f"[ERROR] Reading output: {e}")
                    break
            
            elapsed = time.time() - start_time
            return_code = process.poll()
            
            if return_code == 0:
                safe_print(f"[SUCCESS] Pushed in {elapsed:.1f} seconds")
                return True
            else:
                safe_print(f"[FAILED] Return code: {return_code}")
                return False
                
        except Exception as e:
            safe_print(f"[ERROR] Exception during push: {e}")
            return False
    
    def run_chunk_push(self):
        """Main execution - push commits one at a time."""
        safe_print("Simple Chunk Push System (ASCII-Safe)")
        safe_print("=" * 50)
        safe_print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
        
        # Get unpushed commits
        commits = self.get_commits_to_push()
        if not commits:
            safe_print("[INFO] No commits to push!")
            return True
        
        safe_print(f"\n[INFO] Found {len(commits)} commits to push")
        safe_print(f"[INFO] Will push ONE commit at a time with {self.chunk_delay}s delays")
        
        # Show commit summary
        safe_print("\n[COMMITS]:")
        for i, commit in enumerate(commits):
            msg = self.get_commit_message(commit)
            safe_print(f"  {i+1:2d}. {commit[:8]} - {msg}")
        
        safe_print(f"\n[STRATEGY] Ultra-conservative: 1 commit per push")
        safe_print("Press Ctrl+C to stop at any time")
        
        # Push commits one by one
        success_count = 0
        for i, commit in enumerate(commits):
            safe_print(f"\n[COMMIT {i+1}/{len(commits)}]")
            
            if self.push_single_commit(commit):
                success_count += 1
                safe_print(f"[PROGRESS] {success_count}/{len(commits)} commits successfully pushed")
                
                # Delay between pushes (except last one)
                if i < len(commits) - 1:
                    safe_print(f"[WAIT] Waiting {self.chunk_delay} seconds before next push...")
                    time.sleep(self.chunk_delay)
            else:
                safe_print(f"[FAILED] Stopping at commit {i+1} due to error")
                break
        
        # Final summary
        safe_print(f"\n[FINAL RESULTS]")
        safe_print(f"Successfully pushed: {success_count}/{len(commits)} commits")
        
        if success_count == len(commits):
            safe_print("[SUCCESS] ALL COMMITS PUSHED TO GITHUB!")
            return True
        else:
            remaining = len(commits) - success_count
            safe_print(f"[PARTIAL] {remaining} commits still need to be pushed")
            return False

def main():
    pusher = SimpleChunkPush()
    try:
        pusher.run_chunk_push()
    except KeyboardInterrupt:
        safe_print("\n[STOPPED] Push interrupted by user")
    except Exception as e:
        safe_print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    main()