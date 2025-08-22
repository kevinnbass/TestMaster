#!/usr/bin/env python3
"""
Intelligent Push System with Stall Detection
==========================================

Monitors git push progress and automatically retries when:
1. Push stalls (no progress for 5 minutes)
2. Push fails with timeout or server errors
3. Network issues occur

Keeps trying until successful push is achieved.
"""

import subprocess
import time
import re
import sys
from datetime import datetime
from pathlib import Path

class IntelligentPushSystem:
    def __init__(self):
        self.stall_timeout = 300  # 5 minutes in seconds
        self.max_total_attempts = 20
        self.retry_delay = 30  # seconds between retries
        self.progress_file = "push_progress.log"
        
    def log(self, message):
        """Log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
        
        # Also write to log file
        with open(self.progress_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
    
    def parse_progress(self, line):
        """Parse git push progress from output line."""
        progress_info = {
            'phase': 'unknown',
            'percent': 0,
            'current': 0,
            'total': 0,
            'transferred': '',
            'speed': ''
        }
        
        # Counting objects
        if 'Counting objects:' in line:
            match = re.search(r'Counting objects:\s*(\d+)%\s*\((\d+)/(\d+)\)', line)
            if match:
                progress_info.update({
                    'phase': 'counting',
                    'percent': int(match.group(1)),
                    'current': int(match.group(2)),
                    'total': int(match.group(3))
                })
        
        # Compressing objects
        elif 'Compressing objects:' in line:
            match = re.search(r'Compressing objects:\s*(\d+)%\s*\((\d+)/(\d+)\)', line)
            if match:
                progress_info.update({
                    'phase': 'compressing',
                    'percent': int(match.group(1)),
                    'current': int(match.group(2)),
                    'total': int(match.group(3))
                })
        
        # Writing objects (most important for detecting stalls)
        elif 'Writing objects:' in line:
            match = re.search(r'Writing objects:\s*(\d+)%\s*\((\d+)/(\d+)\)(?:,\s*([\d.]+\s*[KMGT]?iB)\s*\|\s*([\d.]+\s*[KMGT]?iB/s))?', line)
            if match:
                progress_info.update({
                    'phase': 'writing',
                    'percent': int(match.group(1)),
                    'current': int(match.group(2)),
                    'total': int(match.group(3)),
                    'transferred': match.group(4) or '',
                    'speed': match.group(5) or ''
                })
        
        return progress_info
    
    def monitor_push_with_stall_detection(self, attempt_num):
        """Monitor a single push attempt and detect stalls."""
        self.log(f"=== ATTEMPT {attempt_num} - Starting monitored push ===")
        
        try:
            # Start git push process
            process = subprocess.Popen(
                ['git', 'push', 'origin', 'HEAD:master', '--verbose'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            last_progress_time = time.time()
            last_progress_state = None
            
            while True:
                # Check if process has finished
                if process.poll() is not None:
                    break
                
                # Read output line
                line = process.stdout.readline()
                if not line:
                    time.sleep(1)
                    continue
                
                line = line.strip()
                if not line:
                    continue
                
                self.log(f"OUTPUT: {line}")
                
                # Parse progress
                progress = self.parse_progress(line)
                
                # Check for meaningful progress
                if progress['phase'] != 'unknown' and progress['percent'] > 0:
                    current_state = f"{progress['phase']}-{progress['percent']}-{progress['current']}"
                    
                    # If we have real progress, update last progress time
                    if last_progress_state != current_state:
                        last_progress_time = time.time()
                        last_progress_state = current_state
                        
                        # Show detailed progress
                        if progress['transferred']:
                            self.log(f"PROGRESS: {progress['phase'].upper()} {progress['percent']}% "
                                   f"({progress['current']}/{progress['total']}) - "
                                   f"{progress['transferred']} @ {progress['speed']}")
                        else:
                            self.log(f"PROGRESS: {progress['phase'].upper()} {progress['percent']}% "
                                   f"({progress['current']}/{progress['total']})")
                
                # Check for stall (no progress for 5 minutes)
                time_since_progress = time.time() - last_progress_time
                if time_since_progress > self.stall_timeout:
                    self.log(f"STALL DETECTED: No progress for {time_since_progress:.0f} seconds")
                    self.log("Terminating stalled push...")
                    process.terminate()
                    time.sleep(5)
                    if process.poll() is None:
                        process.kill()
                    return False, "STALL_TIMEOUT"
            
            # Process finished, check result
            return_code = process.returncode
            
            if return_code == 0:
                self.log("SUCCESS: Push completed successfully!")
                return True, "SUCCESS"
            else:
                self.log(f"FAILED: Push failed with return code {return_code}")
                return False, f"FAILED_CODE_{return_code}"
                
        except Exception as e:
            self.log(f"ERROR: Exception during push: {e}")
            return False, f"EXCEPTION_{str(e)}"
    
    def run_intelligent_push(self):
        """Main loop - keep trying until success."""
        self.log("INTELLIGENT PUSH SYSTEM STARTING")
        self.log("=" * 60)
        self.log("Will retry on stalls (5min timeout) and failures")
        self.log("Press Ctrl+C to stop")
        self.log("")
        
        # Clear previous log
        with open(self.progress_file, "w") as f:
            f.write(f"Intelligent Push System Log - {datetime.now()}\n")
            f.write("=" * 60 + "\n")
        
        for attempt in range(1, self.max_total_attempts + 1):
            self.log(f"Starting attempt {attempt}/{self.max_total_attempts}")
            
            success, reason = self.monitor_push_with_stall_detection(attempt)
            
            if success:
                self.log("=" * 60)
                self.log(f"MISSION ACCOMPLISHED! Push succeeded on attempt {attempt}")
                return True
            
            self.log(f"Attempt {attempt} failed: {reason}")
            
            if attempt < self.max_total_attempts:
                self.log(f"Waiting {self.retry_delay} seconds before retry...")
                time.sleep(self.retry_delay)
                self.log("")
            
        self.log("=" * 60)
        self.log("FAILED: All attempts exhausted")
        return False

def main():
    pusher = IntelligentPushSystem()
    try:
        pusher.run_intelligent_push()
    except KeyboardInterrupt:
        pusher.log("\nSTOPPED: User interrupted")
    except Exception as e:
        pusher.log(f"\nERROR: {e}")

if __name__ == "__main__":
    main()