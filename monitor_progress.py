#!/usr/bin/env python3
"""Monitor conversion progress."""

import time
from pathlib import Path
from datetime import datetime

def monitor():
    """Monitor test file creation."""
    initial_count = 54  # Starting point
    
    while True:
        current_count = len(list(Path("tests/unit").glob("*_intelligent.py")))
        progress = current_count / 262 * 100
        new_files = current_count - initial_count
        
        print(f"\r[{datetime.now().strftime('%H:%M:%S')}] "
              f"Intelligent tests: {current_count}/262 ({progress:.1f}%) "
              f"| New this session: +{new_files} files", end="", flush=True)
        
        if current_count >= 262:
            print("\n🎉 COMPLETE! All 262 files converted!")
            break
        
        time.sleep(10)  # Check every 10 seconds

if __name__ == "__main__":
    print("Monitoring conversion progress...")
    print("-" * 60)
    monitor()