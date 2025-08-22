#!/usr/bin/env python3
"""
Break Large Commits Into Smaller Chunks
========================================

Breaks down large commits into smaller, pushable pieces.
"""

import subprocess
import os
import sys
from pathlib import Path

def safe_print(msg):
    print(msg)
    sys.stdout.flush()

def get_changed_files_in_commit(commit_hash):
    """Get list of files changed in a specific commit."""
    try:
        result = subprocess.run([
            'git', 'diff-tree', '--no-commit-id', '--name-only', '-r', commit_hash
        ], capture_output=True, text=True, check=True)
        
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        return [f for f in files if f]  # Filter empty strings
    except Exception as e:
        safe_print(f"Error getting files: {e}")
        return []

def reset_to_before_commit(commit_hash):
    """Reset to the commit before the specified one."""
    try:
        # Get the parent commit
        result = subprocess.run([
            'git', 'rev-parse', f'{commit_hash}^'
        ], capture_output=True, text=True, check=True)
        
        parent_commit = result.stdout.strip()
        
        # Reset to parent commit
        subprocess.run(['git', 'reset', '--hard', parent_commit], check=True)
        safe_print(f"Reset to parent commit: {parent_commit[:8]}")
        return True
    except Exception as e:
        safe_print(f"Error resetting: {e}")
        return False

def chunk_files_by_size(files, max_files_per_chunk=100):
    """Break files into chunks."""
    chunks = []
    for i in range(0, len(files), max_files_per_chunk):
        chunks.append(files[i:i + max_files_per_chunk])
    return chunks

def apply_files_from_commit(commit_hash, file_list):
    """Apply specific files from a commit."""
    try:
        # Checkout specific files from the commit
        for file_path in file_list:
            subprocess.run([
                'git', 'checkout', commit_hash, '--', file_path
            ], capture_output=True, check=True)
        
        return True
    except Exception as e:
        safe_print(f"Error applying files: {e}")
        return False

def main():
    safe_print("Commit Breakdown Analysis")
    safe_print("=" * 60)
    
    # The problematic commit
    big_commit = "154482eb2ac24e8d2f92a37ba499efb7e9a28a9a"
    
    safe_print(f"\nAnalyzing commit: {big_commit[:8]}")
    
    # Get all changed files
    files = get_changed_files_in_commit(big_commit)
    safe_print(f"Total files changed: {len(files)}")
    
    if not files:
        safe_print("No files found in commit")
        return
    
    # Group files by directory for logical chunking
    file_groups = {}
    for file_path in files:
        # Get top-level directory
        parts = file_path.split('/')
        if len(parts) > 1:
            top_dir = parts[0]
        else:
            top_dir = "root"
        
        if top_dir not in file_groups:
            file_groups[top_dir] = []
        file_groups[top_dir].append(file_path)
    
    safe_print(f"\nFiles grouped by directory:")
    for dir_name, dir_files in sorted(file_groups.items(), key=lambda x: len(x[1]), reverse=True):
        safe_print(f"  {dir_name}: {len(dir_files)} files")
    
    # Create strategy for breaking down
    safe_print("\n" + "=" * 60)
    safe_print("RECOMMENDED STRATEGY:")
    safe_print("=" * 60)
    
    safe_print("\n1. Reset to before the big commit")
    safe_print("2. Apply and commit files in small batches:")
    
    batch_num = 1
    for dir_name, dir_files in sorted(file_groups.items(), key=lambda x: len(x[1])):
        if len(dir_files) <= 100:
            safe_print(f"   Batch {batch_num}: {dir_name} ({len(dir_files)} files)")
            batch_num += 1
        else:
            # Need to break this directory into smaller chunks
            chunks = chunk_files_by_size(dir_files, 100)
            for i, chunk in enumerate(chunks):
                safe_print(f"   Batch {batch_num}: {dir_name} part {i+1} ({len(chunk)} files)")
                batch_num += 1
    
    safe_print(f"\nTotal batches: {batch_num - 1}")
    safe_print("\n3. Push each batch individually")
    
    # Ask user if they want to proceed
    safe_print("\n" + "=" * 60)
    safe_print("WARNING: This will reset your git history!")
    safe_print("Make sure you have a backup or are okay with rewriting history.")
    safe_print("=" * 60)
    
    response = input("\nDo you want to proceed with breaking down the commit? (yes/no): ")
    
    if response.lower() != 'yes':
        safe_print("Aborted.")
        return
    
    # Create the breakdown script
    safe_print("\nCreating breakdown script...")
    
    script_lines = [
        "#!/bin/bash",
        "# Auto-generated script to break down large commit",
        "",
        f"# Reset to before the big commit",
        f"git reset --hard {big_commit}^",
        "",
        "# Apply files in batches",
    ]
    
    batch_num = 1
    for dir_name, dir_files in sorted(file_groups.items(), key=lambda x: len(x[1])):
        if len(dir_files) <= 100:
            script_lines.append(f"\n# Batch {batch_num}: {dir_name}")
            for file_path in dir_files:
                script_lines.append(f"git checkout {big_commit} -- '{file_path}'")
            script_lines.append(f"git add .")
            script_lines.append(f"git commit -m 'Phase 1 Batch {batch_num}: {dir_name} ({len(dir_files)} files)'")
            script_lines.append(f"git push origin HEAD:master")
            script_lines.append(f"echo 'Pushed batch {batch_num}'")
            script_lines.append(f"sleep 5")
            batch_num += 1
        else:
            chunks = chunk_files_by_size(dir_files, 100)
            for i, chunk in enumerate(chunks):
                script_lines.append(f"\n# Batch {batch_num}: {dir_name} part {i+1}")
                for file_path in chunk:
                    script_lines.append(f"git checkout {big_commit} -- '{file_path}'")
                script_lines.append(f"git add .")
                script_lines.append(f"git commit -m 'Phase 1 Batch {batch_num}: {dir_name} part {i+1} ({len(chunk)} files)'")
                script_lines.append(f"git push origin HEAD:master")
                script_lines.append(f"echo 'Pushed batch {batch_num}'")
                script_lines.append(f"sleep 5")
                batch_num += 1
    
    # Save the script
    script_path = Path("breakdown_and_push.sh")
    with open(script_path, 'w') as f:
        f.write('\n'.join(script_lines))
    
    safe_print(f"\nScript saved to: {script_path}")
    safe_print("You can run this script to break down and push the commit in small batches.")
    safe_print("\nAlternatively, I can start executing the breakdown now.")

if __name__ == "__main__":
    main()