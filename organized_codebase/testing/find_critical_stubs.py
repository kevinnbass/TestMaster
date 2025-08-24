#!/usr/bin/env python3
"""
Find critical stub implementations.
"""

import os
import re
from pathlib import Path

def find_critical_stubs():
    """Find the most critical stub implementations."""
    
    os.chdir('C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster')
    
    critical_areas = [
        'core/',
        'integration/',
        'agents/',
        'dashboard/api/'
    ]
    
    critical_files = []
    
    print("=" * 60)
    print("CRITICAL STUB ANALYSIS")
    print("=" * 60)
    
    for area in critical_areas:
        if not os.path.exists(area):
            continue
            
        print(f"\nAnalyzing {area}:")
        
        for root, dirs, files in os.walk(area):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not any(skip in d for skip in ['__pycache__', '.git', 'test'])]
            
            for file in files:
                if not file.endswith('.py'):
                    continue
                
                file_path = Path(root) / file
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    issues = []
                    lines = content.split('\n')
                    
                    for i, line in enumerate(lines, 1):
                        line_stripped = line.strip()
                        
                        # Look for placeholder patterns
                        if re.search(r'placeholder|stub|TODO|FIXME', line, re.IGNORECASE):
                            issues.append(f"Line {i}: {line_stripped[:80]}")
                        
                        # Pass statements in methods
                        if re.match(r'\s*pass\s*$', line) and i > 1:
                            prev_line = lines[i-2].strip() if i > 1 else ""
                            if 'def ' in prev_line:
                                issues.append(f"Line {i}: Method with only 'pass'")
                        
                        # Empty returns
                        if re.match(r'\s*return\s+({}|\[\]|None)\s*$', line):
                            issues.append(f"Line {i}: Empty return")
                    
                    if issues:
                        print(f"  {file_path}: {len(issues)} issues")
                        critical_files.append((str(file_path), len(issues), issues))
                        
                        # Show first few issues
                        for issue in issues[:3]:
                            print(f"    - {issue}")
                        if len(issues) > 3:
                            print(f"    ... and {len(issues) - 3} more")
                
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    
    # Sort by issue count
    critical_files.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: Found {len(critical_files)} files with stubs")
    print("=" * 60)
    
    print("\nTOP 10 FILES NEEDING ATTENTION:")
    for i, (file_path, count, issues) in enumerate(critical_files[:10], 1):
        print(f"{i:2d}. {file_path} ({count} issues)")
    
    return critical_files

def search_archive_for_file(filename):
    """Search archive for implementations of a specific file."""
    matches = []
    
    base_name = Path(filename).stem
    
    for root, dirs, files in os.walk('archive'):
        for file in files:
            if file.endswith('.py') and (base_name in file or file in filename):
                archive_path = Path(root) / file
                try:
                    with open(archive_path, 'r') as f:
                        content = f.read()
                    if len(content) > 1000:  # Substantial file
                        matches.append(str(archive_path))
                except:
                    pass
    
    return matches

def main():
    """Main analysis."""
    
    critical_files = find_critical_stubs()
    
    print("\n" + "=" * 60)
    print("CHECKING ARCHIVE FOR IMPLEMENTATIONS")
    print("=" * 60)
    
    # Check top 15 critical files
    for file_path, count, issues in critical_files[:15]:
        matches = search_archive_for_file(file_path)
        if matches:
            print(f"\n{file_path}:")
            for match in matches:
                print(f"  -> Found in archive: {match}")
        else:
            print(f"\n{file_path}: No archive match - needs new implementation")
    
    print(f"\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Restore robust implementations from archive where available")
    print("2. Create new robust implementations for files without archive matches")
    print("3. Focus on core/ and integration/ directories first")

if __name__ == '__main__':
    main()