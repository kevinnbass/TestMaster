#!/usr/bin/env python3
"""
Fix Analytics Aggregator
========================
Removes orphaned code from analytics_aggregator.py
"""

import re

def fix_file():
    with open('dashboard/dashboard_core/analytics_aggregator.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find valid methods (starting with "    def ")
    valid_lines = []
    in_orphaned_section = False
    
    for i, line in enumerate(lines):
        # Keep everything up to line 406
        if i < 406:
            valid_lines.append(line)
        # After line 406, only keep properly indented class methods
        elif i == 406:
            valid_lines.append(line)
            in_orphaned_section = True
        elif in_orphaned_section:
            # Check if this is a valid class method
            if line.startswith('    def '):
                in_orphaned_section = False
                valid_lines.append('\n')  # Add blank line before method
                valid_lines.append(line)
            elif not in_orphaned_section:
                valid_lines.append(line)
            # Skip orphaned lines
    
    # Write back
    with open('dashboard/dashboard_core/analytics_aggregator.py', 'w', encoding='utf-8') as f:
        f.writelines(valid_lines)
    
    print(f"Fixed analytics_aggregator.py - removed orphaned code")

if __name__ == "__main__":
    fix_file()