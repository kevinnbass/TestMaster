#!/usr/bin/env python3
"""Fix the syntax error in the HTML file."""

with open('hybrid_intelligence_dashboard_grouped.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find and remove the problematic lines (around line 2585)
# Remove lines from 2584 to 2610 which contain the broken code
cleaned_lines = []
skip_until_line = 0

for i, line in enumerate(lines, 1):
    # Skip the broken section
    if i == 2584:
        cleaned_lines.append("        // Removed simulation - only show real data\n")
        skip_until_line = 2611
    elif i < skip_until_line:
        continue  # Skip these lines
    else:
        cleaned_lines.append(line)

# Write back
with open('hybrid_intelligence_dashboard_grouped.html', 'w', encoding='utf-8') as f:
    f.writelines(cleaned_lines)

print("Fixed syntax error in HTML file")