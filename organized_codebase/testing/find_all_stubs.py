#!/usr/bin/env python3
"""
Comprehensive stub and placeholder detection script.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set

def analyze_file(file_path: Path) -> Dict[str, List[str]]:
    """Analyze a Python file for stubs and placeholders."""
    issues = {
        'placeholder_methods': [],
        'empty_returns': [],
        'pass_statements': [],
        'todo_comments': [],
        'stub_implementations': [],
        'mock_references': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Check each line
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # TODO/FIXME comments
            if re.search(r'TODO|FIXME|XXX|HACK', line, re.IGNORECASE):
                issues['todo_comments'].append(f"Line {i}: {line_stripped}")
            
            # Placeholder/stub comments
            if re.search(r'placeholder|stub|mock|dummy|fake|simplified', line, re.IGNORECASE):
                issues['stub_implementations'].append(f"Line {i}: {line_stripped}")
            
            # Empty returns
            if re.match(r'\s*return\s+({}|\[\]|None)\s*$', line):
                issues['empty_returns'].append(f"Line {i}: {line_stripped}")
            
            # Pass statements
            if re.match(r'\s*pass\s*$', line):
                issues['pass_statements'].append(f"Line {i}: {line_stripped}")
            
            # Mock references
            if re.search(r'mock_|Mock|fake_|dummy_', line):
                issues['mock_references'].append(f"Line {i}: {line_stripped}")
        
        # Parse AST for more sophisticated analysis
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for placeholder methods
                    if any(keyword in node.name.lower() for keyword in ['stub', 'mock', 'dummy', 'fake']):
                        issues['placeholder_methods'].append(f"Function {node.name} at line {node.lineno}")
                    
                    # Check for functions that only return empty values
                    if (len(node.body) == 1 and 
                        isinstance(node.body[0], ast.Return) and
                        node.body[0].value is not None):
                        
                        if isinstance(node.body[0].value, (ast.Dict, ast.List)) and not node.body[0].value.keys and not node.body[0].value.elts:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns empty dict/list")
                        elif isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value is None:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns None")
                    
                    # Check for functions with only pass
                    if (len(node.body) == 1 and 
                        isinstance(node.body[0], ast.Pass)):
                        issues['pass_statements'].append(f"Function {node.name} at line {node.lineno}: only contains pass")
        
        except SyntaxError:
            pass  # Skip files with syntax errors
            
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return {k: v for k, v in issues.items() if v}

def scan_codebase():
    """Scan the entire codebase for stubs and placeholders."""
    
    os.chdir('C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster')
    
    # Files to skip
    skip_patterns = [
        'archive/',
        '__pycache__/',
        '.git/',
        'test_',
        '_test',
        '.pyc',
        'fix_',
        'debug_',
        'find_',
        'enhance_',
        'restore_'
    ]
    
    all_issues = {}
    total_files = 0
    issues_found = 0
    
    print("=" * 80)
    print("COMPREHENSIVE STUB AND PLACEHOLDER ANALYSIS")
    print("=" * 80)
    
    for root, dirs, files in os.walk('.'):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not any(skip in d for skip in skip_patterns)]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = Path(root) / file
            
            # Skip files matching skip patterns
            if any(skip in str(file_path) for skip in skip_patterns):
                continue
            
            total_files += 1
            issues = analyze_file(file_path)
            
            if issues:
                all_issues[str(file_path)] = issues
                issues_found += 1
    
    print(f"\nScanned {total_files} Python files")
    print(f"Found issues in {issues_found} files")
    print("=" * 80)
    
    # Categorize and prioritize issues
    critical_files = []
    important_files = []
    minor_files = []
    
    for file_path, issues in all_issues.items():
        issue_count = sum(len(issue_list) for issue_list in issues.values())
        
        # Categorize by severity and location
        if 'core/' in file_path or 'integration/' in file_path:
            if issue_count > 5:
                critical_files.append((file_path, issues, issue_count))
            else:
                important_files.append((file_path, issues, issue_count))
        else:
            minor_files.append((file_path, issues, issue_count))
    
    # Sort by issue count
    critical_files.sort(key=lambda x: x[2], reverse=True)
    important_files.sort(key=lambda x: x[2], reverse=True)
    minor_files.sort(key=lambda x: x[2], reverse=True)
    
    print("\\n[CRITICAL] CRITICAL FILES (core/integration with many stubs):")
    for file_path, issues, count in critical_files[:10]:
        print(f"\\n[FILE] {file_path} ({count} issues)")
        for category, issue_list in issues.items():
            if issue_list:
                print(f"   {category}: {len(issue_list)} items")
                for issue in issue_list[:3]:  # Show first 3
                    print(f"     - {issue}")
                if len(issue_list) > 3:
                    print(f"     ... and {len(issue_list) - 3} more")
    
    print("\\n[IMPORTANT] IMPORTANT FILES (core/integration with few stubs):")
    for file_path, issues, count in important_files[:10]:
        print(f"\\n[FILE] {file_path} ({count} issues)")
        for category, issue_list in issues.items():
            if issue_list:
                print(f"   {category}: {len(issue_list)} items")
    
    print("\\n[MINOR] MINOR FILES (other areas):")
    for file_path, issues, count in minor_files[:5]:
        print(f"   [FILE] {file_path} ({count} issues)")
    
    print(f"\\n... and {len(minor_files) - 5} more minor files" if len(minor_files) > 5 else "")
    
    return {
        'critical': critical_files,
        'important': important_files, 
        'minor': minor_files
    }

def search_archive_for_implementations(file_list: List[str]) -> Dict[str, List[str]]:
    """Search archive for robust implementations of stub files."""
    
    print("\\n" + "=" * 80)
    print("SEARCHING ARCHIVE FOR ROBUST IMPLEMENTATIONS")
    print("=" * 80)
    
    archive_matches = {}
    
    for file_path in file_list:
        file_name = Path(file_path).name
        base_name = file_name.replace('.py', '')
        
        # Search patterns
        search_patterns = [
            f"**/{file_name}",
            f"**/{base_name}*.py",
            f"**/*{base_name}*.py"
        ]
        
        matches = []
        for pattern in search_patterns:
            try:
                for archive_file in Path('archive').rglob(pattern):
                    if archive_file.is_file() and archive_file.suffix == '.py':
                        # Check if it's more substantial than current
                        try:
                            with open(archive_file, 'r') as f:
                                archive_content = f.read()
                            
                            if len(archive_content) > 1000:  # Substantial implementation
                                matches.append(str(archive_file))
                        except:
                            pass
            except:
                pass
        
        if matches:
            archive_matches[file_path] = matches
    
    for file_path, matches in archive_matches.items():
        print(f"\\n[FILE] {file_path}")
        for match in matches:
            print(f"   [ARCHIVE] Found in archive: {match}")
    
    return archive_matches

def main():
    """Main analysis."""
    
    # Scan for stubs
    results = scan_codebase()
    
    # Get list of critical and important files
    priority_files = [f[0] for f in results['critical']] + [f[0] for f in results['important']]
    
    # Search archive
    archive_matches = search_archive_for_implementations(priority_files)
    
    print("\\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Files with archive matches
    with_archive = [f for f in priority_files if f in archive_matches]
    without_archive = [f for f in priority_files if f not in archive_matches]
    
    print(f"\\n[RESTORABLE] {len(with_archive)} files can be restored from archive:")
    for file_path in with_archive:
        print(f"   [FILE] {file_path}")
    
    print(f"\\n[DE-NOVO] {len(without_archive)} files need new robust implementations:")
    for file_path in without_archive[:10]:  # Show top 10
        print(f"   [FILE] {file_path}")
    
    if len(without_archive) > 10:
        print(f"   ... and {len(without_archive) - 10} more")
    
    print("\\nNext steps:")
    print("1. Restore implementations from archive where available")
    print("2. Create robust implementations for remaining stub files")
    print("3. Focus on critical and important files first")

if __name__ == '__main__':
    main()