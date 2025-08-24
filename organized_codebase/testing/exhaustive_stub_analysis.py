#!/usr/bin/env python3
"""
Exhaustive stub and placeholder analysis with detailed archive searching.
"""

import os
import re
import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

def analyze_file_deeply(file_path: Path) -> Dict[str, List[str]]:
    """Deep analysis of a Python file for any kind of stub/placeholder."""
    issues = {
        'placeholder_methods': [],
        'empty_returns': [],
        'pass_statements': [],
        'todo_comments': [],
        'stub_implementations': [],
        'mock_references': [],
        'minimal_implementations': [],
        'unimplemented_methods': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        
        # Check each line for various stub patterns
        for i, line in enumerate(lines, 1):
            line_stripped = line.strip()
            
            # TODO/FIXME/HACK comments
            if re.search(r'TODO|FIXME|XXX|HACK|PLACEHOLDER|STUB', line, re.IGNORECASE):
                issues['todo_comments'].append(f"Line {i}: {line_stripped[:100]}")
            
            # Placeholder/stub comments and strings
            if re.search(r'placeholder|stub|mock|dummy|fake|simplified|minimal|not implemented', line, re.IGNORECASE):
                issues['stub_implementations'].append(f"Line {i}: {line_stripped[:100]}")
            
            # Empty returns
            if re.match(r'\s*return\s+({}|\[\]|None|""|\'\'|0|False)\s*$', line):
                issues['empty_returns'].append(f"Line {i}: {line_stripped}")
            
            # Pass statements
            if re.match(r'\s*pass\s*$', line):
                issues['pass_statements'].append(f"Line {i}: {line_stripped}")
            
            # Mock references
            if re.search(r'mock_|Mock|fake_|dummy_|MagicMock|patch', line):
                issues['mock_references'].append(f"Line {i}: {line_stripped[:100]}")
        
        # Parse AST for more sophisticated analysis
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for placeholder method names
                    if any(keyword in node.name.lower() for keyword in ['stub', 'mock', 'dummy', 'fake', 'placeholder', 'todo']):
                        issues['placeholder_methods'].append(f"Function {node.name} at line {node.lineno}")
                    
                    # Check for functions that only raise NotImplementedError
                    if (len(node.body) == 1 and 
                        isinstance(node.body[0], ast.Raise) and
                        isinstance(node.body[0].exc, ast.Call) and
                        isinstance(node.body[0].exc.func, ast.Name) and
                        node.body[0].exc.func.id == 'NotImplementedError'):
                        issues['unimplemented_methods'].append(f"Function {node.name} at line {node.lineno}: raises NotImplementedError")
                    
                    # Check for functions that only return empty values
                    if (len(node.body) == 1 and 
                        isinstance(node.body[0], ast.Return)):
                        ret_val = node.body[0].value
                        if ret_val is None:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns None")
                        elif isinstance(ret_val, ast.Dict) and not ret_val.keys:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns empty dict")
                        elif isinstance(ret_val, ast.List) and not ret_val.elts:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns empty list")
                        elif isinstance(ret_val, ast.Constant) and ret_val.value in [None, False, 0, "", []]:
                            issues['empty_returns'].append(f"Function {node.name} at line {node.lineno}: returns {ret_val.value}")
                    
                    # Check for functions with only pass
                    if (len(node.body) == 1 and 
                        isinstance(node.body[0], ast.Pass)):
                        issues['pass_statements'].append(f"Function {node.name} at line {node.lineno}: only contains pass")
                    
                    # Check for minimal implementations (very short functions)
                    if len(node.body) <= 2 and not any(isinstance(stmt, ast.Return) for stmt in node.body):
                        body_lines = [lines[node.lineno + i - 1].strip() for i in range(len(node.body)) if node.lineno + i - 1 < len(lines)]
                        if all(line in ['pass', ''] or line.startswith('#') for line in body_lines):
                            issues['minimal_implementations'].append(f"Function {node.name} at line {node.lineno}: minimal implementation")
        
        except SyntaxError:
            pass  # Skip files with syntax errors
            
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
    
    return {k: v for k, v in issues.items() if v}

def scan_all_files() -> Dict[str, Dict[str, List[str]]]:
    """Scan absolutely all Python files for stubs and placeholders."""
    
    os.chdir('C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster')
    
    # Only skip version control and cache directories
    skip_patterns = [
        '__pycache__/',
        '.git/',
        '.pyc'
    ]
    
    all_issues = {}
    total_files = 0
    issues_found = 0
    
    print("=" * 80)
    print("EXHAUSTIVE STUB AND PLACEHOLDER ANALYSIS")
    print("=" * 80)
    
    for root, dirs, files in os.walk('.'):
        # Only skip cache and git directories
        dirs[:] = [d for d in dirs if not any(skip in d for skip in ['__pycache__', '.git'])]
        
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = Path(root) / file
            
            # Skip only cache files
            if any(skip in str(file_path) for skip in skip_patterns):
                continue
            
            total_files += 1
            if total_files % 100 == 0:
                print(f"Processed {total_files} files...")
            
            issues = analyze_file_deeply(file_path)
            
            if issues:
                all_issues[str(file_path)] = issues
                issues_found += 1
    
    print(f"\nScanned {total_files} Python files")
    print(f"Found issues in {issues_found} files")
    print("=" * 80)
    
    return all_issues

def exhaustive_archive_search(stub_files: List[str]) -> Dict[str, List[Tuple[str, int]]]:
    """Exhaustively search archive for ANY possible matches."""
    
    print("\n" + "=" * 80)
    print("EXHAUSTIVE ARCHIVE SEARCH")
    print("=" * 80)
    
    archive_matches = {}
    
    # Get all Python files in archive
    archive_files = []
    if os.path.exists('archive'):
        for root, dirs, files in os.walk('archive'):
            for file in files:
                if file.endswith('.py'):
                    archive_files.append(Path(root) / file)
    
    print(f"Found {len(archive_files)} Python files in archive")
    print(f"Searching matches for {len(stub_files)} stub files...")
    
    for stub_file in stub_files:
        stub_name = Path(stub_file).name
        stub_base = stub_name.replace('.py', '')
        stub_parts = stub_base.split('_')
        
        matches = []
        
        # Strategy 1: Exact filename match
        for archive_file in archive_files:
            if archive_file.name == stub_name:
                try:
                    size = archive_file.stat().st_size
                    matches.append((str(archive_file), size))
                except:
                    pass
        
        # Strategy 2: Base name match
        for archive_file in archive_files:
            archive_base = archive_file.name.replace('.py', '')
            if archive_base == stub_base:
                try:
                    size = archive_file.stat().st_size
                    matches.append((str(archive_file), size))
                except:
                    pass
        
        # Strategy 3: Partial name matches
        for archive_file in archive_files:
            archive_base = archive_file.name.replace('.py', '')
            if any(part in archive_base for part in stub_parts if len(part) > 3):
                try:
                    size = archive_file.stat().st_size
                    matches.append((str(archive_file), size))
                except:
                    pass
        
        # Strategy 4: Content-based search (look for similar class/function names)
        try:
            with open(stub_file, 'r', encoding='utf-8') as f:
                stub_content = f.read()
            
            # Extract class and function names from stub file
            try:
                stub_tree = ast.parse(stub_content)
                stub_names = set()
                for node in ast.walk(stub_tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                        stub_names.add(node.name)
                
                # Search for archives with similar names
                for archive_file in archive_files:
                    try:
                        with open(archive_file, 'r', encoding='utf-8') as f:
                            archive_content = f.read()
                        
                        archive_tree = ast.parse(archive_content)
                        archive_names = set()
                        for node in ast.walk(archive_tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                archive_names.add(node.name)
                        
                        # If significant overlap in names
                        if stub_names and archive_names:
                            overlap = len(stub_names.intersection(archive_names))
                            if overlap >= min(2, len(stub_names) * 0.5):
                                size = archive_file.stat().st_size
                                matches.append((str(archive_file), size))
                    except:
                        continue
            except:
                pass
        except:
            pass
        
        # Remove duplicates and sort by size (larger files likely more complete)
        unique_matches = {}
        for match, size in matches:
            if match not in unique_matches or size > unique_matches[match]:
                unique_matches[match] = size
        
        if unique_matches:
            sorted_matches = sorted(unique_matches.items(), key=lambda x: x[1], reverse=True)
            archive_matches[stub_file] = sorted_matches
    
    return archive_matches

def categorize_and_prioritize(all_issues: Dict[str, Dict[str, List[str]]]) -> Dict[str, List[Tuple[str, Dict, int]]]:
    """Categorize files by priority and issue severity."""
    
    critical_files = []
    important_files = []
    minor_files = []
    
    for file_path, issues in all_issues.items():
        issue_count = sum(len(issue_list) for issue_list in issues.values())
        
        # Calculate severity score
        severity = 0
        severity += len(issues.get('unimplemented_methods', [])) * 10  # High priority
        severity += len(issues.get('placeholder_methods', [])) * 8
        severity += len(issues.get('pass_statements', [])) * 6
        severity += len(issues.get('empty_returns', [])) * 4
        severity += len(issues.get('minimal_implementations', [])) * 3
        severity += len(issues.get('todo_comments', [])) * 2
        severity += len(issues.get('stub_implementations', [])) * 5
        
        # Prioritize by location and severity
        if 'core/' in file_path or 'integration/' in file_path:
            if severity > 20:
                critical_files.append((file_path, issues, severity))
            elif severity > 5:
                important_files.append((file_path, issues, severity))
            else:
                minor_files.append((file_path, issues, severity))
        elif 'agents/' in file_path or 'dashboard/' in file_path:
            if severity > 15:
                important_files.append((file_path, issues, severity))
            else:
                minor_files.append((file_path, issues, severity))
        else:
            minor_files.append((file_path, issues, severity))
    
    # Sort by severity
    critical_files.sort(key=lambda x: x[2], reverse=True)
    important_files.sort(key=lambda x: x[2], reverse=True)
    minor_files.sort(key=lambda x: x[2], reverse=True)
    
    return {
        'critical': critical_files,
        'important': important_files,
        'minor': minor_files
    }

def main():
    """Main exhaustive analysis."""
    
    # Scan all files
    all_issues = scan_all_files()
    
    # Categorize by priority
    categorized = categorize_and_prioritize(all_issues)
    
    print(f"\n[CRITICAL] {len(categorized['critical'])} CRITICAL FILES:")
    for file_path, issues, severity in categorized['critical'][:20]:
        print(f"\n[FILE] {file_path} (severity: {severity})")
        for category, issue_list in issues.items():
            if issue_list:
                print(f"   {category}: {len(issue_list)} items")
                for issue in issue_list[:2]:  # Show first 2
                    print(f"     - {issue}")
                if len(issue_list) > 2:
                    print(f"     ... and {len(issue_list) - 2} more")
    
    print(f"\n[IMPORTANT] {len(categorized['important'])} IMPORTANT FILES:")
    for file_path, issues, severity in categorized['important'][:15]:
        print(f"\n[FILE] {file_path} (severity: {severity})")
        for category, issue_list in issues.items():
            if issue_list:
                print(f"   {category}: {len(issue_list)} items")
    
    print(f"\n[MINOR] {len(categorized['minor'])} MINOR FILES (showing top 10):")
    for file_path, issues, severity in categorized['minor'][:10]:
        print(f"   [FILE] {file_path} (severity: {severity})")
    
    # Get all stub files for archive search
    all_stub_files = []
    all_stub_files.extend([f[0] for f in categorized['critical']])
    all_stub_files.extend([f[0] for f in categorized['important']])
    all_stub_files.extend([f[0] for f in categorized['minor'][:50]])  # Top 50 minor files
    
    # Exhaustive archive search
    archive_matches = exhaustive_archive_search(all_stub_files)
    
    # Show results
    print(f"\n[ARCHIVE MATCHES] Found archive matches for {len(archive_matches)} files:")
    for stub_file, matches in archive_matches.items():
        print(f"\n[FILE] {stub_file}")
        for match, size in matches[:3]:  # Show top 3 matches
            print(f"   [ARCHIVE] {match} ({size} bytes)")
        if len(matches) > 3:
            print(f"   ... and {len(matches) - 3} more matches")
    
    # Final recommendations
    with_archive = [f for f in all_stub_files if f in archive_matches]
    without_archive = [f for f in all_stub_files if f not in archive_matches]
    
    print(f"\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    print(f"[RESTORABLE] {len(with_archive)} files can be restored from archive")
    print(f"[DE-NOVO] {len(without_archive)} files need new robust implementations")
    
    # Save detailed results
    results = {
        'categorized': {
            'critical': [(f[0], f[2]) for f in categorized['critical']],
            'important': [(f[0], f[2]) for f in categorized['important']],
            'minor': [(f[0], f[2]) for f in categorized['minor']]
        },
        'archive_matches': {k: [match[0] for match in v] for k, v in archive_matches.items()},
        'restorable': with_archive,
        'de_novo_needed': without_archive
    }
    
    with open('exhaustive_stub_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: exhaustive_stub_analysis_results.json")

if __name__ == '__main__':
    main()