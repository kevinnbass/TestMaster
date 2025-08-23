#!/usr/bin/env python3
"""
Script to find all Python modules that are not in the archive
and not in any of the cloned research repositories.

This script will:
1. Scan the current directory and all subdirectories
2. Exclude the 'archive' directory
3. Exclude known research repository directories
4. Find all .py files in the remaining directories
5. Write their full pathnames to a text file
"""

import os
import sys
from pathlib import Path

def find_active_python_modules(root_dir=".", output_file="active_python_modules.txt"):
    """
    Find Python modules that are not in archive or research repos.

    Args:
        root_dir (str): Root directory to start scanning from
        output_file (str): Output file to write the results to

    Returns:
        list: List of full paths to active Python modules
    """

    # Define directories to exclude (archive and research repos)
    excluded_dirs = {
        'archive',
        'archives',  # Sometimes archives are plural
        'agency-swarm',
        'autogen',
        'agent-squad',
        'agentops',
        'agentscope',
        'AgentVerse',
        'crewAI',
        'CodeGraph',
        'falkordb-py',
        'AWorld',
        # Additional research repos to exclude
        'MetaGPT', 'metagpt',
        'PraisonAI', 'praisonai',
        'llama-agents',
        'phidata',
        'swarms',
        '__pycache__',
        '.git',
        'node_modules',
        'htmlcov',
        'docs',
        'tests'  # Usually test files are not considered active modules
    }

    # Use lowercase comparison for case-insensitive directory name matching
    excluded_dirs_lower = {name.lower() for name in excluded_dirs}

    # Additional patterns to exclude (common build/test directories)
    excluded_patterns = [
        'test_*.py',
        '*_test.py',
        'setup.py',
        'conftest.py',
        'manage.py'
    ]

    active_modules = []

    print(f"Scanning directory: {os.path.abspath(root_dir)}")
    print("Excluding directories:", sorted(excluded_dirs))
    print()

    for current_dir, dirs, files in os.walk(root_dir):
        # Get the relative path from root to determine if we're in an excluded dir
        rel_path = os.path.relpath(current_dir, root_dir)

        # Skip if we're in an excluded directory
        path_parts = rel_path.split(os.sep)
        if any(part.lower() in excluded_dirs_lower for part in path_parts):
            print(f"Skipping excluded directory: {current_dir}")
            # Remove subdirectories from dirs to prevent walking into them
            dirs.clear()
            continue

        # Find Python files in this directory
        for file in files:
            if file.endswith('.py'):
                full_path = os.path.join(current_dir, file)

                # Check if file matches exclusion patterns
                should_exclude = False
                for pattern in excluded_patterns:
                    if pattern.startswith('*') and pattern.endswith('*.py'):
                        # Pattern like test_*.py or *_test.py
                        prefix = pattern[1:-4]  # Remove * and .py
                        if file.startswith(prefix) or file.endswith(prefix):
                            should_exclude = True
                            break
                    elif file == pattern:
                        should_exclude = True
                        break

                if not should_exclude:
                    active_modules.append(full_path)
                    print(f"Found active module: {full_path}")

    # Sort the results for consistent output
    active_modules.sort()

    # Write to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Active Python Modules (not in archive or research repos)\n")
            f.write(f"# Generated on: {os.path.abspath(root_dir)}\n")
            f.write(f"# Total modules found: {len(active_modules)}\n")
            f.write("#" + "="*60 + "\n\n")

            for module_path in active_modules:
                f.write(module_path + "\n")

        print(f"\nSuccessfully wrote {len(active_modules)} modules to {output_file}")

    except Exception as e:
        print(f"Error writing to output file: {e}")
        return []

    return active_modules

def main():
    """Main function to run the script."""
    print("Python Module Finder")
    print("=" * 40)

    # Allow custom root directory and output file via command line arguments
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    output_file = sys.argv[2] if len(sys.argv) > 2 else "active_python_modules.txt"

    # Find active Python modules
    modules = find_active_python_modules(root_dir, output_file)

    print(f"\nFound {len(modules)} active Python modules")
    print(f"Results saved to: {output_file}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
