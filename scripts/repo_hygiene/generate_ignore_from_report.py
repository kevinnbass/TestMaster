import json
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
REPORT = os.path.join(ROOT, 'tools', 'codebase_monitor', 'reports', 'latest', 'scan.json')
OUT_IGNORE = os.path.join(ROOT, '.gitignore')
ARTIFACT_MANIFEST = os.path.join(ROOT, 'tools', 'codebase_monitor', 'outputs', 'artifact_manifest.json')

def main():
    if not os.path.exists(REPORT):
        print(f"Report not found at {REPORT}")
        return 1
        
    with open(REPORT, 'r', encoding='utf-8') as f:
        data = json.load(f)

    to_ignore = set()
    artifacts = []

    # Heuristics: common artifact dirs/files (tunable)
    candidates = [
        'PRODUCTION_PACKAGES/',
        'organized_codebase/monitoring/',
        '**/telemetry/*.json',
        '*.gz',
        '*.db', 
        'logs/',
        '*.log',
        'tools/codebase_monitor/reports/',
        '__pycache__/',
        '*.pyc',
        '.venv/',
        'node_modules/',
        'dist/',
        'build/',
        '*.egg-info/',
        '.pytest_cache/',
        'coverage/',
        '.coverage',
        '*.bak',
        '*.tmp',
        '*.swp',
        '.DS_Store',
        'Thumbs.db'
    ]
    to_ignore.update(candidates)

    # Directories with many large binaries (from directory_summaries)
    for ds in data.get('directory_summaries', []):
        size_mb = ds.get('total_size_bytes', 0) / (1024 * 1024)
        rel_dir = ds.get('rel_dir', '')
        
        # Ignore very large directories (>100MB) containing assets/media/data
        if size_mb > 100:
            for pattern in ['assets', 'media', 'data', 'vendor', 'third_party']:
                if pattern.lower() in rel_dir.lower():
                    to_ignore.add(rel_dir.rstrip('/') + '/')
                    break
        
        # Ignore directories with many duplicates
        if rel_dir in ['PRODUCTION_PACKAGES', 'organized_codebase/monitoring', 'telemetry']:
            to_ignore.add(rel_dir.rstrip('/') + '/')

    # Save manifest of artifacts (for relocation planning)
    artifacts = sorted(list(to_ignore))
    os.makedirs(os.path.dirname(ARTIFACT_MANIFEST), exist_ok=True)
    with open(ARTIFACT_MANIFEST, 'w', encoding='utf-8') as f:
        json.dump({'ignore_patterns': artifacts, 'generated_from': REPORT}, f, indent=2)

    # Read current .gitignore
    current_content = ""
    if os.path.exists(OUT_IGNORE):
        with open(OUT_IGNORE, 'r', encoding='utf-8') as f:
            current_content = f.read()
    
    # Find patterns to add
    additions = []
    for pat in artifacts:
        # Check if pattern is already in gitignore (simple string match)
        if pat not in current_content:
            additions.append(pat)
    
    # Append new patterns
    if additions:
        with open(OUT_IGNORE, 'a', encoding='utf-8') as f:
            f.write('\n# Codebase monitor auto-generated ignores\n')
            for a in additions:
                f.write(f'{a}\n')
        print(f'Added {len(additions)} patterns to .gitignore')
        print('New patterns:', additions[:10], '...' if len(additions) > 10 else '')
    else:
        print('No new patterns to add to .gitignore')
    
    print(f'Artifact manifest saved to {ARTIFACT_MANIFEST}')
    return 0

if __name__ == '__main__':
    sys.exit(main())