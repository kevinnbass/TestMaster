#!/usr/bin/env python3
"""
Agent C Hours 57-75: Documentation Architecture & Archive Validation
Complete markdown consolidation and documentation architecture analysis
"""

import os
import json
import hashlib
import time
from collections import defaultdict, Counter
import re

def analyze_documentation_architecture():
    """Analyze documentation architecture and consolidation opportunities"""
    start_time = time.time()
    
    doc_analysis = {
        'markdown_files': [],
        'documentation_types': defaultdict(list),
        'duplicate_content': defaultdict(list),
        'consolidation_opportunities': [],
        'archive_validation': {},
        'organization_metrics': {}
    }
    
    files_analyzed = 0
    total_size = 0
    duplicate_sections = 0
    
    # Documentation patterns
    doc_types = {
        'roadmap': r'roadmap|plan|strategy',
        'readme': r'readme|getting.started',
        'api_docs': r'api|endpoint|swagger',
        'tutorial': r'tutorial|guide|howto',
        'architecture': r'architecture|design|system',
        'changelog': r'changelog|history|release',
        'config': r'config|settings|env',
        'deployment': r'deploy|install|setup'
    }
    
    # Walk through all documentation files
    for root, dirs, files in os.walk('.'):
        if any(skip in root for skip in ['__pycache__', '.git', 'node_modules']):
            continue
            
        for file in files:
            if file.endswith(('.md', '.txt', '.rst', '.adoc')):
                file_path = os.path.join(root, file)
                files_analyzed += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    file_size = len(content.encode('utf-8'))
                    total_size += file_size
                    
                    # Categorize documentation type
                    doc_type = 'general'
                    for type_name, pattern in doc_types.items():
                        if re.search(pattern, file.lower()) or re.search(pattern, content[:1000].lower()):
                            doc_type = type_name
                            break
                    
                    # Create content hash for duplicate detection
                    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
                    
                    file_info = {
                        'path': file_path,
                        'size': file_size,
                        'type': doc_type,
                        'content_hash': content_hash,
                        'lines': len(content.split('\n')),
                        'sections': len(re.findall(r'^#+\s', content, re.MULTILINE))
                    }
                    
                    doc_analysis['markdown_files'].append(file_info)
                    doc_analysis['documentation_types'][doc_type].append(file_info)
                    
                    # Check for duplicate content
                    doc_analysis['duplicate_content'][content_hash].append(file_info)
                    if len(doc_analysis['duplicate_content'][content_hash]) > 1:
                        duplicate_sections += 1
                    
                except Exception as e:
                    continue
    
    # Analyze consolidation opportunities
    for doc_type, files in doc_analysis['documentation_types'].items():
        if len(files) > 3 and doc_type != 'general':
            total_size_type = sum(f['size'] for f in files)
            consolidation_opportunity = {
                'type': doc_type,
                'file_count': len(files),
                'total_size': total_size_type,
                'consolidation_potential': min(total_size_type * 0.7, total_size_type - 10000),
                'recommended_action': f'Consolidate {len(files)} {doc_type} files into unified documentation'
            }
            doc_analysis['consolidation_opportunities'].append(consolidation_opportunity)
    
    # Archive validation metrics
    archive_dirs = []
    for root, dirs, files in os.walk('.'):
        if 'archive' in root.lower() or 'backup' in root.lower():
            archive_dirs.append(root)
    
    doc_analysis['archive_validation'] = {
        'archive_directories_found': len(archive_dirs),
        'archive_paths': archive_dirs[:10],  # First 10 for space
        'total_archive_size': sum(os.path.getsize(os.path.join(root, file)) 
                                 for archive in archive_dirs[:5] 
                                 for root, dirs, files in os.walk(archive) 
                                 for file in files if os.path.exists(os.path.join(root, file))),
        'retrieval_test_status': 'ready_for_validation'
    }
    
    # Organization metrics
    doc_analysis['organization_metrics'] = {
        'total_documentation_files': files_analyzed,
        'total_documentation_size': total_size,
        'duplicate_content_instances': duplicate_sections,
        'documentation_types_count': len(doc_analysis['documentation_types']),
        'consolidation_opportunities_count': len(doc_analysis['consolidation_opportunities']),
        'average_file_size': total_size / max(files_analyzed, 1),
        'organization_health_score': min(100, max(0, 100 - (duplicate_sections * 5) - (len(doc_analysis['consolidation_opportunities']) * 10)))
    }
    
    analysis_time = time.time() - start_time
    
    # Final results
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analysis_metrics': {
            'files_analyzed': files_analyzed,
            'total_documentation_size': total_size,
            'duplicate_sections': duplicate_sections,
            'analysis_duration_seconds': round(analysis_time, 2)
        },
        'documentation_architecture': doc_analysis['organization_metrics'],
        'consolidation_analysis': {
            'opportunities_identified': len(doc_analysis['consolidation_opportunities']),
            'total_consolidation_potential': sum(op['consolidation_potential'] for op in doc_analysis['consolidation_opportunities']),
            'recommended_actions': doc_analysis['consolidation_opportunities']
        },
        'archive_validation': doc_analysis['archive_validation'],
        'documentation_types': {doc_type: len(files) for doc_type, files in doc_analysis['documentation_types'].items()},
        'detailed_analysis': {
            'duplicate_content_groups': {hash_val: len(files) for hash_val, files in doc_analysis['duplicate_content'].items() if len(files) > 1},
            'largest_files': sorted([f for f in doc_analysis['markdown_files']], key=lambda x: x['size'], reverse=True)[:10],
            'documentation_distribution': doc_analysis['documentation_types']
        }
    }
    
    # Save results
    with open('documentation_architecture_hour57.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == '__main__':
    # Run analysis
    results = analyze_documentation_architecture()
    print('DOCUMENTATION ARCHITECTURE ANALYSIS COMPLETE')
    print(f'Files Analyzed: {results["analysis_metrics"]["files_analyzed"]}')
    print(f'Total Size: {results["analysis_metrics"]["total_documentation_size"]} bytes')
    print(f'Duplicate Sections: {results["analysis_metrics"]["duplicate_sections"]}')
    print(f'Consolidation Opportunities: {results["consolidation_analysis"]["opportunities_identified"]}')
    print(f'Analysis Time: {results["analysis_metrics"]["analysis_duration_seconds"]}s')
    print('Results saved to: documentation_architecture_hour57.json')