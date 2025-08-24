#!/usr/bin/env python3
"""
Quick Organization Script
========================

A simple script to help you organize your codebase by creating directories
and showing which files could be moved where, without actually moving them.

Run this to see what a reorganized structure would look like!
"""

import os
from pathlib import Path

def create_organization_preview():
    """Show what the organized structure would look like"""
    print("Codebase Organization Preview")
    print("=" * 35)
    print()
    
    # Define the organization plan based on analysis
    organization_plan = {
        'tests/': {
            'description': 'All test files',
            'files': [
                'test_*.py',  # Pattern - all files starting with test_
                'monitor_db.py',
                'simple_framework_test.py',
                'simple_test.py',
                'system_integration_test.py',
                'agent_c_test_splitter.py'
            ]
        },
        
        'analyzers/': {
            'description': 'Code analysis tools',
            'files': [
                'focus_analyzer.py',
                'dependency_analyzer.py',
                'simple_codebase_analyzer.py',
                'file_breakdown_advisor.py',
                'organization_advisor.py',
                'codebase_inventory_analysis.py',
                'functional_linkage_analyzer.py',
                'query_analyzer.py',
                'legacy_code_preservation_analyzer.py',
                'debug_pattern_recognizer.py',
                'agent_a_redundancy_analyzer.py',
                'agent_a_size_analyzer.py',
                'agent_b_analysis.py',
                'api_dependency_analyzer.py',
                'documentation_architecture_analyzer.py'
            ]
        },
        
        'web/': {
            'description': 'Web dashboards and interfaces',
            'files': [
                'enhanced_linkage_dashboard.py',  # Your 4,429 line giant!
                'enhanced_linkage_dashboard_BACKUP_20250822_011701.py',
                'enhanced_intelligence_linkage.py',
                'agent_coordination_dashboard.py',
                'enhanced_dashboard.py',
                'complete_dashboard.py',
                'working_dashboard.py',
                'simple_working_dashboard.py',
                'gamma_visualization_enhancements.py',
                'hybrid_dashboard_integration.py',
                'launch_live_dashboard.py',
                'debug_server.py',
                'test_template.py'
            ]
        },
        
        'utils/': {
            'description': 'Utility scripts and tools',
            'files': [
                'codebase_toolkit.py',  # Your new command-line interface
                'break_commits.py',
                'chunked_git_push.py',
                'simple_chunk_push.py',
                'smart_chunk_push.py',
                'persistent_push_system.py',
                'git_push_monitor.py',
                'enhanced_git_monitor.py',
                'current_push_monitor.py',
                'live_push_tracker.py',
                'simple_push_monitor.py',
                'add_linkage_to_dashboard.py'
            ]
        },
        
        'deployment/': {
            'description': 'Production deployment scripts',
            'files': [
                'PRODUCTION_DEPLOYMENT_PACKAGE.py',
                'production_deployment.py',
                'DEPLOY_SECURITY_FIXES.py',
                'DEPLOY_SECURITY_FIXES_PHASE2.py',
                'CONTINUOUS_MONITORING_SYSTEM.py'
            ]
        },
        
        'monitoring/': {
            'description': 'System monitoring tools',
            'files': [
                'monitoring_system.py',
                'backup_monitor.py',
                'db_monitor_standalone.py',
                'db_growth_tracker.py',
                'unified_enhanced_monitor.py',
                'monitor_now.py'
            ]
        }
    }
    
    # Show the plan
    current_files = set(f.name for f in Path('.').glob('*.py'))
    total_organized = 0
    
    for directory, info in organization_plan.items():
        print(f"{directory}")
        print(f"   {info['description']}")
        print()
        
        actual_files = []
        for pattern_or_file in info['files']:
            if pattern_or_file == 'test_*.py':
                # Handle pattern
                test_files = [f for f in current_files if f.startswith('test_') and f.endswith('.py')]
                actual_files.extend(test_files)
            else:
                if pattern_or_file in current_files:
                    actual_files.append(pattern_or_file)
        
        actual_files = list(set(actual_files))  # Remove duplicates
        actual_files.sort()
        
        for file in actual_files:
            print(f"   - {file}")
            
        print(f"   Total: {len(actual_files)} files")
        print()
        total_organized += len(actual_files)
    
    # Show remaining files
    organized_files = set()
    for info in organization_plan.values():
        for pattern_or_file in info['files']:
            if pattern_or_file == 'test_*.py':
                test_files = [f for f in current_files if f.startswith('test_') and f.endswith('.py')]
                organized_files.update(test_files)
            else:
                organized_files.add(pattern_or_file)
    
    remaining_files = current_files - organized_files
    remaining_files = sorted([f for f in remaining_files if f.endswith('.py')])
    
    if remaining_files:
        print("core/ (remaining files)")
        print("   Files that need individual assessment")
        print()
        for file in remaining_files:
            print(f"   - {file}")
        print(f"   Total: {len(remaining_files)} files")
        print()
    
    # Summary
    print("=" * 50)
    print("ORGANIZATION SUMMARY")
    print("=" * 50)
    print(f"Total Python files found: {len(current_files)}")
    print(f"Files with clear organization: {total_organized}")
    print(f"Files needing assessment: {len(remaining_files)}")
    print(f"Organization coverage: {(total_organized/len(current_files))*100:.1f}%")
    print()
    
    print("IMPACT:")
    print(f"   - Root directory would go from {len(current_files)} files to {len(remaining_files)} files")
    print(f"   - That's a {((len(current_files) - len(remaining_files))/len(current_files))*100:.1f}% reduction in root clutter!")
    print()
    
    print("NEXT STEPS:")
    print("   1. Create the directories: mkdir tests analyzers web utils deployment monitoring")
    print("   2. Start with safest moves (tests/ directory)")
    print("   3. Move files one directory at a time")
    print("   4. Update any import statements that break")
    print("   5. Test functionality after each move")

def show_biggest_wins():
    """Show which moves would have the biggest impact"""
    print("\n" + "=" * 50)
    print("BIGGEST WINS - Start Here!")
    print("=" * 50)
    
    wins = [
        {
            'directory': 'tests/',
            'impact': 'Move ~34 test files out of root',
            'risk': 'LOW - tests rarely imported by main code',
            'benefit': 'Immediate visual cleanup, tests clearly separated'
        },
        {
            'directory': 'web/',
            'impact': 'Move 13 dashboard files (including 4,429-line monster)',
            'risk': 'MEDIUM - may have import dependencies',
            'benefit': 'Huge file size reduction, web files grouped logically'
        },
        {
            'directory': 'analyzers/',
            'impact': 'Move 15 analysis tools (your new personal tools!)',
            'risk': 'LOW - mostly standalone analysis scripts',
            'benefit': 'Your analysis toolkit cleanly organized'
        }
    ]
    
    for i, win in enumerate(wins, 1):
        print(f"{i}. {win['directory']}")
        print(f"   Impact: {win['impact']}")
        print(f"   Risk: {win['risk']}")
        print(f"   Benefit: {win['benefit']}")
        print()

if __name__ == "__main__":
    create_organization_preview()
    show_biggest_wins()