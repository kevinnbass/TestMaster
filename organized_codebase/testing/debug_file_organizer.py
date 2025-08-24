#!/usr/bin/env python3
"""
Debug File Organization & Analysis Tool - Agent C Hours 51-53
Analyzes debug files, logs, temporary files, and development artifacts for intelligent organization.
Identifies consolidation opportunities and cleanup strategies.
"""

import ast
import json
import argparse
import re
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys
from datetime import datetime

class DebugFileOrganizer:
    """Analyzes and organizes debug files, logs, and development artifacts."""
    
    def __init__(self):
        self.debug_files = []
        self.log_files = []
        self.temp_files = []
        self.backup_files = []
        self.development_artifacts = []
        self.debug_patterns = defaultdict(int)
        self.file_categories = {}
        self.cleanup_opportunities = []
        self.organization_strategy = {}
        
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze directory for debug and development files."""
        results = {
            'files_analyzed': 0,
            'debug_files': 0,
            'log_files': 0,
            'temp_files': 0,
            'backup_files': 0,
            'development_artifacts': 0,
            'total_size': 0
        }
        
        print("=== Agent C Hours 51-53: Debug File Organization & Analysis ===")
        print(f"Scanning directory: {root_path}")
        
        # Scan all files in directory
        all_files = list(root_path.rglob('*'))
        print(f"Total files found: {len(all_files)}")
        
        for file_path in all_files:
            if file_path.is_file():
                results['files_analyzed'] += 1
                file_size = self._get_file_size(file_path)
                results['total_size'] += file_size
                
                # Categorize file
                category = self._categorize_file(file_path, file_size)
                
                if category == 'debug':
                    self.debug_files.append(self._analyze_debug_file(file_path, file_size))
                    results['debug_files'] += 1
                elif category == 'log':
                    self.log_files.append(self._analyze_log_file(file_path, file_size))
                    results['log_files'] += 1
                elif category == 'temp':
                    self.temp_files.append(self._analyze_temp_file(file_path, file_size))
                    results['temp_files'] += 1
                elif category == 'backup':
                    self.backup_files.append(self._analyze_backup_file(file_path, file_size))
                    results['backup_files'] += 1
                elif category == 'development':
                    self.development_artifacts.append(self._analyze_development_file(file_path, file_size))
                    results['development_artifacts'] += 1
        
        print(f"Categorization complete:")
        print(f"  Debug files: {results['debug_files']}")
        print(f"  Log files: {results['log_files']}")
        print(f"  Temp files: {results['temp_files']}")
        print(f"  Backup files: {results['backup_files']}")
        print(f"  Development artifacts: {results['development_artifacts']}")
        print(f"  Total size: {self._format_size(results['total_size'])}")
        
        return results
    
    def _categorize_file(self, file_path: Path, file_size: int) -> str:
        """Categorize file based on name patterns and characteristics."""
        filename = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Debug file patterns
        debug_patterns = [
            'debug', 'test_', '_test', 'tmp_', 'temp_', 'scratch',
            'playground', 'experiment', 'trial', 'draft'
        ]
        
        # Log file patterns
        log_patterns = [
            '.log', '.out', '.err', 'error', 'trace', 'audit',
            'access.log', 'debug.log', 'error.log'
        ]
        
        # Temp file patterns
        temp_patterns = [
            '.tmp', '.temp', '~', '.swp', '.swo', '.pyc', '__pycache__',
            '.DS_Store', 'Thumbs.db', '.git', '.cache'
        ]
        
        # Backup file patterns
        backup_patterns = [
            '.bak', '.backup', '.old', '_old', '.orig', '_orig',
            '_backup', 'backup_', '.copy', '_copy'
        ]
        
        # Development artifact patterns
        dev_patterns = [
            'TODO', 'FIXME', 'NOTE', '.md', 'README', 'CHANGELOG',
            'requirements', 'setup.py', 'Makefile', '.yml', '.yaml',
            '.json', '.ini', '.cfg', '.config'
        ]
        
        # Check patterns
        if any(pattern in filename for pattern in debug_patterns):
            return 'debug'
        elif any(pattern in filename for pattern in log_patterns):
            return 'log'
        elif any(pattern in filename for pattern in temp_patterns):
            return 'temp'
        elif any(pattern in filename for pattern in backup_patterns):
            return 'backup'
        elif any(pattern in filename for pattern in dev_patterns):
            return 'development'
        
        # Check directory context
        if any(pattern in path_str for pattern in ['debug', 'logs', 'temp', 'tmp']):
            if 'debug' in path_str:
                return 'debug'
            elif any(log_pattern in path_str for log_pattern in ['log', 'logs']):
                return 'log'
            elif any(temp_pattern in path_str for temp_pattern in ['temp', 'tmp']):
                return 'temp'
        
        return 'other'
    
    def _analyze_debug_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Analyze a debug file."""
        analysis = {
            'path': str(file_path),
            'name': file_path.name,
            'size': file_size,
            'extension': file_path.suffix,
            'last_modified': self._get_last_modified(file_path),
            'debug_type': self._determine_debug_type(file_path),
            'cleanup_priority': self._calculate_cleanup_priority(file_path, file_size)
        }
        
        # Try to analyze content for Python files
        if file_path.suffix == '.py':
            content_analysis = self._analyze_python_debug_content(file_path)
            analysis.update(content_analysis)
        
        return analysis
    
    def _analyze_log_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Analyze a log file."""
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': file_size,
            'extension': file_path.suffix,
            'last_modified': self._get_last_modified(file_path),
            'log_type': self._determine_log_type(file_path),
            'retention_recommendation': self._recommend_log_retention(file_path, file_size)
        }
    
    def _analyze_temp_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Analyze a temporary file."""
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': file_size,
            'extension': file_path.suffix,
            'last_modified': self._get_last_modified(file_path),
            'temp_type': self._determine_temp_type(file_path),
            'safe_to_delete': self._is_safe_to_delete(file_path)
        }
    
    def _analyze_backup_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Analyze a backup file."""
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': file_size,
            'extension': file_path.suffix,
            'last_modified': self._get_last_modified(file_path),
            'backup_type': self._determine_backup_type(file_path),
            'archive_recommendation': self._recommend_backup_action(file_path, file_size)
        }
    
    def _analyze_development_file(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Analyze a development artifact."""
        return {
            'path': str(file_path),
            'name': file_path.name,
            'size': file_size,
            'extension': file_path.suffix,
            'last_modified': self._get_last_modified(file_path),
            'artifact_type': self._determine_artifact_type(file_path),
            'consolidation_opportunity': self._assess_consolidation_opportunity(file_path)
        }
    
    def _analyze_python_debug_content(self, file_path: Path) -> Dict[str, Any]:
        """Analyze Python debug file content."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            analysis = {
                'has_main': '__main__' in content,
                'has_print_debug': 'print(' in content,
                'has_imports': re.search(r'^import |^from .+ import', content, re.MULTILINE) is not None,
                'has_functions': 'def ' in content,
                'has_classes': 'class ' in content,
                'line_count': len(content.split('\n')),
                'todo_count': content.lower().count('todo'),
                'fixme_count': content.lower().count('fixme'),
                'debug_statements': content.count('print(') + content.count('pprint(') + content.count('logging.')
            }
            
            # Check if it's likely a test or experiment file
            if any(keyword in content.lower() for keyword in ['test', 'experiment', 'trial', 'scratch']):
                analysis['is_experimental'] = True
            else:
                analysis['is_experimental'] = False
            
            return analysis
            
        except Exception:
            return {'content_analysis_failed': True}
    
    def _get_file_size(self, file_path: Path) -> int:
        """Get file size safely."""
        try:
            return file_path.stat().st_size
        except (OSError, FileNotFoundError):
            return 0
    
    def _get_last_modified(self, file_path: Path) -> str:
        """Get last modified time."""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).isoformat()
        except (OSError, FileNotFoundError):
            return 'unknown'
    
    def _determine_debug_type(self, file_path: Path) -> str:
        """Determine type of debug file."""
        name = file_path.name.lower()
        
        if 'test' in name:
            return 'test_file'
        elif any(word in name for word in ['experiment', 'trial', 'scratch']):
            return 'experimental'
        elif any(word in name for word in ['debug', 'trace']):
            return 'debug_output'
        elif any(word in name for word in ['tmp', 'temp']):
            return 'temporary_debug'
        else:
            return 'unknown_debug'
    
    def _determine_log_type(self, file_path: Path) -> str:
        """Determine type of log file."""
        name = file_path.name.lower()
        
        if 'error' in name:
            return 'error_log'
        elif 'access' in name:
            return 'access_log'
        elif 'debug' in name:
            return 'debug_log'
        elif 'audit' in name:
            return 'audit_log'
        else:
            return 'general_log'
    
    def _determine_temp_type(self, file_path: Path) -> str:
        """Determine type of temporary file."""
        name = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        if extension in ['.pyc', '.pyo']:
            return 'python_cache'
        elif '__pycache__' in str(file_path):
            return 'python_cache'
        elif extension in ['.swp', '.swo', '.tmp']:
            return 'editor_temp'
        elif name in ['.ds_store', 'thumbs.db']:
            return 'system_temp'
        else:
            return 'unknown_temp'
    
    def _determine_backup_type(self, file_path: Path) -> str:
        """Determine type of backup file."""
        name = file_path.name.lower()
        
        if any(word in name for word in ['.bak', '.backup']):
            return 'explicit_backup'
        elif any(word in name for word in ['.old', '_old']):
            return 'version_backup'
        elif any(word in name for word in ['.orig', '_orig']):
            return 'original_backup'
        else:
            return 'unknown_backup'
    
    def _determine_artifact_type(self, file_path: Path) -> str:
        """Determine type of development artifact."""
        name = file_path.name.lower()
        extension = file_path.suffix.lower()
        
        if extension == '.md':
            return 'documentation'
        elif name in ['readme', 'readme.md', 'readme.txt']:
            return 'readme'
        elif name in ['todo', 'todo.md', 'todo.txt']:
            return 'todo_list'
        elif extension in ['.yml', '.yaml']:
            return 'configuration'
        elif extension == '.json':
            return 'json_config'
        elif name in ['makefile', 'dockerfile']:
            return 'build_script'
        else:
            return 'unknown_artifact'
    
    def _calculate_cleanup_priority(self, file_path: Path, file_size: int) -> str:
        """Calculate cleanup priority for debug files."""
        name = file_path.name.lower()
        age_days = self._get_file_age_days(file_path)
        
        # High priority for large, old files
        if file_size > 1024 * 1024 and age_days > 30:  # >1MB and >30 days
            return 'high'
        elif 'temp' in name or 'tmp' in name:
            return 'high'
        elif age_days > 90:  # >3 months
            return 'medium'
        else:
            return 'low'
    
    def _recommend_log_retention(self, file_path: Path, file_size: int) -> str:
        """Recommend log retention strategy."""
        log_type = self._determine_log_type(file_path)
        age_days = self._get_file_age_days(file_path)
        
        if log_type == 'audit_log':
            return 'keep_long_term' if age_days < 365 else 'archive'
        elif log_type == 'error_log':
            return 'keep_medium_term' if age_days < 90 else 'archive'
        elif file_size > 10 * 1024 * 1024:  # >10MB
            return 'compress_and_archive'
        else:
            return 'keep_short_term' if age_days < 30 else 'delete'
    
    def _is_safe_to_delete(self, file_path: Path) -> bool:
        """Determine if temporary file is safe to delete."""
        temp_type = self._determine_temp_type(file_path)
        age_days = self._get_file_age_days(file_path)
        
        # Python cache files are usually safe to delete
        if temp_type == 'python_cache':
            return True
        
        # System temp files are usually safe
        if temp_type == 'system_temp':
            return True
        
        # Old temp files are usually safe
        if age_days > 7:
            return True
        
        return False
    
    def _recommend_backup_action(self, file_path: Path, file_size: int) -> str:
        """Recommend action for backup files."""
        backup_type = self._determine_backup_type(file_path)
        age_days = self._get_file_age_days(file_path)
        
        if backup_type == 'explicit_backup' and age_days < 30:
            return 'keep'
        elif age_days > 180:  # >6 months
            return 'archive_or_delete'
        elif file_size > 5 * 1024 * 1024:  # >5MB
            return 'compress'
        else:
            return 'review'
    
    def _assess_consolidation_opportunity(self, file_path: Path) -> str:
        """Assess consolidation opportunity for development artifacts."""
        artifact_type = self._determine_artifact_type(file_path)
        
        if artifact_type == 'documentation':
            return 'consolidate_with_main_docs'
        elif artifact_type == 'todo_list':
            return 'merge_with_project_todos'
        elif artifact_type == 'configuration':
            return 'review_for_centralization'
        else:
            return 'review_individually'
    
    def _get_file_age_days(self, file_path: Path) -> int:
        """Get file age in days."""
        try:
            timestamp = file_path.stat().st_mtime
            age_seconds = datetime.now().timestamp() - timestamp
            return int(age_seconds / (24 * 60 * 60))
        except (OSError, FileNotFoundError):
            return 0
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
    
    def generate_organization_strategy(self) -> Dict[str, Any]:
        """Generate comprehensive organization strategy."""
        print("\nGenerating organization strategy...")
        
        strategy = {
            'cleanup_recommendations': self._generate_cleanup_recommendations(),
            'organization_structure': self._design_organization_structure(),
            'retention_policies': self._create_retention_policies(),
            'automation_opportunities': self._identify_automation_opportunities(),
            'space_savings': self._calculate_space_savings(),
            'implementation_plan': self._create_implementation_plan()
        }
        
        return strategy
    
    def _generate_cleanup_recommendations(self) -> List[Dict[str, Any]]:
        """Generate cleanup recommendations."""
        recommendations = []
        
        # High priority debug files
        high_priority_debug = [f for f in self.debug_files if f.get('cleanup_priority') == 'high']
        if high_priority_debug:
            total_size = sum(f['size'] for f in high_priority_debug)
            recommendations.append({
                'category': 'debug_files',
                'action': 'cleanup_high_priority',
                'file_count': len(high_priority_debug),
                'space_savings': total_size,
                'priority': 'high'
            })
        
        # Safe to delete temp files
        safe_temp_files = [f for f in self.temp_files if f.get('safe_to_delete', False)]
        if safe_temp_files:
            total_size = sum(f['size'] for f in safe_temp_files)
            recommendations.append({
                'category': 'temp_files',
                'action': 'delete_safe_files',
                'file_count': len(safe_temp_files),
                'space_savings': total_size,
                'priority': 'high'
            })
        
        # Old backup files
        old_backups = [f for f in self.backup_files 
                      if f.get('archive_recommendation') == 'archive_or_delete']
        if old_backups:
            total_size = sum(f['size'] for f in old_backups)
            recommendations.append({
                'category': 'backup_files',
                'action': 'archive_or_delete_old',
                'file_count': len(old_backups),
                'space_savings': total_size,
                'priority': 'medium'
            })
        
        return recommendations
    
    def _design_organization_structure(self) -> Dict[str, Any]:
        """Design improved organization structure."""
        return {
            'debug_directory': {
                'path': 'debug/',
                'subdirectories': ['experiments/', 'tests/', 'scratch/', 'archived/'],
                'purpose': 'Centralized debug and experimental files'
            },
            'logs_directory': {
                'path': 'logs/',
                'subdirectories': ['current/', 'archived/', 'error/', 'audit/'],
                'purpose': 'Structured log file organization'
            },
            'temp_directory': {
                'path': 'temp/',
                'subdirectories': ['cache/', 'working/', 'downloads/'],
                'purpose': 'Temporary files with automated cleanup'
            },
            'backups_directory': {
                'path': 'backups/',
                'subdirectories': ['recent/', 'monthly/', 'archived/'],
                'purpose': 'Organized backup retention'
            },
            'docs_directory': {
                'path': 'docs/',
                'subdirectories': ['technical/', 'user/', 'development/', 'archived/'],
                'purpose': 'Consolidated documentation'
            }
        }
    
    def _create_retention_policies(self) -> Dict[str, Any]:
        """Create retention policies for different file types."""
        return {
            'debug_files': {
                'experimental': '30 days',
                'test_files': '90 days',
                'debug_output': '14 days',
                'temporary_debug': '7 days'
            },
            'log_files': {
                'error_log': '90 days',
                'access_log': '30 days',
                'debug_log': '14 days',
                'audit_log': '1 year'
            },
            'temp_files': {
                'python_cache': 'auto-cleanup',
                'editor_temp': '1 day',
                'system_temp': 'auto-cleanup',
                'unknown_temp': '7 days'
            },
            'backup_files': {
                'explicit_backup': '6 months',
                'version_backup': '3 months',
                'original_backup': '1 year'
            }
        }
    
    def _identify_automation_opportunities(self) -> List[Dict[str, Any]]:
        """Identify automation opportunities."""
        return [
            {
                'automation': 'scheduled_cleanup',
                'description': 'Daily cleanup of safe-to-delete temp files',
                'impact': 'high',
                'implementation': 'cron job or scheduled task'
            },
            {
                'automation': 'log_rotation',
                'description': 'Automatic log rotation and compression',
                'impact': 'medium',
                'implementation': 'logrotate or custom script'
            },
            {
                'automation': 'debug_file_archival',
                'description': 'Weekly archival of old debug files',
                'impact': 'medium',
                'implementation': 'scheduled archive script'
            },
            {
                'automation': 'backup_retention',
                'description': 'Automated backup retention policy enforcement',
                'impact': 'high',
                'implementation': 'backup management script'
            }
        ]
    
    def _calculate_space_savings(self) -> Dict[str, Any]:
        """Calculate potential space savings."""
        cleanup_recs = self._generate_cleanup_recommendations()
        
        total_savings = sum(rec['space_savings'] for rec in cleanup_recs)
        
        return {
            'total_potential_savings': total_savings,
            'formatted_savings': self._format_size(total_savings),
            'savings_by_category': {
                rec['category']: rec['space_savings'] for rec in cleanup_recs
            },
            'percentage_of_total': self._calculate_savings_percentage(total_savings)
        }
    
    def _calculate_savings_percentage(self, savings: int) -> float:
        """Calculate savings as percentage of total analyzed size."""
        total_size = sum(f['size'] for f in self.debug_files + self.log_files + 
                        self.temp_files + self.backup_files + self.development_artifacts)
        
        if total_size > 0:
            return round((savings / total_size) * 100, 2)
        return 0.0
    
    def _create_implementation_plan(self) -> List[Dict[str, Any]]:
        """Create implementation plan for organization strategy."""
        return [
            {
                'phase': 'Phase 1: Safety & Cleanup',
                'duration': '1-2 days',
                'actions': [
                    'Delete safe temp files and caches',
                    'Archive old backup files',
                    'Compress large log files'
                ]
            },
            {
                'phase': 'Phase 2: Organization',
                'duration': '2-3 days',
                'actions': [
                    'Create new directory structure',
                    'Move files to appropriate locations',
                    'Update documentation and scripts'
                ]
            },
            {
                'phase': 'Phase 3: Automation',
                'duration': '1-2 days',
                'actions': [
                    'Implement automated cleanup scripts',
                    'Set up log rotation',
                    'Create retention policy enforcement'
                ]
            }
        ]
    
    def generate_summary(self, analysis_results: Dict[str, Any], organization_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        return {
            'analysis_metadata': {
                'tool': 'debug_file_organizer',
                'version': '1.0',
                'agent': 'Agent_C',
                'hours': '51-53',
                'phase': 'Debug_Markdown_Stowage'
            },
            'analysis_results': analysis_results,
            'file_categories': {
                'debug_files': len(self.debug_files),
                'log_files': len(self.log_files),
                'temp_files': len(self.temp_files),
                'backup_files': len(self.backup_files),
                'development_artifacts': len(self.development_artifacts)
            },
            'organization_strategy': organization_strategy,
            'space_analysis': {
                'total_analyzed_size': analysis_results['total_size'],
                'potential_savings': organization_strategy['space_savings']['total_potential_savings'],
                'savings_percentage': organization_strategy['space_savings']['percentage_of_total']
            },
            'implementation_readiness': self._assess_implementation_readiness(organization_strategy),
            'recommendations': self._generate_final_recommendations(organization_strategy)
        }
    
    def _assess_implementation_readiness(self, strategy: Dict[str, Any]) -> str:
        """Assess readiness for implementation."""
        cleanup_recs = strategy['cleanup_recommendations']
        
        if len(cleanup_recs) > 0:
            high_priority = len([r for r in cleanup_recs if r['priority'] == 'high'])
            if high_priority > 0:
                return 'ready_with_caution'
            else:
                return 'ready'
        else:
            return 'minimal_changes_needed'
    
    def _generate_final_recommendations(self, strategy: Dict[str, Any]) -> List[str]:
        """Generate final recommendations."""
        recommendations = []
        
        space_savings = strategy['space_savings']['percentage_of_total']
        
        if space_savings > 20:
            recommendations.append("High space savings potential - prioritize immediate cleanup")
        elif space_savings > 10:
            recommendations.append("Moderate space savings available - schedule cleanup")
        
        if len(self.temp_files) > 100:
            recommendations.append("High number of temp files - implement automated cleanup")
        
        if len(self.debug_files) > 50:
            recommendations.append("Many debug files found - establish organization structure")
        
        if len(self.backup_files) > 20:
            recommendations.append("Multiple backup files - implement retention policy")
        
        return recommendations


def main():
    parser = argparse.ArgumentParser(description='Debug File Organization & Analysis Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    organizer = DebugFileOrganizer()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = organizer.analyze_directory(root_path)
    
    # Generate organization strategy
    organization_strategy = organizer.generate_organization_strategy()
    
    # Generate summary
    summary = organizer.generate_summary(analysis_results, organization_strategy)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== DEBUG FILE ORGANIZATION COMPLETE ===")
    print(f"Files analyzed: {analysis_results['files_analyzed']}")
    print(f"Debug files: {analysis_results['debug_files']}")
    print(f"Log files: {analysis_results['log_files']}")
    print(f"Temp files: {analysis_results['temp_files']}")
    print(f"Backup files: {analysis_results['backup_files']}")
    print(f"Development artifacts: {analysis_results['development_artifacts']}")
    print(f"Total size analyzed: {organizer._format_size(analysis_results['total_size'])}")
    print(f"Potential space savings: {organization_strategy['space_savings']['formatted_savings']}")
    print(f"Implementation readiness: {summary['implementation_readiness']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()