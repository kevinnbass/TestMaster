#!/usr/bin/env python3
"""
Agent C Hours 60-62: Legacy Code Preservation Analyzer
Advanced legacy code analysis and preservation strategy development
"""

import ast
import json
import os
import re
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse

class LegacyCodePreservationAnalyzer:
    """Analyzes legacy code patterns and creates preservation strategies."""
    
    def __init__(self):
        self.legacy_files = []
        self.legacy_patterns = defaultdict(list)
        self.preservation_strategies = defaultdict(list)
        self.code_archaeology = defaultdict(dict)
        self.migration_candidates = []
        self.preservation_priority = []
        
    def analyze_legacy_patterns(self, root_path):
        """Analyze codebase for legacy patterns and preservation needs."""
        print(f"Analyzing legacy code patterns in: {root_path}")
        
        for file_path in Path(root_path).rglob("*.py"):
            try:
                self._analyze_file_for_legacy_patterns(file_path)
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
                
        self._identify_preservation_strategies()
        self._calculate_preservation_priority()
        
        return {
            'legacy_files': len(self.legacy_files),
            'legacy_patterns': dict(self.legacy_patterns),
            'preservation_strategies': dict(self.preservation_strategies),
            'migration_candidates': len(self.migration_candidates),
            'preservation_priority': len(self.preservation_priority)
        }
    
    def _analyze_file_for_legacy_patterns(self, file_path):
        """Analyze individual file for legacy code patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Check for legacy indicators
            legacy_indicators = self._detect_legacy_indicators(content, file_path)
            
            if legacy_indicators:
                self.legacy_files.append({
                    'file': str(file_path),
                    'size': len(content),
                    'indicators': legacy_indicators,
                    'last_modified': os.path.getmtime(file_path)
                })
                
                # Perform code archaeology
                self._perform_code_archaeology(file_path, content)
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    def _detect_legacy_indicators(self, content, file_path):
        """Detect various legacy code indicators."""
        indicators = []
        
        # Python version indicators
        if re.search(r'print\s+[^(]', content):
            indicators.append('python2_print_statements')
        
        if re.search(r'from __future__ import', content):
            indicators.append('future_imports')
            
        if re.search(r'xrange\(', content):
            indicators.append('python2_xrange')
            
        # Legacy libraries and patterns
        legacy_imports = [
            'imp ', 'optparse', 'sets', 'md5', 'sha', 'cPickle',
            'ConfigParser', 'StringIO', 'urllib2', 'urlparse'
        ]
        
        for legacy_import in legacy_imports:
            if legacy_import in content:
                indicators.append(f'legacy_import_{legacy_import.strip()}')
        
        # Deprecated patterns
        if re.search(r'has_key\(', content):
            indicators.append('deprecated_has_key')
            
        if re.search(r'apply\(', content):
            indicators.append('deprecated_apply')
            
        # Old-style classes (Python 2)
        if re.search(r'class\s+\w+:', content):
            indicators.append('old_style_class')
            
        # Legacy string formatting
        if re.search(r'%[sd]', content):
            indicators.append('old_string_formatting')
            
        # File modification date analysis
        file_age_days = (datetime.now().timestamp() - os.path.getmtime(file_path)) / 86400
        if file_age_days > 365:
            indicators.append('old_file_over_1_year')
        if file_age_days > 1095:
            indicators.append('old_file_over_3_years')
            
        # Large uncommented blocks
        lines = content.split('\n')
        comment_ratio = sum(1 for line in lines if line.strip().startswith('#')) / max(len(lines), 1)
        if comment_ratio < 0.1 and len(lines) > 100:
            indicators.append('poorly_documented_legacy')
            
        return indicators
    
    def _perform_code_archaeology(self, file_path, content):
        """Perform archaeological analysis of code patterns."""
        archaeology = {
            'file_hash': hashlib.md5(content.encode()).hexdigest(),
            'line_count': len(content.split('\n')),
            'character_count': len(content),
            'function_count': len(re.findall(r'def\s+\w+\(', content)),
            'class_count': len(re.findall(r'class\s+\w+', content)),
            'import_count': len(re.findall(r'^import\s+|^from\s+\w+\s+import', content, re.MULTILINE)),
            'comment_lines': len([line for line in content.split('\n') if line.strip().startswith('#')]),
            'blank_lines': len([line for line in content.split('\n') if not line.strip()]),
            'complexity_indicators': self._calculate_complexity_indicators(content)
        }
        
        self.code_archaeology[str(file_path)] = archaeology
    
    def _calculate_complexity_indicators(self, content):
        """Calculate various complexity indicators for legacy assessment."""
        indicators = {}
        
        # Cyclomatic complexity approximation
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally', 'with']
        indicators['cyclomatic_approximation'] = sum(
            len(re.findall(rf'\b{keyword}\b', content)) for keyword in complexity_keywords
        )
        
        # Nesting depth approximation
        max_indent = 0
        for line in content.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent // 4)  # Assuming 4-space indentation
        indicators['max_nesting_depth'] = max_indent
        
        # TODO and FIXME counts
        indicators['todo_count'] = len(re.findall(r'TODO|FIXME|XXX|HACK', content, re.IGNORECASE))
        
        # Magic number count
        indicators['magic_numbers'] = len(re.findall(r'\b\d{2,}\b', content))
        
        # Long line count
        long_lines = [line for line in content.split('\n') if len(line) > 100]
        indicators['long_lines'] = len(long_lines)
        
        return indicators
    
    def _identify_preservation_strategies(self):
        """Identify appropriate preservation strategies for different types of legacy code."""
        
        for legacy_file in self.legacy_files:
            file_path = legacy_file['file']
            indicators = legacy_file['indicators']
            
            # Strategy 1: Modernization candidates
            if any(indicator in indicators for indicator in ['python2_print_statements', 'old_string_formatting', 'legacy_import_']):
                self.preservation_strategies['modernization_candidates'].append({
                    'file': file_path,
                    'strategy': 'modernize_syntax',
                    'effort': 'low',
                    'risk': 'low'
                })
            
            # Strategy 2: Documentation candidates
            if 'poorly_documented_legacy' in indicators:
                self.preservation_strategies['documentation_candidates'].append({
                    'file': file_path,
                    'strategy': 'enhance_documentation',
                    'effort': 'medium',
                    'risk': 'low'
                })
            
            # Strategy 3: Refactoring candidates
            archaeology = self.code_archaeology.get(file_path, {})
            complexity = archaeology.get('complexity_indicators', {})
            
            if complexity.get('cyclomatic_approximation', 0) > 20 or complexity.get('max_nesting_depth', 0) > 5:
                self.preservation_strategies['refactoring_candidates'].append({
                    'file': file_path,
                    'strategy': 'reduce_complexity',
                    'effort': 'high',
                    'risk': 'medium'
                })
            
            # Strategy 4: Archive candidates
            if any(indicator in indicators for indicator in ['old_file_over_3_years']) and legacy_file['size'] < 1000:
                self.preservation_strategies['archive_candidates'].append({
                    'file': file_path,
                    'strategy': 'archive_with_documentation',
                    'effort': 'low',
                    'risk': 'low'
                })
            
            # Strategy 5: Migration candidates
            if len(indicators) > 3:
                self.migration_candidates.append({
                    'file': file_path,
                    'indicators': indicators,
                    'migration_priority': len(indicators),
                    'estimated_effort': self._estimate_migration_effort(indicators, archaeology)
                })
    
    def _estimate_migration_effort(self, indicators, archaeology):
        """Estimate effort required for migrating legacy code."""
        base_effort = 1
        
        # Increase effort based on indicators
        effort_multipliers = {
            'python2_print_statements': 1.2,
            'legacy_import_': 1.5,
            'deprecated_has_key': 1.3,
            'old_style_class': 2.0,
            'poorly_documented_legacy': 1.8
        }
        
        for indicator in indicators:
            for pattern, multiplier in effort_multipliers.items():
                if pattern in indicator:
                    base_effort *= multiplier
        
        # Adjust for complexity
        complexity = archaeology.get('complexity_indicators', {})
        if complexity.get('cyclomatic_approximation', 0) > 30:
            base_effort *= 2.0
        if complexity.get('max_nesting_depth', 0) > 6:
            base_effort *= 1.5
            
        return min(base_effort, 10.0)  # Cap at 10x effort
    
    def _calculate_preservation_priority(self):
        """Calculate preservation priority based on various factors."""
        
        for legacy_file in self.legacy_files:
            file_path = legacy_file['file']
            indicators = legacy_file['indicators']
            archaeology = self.code_archaeology.get(file_path, {})
            
            # Calculate priority score
            priority_score = 0
            
            # Age factor
            if 'old_file_over_3_years' in indicators:
                priority_score += 3
            elif 'old_file_over_1_year' in indicators:
                priority_score += 1
                
            # Complexity factor
            complexity = archaeology.get('complexity_indicators', {})
            if complexity.get('cyclomatic_approximation', 0) > 20:
                priority_score += 2
            if complexity.get('todo_count', 0) > 5:
                priority_score += 1
                
            # Size factor
            if legacy_file['size'] > 10000:
                priority_score += 2
            elif legacy_file['size'] > 5000:
                priority_score += 1
                
            # Legacy pattern factor
            priority_score += len(indicators)
            
            self.preservation_priority.append({
                'file': file_path,
                'priority_score': priority_score,
                'preservation_urgency': 'high' if priority_score > 8 else 'medium' if priority_score > 4 else 'low'
            })
        
        # Sort by priority score
        self.preservation_priority.sort(key=lambda x: x['priority_score'], reverse=True)
    
    def generate_preservation_plan(self):
        """Generate comprehensive preservation implementation plan."""
        plan = {
            'preservation_overview': {
                'total_legacy_files': len(self.legacy_files),
                'preservation_strategies': len(self.preservation_strategies),
                'migration_candidates': len(self.migration_candidates),
                'high_priority_files': len([p for p in self.preservation_priority if p['preservation_urgency'] == 'high'])
            },
            'implementation_phases': [
                {
                    'phase': 'Phase 1: Critical Preservation',
                    'duration': '1-2 weeks',
                    'focus': 'High-priority legacy files and security concerns',
                    'actions': [
                        'Document critical legacy functions',
                        'Create preservation archives',
                        'Fix immediate security issues',
                        'Establish version control for legacy code'
                    ]
                },
                {
                    'phase': 'Phase 2: Modernization',
                    'duration': '3-4 weeks',
                    'focus': 'Syntax modernization and simple migrations',
                    'actions': [
                        'Update Python 2 syntax to Python 3',
                        'Replace deprecated libraries',
                        'Modernize string formatting',
                        'Update import statements'
                    ]
                },
                {
                    'phase': 'Phase 3: Refactoring',
                    'duration': '4-6 weeks',
                    'focus': 'Complexity reduction and architecture improvement',
                    'actions': [
                        'Reduce cyclomatic complexity',
                        'Improve code organization',
                        'Extract reusable components',
                        'Enhance error handling'
                    ]
                },
                {
                    'phase': 'Phase 4: Documentation & Testing',
                    'duration': '2-3 weeks',
                    'focus': 'Comprehensive documentation and test coverage',
                    'actions': [
                        'Create comprehensive documentation',
                        'Add unit tests for legacy functions',
                        'Create migration guides',
                        'Establish maintenance procedures'
                    ]
                }
            ],
            'risk_mitigation': {
                'backup_strategy': 'Create full backups before any modifications',
                'testing_strategy': 'Implement comprehensive regression testing',
                'rollback_plan': 'Maintain rollback capabilities for all changes',
                'monitoring': 'Monitor system behavior after modifications'
            }
        }
        
        return plan
    
    def export_results(self, output_file):
        """Export analysis results to JSON file."""
        results = {
            'analysis_metadata': {
                'tool': 'legacy_code_preservation_analyzer',
                'version': '1.0',
                'agent': 'Agent_C',
                'hours': '60-62',
                'phase': 'Debug_Markdown_Stowage',
                'timestamp': datetime.now().isoformat()
            },
            'analysis_results': {
                'legacy_files_count': len(self.legacy_files),
                'legacy_patterns': dict(self.legacy_patterns),
                'preservation_strategies': dict(self.preservation_strategies),
                'migration_candidates_count': len(self.migration_candidates),
                'preservation_priority_count': len(self.preservation_priority)
            },
            'legacy_files': self.legacy_files,
            'preservation_strategies': dict(self.preservation_strategies),
            'migration_candidates': self.migration_candidates,
            'preservation_priority': self.preservation_priority,
            'code_archaeology': dict(self.code_archaeology),
            'preservation_plan': self.generate_preservation_plan(),
            'implementation_readiness': self._assess_implementation_readiness()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _assess_implementation_readiness(self):
        """Assess readiness for implementation."""
        total_files = len(self.legacy_files)
        high_priority = len([p for p in self.preservation_priority if p['preservation_urgency'] == 'high'])
        
        if high_priority > total_files * 0.3:
            return 'needs_immediate_attention'
        elif high_priority > total_files * 0.1:
            return 'ready_with_planning'
        else:
            return 'ready_for_implementation'

def main():
    parser = argparse.ArgumentParser(description='Analyze legacy code for preservation strategies')
    parser.add_argument('--root', default='.', help='Root directory to analyze')
    parser.add_argument('--output', default='legacy_preservation_hour60.json', help='Output file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 60-62: Legacy Code Preservation Analysis ===")
    
    analyzer = LegacyCodePreservationAnalyzer()
    results = analyzer.analyze_legacy_patterns(args.root)
    
    print(f"Legacy files found: {results['legacy_files']}")
    print(f"Migration candidates: {results['migration_candidates']}")
    print(f"Preservation strategies: {len(results['preservation_strategies'])}")
    
    export_results = analyzer.export_results(args.output)
    
    print(f"\n=== LEGACY CODE PRESERVATION ANALYSIS COMPLETE ===")
    print(f"Legacy files: {export_results['analysis_results']['legacy_files_count']}")
    print(f"Preservation strategies: {len(export_results['preservation_strategies'])}")
    print(f"Migration candidates: {export_results['analysis_results']['migration_candidates_count']}")
    print(f"Implementation readiness: {export_results['implementation_readiness']}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()