#!/usr/bin/env python3
"""
Agent C Hours 63-65: Debug Pattern Recognition Analyzer
Advanced debug pattern detection and development artifact analysis
"""

import ast
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import argparse

class DebugPatternRecognizer:
    """Recognizes debug patterns and development artifacts in codebase."""
    
    def __init__(self):
        self.debug_patterns = defaultdict(list)
        self.development_artifacts = defaultdict(list)
        self.debug_hotspots = []
        self.pattern_statistics = defaultdict(int)
        self.cleanup_recommendations = []
        
    def analyze_debug_patterns(self, root_path):
        """Analyze codebase for debug patterns and development artifacts."""
        print(f"Analyzing debug patterns in: {root_path}")
        
        for file_path in Path(root_path).rglob("*"):
            if file_path.is_file():
                try:
                    self._analyze_file_for_debug_patterns(file_path)
                except Exception as e:
                    print(f"Error analyzing {file_path}: {e}")
                
        self._identify_debug_hotspots()
        self._generate_cleanup_recommendations()
        
        return {
            'debug_patterns': len(self.debug_patterns),
            'development_artifacts': len(self.development_artifacts),
            'debug_hotspots': len(self.debug_hotspots),
            'pattern_statistics': dict(self.pattern_statistics)
        }
    
    def _analyze_file_for_debug_patterns(self, file_path):
        """Analyze individual file for debug patterns."""
        try:
            # Skip binary files and certain extensions
            skip_extensions = {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin', '.jpg', '.png', '.gif', '.pdf', '.zip', '.gz', '.tar'}
            if file_path.suffix.lower() in skip_extensions:
                return
                
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Detect debug patterns
            debug_patterns = self._detect_debug_patterns(content, file_path)
            development_artifacts = self._detect_development_artifacts(content, file_path)
            
            if debug_patterns:
                self.debug_patterns[str(file_path)] = debug_patterns
                
            if development_artifacts:
                self.development_artifacts[str(file_path)] = development_artifacts
                
        except Exception as e:
            # Skip files we can't read
            pass
    
    def _detect_debug_patterns(self, content, file_path):
        """Detect various debug patterns in code."""
        patterns = []
        
        # Python debug patterns
        if file_path.suffix == '.py':
            # Print debugging
            print_debug = re.findall(r'print\s*\(["\'].*?debug.*?["\']', content, re.IGNORECASE)
            if print_debug:
                patterns.append({
                    'type': 'print_debugging',
                    'count': len(print_debug),
                    'examples': print_debug[:3]
                })
                self.pattern_statistics['print_debugging'] += len(print_debug)
            
            # PDB debugging
            pdb_patterns = re.findall(r'import\s+pdb|pdb\.set_trace\(\)', content)
            if pdb_patterns:
                patterns.append({
                    'type': 'pdb_debugging',
                    'count': len(pdb_patterns),
                    'examples': pdb_patterns[:3]
                })
                self.pattern_statistics['pdb_debugging'] += len(pdb_patterns)
            
            # IPython debugging
            ipdb_patterns = re.findall(r'import\s+ipdb|ipdb\.set_trace\(\)', content)
            if ipdb_patterns:
                patterns.append({
                    'type': 'ipdb_debugging',
                    'count': len(ipdb_patterns),
                    'examples': ipdb_patterns[:3]
                })
                self.pattern_statistics['ipdb_debugging'] += len(ipdb_patterns)
        
        # General debug patterns (all file types)
        
        # Debug comments
        debug_comments = re.findall(r'#.*?(?:debug|DEBUG|Debug|test|TEST|Test|hack|HACK|Hack|todo|TODO|Todo|fixme|FIXME|Fixme)', content)
        if debug_comments:
            patterns.append({
                'type': 'debug_comments',
                'count': len(debug_comments),
                'examples': debug_comments[:5]
            })
            self.pattern_statistics['debug_comments'] += len(debug_comments)
        
        # Console logging patterns
        console_patterns = re.findall(r'console\.log|console\.debug|console\.error|console\.warn', content, re.IGNORECASE)
        if console_patterns:
            patterns.append({
                'type': 'console_logging',
                'count': len(console_patterns),
                'examples': console_patterns[:3]
            })
            self.pattern_statistics['console_logging'] += len(console_patterns)
        
        # Commented out code blocks
        commented_code = re.findall(r'(?:^|\n)\s*#[^#\n]*(?:def |class |import |from |if |for |while )', content, re.MULTILINE)
        if commented_code:
            patterns.append({
                'type': 'commented_code',
                'count': len(commented_code),
                'examples': commented_code[:3]
            })
            self.pattern_statistics['commented_code'] += len(commented_code)
        
        # Assert statements for debugging
        assert_debug = re.findall(r'assert\s+.*?["\'].*?(?:debug|test)', content, re.IGNORECASE)
        if assert_debug:
            patterns.append({
                'type': 'assert_debugging',
                'count': len(assert_debug),
                'examples': assert_debug[:3]
            })
            self.pattern_statistics['assert_debugging'] += len(assert_debug)
        
        # Temporary variable patterns
        temp_vars = re.findall(r'\b(?:temp|tmp|debug|test)_\w+\b', content, re.IGNORECASE)
        if temp_vars:
            patterns.append({
                'type': 'temporary_variables',
                'count': len(temp_vars),
                'examples': temp_vars[:5]
            })
            self.pattern_statistics['temporary_variables'] += len(temp_vars)
        
        return patterns
    
    def _detect_development_artifacts(self, content, file_path):
        """Detect development artifacts that might need cleanup."""
        artifacts = []
        
        # Check filename patterns
        filename = file_path.name.lower()
        
        # Development file patterns
        dev_patterns = [
            'test', 'debug', 'temp', 'tmp', 'scratch', 'experiment',
            'backup', 'old', 'copy', 'draft', 'wip', 'work_in_progress'
        ]
        
        for pattern in dev_patterns:
            if pattern in filename:
                artifacts.append({
                    'type': 'development_filename',
                    'pattern': pattern,
                    'severity': 'medium'
                })
                self.pattern_statistics['development_filename'] += 1
        
        # Backup file patterns
        backup_extensions = ['.bak', '.backup', '.old', '.orig', '.save']
        if any(filename.endswith(ext) for ext in backup_extensions):
            artifacts.append({
                'type': 'backup_file',
                'severity': 'low'
            })
            self.pattern_statistics['backup_file'] += 1
        
        # Version control artifacts
        if any(pattern in str(file_path) for pattern in ['.git', '.svn', '.hg', '__pycache__']):
            artifacts.append({
                'type': 'version_control_artifact',
                'severity': 'low'
            })
            self.pattern_statistics['version_control_artifact'] += 1
        
        # Large comment blocks (potential dead code)
        large_comment_blocks = re.findall(r'(?:^|\n)\s*#{3,}.*?#{3,}', content, re.MULTILINE | re.DOTALL)
        if large_comment_blocks:
            artifacts.append({
                'type': 'large_comment_blocks',
                'count': len(large_comment_blocks),
                'severity': 'medium'
            })
            self.pattern_statistics['large_comment_blocks'] += len(large_comment_blocks)
        
        # Empty or minimal files
        lines = content.strip().split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        
        if len(non_empty_lines) < 3 and len(content) > 0:
            artifacts.append({
                'type': 'minimal_file',
                'line_count': len(non_empty_lines),
                'severity': 'low'
            })
            self.pattern_statistics['minimal_file'] += 1
        
        # Files with only imports (potentially unused)
        if file_path.suffix == '.py' and len(non_empty_lines) > 0:
            import_lines = [line for line in non_empty_lines if line.strip().startswith(('import ', 'from '))]
            if len(import_lines) == len(non_empty_lines):
                artifacts.append({
                    'type': 'import_only_file',
                    'import_count': len(import_lines),
                    'severity': 'medium'
                })
                self.pattern_statistics['import_only_file'] += 1
        
        return artifacts
    
    def _identify_debug_hotspots(self):
        """Identify files/areas with high concentration of debug patterns."""
        
        file_debug_scores = {}
        
        for file_path, patterns in self.debug_patterns.items():
            debug_score = 0
            for pattern in patterns:
                debug_score += pattern['count']
            
            # Add development artifact score
            if file_path in self.development_artifacts:
                debug_score += len(self.development_artifacts[file_path])
            
            file_debug_scores[file_path] = debug_score
        
        # Sort by debug score and identify hotspots
        sorted_files = sorted(file_debug_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Top 20% are considered hotspots
        hotspot_threshold = max(1, len(sorted_files) // 5)
        
        for file_path, score in sorted_files[:hotspot_threshold]:
            if score > 0:  # Only include files with actual debug patterns
                self.debug_hotspots.append({
                    'file': file_path,
                    'debug_score': score,
                    'priority': 'high' if score > 10 else 'medium' if score > 5 else 'low',
                    'patterns': self.debug_patterns.get(file_path, []),
                    'artifacts': self.development_artifacts.get(file_path, [])
                })
    
    def _generate_cleanup_recommendations(self):
        """Generate recommendations for cleaning up debug patterns and artifacts."""
        
        # Recommendation 1: Remove print debugging
        if self.pattern_statistics['print_debugging'] > 0:
            self.cleanup_recommendations.append({
                'type': 'remove_print_debugging',
                'priority': 'high',
                'description': f"Remove {self.pattern_statistics['print_debugging']} print debugging statements",
                'effort': 'low',
                'risk': 'low'
            })
        
        # Recommendation 2: Remove PDB statements
        if self.pattern_statistics['pdb_debugging'] > 0:
            self.cleanup_recommendations.append({
                'type': 'remove_pdb_debugging',
                'priority': 'high',
                'description': f"Remove {self.pattern_statistics['pdb_debugging']} PDB debugging statements",
                'effort': 'low',
                'risk': 'low'
            })
        
        # Recommendation 3: Clean up commented code
        if self.pattern_statistics['commented_code'] > 10:
            self.cleanup_recommendations.append({
                'type': 'clean_commented_code',
                'priority': 'medium',
                'description': f"Review and clean up {self.pattern_statistics['commented_code']} commented code blocks",
                'effort': 'medium',
                'risk': 'medium'
            })
        
        # Recommendation 4: Remove development files
        if self.pattern_statistics['development_filename'] > 0:
            self.cleanup_recommendations.append({
                'type': 'remove_development_files',
                'priority': 'medium',
                'description': f"Review {self.pattern_statistics['development_filename']} development/temporary files for removal",
                'effort': 'low',
                'risk': 'medium'
            })
        
        # Recommendation 5: Clean up backup files
        if self.pattern_statistics['backup_file'] > 0:
            self.cleanup_recommendations.append({
                'type': 'clean_backup_files',
                'priority': 'low',
                'description': f"Remove {self.pattern_statistics['backup_file']} backup files",
                'effort': 'low',
                'risk': 'low'
            })
        
        # Recommendation 6: Address debug hotspots
        high_priority_hotspots = [h for h in self.debug_hotspots if h['priority'] == 'high']
        if high_priority_hotspots:
            self.cleanup_recommendations.append({
                'type': 'address_debug_hotspots',
                'priority': 'high',
                'description': f"Address {len(high_priority_hotspots)} high-priority debug hotspots",
                'effort': 'high',
                'risk': 'medium'
            })
    
    def generate_cleanup_strategy(self):
        """Generate comprehensive cleanup implementation strategy."""
        strategy = {
            'cleanup_overview': {
                'total_debug_patterns': sum(self.pattern_statistics.values()),
                'debug_hotspots': len(self.debug_hotspots),
                'cleanup_recommendations': len(self.cleanup_recommendations),
                'estimated_cleanup_time': self._estimate_cleanup_time()
            },
            'cleanup_phases': [
                {
                    'phase': 'Phase 1: Quick Wins',
                    'duration': '1-2 days',
                    'focus': 'Remove obvious debug statements and temporary files',
                    'actions': [
                        'Remove print debugging statements',
                        'Remove PDB/IPDB debugging statements',
                        'Delete obvious temporary files',
                        'Clean up simple backup files'
                    ]
                },
                {
                    'phase': 'Phase 2: Code Review',
                    'duration': '3-5 days',
                    'focus': 'Review and clean commented code and development artifacts',
                    'actions': [
                        'Review commented code blocks',
                        'Evaluate development files for removal',
                        'Clean up debug comments',
                        'Address import-only files'
                    ]
                },
                {
                    'phase': 'Phase 3: Hotspot Resolution',
                    'duration': '1-2 weeks',
                    'focus': 'Address high-priority debug hotspots',
                    'actions': [
                        'Refactor files with high debug concentrations',
                        'Implement proper logging where needed',
                        'Add proper testing for debug-heavy areas',
                        'Document any preserved debug patterns'
                    ]
                }
            ],
            'automation_opportunities': [
                {
                    'automation': 'debug_pattern_linting',
                    'description': 'Add linting rules to prevent debug patterns in commits',
                    'impact': 'high',
                    'implementation': 'pre-commit hooks'
                },
                {
                    'automation': 'temporary_file_cleanup',
                    'description': 'Automated cleanup of temporary and backup files',
                    'impact': 'medium',
                    'implementation': 'scheduled cleanup script'
                }
            ]
        }
        
        return strategy
    
    def _estimate_cleanup_time(self):
        """Estimate time required for cleanup activities."""
        base_time = 0
        
        # Time estimates in hours
        time_estimates = {
            'print_debugging': 0.1,  # 6 minutes per statement
            'pdb_debugging': 0.1,
            'commented_code': 0.5,   # 30 minutes per block
            'development_filename': 0.25,  # 15 minutes per file
            'backup_file': 0.05,     # 3 minutes per file
            'debug_hotspots': 2.0    # 2 hours per hotspot
        }
        
        for pattern, count in self.pattern_statistics.items():
            if pattern in time_estimates:
                base_time += count * time_estimates[pattern]
        
        # Add hotspot time
        base_time += len(self.debug_hotspots) * time_estimates['debug_hotspots']
        
        return f"{base_time:.1f} hours"
    
    def export_results(self, output_file):
        """Export analysis results to JSON file."""
        results = {
            'analysis_metadata': {
                'tool': 'debug_pattern_recognizer',
                'version': '1.0',
                'agent': 'Agent_C',
                'hours': '63-65',
                'phase': 'Debug_Markdown_Stowage',
                'timestamp': datetime.now().isoformat()
            },
            'analysis_results': {
                'debug_patterns_count': len(self.debug_patterns),
                'development_artifacts_count': len(self.development_artifacts),
                'debug_hotspots_count': len(self.debug_hotspots),
                'pattern_statistics': dict(self.pattern_statistics),
                'cleanup_recommendations_count': len(self.cleanup_recommendations)
            },
            'debug_patterns': dict(self.debug_patterns),
            'development_artifacts': dict(self.development_artifacts),
            'debug_hotspots': self.debug_hotspots,
            'pattern_statistics': dict(self.pattern_statistics),
            'cleanup_recommendations': self.cleanup_recommendations,
            'cleanup_strategy': self.generate_cleanup_strategy(),
            'implementation_readiness': self._assess_implementation_readiness()
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def _assess_implementation_readiness(self):
        """Assess readiness for cleanup implementation."""
        total_patterns = sum(self.pattern_statistics.values())
        high_priority_hotspots = len([h for h in self.debug_hotspots if h['priority'] == 'high'])
        
        if total_patterns > 1000 or high_priority_hotspots > 50:
            return 'needs_significant_planning'
        elif total_patterns > 100 or high_priority_hotspots > 10:
            return 'ready_with_planning'
        else:
            return 'ready_for_immediate_implementation'

def main():
    parser = argparse.ArgumentParser(description='Recognize debug patterns and development artifacts')
    parser.add_argument('--root', default='.', help='Root directory to analyze')
    parser.add_argument('--output', default='debug_patterns_hour63.json', help='Output file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 63-65: Debug Pattern Recognition ===")
    
    recognizer = DebugPatternRecognizer()
    results = recognizer.analyze_debug_patterns(args.root)
    
    print(f"Debug patterns found: {results['debug_patterns']}")
    print(f"Development artifacts: {results['development_artifacts']}")
    print(f"Debug hotspots: {results['debug_hotspots']}")
    
    export_results = recognizer.export_results(args.output)
    
    print(f"\n=== DEBUG PATTERN RECOGNITION COMPLETE ===")
    print(f"Debug patterns: {export_results['analysis_results']['debug_patterns_count']}")
    print(f"Development artifacts: {export_results['analysis_results']['development_artifacts_count']}")
    print(f"Debug hotspots: {export_results['analysis_results']['debug_hotspots_count']}")
    print(f"Cleanup recommendations: {export_results['analysis_results']['cleanup_recommendations_count']}")
    print(f"Implementation readiness: {export_results['implementation_readiness']}")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()