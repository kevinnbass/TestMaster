"""
Comprehensive Codebase Analysis for Deep Optimization
====================================================

Phase 1: Complete codebase mapping and redundancy detection
"""

import os
import ast
import hashlib
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import json
from datetime import datetime


class CodebaseAnalyzer:
    """Comprehensive analyzer for redundancy detection and architecture optimization."""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.files_analyzed = 0
        self.total_lines = 0
        self.function_signatures = defaultdict(list)  # signature -> [files containing it]
        self.class_definitions = defaultdict(list)
        self.import_patterns = defaultdict(list)
        self.file_purposes = {}  # file -> purpose analysis
        self.redundancy_candidates = []
        self.architecture_opportunities = {}
        
    def analyze_complete_codebase(self) -> Dict[str, Any]:
        """Perform comprehensive codebase analysis."""
        print("Starting comprehensive codebase analysis...")
        
        # Get all Python files
        py_files = list(self.root_path.glob("**/*.py"))
        print(f"Found {len(py_files)} Python files")
        
        # Analyze each file
        for file_path in py_files:
            if self._should_analyze_file(file_path):
                self._analyze_file(file_path)
        
        # Detect redundancies
        self._detect_redundancies()
        
        # Analyze architecture opportunities
        self._analyze_architecture_opportunities()
        
        # Generate comprehensive report
        return self._generate_analysis_report()
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if file should be analyzed."""
        # Skip certain directories
        skip_dirs = {'.git', '__pycache__', '.pytest_cache', 'node_modules', 'venv', 'env'}
        
        # Skip if in excluded directory
        for skip_dir in skip_dirs:
            if skip_dir in file_path.parts:
                return False
        
        # Skip empty files
        try:
            if file_path.stat().st_size == 0:
                return False
        except OSError:
            return False
        
        return True
    
    def _analyze_file(self, file_path: Path) -> None:
        """Analyze individual file for patterns and functionality."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            self.files_analyzed += 1
            self.total_lines += len(content.splitlines())
            
            # Parse AST
            try:
                tree = ast.parse(content)
                self._extract_ast_patterns(file_path, tree)
            except SyntaxError:
                print(f"Syntax error in {file_path}")
                return
            
            # Analyze file purpose
            self._analyze_file_purpose(file_path, content)
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
    
    def _extract_ast_patterns(self, file_path: Path, tree: ast.AST) -> None:
        """Extract patterns from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Extract function signature
                args = [arg.arg for arg in node.args.args]
                signature = f"{node.name}({', '.join(args)})"
                self.function_signatures[signature].append(str(file_path))
                
            elif isinstance(node, ast.ClassDef):
                # Extract class definition
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                class_info = {
                    'name': node.name,
                    'methods': methods,
                    'file': str(file_path)
                }
                self.class_definitions[node.name].append(class_info)
                
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # Extract import patterns
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.import_patterns[alias.name].append(str(file_path))
                else:
                    module = node.module or ""
                    for alias in node.names:
                        import_str = f"from {module} import {alias.name}"
                        self.import_patterns[import_str].append(str(file_path))
    
    def _analyze_file_purpose(self, file_path: Path, content: str) -> None:
        """Analyze the purpose and functionality of a file."""
        lines = content.splitlines()
        
        purpose_indicators = {
            'test': ['test_', 'import pytest', 'import unittest', 'TestCase'],
            'api': ['Flask', 'Blueprint', '@app.route', '@bp.route', 'jsonify'],
            'database': ['SQLAlchemy', 'database', 'db.', 'CREATE TABLE'],
            'config': ['config', 'settings', 'CONFIG', 'SETTINGS'],
            'utility': ['utils', 'helper', 'common', 'shared'],
            'monitoring': ['monitor', 'metric', 'log', 'track'],
            'security': ['auth', 'security', 'encrypt', 'decrypt'],
            'analysis': ['analyze', 'analyzer', 'analysis'],
            'orchestration': ['orchestrat', 'coordinat', 'workflow'],
            'intelligence': ['intelligence', 'smart', 'ai', 'ml']
        }
        
        file_content_lower = content.lower()
        file_name_lower = file_path.name.lower()
        
        detected_purposes = []
        for purpose, indicators in purpose_indicators.items():
            for indicator in indicators:
                if (indicator in file_content_lower or 
                    indicator in file_name_lower):
                    detected_purposes.append(purpose)
                    break
        
        self.file_purposes[str(file_path)] = {
            'purposes': detected_purposes,
            'line_count': len(lines),
            'size_category': self._categorize_file_size(len(lines))
        }
    
    def _categorize_file_size(self, line_count: int) -> str:
        """Categorize file size."""
        if line_count < 50:
            return 'tiny'
        elif line_count < 150:
            return 'small'
        elif line_count < 300:
            return 'medium'
        elif line_count < 500:
            return 'large'
        else:
            return 'huge'
    
    def _detect_redundancies(self) -> None:
        """Detect potential redundancies in functionality."""
        print("Detecting redundancies...")
        
        # Find functions that appear in multiple files
        redundant_functions = {
            sig: files for sig, files in self.function_signatures.items()
            if len(files) > 1
        }
        
        # Find similar classes
        redundant_classes = {}
        for class_name, instances in self.class_definitions.items():
            if len(instances) > 1:
                # Check if they have similar method sets
                method_sets = [set(instance['methods']) for instance in instances]
                if len(method_sets) > 1:
                    # Calculate similarity
                    similarities = []
                    for i in range(len(method_sets)):
                        for j in range(i + 1, len(method_sets)):
                            intersection = len(method_sets[i] & method_sets[j])
                            union = len(method_sets[i] | method_sets[j])
                            similarity = intersection / union if union > 0 else 0
                            similarities.append(similarity)
                    
                    avg_similarity = sum(similarities) / len(similarities)
                    if avg_similarity > 0.7:  # 70% similarity threshold
                        redundant_classes[class_name] = {
                            'instances': instances,
                            'similarity': avg_similarity
                        }
        
        self.redundancy_candidates = [
            {
                'type': 'function',
                'name': sig,
                'files': files,
                'severity': 'high' if len(files) > 3 else 'medium'
            }
            for sig, files in redundant_functions.items()
        ]
        
        self.redundancy_candidates.extend([
            {
                'type': 'class',
                'name': name,
                'files': [inst['file'] for inst in data['instances']],
                'similarity': data['similarity'],
                'severity': 'high' if data['similarity'] > 0.8 else 'medium'
            }
            for name, data in redundant_classes.items()
        ])
    
    def _analyze_architecture_opportunities(self) -> None:
        """Analyze opportunities for better architecture organization."""
        print("Analyzing architecture opportunities...")
        
        # Group files by purpose
        purpose_groups = defaultdict(list)
        for file_path, info in self.file_purposes.items():
            for purpose in info['purposes']:
                purpose_groups[purpose].append({
                    'file': file_path,
                    'line_count': info['line_count'],
                    'size_category': info['size_category']
                })
        
        # Identify opportunities
        opportunities = {}
        
        for purpose, files in purpose_groups.items():
            if len(files) > 3:  # Multiple files with same purpose
                total_lines = sum(f['line_count'] for f in files)
                
                # Check if files are scattered
                directories = set(Path(f['file']).parent for f in files)
                
                opportunities[purpose] = {
                    'file_count': len(files),
                    'total_lines': total_lines,
                    'scattered_across': len(directories),
                    'directories': list(str(d) for d in directories),
                    'consolidation_opportunity': len(directories) > 2,
                    'files': files
                }
        
        self.architecture_opportunities = opportunities
    
    def _generate_analysis_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        return {
            'summary': {
                'files_analyzed': self.files_analyzed,
                'total_lines': self.total_lines,
                'redundancy_candidates': len(self.redundancy_candidates),
                'architecture_opportunities': len(self.architecture_opportunities)
            },
            'redundancy_analysis': {
                'high_priority': [r for r in self.redundancy_candidates if r['severity'] == 'high'],
                'medium_priority': [r for r in self.redundancy_candidates if r['severity'] == 'medium'],
                'total_candidates': len(self.redundancy_candidates)
            },
            'architecture_analysis': self.architecture_opportunities,
            'file_distribution': {
                purpose: {
                    'count': len([f for f, info in self.file_purposes.items() 
                                if purpose in info['purposes']]),
                    'total_lines': sum(info['line_count'] for f, info in self.file_purposes.items()
                                     if purpose in info['purposes'])
                }
                for purpose in set(p for info in self.file_purposes.values() 
                                 for p in info['purposes'])
            },
            'size_distribution': dict(Counter(
                info['size_category'] for info in self.file_purposes.values()
            ))
        }


def main():
    """Run comprehensive codebase analysis."""
    analyzer = CodebaseAnalyzer()
    
    print("Starting comprehensive codebase analysis...")
    print("="*60)
    
    results = analyzer.analyze_complete_codebase()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"comprehensive_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Files analyzed: {results['summary']['files_analyzed']}")
    print(f"Total lines: {results['summary']['total_lines']:,}")
    print(f"Redundancy candidates: {results['summary']['redundancy_candidates']}")
    print(f"Architecture opportunities: {results['summary']['architecture_opportunities']}")
    
    print(f"\nREDUNDANCY ANALYSIS")
    print(f"High priority: {len(results['redundancy_analysis']['high_priority'])}")
    print(f"Medium priority: {len(results['redundancy_analysis']['medium_priority'])}")
    
    print(f"\nARCHITECTURE OPPORTUNITIES")
    for purpose, data in results['architecture_analysis'].items():
        if data['consolidation_opportunity']:
            print(f"  {purpose}: {data['file_count']} files scattered across {data['scattered_across']} directories")
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    main()