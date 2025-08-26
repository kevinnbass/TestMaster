"""
Agent A - Redundancy Pattern Analyzer
Phase 2: Hours 16-20 - Detect Initial Redundancy Patterns
"""

import os
import ast
import hashlib
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import difflib

class RedundancyAnalyzer:
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.redundancy_groups = defaultdict(list)
        self.naming_patterns = defaultdict(list)
        self.structural_patterns = defaultdict(list)
        self.semantic_duplicates = []
        self.statistics = {
            'files_analyzed': 0,
            'exact_duplicates': 0,
            'naming_redundancies': 0,
            'structural_redundancies': 0,
            'semantic_similarities': 0,
            'total_redundancy_groups': 0
        }
        
    def analyze_directory(self, directory: str, max_files: int = 200) -> Dict:
        """Analyze Python files for redundancy patterns"""
        dir_path = self.root_path / directory
        python_files = list(dir_path.rglob("*.py"))[:max_files]
        
        print(f"Analyzing {len(python_files)} files for redundancy patterns...")
        
        # Phase 1: Collect file signatures
        file_signatures = {}
        for file_path in python_files:
            self.statistics['files_analyzed'] += 1
            sig = self._get_file_signature(file_path)
            if sig:
                file_signatures[file_path] = sig
        
        # Phase 2: Detect exact duplicates
        self._detect_exact_duplicates(file_signatures)
        
        # Phase 3: Detect naming pattern redundancies
        self._detect_naming_redundancies(python_files)
        
        # Phase 4: Detect structural redundancies
        self._detect_structural_redundancies(file_signatures)
        
        # Phase 5: Detect semantic similarities
        self._detect_semantic_similarities(python_files)
        
        return self.generate_report()
    
    def _get_file_signature(self, file_path: Path) -> Dict:
        """Generate signature for a file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Calculate content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Parse AST for structural analysis
            try:
                tree = ast.parse(content)
                functions = []
                classes = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        functions.append(node.name)
                    elif isinstance(node, ast.ClassDef):
                        classes.append(node.name)
                        
                return {
                    'path': str(file_path.relative_to(self.root_path)),
                    'hash': content_hash,
                    'size': len(content),
                    'lines': content.count('\n'),
                    'functions': functions,
                    'classes': classes,
                    'content': content[:1000]  # First 1000 chars for similarity
                }
            except:
                return None
        except:
            return None
    
    def _detect_exact_duplicates(self, signatures: Dict) -> None:
        """Detect files with identical content"""
        hash_groups = defaultdict(list)
        
        for file_path, sig in signatures.items():
            if sig:
                hash_groups[sig['hash']].append(sig['path'])
        
        for hash_val, files in hash_groups.items():
            if len(files) > 1:
                self.redundancy_groups['exact_duplicates'].append({
                    'type': 'exact',
                    'files': files,
                    'count': len(files)
                })
                self.statistics['exact_duplicates'] += len(files) - 1
    
    def _detect_naming_redundancies(self, files: List[Path]) -> None:
        """Detect files with similar naming patterns"""
        base_names = defaultdict(list)
        
        for file_path in files:
            # Extract base name patterns
            name = file_path.stem
            
            # Remove common suffixes
            for suffix in ['_v2', '_v3', '_old', '_new', '_backup', '_copy', '_temp', '_test']:
                if name.endswith(suffix):
                    base_name = name[:-len(suffix)]
                    base_names[base_name].append(str(file_path.relative_to(self.root_path)))
                    break
            else:
                # Check for numbered versions
                import re
                match = re.match(r'(.+?)_?\d+$', name)
                if match:
                    base_name = match.group(1)
                    base_names[base_name].append(str(file_path.relative_to(self.root_path)))
        
        for base_name, file_list in base_names.items():
            if len(file_list) > 1:
                self.naming_patterns[base_name] = file_list
                self.statistics['naming_redundancies'] += len(file_list) - 1
    
    def _detect_structural_redundancies(self, signatures: Dict) -> None:
        """Detect files with similar structure"""
        structure_groups = defaultdict(list)
        
        for file_path, sig in signatures.items():
            if sig and sig['functions']:
                # Create structural signature
                struct_sig = f"funcs:{','.join(sorted(sig['functions'][:5]))}_classes:{','.join(sorted(sig['classes'][:3]))}"
                structure_groups[struct_sig].append(sig['path'])
        
        for struct_sig, files in structure_groups.items():
            if len(files) > 1:
                self.structural_patterns[struct_sig] = files
                self.statistics['structural_redundancies'] += len(files) - 1
    
    def _detect_semantic_similarities(self, files: List[Path]) -> None:
        """Detect semantically similar files"""
        # Sample implementation - check files with similar names
        analyzed_pairs = set()
        
        for i, file1 in enumerate(files[:50]):  # Limit for performance
            for file2 in files[i+1:51]:
                pair = tuple(sorted([str(file1), str(file2)]))
                if pair not in analyzed_pairs:
                    analyzed_pairs.add(pair)
                    
                    similarity = self._calculate_similarity(file1, file2)
                    if similarity > 0.7:  # 70% similarity threshold
                        self.semantic_duplicates.append({
                            'file1': str(file1.relative_to(self.root_path)),
                            'file2': str(file2.relative_to(self.root_path)),
                            'similarity': similarity
                        })
                        self.statistics['semantic_similarities'] += 1
    
    def _calculate_similarity(self, file1: Path, file2: Path) -> float:
        """Calculate semantic similarity between two files"""
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()[:2000]
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()[:2000]
            
            # Simple similarity using difflib
            return difflib.SequenceMatcher(None, content1, content2).ratio()
        except:
            return 0.0
    
    def generate_report(self) -> Dict:
        """Generate comprehensive redundancy report"""
        self.statistics['total_redundancy_groups'] = (
            len(self.redundancy_groups['exact_duplicates']) +
            len(self.naming_patterns) +
            len(self.structural_patterns)
        )
        
        return {
            'statistics': self.statistics,
            'exact_duplicates': self.redundancy_groups['exact_duplicates'][:5],
            'naming_patterns': dict(list(self.naming_patterns.items())[:5]),
            'structural_patterns': dict(list(self.structural_patterns.items())[:3]),
            'semantic_similarities': self.semantic_duplicates[:5],
            'consolidation_opportunities': {
                'exact_duplicates_removal': self.statistics['exact_duplicates'],
                'naming_pattern_consolidation': self.statistics['naming_redundancies'],
                'structural_consolidation': self.statistics['structural_redundancies'],
                'total_files_reducible': sum([
                    self.statistics['exact_duplicates'],
                    self.statistics['naming_redundancies'],
                    self.statistics['structural_redundancies']
                ])
            }
        }

# Execute redundancy analysis
if __name__ == "__main__":
    analyzer = RedundancyAnalyzer()
    report = analyzer.analyze_directory("TestMaster", max_files=200)
    
    print("\n=== REDUNDANCY ANALYSIS REPORT ===")
    print(f"Files Analyzed: {report['statistics']['files_analyzed']}")
    print(f"Total Redundancy Groups: {report['statistics']['total_redundancy_groups']}")
    
    print("\n=== REDUNDANCY PATTERNS ===")
    print(f"Exact Duplicates: {report['statistics']['exact_duplicates']} files")
    print(f"Naming Redundancies: {report['statistics']['naming_redundancies']} files")
    print(f"Structural Redundancies: {report['statistics']['structural_redundancies']} files")
    print(f"Semantic Similarities: {report['statistics']['semantic_similarities']} pairs")
    
    print("\n=== CONSOLIDATION OPPORTUNITIES ===")
    print(f"Total Files Reducible: {report['consolidation_opportunities']['total_files_reducible']}")
    print(f"Potential Reduction: {report['consolidation_opportunities']['total_files_reducible'] / max(1, report['statistics']['files_analyzed']) * 100:.1f}%")
    
    if report['exact_duplicates']:
        print("\n=== EXACT DUPLICATES (First 5) ===")
        for group in report['exact_duplicates'][:5]:
            print(f"  Duplicate group ({group['count']} files):")
            for file in group['files'][:3]:
                print(f"    - {file}")
    
    if report['naming_patterns']:
        print("\n=== NAMING PATTERN REDUNDANCIES (First 5) ===")
        for base_name, files in list(report['naming_patterns'].items())[:5]:
            print(f"  Pattern '{base_name}' ({len(files)} variants):")
            for file in files[:3]:
                print(f"    - {file}")
    
    if report['semantic_similarities']:
        print("\n=== SEMANTIC SIMILARITIES (First 5) ===")
        for sim in report['semantic_similarities'][:5]:
            print(f"  {sim['similarity']*100:.1f}% similarity:")
            print(f"    - {sim['file1']}")
            print(f"    - {sim['file2']}")