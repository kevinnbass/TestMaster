"""
Archive Intelligence Mining Framework
Phase 2 of TestMaster intelligence extraction - mining test patterns from archived systems.
"""

import os
import re
import ast
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from datetime import datetime
import time


@dataclass
class ArchivedTestPattern:
    """Pattern extracted from archived test files"""
    pattern_type: str  # 'self_healing', 'verification', 'generation', 'monitoring'
    source_file: str
    pattern_name: str
    code_signature: str
    complexity_score: float
    usage_frequency: int = 0
    dependencies: List[str] = field(default_factory=list)
    effectiveness_metrics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def priority_score(self) -> float:
        """Calculate pattern priority based on complexity and usage"""
        return (self.complexity_score * 0.6) + (self.usage_frequency * 0.4)


@dataclass
class TestIntelligenceReport:
    """Comprehensive report of extracted test intelligence"""
    total_files_analyzed: int = 0
    total_patterns_extracted: int = 0
    pattern_categories: Dict[str, int] = field(default_factory=dict)
    high_value_patterns: List[ArchivedTestPattern] = field(default_factory=list)
    redundancy_analysis: Dict[str, List[str]] = field(default_factory=dict)
    evolution_timeline: List[Dict[str, Any]] = field(default_factory=list)
    implementation_recommendations: List[str] = field(default_factory=list)


class ArchiveCodeAnalyzer:
    """Analyze archived code for intelligent patterns"""
    
    def __init__(self):
        self.function_patterns = {}
        self.class_patterns = {}
        self.import_dependencies = defaultdict(set)
        self.complexity_metrics = {}
        self.api_usage_patterns = defaultdict(int)
    
    def analyze_file_ast(self, file_path: str, content: str) -> Dict[str, Any]:
        """Deep AST analysis of Python file"""
        try:
            tree = ast.parse(content)
            
            analysis = {
                'functions': [],
                'classes': [],
                'imports': [],
                'complexity_score': 0,
                'api_calls': [],
                'decorators': [],
                'error_handling': []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                        'docstring': ast.get_docstring(node),
                        'line_count': node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0,
                        'async': isinstance(node, ast.AsyncFunctionDef)
                    }
                    analysis['functions'].append(func_info)
                    analysis['complexity_score'] += len(node.body) * 0.1
                
                elif isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'bases': [base.id if isinstance(base, ast.Name) else str(base) for base in node.bases],
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                        'docstring': ast.get_docstring(node)
                    }
                    analysis['classes'].append(class_info)
                    analysis['complexity_score'] += len(node.body) * 0.2
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis['imports'].append({
                                'type': 'import',
                                'module': alias.name,
                                'alias': alias.asname
                            })
                    else:  # ImportFrom
                        module = node.module or ''
                        for alias in node.names:
                            analysis['imports'].append({
                                'type': 'from_import',
                                'module': module,
                                'name': alias.name,
                                'alias': alias.asname
                            })
                
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        analysis['api_calls'].append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        analysis['api_calls'].append(f"{node.func.attr}")
                
                elif isinstance(node, ast.Try):
                    analysis['error_handling'].append({
                        'handlers': len(node.handlers),
                        'has_finally': bool(node.finalbody),
                        'has_else': bool(node.orelse)
                    })
            
            return analysis
            
        except SyntaxError as e:
            return {'error': f"Syntax error: {e}", 'complexity_score': 0}
        except Exception as e:
            return {'error': f"Analysis error: {e}", 'complexity_score': 0}
    
    def extract_test_patterns(self, analysis: Dict[str, Any], file_path: str) -> List[ArchivedTestPattern]:
        """Extract specific test patterns from analyzed code"""
        patterns = []
        
        # Pattern 1: Self-healing patterns
        for func in analysis.get('functions', []):
            if any(keyword in func['name'].lower() for keyword in ['heal', 'fix', 'repair', 'recover']):
                pattern = ArchivedTestPattern(
                    pattern_type='self_healing',
                    source_file=file_path,
                    pattern_name=func['name'],
                    code_signature=f"def {func['name']}({', '.join(func['args'])})",
                    complexity_score=func['line_count'] * 0.1,
                    dependencies=analysis.get('imports', [])[:5]
                )
                patterns.append(pattern)
        
        # Pattern 2: Verification patterns  
        for func in analysis.get('functions', []):
            if any(keyword in func['name'].lower() for keyword in ['verify', 'validate', 'check', 'test']):
                pattern = ArchivedTestPattern(
                    pattern_type='verification',
                    source_file=file_path,
                    pattern_name=func['name'],
                    code_signature=f"def {func['name']}({', '.join(func['args'])})",
                    complexity_score=func['line_count'] * 0.1 + len(func['decorators']) * 0.2,
                    dependencies=[imp['module'] for imp in analysis.get('imports', [])][:3]
                )
                patterns.append(pattern)
        
        # Pattern 3: Generation patterns
        for func in analysis.get('functions', []):
            if any(keyword in func['name'].lower() for keyword in ['generate', 'create', 'build', 'make']):
                pattern = ArchivedTestPattern(
                    pattern_type='generation',
                    source_file=file_path,
                    pattern_name=func['name'],
                    code_signature=f"def {func['name']}({', '.join(func['args'])})",
                    complexity_score=func['line_count'] * 0.15 + len(analysis.get('api_calls', [])) * 0.05,
                    dependencies=[imp['module'] for imp in analysis.get('imports', [])][:4]
                )
                patterns.append(pattern)
        
        # Pattern 4: Monitoring patterns
        for func in analysis.get('functions', []):
            if any(keyword in func['name'].lower() for keyword in ['monitor', 'watch', 'track', 'observe']):
                pattern = ArchivedTestPattern(
                    pattern_type='monitoring',
                    source_file=file_path,
                    pattern_name=func['name'],
                    code_signature=f"def {func['name']}({', '.join(func['args'])})",
                    complexity_score=func['line_count'] * 0.12 + len(func['decorators']) * 0.3,
                    dependencies=[imp['module'] for imp in analysis.get('imports', [])][:3]
                )
                patterns.append(pattern)
        
        return patterns


class ArchiveIntelligenceMiner:
    """Main class for mining intelligence from archived test systems"""
    
    def __init__(self, archive_root: str):
        self.archive_root = Path(archive_root)
        self.analyzer = ArchiveCodeAnalyzer()
        self.patterns_database = []
        self.file_analysis_cache = {}
        self.intelligence_metrics = {
            'files_processed': 0,
            'patterns_found': 0,
            'processing_time': 0,
            'error_count': 0
        }
    
    def discover_archived_files(self, file_extensions: List[str] = None) -> List[Path]:
        """Discover all relevant files in archive directories"""
        if file_extensions is None:
            file_extensions = ['.py']
        
        archived_files = []
        
        for ext in file_extensions:
            pattern = f"**/*{ext}"
            files = list(self.archive_root.glob(pattern))
            archived_files.extend(files)
        
        # Filter out non-test related files
        test_related_files = []
        for file_path in archived_files:
            file_name = file_path.name.lower()
            if any(keyword in file_name for keyword in [
                'test', 'verif', 'heal', 'generat', 'monitor', 'cover',
                'build', 'convert', 'fix', 'analy', 'intellig'
            ]):
                test_related_files.append(file_path)
        
        return sorted(test_related_files)
    
    def analyze_archived_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze single archived file for intelligence patterns"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic file metrics
            file_info = {
                'path': str(file_path),
                'size_bytes': len(content.encode('utf-8')),
                'line_count': content.count('\n') + 1,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            # AST analysis
            ast_analysis = self.analyzer.analyze_file_ast(str(file_path), content)
            file_info['ast_analysis'] = ast_analysis
            
            # Extract patterns
            patterns = self.analyzer.extract_test_patterns(ast_analysis, str(file_path))
            file_info['extracted_patterns'] = patterns
            
            # Pattern frequency analysis
            pattern_types = Counter([p.pattern_type for p in patterns])
            file_info['pattern_distribution'] = dict(pattern_types)
            
            self.intelligence_metrics['files_processed'] += 1
            self.intelligence_metrics['patterns_found'] += len(patterns)
            
            return file_info
            
        except Exception as e:
            self.intelligence_metrics['error_count'] += 1
            return {
                'path': str(file_path),
                'error': str(e),
                'analysis_timestamp': datetime.now().isoformat()
            }
    
    def mine_archive_intelligence(self, max_files: Optional[int] = None) -> TestIntelligenceReport:
        """Comprehensive intelligence mining from archive"""
        start_time = time.time()
        
        # Discover files
        archived_files = self.discover_archived_files()
        if max_files:
            archived_files = archived_files[:max_files]
        
        print(f"Mining intelligence from {len(archived_files)} archived files...")
        
        # Analyze each file
        all_patterns = []
        pattern_categories = defaultdict(int)
        redundancy_tracker = defaultdict(list)
        evolution_timeline = []
        
        for i, file_path in enumerate(archived_files):
            print(f"Analyzing [{i+1}/{len(archived_files)}]: {file_path.name}")
            
            file_analysis = self.analyze_archived_file(file_path)
            self.file_analysis_cache[str(file_path)] = file_analysis
            
            # Collect patterns
            patterns = file_analysis.get('extracted_patterns', [])
            all_patterns.extend(patterns)
            
            # Track categories
            for pattern in patterns:
                pattern_categories[pattern.pattern_type] += 1
            
            # Track redundancy
            for pattern in patterns:
                signature = pattern.code_signature
                redundancy_tracker[signature].append(pattern.source_file)
            
            # Evolution timeline
            if 'last_modified' in file_analysis:
                evolution_timeline.append({
                    'file': str(file_path),
                    'timestamp': file_analysis['last_modified'],
                    'pattern_count': len(patterns),
                    'complexity': file_analysis.get('ast_analysis', {}).get('complexity_score', 0)
                })
        
        # Identify high-value patterns
        high_value_patterns = []
        for pattern in all_patterns:
            pattern.usage_frequency = len(redundancy_tracker.get(pattern.code_signature, []))
            if pattern.priority_score > 1.0:  # High priority threshold
                high_value_patterns.append(pattern)
        
        # Sort by priority
        high_value_patterns.sort(key=lambda x: x.priority_score, reverse=True)
        
        # Create redundancy analysis
        redundancy_analysis = {}
        for signature, files in redundancy_tracker.items():
            if len(files) > 1:  # Found redundant patterns
                redundancy_analysis[signature] = files
        
        # Generate implementation recommendations
        recommendations = self._generate_recommendations(
            all_patterns, pattern_categories, redundancy_analysis
        )
        
        # Create comprehensive report
        report = TestIntelligenceReport(
            total_files_analyzed=len(archived_files),
            total_patterns_extracted=len(all_patterns),
            pattern_categories=dict(pattern_categories),
            high_value_patterns=high_value_patterns[:20],  # Top 20
            redundancy_analysis=redundancy_analysis,
            evolution_timeline=sorted(evolution_timeline, key=lambda x: x['timestamp']),
            implementation_recommendations=recommendations
        )
        
        self.intelligence_metrics['processing_time'] = time.time() - start_time
        
        return report
    
    def _generate_recommendations(self, patterns: List[ArchivedTestPattern], 
                                categories: Dict[str, int], 
                                redundancies: Dict[str, List[str]]) -> List[str]:
        """Generate implementation recommendations based on analysis"""
        recommendations = []
        
        # Pattern category recommendations
        if categories.get('self_healing', 0) > 5:
            recommendations.append(
                "High concentration of self-healing patterns detected. "
                "Consider implementing unified self-healing framework."
            )
        
        if categories.get('verification', 0) > 10:
            recommendations.append(
                "Multiple verification patterns found. "
                "Recommend creating centralized verification engine."
            )
        
        if categories.get('generation', 0) > 8:
            recommendations.append(
                "Significant test generation capability detected. "
                "Consider consolidating into intelligent test generator."
            )
        
        # Redundancy recommendations
        if len(redundancies) > 10:
            recommendations.append(
                f"Found {len(redundancies)} redundant patterns. "
                "High priority for deduplication and consolidation."
            )
        
        # High-value pattern recommendations
        high_value_count = sum(1 for p in patterns if p.priority_score > 1.5)
        if high_value_count > 5:
            recommendations.append(
                f"Identified {high_value_count} high-value patterns. "
                "Prioritize integration into new testing framework."
            )
        
        return recommendations
    
    def export_intelligence_report(self, report: TestIntelligenceReport, 
                                 output_path: str) -> None:
        """Export intelligence report to file"""
        report_data = {
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'archive_root': str(self.archive_root),
                'processing_metrics': self.intelligence_metrics
            },
            'summary': {
                'files_analyzed': report.total_files_analyzed,
                'patterns_extracted': report.total_patterns_extracted,
                'pattern_categories': report.pattern_categories,
                'redundant_patterns': len(report.redundancy_analysis)
            },
            'high_value_patterns': [
                {
                    'type': p.pattern_type,
                    'name': p.pattern_name,
                    'source': p.source_file,
                    'signature': p.code_signature,
                    'priority_score': p.priority_score,
                    'usage_frequency': p.usage_frequency
                }
                for p in report.high_value_patterns
            ],
            'redundancy_analysis': report.redundancy_analysis,
            'evolution_timeline': report.evolution_timeline,
            'recommendations': report.implementation_recommendations
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"Intelligence report exported to: {output_path}")


class ArchiveTestingFramework:
    """Testing framework for archive intelligence mining"""
    
    def __init__(self):
        self.test_results = []
    
    def test_archive_discovery(self, archive_path: str) -> bool:
        """Test archive file discovery"""
        try:
            miner = ArchiveIntelligenceMiner(archive_path)
            files = miner.discover_archived_files()
            
            assert len(files) > 0, "No archived files found"
            assert all(f.exists() for f in files), "Some files don't exist"
            assert all(f.suffix == '.py' for f in files), "Non-Python files found"
            
            return True
        except Exception as e:
            print(f"Archive discovery test failed: {e}")
            return False
    
    def test_file_analysis(self, archive_path: str) -> bool:
        """Test individual file analysis"""
        try:
            miner = ArchiveIntelligenceMiner(archive_path)
            files = miner.discover_archived_files()
            
            if not files:
                return True  # No files to analyze
            
            # Test first file
            analysis = miner.analyze_archived_file(files[0])
            
            assert 'path' in analysis, "Missing path in analysis"
            assert 'analysis_timestamp' in analysis, "Missing timestamp"
            
            if 'ast_analysis' in analysis:
                ast_data = analysis['ast_analysis']
                assert isinstance(ast_data.get('functions', []), list)
                assert isinstance(ast_data.get('classes', []), list)
            
            return True
        except Exception as e:
            print(f"File analysis test failed: {e}")
            return False
    
    def test_pattern_extraction(self, archive_path: str) -> bool:
        """Test pattern extraction functionality"""
        try:
            miner = ArchiveIntelligenceMiner(archive_path)
            files = miner.discover_archived_files()
            
            if not files:
                return True
            
            analysis = miner.analyze_archived_file(files[0])
            patterns = analysis.get('extracted_patterns', [])
            
            # Validate pattern structure
            for pattern in patterns[:3]:  # Test first 3 patterns
                assert hasattr(pattern, 'pattern_type')
                assert hasattr(pattern, 'source_file')
                assert hasattr(pattern, 'pattern_name')
                assert hasattr(pattern, 'complexity_score')
                assert pattern.pattern_type in ['self_healing', 'verification', 'generation', 'monitoring']
            
            return True
        except Exception as e:
            print(f"Pattern extraction test failed: {e}")
            return False
    
    def test_intelligence_mining(self, archive_path: str) -> bool:
        """Test complete intelligence mining process"""
        try:
            miner = ArchiveIntelligenceMiner(archive_path)
            report = miner.mine_archive_intelligence(max_files=5)  # Limit for testing
            
            assert report.total_files_analyzed >= 0
            assert report.total_patterns_extracted >= 0
            assert isinstance(report.pattern_categories, dict)
            assert isinstance(report.high_value_patterns, list)
            assert isinstance(report.implementation_recommendations, list)
            
            return True
        except Exception as e:
            print(f"Intelligence mining test failed: {e}")
            return False
    
    def run_comprehensive_tests(self, archive_path: str) -> Dict[str, bool]:
        """Run all archive testing framework tests"""
        tests = [
            'test_archive_discovery',
            'test_file_analysis', 
            'test_pattern_extraction',
            'test_intelligence_mining'
        ]
        
        results = {}
        for test_name in tests:
            try:
                result = getattr(self, test_name)(archive_path)
                results[test_name] = result
                print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                results[test_name] = False
                print(f"❌ {test_name}: FAILED - {e}")
        
        return results


# Main execution for testing
if __name__ == "__main__":
    import sys
    
    # Default archive path
    default_archive = r"C:\Users\kbass\OneDrive\Documents\testmaster\TestMaster\archive\original_backup"
    
    if len(sys.argv) > 1:
        archive_path = sys.argv[1]
    else:
        archive_path = default_archive
    
    print("🔍 Archive Intelligence Mining Framework")
    print(f"Archive path: {archive_path}")
    
    # Run comprehensive tests
    framework = ArchiveTestingFramework()
    results = framework.run_comprehensive_tests(archive_path)
    
    # Summary
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All archive intelligence tests passed!")
        
        # Run actual intelligence mining
        print("\n🚀 Running intelligence mining...")
        miner = ArchiveIntelligenceMiner(archive_path)
        report = miner.mine_archive_intelligence(max_files=20)
        
        # Export report
        output_path = f"archive_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        miner.export_intelligence_report(report, output_path)
        
        print(f"\n📈 Intelligence Mining Complete:")
        print(f"  Files analyzed: {report.total_files_analyzed}")
        print(f"  Patterns extracted: {report.total_patterns_extracted}")
        print(f"  High-value patterns: {len(report.high_value_patterns)}")
        print(f"  Redundant patterns: {len(report.redundancy_analysis)}")
    else:
        print("❌ Some tests failed. Check the output above.")