#!/usr/bin/env python3
"""
Core Library Analysis Tool - Agent C Hours 35-37
Analyzes core libraries, frameworks, and third-party dependencies used across the codebase.
Identifies consolidation opportunities and version conflicts.
"""

import ast
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys

class CoreLibraryAnalyzer(ast.NodeVisitor):
    """Analyzes core libraries and framework usage across the codebase."""
    
    def __init__(self):
        self.stdlib_modules = self._get_stdlib_modules()
        self.third_party_imports = defaultdict(set)
        self.framework_usage = defaultdict(int)
        self.version_patterns = defaultdict(set)
        self.import_locations = defaultdict(list)
        self.dependency_conflicts = []
        self.consolidation_opportunities = []
        self.framework_patterns = self._get_framework_patterns()
        self.current_file = None
        self.line_number = 0
        
    def _get_stdlib_modules(self) -> Set[str]:
        """Return set of Python standard library modules."""
        return {
            'os', 'sys', 'json', 'ast', 'pathlib', 'collections', 'typing',
            'datetime', 'time', 'math', 'random', 'itertools', 'functools',
            'operator', 'copy', 'pickle', 'sqlite3', 'csv', 'xml', 'html',
            'urllib', 're', 'string', 'io', 'base64', 'hashlib', 'hmac',
            'logging', 'unittest', 'subprocess', 'threading', 'multiprocessing',
            'concurrent', 'asyncio', 'socket', 'http', 'email', 'zipfile',
            'tarfile', 'gzip', 'shutil', 'tempfile', 'glob', 'fnmatch',
            'argparse', 'configparser', 'contextlib', 'weakref', 'gc',
            'inspect', 'dis', 'traceback', 'warnings', 'platform', 'site'
        }
    
    def _get_framework_patterns(self) -> Dict[str, List[str]]:
        """Define patterns for detecting framework usage."""
        return {
            'web_frameworks': [
                'flask', 'django', 'fastapi', 'tornado', 'bottle', 'pyramid',
                'starlette', 'sanic', 'quart', 'aiohttp', 'cherrypy'
            ],
            'testing_frameworks': [
                'pytest', 'unittest', 'nose', 'doctest', 'hypothesis',
                'mock', 'faker', 'factory_boy', 'responses'
            ],
            'data_frameworks': [
                'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn',
                'plotly', 'bokeh', 'scikit-learn', 'tensorflow', 'pytorch'
            ],
            'database_frameworks': [
                'sqlalchemy', 'django.db', 'peewee', 'pymongo', 'redis',
                'psycopg2', 'mysql', 'sqlite3', 'alembic', 'mongoengine'
            ],
            'async_frameworks': [
                'asyncio', 'aiohttp', 'asyncpg', 'aiomysql', 'aioredis',
                'celery', 'rq', 'kombu', 'pika', 'kafka'
            ],
            'utility_frameworks': [
                'requests', 'beautifulsoup4', 'lxml', 'pyyaml', 'toml',
                'click', 'typer', 'rich', 'colorama', 'tqdm', 'joblib'
            ]
        }
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single Python file for library usage."""
        self.current_file = str(file_path)
        self.line_number = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            self.visit(tree)
            
            # Also analyze requirements/setup patterns
            self._analyze_dependency_declarations(content)
            
            return {
                'file': str(file_path),
                'imports_analyzed': True,
                'dependency_declarations': self._find_dependency_declarations(content)
            }
            
        except Exception as e:
            return {
                'file': str(file_path),
                'error': str(e),
                'imports_analyzed': False
            }
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import statements."""
        self.line_number = node.lineno
        for alias in node.names:
            self._process_import(alias.name, alias.asname, 'import')
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit from...import statements."""
        self.line_number = node.lineno
        module = node.module or ''
        for alias in node.names:
            full_name = f"{module}.{alias.name}" if module else alias.name
            self._process_import(full_name, alias.asname, 'from_import', module)
        self.generic_visit(node)
    
    def _process_import(self, module_name: str, alias: Optional[str], import_type: str, parent_module: str = None):
        """Process and categorize an import."""
        base_module = module_name.split('.')[0]
        
        # Record import location
        self.import_locations[module_name].append({
            'file': self.current_file,
            'line': self.line_number,
            'type': import_type,
            'alias': alias,
            'parent_module': parent_module
        })
        
        # Categorize import
        if base_module not in self.stdlib_modules:
            self.third_party_imports[base_module].add(module_name)
            
            # Check framework patterns
            for category, frameworks in self.framework_patterns.items():
                for framework in frameworks:
                    if framework in module_name.lower():
                        self.framework_usage[f"{category}::{framework}"] += 1
        
        # Extract version patterns from comments or nearby code
        self._extract_version_info(module_name)
    
    def _extract_version_info(self, module_name: str):
        """Extract version information from the context."""
        # This would be enhanced to parse requirements.txt, setup.py, etc.
        # For now, we'll track module usage patterns
        pass
    
    def _analyze_dependency_declarations(self, content: str):
        """Analyze dependency declarations in setup.py, requirements.txt patterns."""
        # Look for setup() calls
        setup_pattern = r'setup\s*\([^)]*install_requires\s*=\s*\[(.*?)\]'
        matches = re.findall(setup_pattern, content, re.DOTALL)
        for match in matches:
            self._parse_requirements_list(match)
        
        # Look for requirements list patterns
        req_pattern = r'requirements\s*=\s*\[(.*?)\]'
        matches = re.findall(req_pattern, content, re.DOTALL)
        for match in matches:
            self._parse_requirements_list(match)
    
    def _parse_requirements_list(self, requirements_text: str):
        """Parse a requirements list and extract dependencies."""
        lines = requirements_text.split('\n')
        for line in lines:
            line = line.strip().strip('"\'",')
            if line and not line.startswith('#'):
                # Extract package name and version
                if '>=' in line or '==' in line or '>' in line or '<' in line:
                    pkg_name = re.split(r'[><=!]', line)[0].strip()
                    version_spec = line[len(pkg_name):].strip()
                    self.version_patterns[pkg_name].add(version_spec)
                else:
                    self.version_patterns[line].add('*')
    
    def _find_dependency_declarations(self, content: str) -> List[str]:
        """Find all dependency declaration patterns."""
        declarations = []
        
        # setup.py patterns
        if 'install_requires' in content:
            declarations.append('setup.py_install_requires')
        if 'requirements_dev' in content:
            declarations.append('dev_requirements')
        if 'extras_require' in content:
            declarations.append('extras_require')
        
        # Poetry patterns
        if '[tool.poetry.dependencies]' in content:
            declarations.append('pyproject.toml_poetry')
        
        # Pipenv patterns
        if 'Pipfile' in content:
            declarations.append('Pipfile')
        
        return declarations
    
    def analyze_directory(self, root_path: Path) -> Dict[str, Any]:
        """Analyze all Python files in directory."""
        results = {
            'files_analyzed': [],
            'total_files': 0,
            'successful_analyses': 0,
            'errors': []
        }
        
        python_files = list(root_path.rglob('*.py'))
        results['total_files'] = len(python_files)
        
        print(f"Analyzing {len(python_files)} Python files for core library usage...")
        
        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(python_files)} files analyzed")
            
            analysis = self.analyze_file(file_path)
            results['files_analyzed'].append(analysis)
            
            if analysis.get('imports_analyzed', False):
                results['successful_analyses'] += 1
            else:
                results['errors'].append(analysis)
        
        print(f"Analysis complete: {results['successful_analyses']}/{results['total_files']} files analyzed successfully")
        return results
    
    def detect_consolidation_opportunities(self):
        """Detect opportunities for library consolidation."""
        # Find similar/overlapping libraries
        overlapping_libs = self._find_overlapping_libraries()
        
        # Find version conflicts
        version_conflicts = self._find_version_conflicts()
        
        # Find unused imports
        potential_unused = self._find_potential_unused_imports()
        
        self.consolidation_opportunities = {
            'overlapping_libraries': overlapping_libs,
            'version_conflicts': version_conflicts,
            'potential_unused': potential_unused,
            'framework_consolidation': self._suggest_framework_consolidation()
        }
    
    def _find_overlapping_libraries(self) -> List[Dict[str, Any]]:
        """Find libraries that provide similar functionality."""
        overlaps = []
        
        # Web framework overlaps
        web_frameworks = [fw for fw in self.framework_usage.keys() if 'web_frameworks' in fw]
        if len(web_frameworks) > 1:
            overlaps.append({
                'category': 'web_frameworks',
                'libraries': web_frameworks,
                'suggestion': 'Consider standardizing on a single web framework'
            })
        
        # Testing framework overlaps
        test_frameworks = [fw for fw in self.framework_usage.keys() if 'testing_frameworks' in fw]
        if len(test_frameworks) > 2:  # Allow pytest + unittest
            overlaps.append({
                'category': 'testing_frameworks',
                'libraries': test_frameworks,
                'suggestion': 'Consider consolidating testing frameworks'
            })
        
        return overlaps
    
    def _find_version_conflicts(self) -> List[Dict[str, Any]]:
        """Find potential version conflicts."""
        conflicts = []
        for package, versions in self.version_patterns.items():
            if len(versions) > 1:
                conflicts.append({
                    'package': package,
                    'versions': list(versions),
                    'locations': len(self.import_locations.get(package, []))
                })
        return conflicts
    
    def _find_potential_unused_imports(self) -> List[Dict[str, Any]]:
        """Find imports that might be unused (basic heuristic)."""
        # This is a simplified heuristic - would need more sophisticated analysis
        single_use_imports = []
        for module, locations in self.import_locations.items():
            if len(locations) == 1 and not module.split('.')[0] in self.stdlib_modules:
                single_use_imports.append({
                    'module': module,
                    'location': locations[0],
                    'reason': 'only_imported_once'
                })
        return single_use_imports[:20]  # Limit to top 20
    
    def _suggest_framework_consolidation(self) -> List[Dict[str, Any]]:
        """Suggest framework consolidation opportunities."""
        suggestions = []
        
        # Suggest async framework consolidation
        async_frameworks = [fw for fw in self.framework_usage.keys() if 'async_frameworks' in fw]
        if len(async_frameworks) > 2:
            suggestions.append({
                'category': 'async_frameworks',
                'current': async_frameworks,
                'suggestion': 'Consider consolidating around asyncio + one async library'
            })
        
        return suggestions
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive analysis summary."""
        # Detect consolidation opportunities
        self.detect_consolidation_opportunities()
        
        summary = {
            'library_usage_statistics': {
                'total_third_party_libraries': len(self.third_party_imports),
                'total_import_statements': sum(len(locs) for locs in self.import_locations.values()),
                'framework_usage_by_category': self._categorize_framework_usage(),
                'most_used_libraries': self._get_most_used_libraries(),
                'stdlib_vs_third_party_ratio': self._calculate_stdlib_ratio()
            },
            'dependency_analysis': {
                'version_conflicts': len(self.consolidation_opportunities.get('version_conflicts', [])),
                'overlapping_libraries': len(self.consolidation_opportunities.get('overlapping_libraries', [])),
                'potential_unused_imports': len(self.consolidation_opportunities.get('potential_unused', []))
            },
            'consolidation_opportunities': self.consolidation_opportunities,
            'framework_recommendations': self._generate_framework_recommendations(),
            'dependency_health_score': self._calculate_dependency_health_score()
        }
        
        return summary
    
    def _categorize_framework_usage(self) -> Dict[str, int]:
        """Categorize framework usage by type."""
        categories = defaultdict(int)
        for framework_key, count in self.framework_usage.items():
            category = framework_key.split('::')[0]
            categories[category] += count
        return dict(categories)
    
    def _get_most_used_libraries(self) -> List[Dict[str, Any]]:
        """Get the most frequently used libraries."""
        library_counts = defaultdict(int)
        for module, locations in self.import_locations.items():
            base_module = module.split('.')[0]
            if base_module not in self.stdlib_modules:
                library_counts[base_module] += len(locations)
        
        sorted_libraries = sorted(library_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'library': lib, 'usage_count': count} for lib, count in sorted_libraries[:15]]
    
    def _calculate_stdlib_ratio(self) -> float:
        """Calculate ratio of stdlib to third-party imports."""
        stdlib_count = 0
        third_party_count = 0
        
        for module, locations in self.import_locations.items():
            base_module = module.split('.')[0]
            count = len(locations)
            if base_module in self.stdlib_modules:
                stdlib_count += count
            else:
                third_party_count += count
        
        total = stdlib_count + third_party_count
        return round(stdlib_count / total if total > 0 else 0, 3)
    
    def _generate_framework_recommendations(self) -> List[str]:
        """Generate recommendations based on framework analysis."""
        recommendations = []
        
        framework_counts = self._categorize_framework_usage()
        
        if framework_counts.get('web_frameworks', 0) > 1:
            recommendations.append("Consider standardizing on a single web framework for consistency")
        
        if framework_counts.get('testing_frameworks', 0) > 2:
            recommendations.append("Consolidate testing frameworks to reduce complexity")
        
        if len(self.version_patterns) > 20:
            recommendations.append("High number of versioned dependencies - consider dependency audit")
        
        return recommendations
    
    def _calculate_dependency_health_score(self) -> float:
        """Calculate overall dependency health score (0-100)."""
        score = 100.0
        
        # Penalize version conflicts
        conflicts = len(self.consolidation_opportunities.get('version_conflicts', []))
        score -= min(conflicts * 5, 20)
        
        # Penalize overlapping libraries
        overlaps = len(self.consolidation_opportunities.get('overlapping_libraries', []))
        score -= min(overlaps * 10, 30)
        
        # Penalize too many third-party dependencies
        if len(self.third_party_imports) > 50:
            score -= min((len(self.third_party_imports) - 50) * 0.5, 25)
        
        return max(score, 0)


def main():
    parser = argparse.ArgumentParser(description='Core Library Analysis Tool')
    parser.add_argument('--root', type=str, required=True, help='Root directory to analyze')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    print("=== Agent C Hours 35-37: Core Library Analysis ===")
    print(f"Analyzing directory: {args.root}")
    
    analyzer = CoreLibraryAnalyzer()
    root_path = Path(args.root)
    
    # Analyze directory
    analysis_results = analyzer.analyze_directory(root_path)
    
    # Generate summary
    summary = analyzer.generate_summary()
    
    # Combine results
    final_results = {
        'analysis_metadata': {
            'tool': 'core_library_analyzer',
            'version': '1.0',
            'agent': 'Agent_C',
            'hours': '35-37',
            'phase': 'Utility_Component_Extraction'
        },
        'analysis_results': analysis_results,
        'summary': summary,
        'raw_data': {
            'third_party_imports': {k: list(v) for k, v in analyzer.third_party_imports.items()},
            'framework_usage': dict(analyzer.framework_usage),
            'version_patterns': {k: list(v) for k, v in analyzer.version_patterns.items()},
            'import_locations': dict(analyzer.import_locations)
        }
    }
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== CORE LIBRARY ANALYSIS COMPLETE ===")
    print(f"Files analyzed: {analysis_results['successful_analyses']}/{analysis_results['total_files']}")
    print(f"Third-party libraries found: {len(analyzer.third_party_imports)}")
    print(f"Framework categories detected: {len(summary['library_usage_statistics']['framework_usage_by_category'])}")
    print(f"Version conflicts: {summary['dependency_analysis']['version_conflicts']}")
    print(f"Consolidation opportunities: {summary['dependency_analysis']['overlapping_libraries']}")
    print(f"Dependency health score: {summary['dependency_health_score']:.1f}/100")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()