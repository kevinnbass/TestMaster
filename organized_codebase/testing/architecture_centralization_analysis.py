#!/usr/bin/env python3
"""
Architecture Centralization Analysis
====================================

Analyzes the current TestMaster architecture for centralization opportunities,
particularly around testing, analytics, and intelligence systems that are
scattered across different parts of the codebase.
"""

import os
import ast
import json
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, field

@dataclass
class ComponentAnalysis:
    """Analysis of a system component."""
    name: str
    file_path: str
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    functionality_type: str = ""
    complexity_score: int = 0
    line_count: int = 0

@dataclass
class CentralizationOpportunity:
    """Represents a centralization opportunity."""
    category: str
    description: str
    components: List[str]
    current_locations: List[str]
    proposed_location: str
    benefits: List[str]
    complexity: str  # "low", "medium", "high"
    priority: int  # 1-10, 10 being highest

def analyze_codebase_architecture() -> Dict[str, Any]:
    """Analyze the current codebase architecture."""
    
    print("=" * 80)
    print("ARCHITECTURE CENTRALIZATION ANALYSIS")
    print("=" * 80)
    
    os.chdir('C:/Users/kbass/OneDrive/Documents/testmaster/TestMaster')
    
    # Component categories to analyze
    categories = {
        'testing': ['test', 'coverage', 'framework', 'execution', 'validation'],
        'analytics': ['analytics', 'analysis', 'intelligence', 'metrics', 'reporting'],
        'monitoring': ['monitor', 'observability', 'tracking', 'health', 'status'],
        'orchestration': ['orchestration', 'coordination', 'workflow', 'scheduling'],
        'state_management': ['state', 'data', 'storage', 'persistence', 'cache'],
        'backup_recovery': ['backup', 'recovery', 'disaster', 'restore', 'emergency'],
        'integration': ['integration', 'cross', 'unified', 'bridge', 'connector']
    }
    
    # Analyze all Python files
    components = []
    file_categorization = defaultdict(list)
    
    for py_file in Path('.').rglob("*.py"):
        if _should_analyze_file(py_file):
            analysis = _analyze_component(py_file)
            if analysis:
                components.append(analysis)
                
                # Categorize based on functionality
                for category, keywords in categories.items():
                    if any(keyword in analysis.functionality_type.lower() or 
                          keyword in str(py_file).lower() or
                          any(keyword in func.lower() for func in analysis.functions) or
                          any(keyword in cls.lower() for cls in analysis.classes)
                          for keyword in keywords):
                        file_categorization[category].append(analysis)
    
    print(f"Analyzed {len(components)} components across {len(file_categorization)} categories")
    
    # Identify centralization opportunities
    opportunities = _identify_centralization_opportunities(file_categorization, components)
    
    # Analyze current architecture strengths and weaknesses
    architecture_analysis = _analyze_current_architecture(file_categorization, components)
    
    # Generate recommendations
    recommendations = _generate_centralization_recommendations(opportunities, architecture_analysis)
    
    results = {
        'components_analyzed': len(components),
        'categories': {cat: len(comps) for cat, comps in file_categorization.items()},
        'centralization_opportunities': [opp.__dict__ for opp in opportunities],
        'architecture_analysis': architecture_analysis,
        'recommendations': recommendations,
        'detailed_components': [comp.__dict__ for comp in components[:50]]  # Top 50 for brevity
    }
    
    # Print summary
    _print_analysis_summary(results, opportunities)
    
    return results

def _should_analyze_file(file_path: Path) -> bool:
    """Determine if file should be analyzed."""
    file_str = str(file_path)
    
    # Skip certain directories and files
    skip_patterns = [
        '__pycache__', '.git', 'archive/legacy_scripts', 'archive/original_backup',
        'nul', '.pyc', 'backup_', 'test_final', 'test_frontend', 'test_ultra'
    ]
    
    return not any(skip in file_str for skip in skip_patterns)

def _analyze_component(file_path: Path) -> ComponentAnalysis:
    """Analyze a single component file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST
        tree = ast.parse(content)
        
        functions = []
        classes = []
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                else:
                    imports.append(node.module or "")
        
        # Determine functionality type
        functionality_type = _determine_functionality_type(file_path, functions, classes, content)
        
        # Calculate complexity
        complexity_score = len(functions) + len(classes) * 2
        line_count = len(content.split('\n'))
        
        return ComponentAnalysis(
            name=file_path.stem,
            file_path=str(file_path),
            functions=functions,
            classes=classes,
            imports=imports,
            functionality_type=functionality_type,
            complexity_score=complexity_score,
            line_count=line_count
        )
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def _determine_functionality_type(file_path: Path, functions: List[str], classes: List[str], content: str) -> str:
    """Determine the primary functionality type of a component."""
    
    # Keywords that indicate functionality type
    type_indicators = {
        'testing': ['test', 'coverage', 'assert', 'validate', 'verify', 'check'],
        'analytics': ['analyze', 'analysis', 'metric', 'report', 'intelligence', 'insight'],
        'monitoring': ['monitor', 'observe', 'track', 'health', 'status', 'alert'],
        'orchestration': ['orchestrate', 'coordinate', 'workflow', 'schedule', 'execute'],
        'state_management': ['state', 'store', 'cache', 'persist', 'manage'],
        'backup_recovery': ['backup', 'recover', 'restore', 'emergency', 'disaster'],
        'integration': ['integrate', 'connect', 'bridge', 'cross', 'unified'],
        'utility': ['util', 'helper', 'tool', 'common', 'shared']
    }
    
    # Score each type
    type_scores = defaultdict(int)
    
    # Check file path
    file_str = str(file_path).lower()
    for func_type, keywords in type_indicators.items():
        for keyword in keywords:
            if keyword in file_str:
                type_scores[func_type] += 3
    
    # Check function and class names
    all_names = ' '.join(functions + classes).lower()
    for func_type, keywords in type_indicators.items():
        for keyword in keywords:
            if keyword in all_names:
                type_scores[func_type] += 2
    
    # Check content (docstrings, comments)
    content_lower = content.lower()
    for func_type, keywords in type_indicators.items():
        for keyword in keywords:
            type_scores[func_type] += content_lower.count(keyword)
    
    # Return highest scoring type
    if type_scores:
        return max(type_scores.items(), key=lambda x: x[1])[0]
    else:
        return 'unknown'

def _identify_centralization_opportunities(file_categorization: Dict[str, List[ComponentAnalysis]], 
                                         all_components: List[ComponentAnalysis]) -> List[CentralizationOpportunity]:
    """Identify centralization opportunities."""
    
    opportunities = []
    
    # 1. Testing and Analytics Intelligence
    testing_files = file_categorization.get('testing', [])
    analytics_files = file_categorization.get('analytics', [])
    
    if len(testing_files) > 3 or len(analytics_files) > 3:
        scattered_locations = set()
        for comp in testing_files + analytics_files:
            scattered_locations.add(str(Path(comp.file_path).parent))
        
        if len(scattered_locations) > 2:
            opportunities.append(CentralizationOpportunity(
                category="Testing & Analytics Intelligence",
                description="Multiple testing and analytics components are scattered across different directories",
                components=[comp.name for comp in testing_files + analytics_files],
                current_locations=list(scattered_locations),
                proposed_location="core/intelligence/",
                benefits=[
                    "Unified testing and analytics interface",
                    "Reduced code duplication",
                    "Better maintainability",
                    "Enhanced feature integration"
                ],
                complexity="medium",
                priority=9
            ))
    
    # 2. Monitoring and Observability
    monitoring_files = file_categorization.get('monitoring', [])
    
    if len(monitoring_files) > 2:
        scattered_locations = set()
        for comp in monitoring_files:
            scattered_locations.add(str(Path(comp.file_path).parent))
        
        if len(scattered_locations) > 1:
            opportunities.append(CentralizationOpportunity(
                category="Monitoring & Observability",
                description="Monitoring components are distributed across multiple locations",
                components=[comp.name for comp in monitoring_files],
                current_locations=list(scattered_locations),
                proposed_location="core/observability/",
                benefits=[
                    "Centralized monitoring interface",
                    "Consistent metrics collection",
                    "Unified alerting system",
                    "Better performance tracking"
                ],
                complexity="low",
                priority=7
            ))
    
    # 3. State and Data Management
    state_files = file_categorization.get('state_management', [])
    
    if len(state_files) > 3:
        scattered_locations = set()
        for comp in state_files:
            scattered_locations.add(str(Path(comp.file_path).parent))
        
        if len(scattered_locations) > 2:
            opportunities.append(CentralizationOpportunity(
                category="State & Data Management",
                description="State management components are fragmented across the codebase",
                components=[comp.name for comp in state_files],
                current_locations=list(scattered_locations),
                proposed_location="core/data/",
                benefits=[
                    "Unified state management API",
                    "Consistent data persistence",
                    "Better transaction management",
                    "Reduced state conflicts"
                ],
                complexity="high",
                priority=6
            ))
    
    # 4. Integration and Cross-System Components
    integration_files = file_categorization.get('integration', [])
    
    if len(integration_files) > 5:
        opportunities.append(CentralizationOpportunity(
            category="Integration & Cross-System",
            description="Integration components could benefit from centralized management",
            components=[comp.name for comp in integration_files],
            current_locations=["integration/", "dashboard/", "core/"],
            proposed_location="core/integration/",
            benefits=[
                "Centralized integration patterns",
                "Consistent API interfaces",
                "Better dependency management",
                "Unified configuration"
            ],
            complexity="medium",
            priority=5
        ))
    
    # 5. Duplicate Functionality Detection
    function_names = defaultdict(list)
    for comp in all_components:
        for func in comp.functions:
            function_names[func].append(comp)
    
    # Find functions that appear in multiple components
    duplicate_functions = {name: comps for name, comps in function_names.items() 
                          if len(comps) > 1 and not name.startswith('_')}
    
    if len(duplicate_functions) > 10:
        opportunities.append(CentralizationOpportunity(
            category="Duplicate Functionality",
            description=f"Found {len(duplicate_functions)} function names appearing in multiple components",
            components=list(set(comp.name for comps in duplicate_functions.values() for comp in comps)),
            current_locations=["Various"],
            proposed_location="core/common/",
            benefits=[
                "Eliminate code duplication",
                "Single source of truth",
                "Easier maintenance",
                "Consistent behavior"
            ],
            complexity="high",
            priority=4
        ))
    
    return sorted(opportunities, key=lambda x: x.priority, reverse=True)

def _analyze_current_architecture(file_categorization: Dict[str, List[ComponentAnalysis]], 
                                all_components: List[ComponentAnalysis]) -> Dict[str, Any]:
    """Analyze the current architecture strengths and weaknesses."""
    
    # Calculate distribution metrics
    directory_distribution = defaultdict(int)
    for comp in all_components:
        parent_dir = str(Path(comp.file_path).parent)
        directory_distribution[parent_dir] += 1
    
    # Identify hotspots (directories with many components)
    hotspots = {dir_name: count for dir_name, count in directory_distribution.items() 
                if count > 5}
    
    # Calculate complexity distribution
    complexity_stats = {
        'avg_complexity': sum(comp.complexity_score for comp in all_components) / len(all_components),
        'high_complexity_files': [comp.name for comp in all_components if comp.complexity_score > 50],
        'total_lines': sum(comp.line_count for comp in all_components)
    }
    
    # Analyze dependencies and coupling
    import_patterns = defaultdict(int)
    for comp in all_components:
        for imp in comp.imports:
            if imp and not imp.startswith('.'):
                import_patterns[imp] += 1
    
    return {
        'directory_distribution': dict(directory_distribution),
        'hotspots': hotspots,
        'complexity_stats': complexity_stats,
        'top_imports': dict(Counter(import_patterns).most_common(10)),
        'category_distribution': {cat: len(comps) for cat, comps in file_categorization.items()},
        'architecture_patterns': _identify_architecture_patterns(file_categorization)
    }

def _identify_architecture_patterns(file_categorization: Dict[str, List[ComponentAnalysis]]) -> Dict[str, str]:
    """Identify current architecture patterns."""
    
    patterns = {}
    
    # Check for existing centralization
    core_components = sum(1 for cat, comps in file_categorization.items() 
                         for comp in comps if 'core' in comp.file_path)
    
    if core_components > 20:
        patterns['centralization_level'] = 'high'
    elif core_components > 10:
        patterns['centralization_level'] = 'medium'
    else:
        patterns['centralization_level'] = 'low'
    
    # Check for layered architecture
    has_core = any('core' in comp.file_path for comps in file_categorization.values() for comp in comps)
    has_integration = any('integration' in comp.file_path for comps in file_categorization.values() for comp in comps)
    has_dashboard = any('dashboard' in comp.file_path for comps in file_categorization.values() for comp in comps)
    
    if has_core and has_integration and has_dashboard:
        patterns['architecture_style'] = 'layered'
    else:
        patterns['architecture_style'] = 'mixed'
    
    return patterns

def _generate_centralization_recommendations(opportunities: List[CentralizationOpportunity],
                                           architecture_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate specific centralization recommendations."""
    
    recommendations = []
    
    # High priority recommendations
    high_priority_opportunities = [opp for opp in opportunities if opp.priority >= 8]
    
    if high_priority_opportunities:
        recommendations.append({
            'type': 'immediate_action',
            'title': 'High Priority Centralization',
            'description': f'Centralize {len(high_priority_opportunities)} high-priority systems',
            'opportunities': [opp.category for opp in high_priority_opportunities],
            'estimated_effort': 'Medium',
            'expected_benefits': 'High'
        })
    
    # Architecture enhancement recommendations
    current_centralization = architecture_analysis.get('architecture_patterns', {}).get('centralization_level', 'low')
    
    if current_centralization == 'low':
        recommendations.append({
            'type': 'architecture_enhancement',
            'title': 'Establish Central Intelligence Hub',
            'description': 'Create core/intelligence/ to centralize testing, analytics, and monitoring',
            'rationale': 'Currently scattered across multiple directories',
            'estimated_effort': 'High',
            'expected_benefits': 'Very High'
        })
    
    # Duplicate code elimination
    recommendations.append({
        'type': 'code_quality',
        'title': 'Eliminate Duplicate Functionality',
        'description': 'Consolidate common functions into shared modules',
        'rationale': 'Multiple components implement similar functionality',
        'estimated_effort': 'Medium',
        'expected_benefits': 'Medium'
    })
    
    # Interface standardization
    recommendations.append({
        'type': 'interface_design',
        'title': 'Standardize Component Interfaces',
        'description': 'Create consistent APIs across similar components',
        'rationale': 'Improve interoperability and reduce learning curve',
        'estimated_effort': 'Low',
        'expected_benefits': 'Medium'
    })
    
    return recommendations

def _print_analysis_summary(results: Dict[str, Any], opportunities: List[CentralizationOpportunity]):
    """Print analysis summary."""
    
    print(f"\n📊 ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Components analyzed: {results['components_analyzed']}")
    print(f"Categories identified: {len(results['categories'])}")
    print(f"Centralization opportunities: {len(opportunities)}")
    
    print(f"\n🎯 TOP CENTRALIZATION OPPORTUNITIES")
    print("=" * 50)
    
    for i, opp in enumerate(opportunities[:5], 1):
        print(f"{i}. {opp.category} (Priority: {opp.priority}/10)")
        print(f"   {opp.description}")
        print(f"   Complexity: {opp.complexity.title()}")
        print(f"   Proposed location: {opp.proposed_location}")
        print(f"   Components affected: {len(opp.components)}")
        print()
    
    print(f"📋 CATEGORY DISTRIBUTION")
    print("=" * 50)
    for category, count in sorted(results['categories'].items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"  {category}: {count} components")
    
    print(f"\n💡 KEY RECOMMENDATIONS")
    print("=" * 50)
    for i, rec in enumerate(results['recommendations'][:3], 1):
        print(f"{i}. {rec['title']}")
        print(f"   {rec['description']}")
        print(f"   Effort: {rec['estimated_effort']}, Benefits: {rec['expected_benefits']}")
        print()

def main():
    """Run the architecture centralization analysis."""
    
    results = analyze_codebase_architecture()
    
    # Save detailed results
    with open('architecture_centralization_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete! Detailed results saved to architecture_centralization_analysis.json")
    
    return results

if __name__ == '__main__':
    main()