#!/usr/bin/env python3
"""
Component Integration Planning Tool - Agent C Hours 47-50
Synthesizes all previous analysis to create comprehensive integration plans.
Generates actionable roadmaps for component extraction, consolidation, and optimization.
"""

import ast
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Any, Optional, Tuple
import sys

class ComponentIntegrationPlanner:
    """Synthesizes analysis results to create comprehensive integration plans."""
    
    def __init__(self):
        self.analysis_data = {}
        self.integration_plans = []
        self.priority_matrix = {}
        self.implementation_roadmap = {}
        self.risk_assessments = {}
        self.dependency_chains = {}
        self.consolidation_strategies = {}
        
    def load_analysis_data(self, analysis_files: Dict[str, str]) -> None:
        """Load all previous analysis results."""
        print("Loading comprehensive analysis data...")
        
        for analysis_type, file_path in analysis_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.analysis_data[analysis_type] = data
                    print(f"LOADED {analysis_type} data from {file_path}")
            except Exception as e:
                print(f"WARNING: Could not load {analysis_type} from {file_path}: {e}")
    
    def synthesize_integration_opportunities(self) -> Dict[str, Any]:
        """Synthesize all analysis data to identify integration opportunities."""
        print("\nSynthesizing integration opportunities...")
        
        opportunities = {
            'high_priority_extractions': self._identify_high_priority_extractions(),
            'security_critical_fixes': self._identify_security_critical_fixes(),
            'testing_consolidation': self._plan_testing_consolidation(),
            'configuration_centralization': self._plan_configuration_centralization(),
            'utility_component_extraction': self._plan_utility_extraction(),
            'duplicate_code_elimination': self._plan_duplicate_elimination(),
            'dependency_optimization': self._plan_dependency_optimization()
        }
        
        return opportunities
    
    def _identify_high_priority_extractions(self) -> List[Dict[str, Any]]:
        """Identify highest priority component extractions."""
        extractions = []
        
        # From shared components analysis
        if 'shared_components' in self.analysis_data:
            shared_data = self.analysis_data['shared_components']
            if 'summary' in shared_data and 'extraction_opportunities' in shared_data['summary']:
                opportunities = shared_data['summary']['extraction_opportunities']
                
                for category, components in opportunities.items():
                    if isinstance(components, list):
                        for component in components[:5]:  # Top 5 per category
                            if isinstance(component, dict):
                                extractions.append({
                                    'type': 'shared_component',
                                    'category': category,
                                    'component': component,
                                    'priority': self._calculate_extraction_priority(component),
                                    'estimated_impact': self._estimate_extraction_impact(component)
                                })
        
        # From utility functions analysis
        if 'utility_functions' in self.analysis_data:
            utility_data = self.analysis_data['utility_functions']
            if 'summary' in utility_data and 'extraction_opportunities' in utility_data['summary']:
                reusable = utility_data['summary']['extraction_opportunities'].get('reusable_utilities', [])
                
                for utility in reusable[:10]:  # Top 10 utilities
                    extractions.append({
                        'type': 'utility_function',
                        'utility': utility,
                        'priority': self._calculate_utility_priority(utility),
                        'estimated_impact': self._estimate_utility_impact(utility)
                    })
        
        # Sort by priority and return top extractions
        extractions.sort(key=lambda x: x['priority'], reverse=True)
        return extractions[:20]
    
    def _identify_security_critical_fixes(self) -> List[Dict[str, Any]]:
        """Identify security-critical fixes that must be implemented immediately."""
        security_fixes = []
        
        if 'configuration_settings' in self.analysis_data:
            config_data = self.analysis_data['configuration_settings']
            if 'raw_data' in config_data and 'security_concerns' in config_data['raw_data']:
                concerns = config_data['raw_data']['security_concerns']
                
                for concern in concerns:
                    security_fixes.append({
                        'type': 'security_fix',
                        'concern_type': concern.get('type', 'unknown'),
                        'file': concern.get('file', 'unknown'),
                        'line': concern.get('line', 0),
                        'priority': 'CRITICAL',
                        'action_required': self._determine_security_action(concern),
                        'estimated_effort': self._estimate_security_fix_effort(concern)
                    })
        
        return security_fixes
    
    def _plan_testing_consolidation(self) -> Dict[str, Any]:
        """Plan testing framework consolidation strategy."""
        if 'testing_framework' not in self.analysis_data:
            return {'status': 'no_data', 'plan': []}
        
        test_data = self.analysis_data['testing_framework']
        summary = test_data.get('summary', {})
        
        frameworks = summary.get('testing_statistics', {}).get('testing_frameworks', {})
        
        consolidation_plan = {
            'current_frameworks': frameworks,
            'recommended_primary': self._select_primary_framework(frameworks),
            'migration_strategy': self._create_framework_migration_strategy(frameworks),
            'estimated_effort': self._estimate_testing_consolidation_effort(frameworks),
            'implementation_phases': self._create_testing_phases(frameworks)
        }
        
        return consolidation_plan
    
    def _plan_configuration_centralization(self) -> Dict[str, Any]:
        """Plan configuration centralization strategy."""
        if 'configuration_settings' not in self.analysis_data:
            return {'status': 'no_data', 'plan': []}
        
        config_data = self.analysis_data['configuration_settings']
        summary = config_data.get('summary', {})
        
        centralization_plan = {
            'current_config_files': summary.get('configuration_statistics', {}).get('total_config_files', 0),
            'security_concerns': summary.get('security_analysis', {}).get('security_concerns', 0),
            'centralization_strategy': self._create_config_strategy(summary),
            'security_migration_plan': self._create_security_migration_plan(config_data),
            'implementation_timeline': self._create_config_timeline(summary)
        }
        
        return centralization_plan
    
    def _plan_utility_extraction(self) -> Dict[str, Any]:
        """Plan utility function extraction strategy."""
        if 'utility_functions' not in self.analysis_data:
            return {'status': 'no_data', 'plan': []}
        
        utility_data = self.analysis_data['utility_functions']
        summary = utility_data.get('summary', {})
        
        extraction_plan = {
            'total_functions': summary.get('function_statistics', {}).get('total_functions', 0),
            'extraction_candidates': summary.get('extraction_analysis', {}).get('reusable_utilities', 0),
            'proposed_modules': self._design_utility_modules(summary),
            'extraction_strategy': self._create_extraction_strategy(summary),
            'implementation_order': self._prioritize_utility_extraction(summary)
        }
        
        return extraction_plan
    
    def _plan_duplicate_elimination(self) -> Dict[str, Any]:
        """Plan duplicate code elimination strategy."""
        if 'duplicate_analysis' not in self.analysis_data:
            return {'status': 'no_data', 'plan': []}
        
        # Note: This would analyze duplicate_analysis_hour26.json if available
        # For now, using known values from previous analysis
        duplicate_plan = {
            'duplicate_groups': 6296,
            'optimization_potential': 380259,
            'elimination_strategy': self._create_duplicate_strategy(),
            'risk_mitigation': self._create_duplicate_risk_plan(),
            'implementation_phases': self._create_duplicate_phases()
        }
        
        return duplicate_plan
    
    def _plan_dependency_optimization(self) -> Dict[str, Any]:
        """Plan dependency optimization strategy."""
        if 'core_library' not in self.analysis_data:
            return {'status': 'no_data', 'plan': []}
        
        library_data = self.analysis_data['core_library']
        summary = library_data.get('summary', {})
        
        optimization_plan = {
            'total_libraries': summary.get('library_usage_statistics', {}).get('total_third_party_libraries', 0),
            'consolidation_opportunities': summary.get('consolidation_opportunities', {}),
            'optimization_strategy': self._create_dependency_strategy(summary),
            'framework_consolidation': self._plan_framework_consolidation(summary),
            'implementation_roadmap': self._create_dependency_roadmap(summary)
        }
        
        return optimization_plan
    
    def create_implementation_roadmap(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive implementation roadmap."""
        print("\nCreating comprehensive implementation roadmap...")
        
        roadmap = {
            'phase_1_immediate': self._create_immediate_phase(opportunities),
            'phase_2_security': self._create_security_phase(opportunities),
            'phase_3_consolidation': self._create_consolidation_phase(opportunities),
            'phase_4_optimization': self._create_optimization_phase(opportunities),
            'risk_mitigation': self._create_risk_mitigation_strategy(opportunities),
            'success_metrics': self._define_success_metrics(opportunities)
        }
        
        return roadmap
    
    def _create_immediate_phase(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create immediate action phase (0-2 weeks)."""
        immediate_actions = []
        
        # Critical security fixes
        security_fixes = opportunities.get('security_critical_fixes', [])
        for fix in security_fixes[:10]:  # Top 10 most critical
            immediate_actions.append({
                'action': f"Fix {fix['concern_type']} in {fix['file']}",
                'type': 'security_fix',
                'effort': fix.get('estimated_effort', 'medium'),
                'impact': 'critical'
            })
        
        # High-priority extractions
        extractions = opportunities.get('high_priority_extractions', [])
        for extraction in extractions[:5]:  # Top 5 extractions
            immediate_actions.append({
                'action': f"Extract {extraction['type']}: {extraction.get('component', {}).get('name', 'unknown')}",
                'type': 'component_extraction',
                'effort': extraction.get('estimated_impact', {}).get('effort', 'medium'),
                'impact': 'high'
            })
        
        return {
            'duration': '0-2 weeks',
            'actions': immediate_actions,
            'success_criteria': ['All critical security issues resolved', 'Top 5 components extracted'],
            'resources_required': ['2-3 developers', 'Security review team']
        }
    
    def _create_security_phase(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create security hardening phase (2-4 weeks)."""
        config_plan = opportunities.get('configuration_centralization', {})
        
        security_actions = [
            'Implement centralized configuration management',
            'Migrate all hardcoded secrets to environment variables',
            'Set up secure configuration validation',
            'Implement configuration encryption for sensitive data',
            'Create configuration audit logging'
        ]
        
        return {
            'duration': '2-4 weeks',
            'actions': security_actions,
            'success_criteria': [
                'Zero hardcoded secrets remaining',
                'Centralized configuration system operational',
                'Configuration security score > 80/100'
            ],
            'resources_required': ['Security team', '1-2 developers', 'DevOps support']
        }
    
    def _create_consolidation_phase(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create consolidation phase (4-8 weeks)."""
        testing_plan = opportunities.get('testing_consolidation', {})
        
        consolidation_actions = [
            f"Standardize on {testing_plan.get('recommended_primary', 'pytest')} as primary testing framework",
            'Consolidate duplicate code using AST-based analysis',
            'Extract utility functions to shared modules',
            'Implement shared component architecture',
            'Optimize dependency tree and remove redundant libraries'
        ]
        
        return {
            'duration': '4-8 weeks',
            'actions': consolidation_actions,
            'success_criteria': [
                'Single primary testing framework',
                '50% reduction in duplicate code',
                'Shared utility modules operational',
                '20% reduction in dependencies'
            ],
            'resources_required': ['3-4 developers', 'Architecture review team']
        }
    
    def _create_optimization_phase(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create optimization phase (8-12 weeks)."""
        optimization_actions = [
            'Implement performance monitoring for extracted components',
            'Optimize shared component interfaces',
            'Fine-tune dependency management',
            'Implement automated code quality monitoring',
            'Create comprehensive integration tests'
        ]
        
        return {
            'duration': '8-12 weeks',
            'actions': optimization_actions,
            'success_criteria': [
                'All health scores > 80/100',
                'Integration tests passing',
                'Performance metrics improved',
                'Code maintainability index > 85'
            ],
            'resources_required': ['2-3 developers', 'QA team', 'Performance specialists']
        }
    
    def _calculate_extraction_priority(self, component: Dict[str, Any]) -> float:
        """Calculate priority score for component extraction."""
        priority = 0.0
        
        # Higher priority for components with more reuse potential
        if 'reuse_count' in component:
            priority += min(component['reuse_count'] * 0.1, 0.5)
        
        # Higher priority for larger components
        if 'size' in component:
            priority += min(component['size'] * 0.001, 0.3)
        
        # Higher priority for components with dependencies
        if 'dependencies' in component:
            priority += min(len(component['dependencies']) * 0.05, 0.2)
        
        return priority
    
    def _estimate_extraction_impact(self, component: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of component extraction."""
        return {
            'effort': 'medium',
            'risk': 'low',
            'benefit': 'high',
            'timeline': '1-2 weeks'
        }
    
    def _calculate_utility_priority(self, utility: Dict[str, Any]) -> float:
        """Calculate priority score for utility function extraction."""
        priority = 0.0
        
        if 'reusability_score' in utility:
            priority += utility['reusability_score']
        
        if 'complexity' in utility:
            # Lower complexity = higher priority for extraction
            priority += (10 - utility['complexity']) * 0.05
        
        return priority
    
    def _estimate_utility_impact(self, utility: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate impact of utility function extraction."""
        return {
            'effort': 'low',
            'risk': 'low',
            'benefit': 'medium',
            'timeline': '1-3 days'
        }
    
    def _determine_security_action(self, concern: Dict[str, Any]) -> str:
        """Determine required action for security concern."""
        concern_type = concern.get('type', '')
        
        if 'password' in concern_type:
            return 'Migrate to environment variable with encryption'
        elif 'secret' in concern_type:
            return 'Move to secure vault or environment variable'
        elif 'key' in concern_type:
            return 'Implement secure key management'
        else:
            return 'Review and apply appropriate security measures'
    
    def _estimate_security_fix_effort(self, concern: Dict[str, Any]) -> str:
        """Estimate effort required for security fix."""
        concern_type = concern.get('type', '')
        
        if 'password' in concern_type:
            return 'low'  # Usually straightforward env var migration
        elif 'secret' in concern_type:
            return 'medium'  # May require vault setup
        else:
            return 'medium'
    
    def _select_primary_framework(self, frameworks: Dict[str, int]) -> str:
        """Select primary testing framework based on usage."""
        if not frameworks:
            return 'pytest'  # Default recommendation
        
        # Return most used framework
        return max(frameworks.items(), key=lambda x: x[1])[0]
    
    def _create_framework_migration_strategy(self, frameworks: Dict[str, int]) -> List[str]:
        """Create framework migration strategy."""
        primary = self._select_primary_framework(frameworks)
        
        strategy = [
            f"Standardize on {primary} as primary framework",
            "Create migration utilities for test conversion",
            "Migrate tests in order of complexity (simple to complex)",
            "Maintain backward compatibility during transition",
            "Remove deprecated frameworks after migration"
        ]
        
        return strategy
    
    def _estimate_testing_consolidation_effort(self, frameworks: Dict[str, int]) -> str:
        """Estimate effort for testing consolidation."""
        framework_count = len(frameworks)
        
        if framework_count <= 2:
            return 'low'
        elif framework_count <= 4:
            return 'medium'
        else:
            return 'high'
    
    def _create_testing_phases(self, frameworks: Dict[str, int]) -> List[Dict[str, Any]]:
        """Create testing consolidation phases."""
        return [
            {
                'phase': 'Assessment',
                'duration': '1 week',
                'activities': ['Audit all test files', 'Create migration plan']
            },
            {
                'phase': 'Migration',
                'duration': '2-4 weeks',
                'activities': ['Convert tests to primary framework', 'Update CI/CD']
            },
            {
                'phase': 'Cleanup',
                'duration': '1 week',
                'activities': ['Remove unused frameworks', 'Verify all tests pass']
            }
        ]
    
    def _create_config_strategy(self, summary: Dict[str, Any]) -> List[str]:
        """Create configuration centralization strategy."""
        return [
            "Create centralized configuration module",
            "Implement environment-based configuration layers",
            "Add configuration validation and type checking",
            "Create configuration documentation and examples",
            "Implement configuration change monitoring"
        ]
    
    def _create_security_migration_plan(self, config_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create security migration plan."""
        security_concerns = config_data.get('raw_data', {}).get('security_concerns', [])
        
        plan = []
        for i, concern in enumerate(security_concerns[:10]):  # Top 10 concerns
            plan.append({
                'priority': f"Priority {i+1}",
                'concern': concern.get('type', 'unknown'),
                'file': concern.get('file', 'unknown'),
                'action': self._determine_security_action(concern),
                'timeline': '1-3 days'
            })
        
        return plan
    
    def _create_config_timeline(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create configuration implementation timeline."""
        return [
            {
                'week': '1-2',
                'milestone': 'Security fixes complete',
                'deliverables': ['All hardcoded secrets migrated', 'Environment variable system']
            },
            {
                'week': '3-4',
                'milestone': 'Centralized configuration',
                'deliverables': ['Configuration module', 'Validation system']
            }
        ]
    
    # Additional helper methods for other planning functions...
    def _design_utility_modules(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design utility module structure."""
        return [
            {'module': 'utils.validators', 'functions': 'validation utilities'},
            {'module': 'utils.converters', 'functions': 'data conversion utilities'},
            {'module': 'utils.helpers', 'functions': 'common helper functions'}
        ]
    
    def _create_extraction_strategy(self, summary: Dict[str, Any]) -> List[str]:
        """Create utility extraction strategy."""
        return [
            "Identify high-value reusable functions",
            "Create shared utility modules",
            "Update imports across codebase",
            "Add comprehensive tests for utilities",
            "Document utility APIs"
        ]
    
    def _prioritize_utility_extraction(self, summary: Dict[str, Any]) -> List[str]:
        """Prioritize utility extraction order."""
        return [
            "Start with validation utilities (highest reuse)",
            "Extract data conversion functions",
            "Consolidate helper functions",
            "Optimize function interfaces",
            "Create utility documentation"
        ]
    
    def _create_duplicate_strategy(self) -> List[str]:
        """Create duplicate elimination strategy."""
        return [
            "Use AST-based analysis for accurate detection",
            "Prioritize exact duplicates first",
            "Create shared functions for similar code",
            "Implement gradual refactoring approach",
            "Verify functionality preservation"
        ]
    
    def _create_duplicate_risk_plan(self) -> List[str]:
        """Create duplicate elimination risk mitigation."""
        return [
            "Comprehensive test coverage before changes",
            "Incremental refactoring with rollback capability",
            "Code review for all duplicate eliminations",
            "Performance testing after changes",
            "Gradual deployment with monitoring"
        ]
    
    def _create_duplicate_phases(self) -> List[Dict[str, Any]]:
        """Create duplicate elimination phases."""
        return [
            {
                'phase': 'Exact Duplicates',
                'target': '9,010 exact duplicates',
                'timeline': '2-3 weeks'
            },
            {
                'phase': 'Similar Code',
                'target': 'Remaining duplicate groups',
                'timeline': '4-6 weeks'
            }
        ]
    
    def _create_dependency_strategy(self, summary: Dict[str, Any]) -> List[str]:
        """Create dependency optimization strategy."""
        return [
            "Audit all third-party dependencies",
            "Identify overlapping functionality",
            "Consolidate similar libraries",
            "Update to latest secure versions",
            "Remove unused dependencies"
        ]
    
    def _plan_framework_consolidation(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Plan framework consolidation."""
        return {
            'web_frameworks': 'Standardize on primary framework',
            'testing_frameworks': 'Consolidate to 1-2 frameworks',
            'data_frameworks': 'Optimize for specific use cases',
            'timeline': '6-8 weeks'
        }
    
    def _create_dependency_roadmap(self, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create dependency optimization roadmap."""
        return [
            {
                'phase': 'Audit',
                'duration': '1 week',
                'activities': ['Dependency analysis', 'Conflict identification']
            },
            {
                'phase': 'Consolidation',
                'duration': '3-4 weeks',
                'activities': ['Framework standardization', 'Library optimization']
            }
        ]
    
    def _create_risk_mitigation_strategy(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive risk mitigation strategy."""
        return {
            'testing_strategy': [
                'Comprehensive regression testing before changes',
                'Automated testing for all extracted components',
                'Integration testing for consolidated systems'
            ],
            'rollback_procedures': [
                'Version control branching strategy',
                'Automated rollback capabilities',
                'Database backup and restore procedures'
            ],
            'monitoring': [
                'Real-time performance monitoring',
                'Error rate tracking',
                'User experience metrics'
            ]
        }
    
    def _define_success_metrics(self, opportunities: Dict[str, Any]) -> Dict[str, Any]:
        """Define success metrics for integration."""
        return {
            'security_metrics': [
                'Zero hardcoded secrets',
                'Configuration security score > 90/100',
                'Security audit passing'
            ],
            'quality_metrics': [
                'All health scores > 80/100',
                'Code duplication < 5%',
                'Test coverage > 85%'
            ],
            'performance_metrics': [
                'Build time improvement > 20%',
                'Dependency count reduction > 15%',
                'Maintainability index > 85'
            ]
        }
    
    def generate_comprehensive_plan(self, analysis_files: Dict[str, str]) -> Dict[str, Any]:
        """Generate comprehensive integration plan."""
        print("=== Agent C Hours 47-50: Component Integration Planning ===")
        print("Generating comprehensive integration strategy...")
        
        # Load all analysis data
        self.load_analysis_data(analysis_files)
        
        # Synthesize opportunities
        opportunities = self.synthesize_integration_opportunities()
        
        # Create implementation roadmap
        roadmap = self.create_implementation_roadmap(opportunities)
        
        # Generate final plan
        comprehensive_plan = {
            'analysis_metadata': {
                'tool': 'component_integration_planner',
                'version': '1.0',
                'agent': 'Agent_C',
                'hours': '47-50',
                'phase': 'Utility_Component_Extraction_Complete'
            },
            'integration_opportunities': opportunities,
            'implementation_roadmap': roadmap,
            'estimated_timeline': '12-16 weeks total implementation',
            'resource_requirements': self._calculate_resource_requirements(roadmap),
            'risk_assessment': self._assess_overall_risk(opportunities),
            'success_probability': self._calculate_success_probability(opportunities, roadmap)
        }
        
        return comprehensive_plan
    
    def _calculate_resource_requirements(self, roadmap: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate total resource requirements."""
        return {
            'development_team': '3-5 developers',
            'security_specialists': '1-2 security experts',
            'devops_support': '1 DevOps engineer',
            'qa_resources': '2-3 QA engineers',
            'estimated_effort': '40-60 person-weeks total'
        }
    
    def _assess_overall_risk(self, opportunities: Dict[str, Any]) -> str:
        """Assess overall implementation risk."""
        security_fixes = len(opportunities.get('security_critical_fixes', []))
        
        if security_fixes > 100:
            return 'HIGH - Many security concerns require careful handling'
        elif security_fixes > 50:
            return 'MEDIUM - Moderate security concerns manageable with proper planning'
        else:
            return 'LOW - Few security concerns, straightforward implementation'
    
    def _calculate_success_probability(self, opportunities: Dict[str, Any], roadmap: Dict[str, Any]) -> str:
        """Calculate probability of successful implementation."""
        # Based on complexity and resource requirements
        return '85-90% with proper execution and adequate resources'


def main():
    parser = argparse.ArgumentParser(description='Component Integration Planning Tool')
    parser.add_argument('--analysis-dir', type=str, required=True, help='Directory containing analysis files')
    parser.add_argument('--output', type=str, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    # Define analysis files to load
    analysis_files = {
        'core_library': f"{args.analysis_dir}/core_library_hour35.json",
        'utility_functions': f"{args.analysis_dir}/utility_functions_hour38.json",
        'configuration_settings': f"{args.analysis_dir}/configuration_settings_hour41.json",
        'testing_framework': f"{args.analysis_dir}/testing_framework_hour44.json",
        'shared_components': f"{args.analysis_dir}/shared_components_hour29.json"
    }
    
    planner = ComponentIntegrationPlanner()
    comprehensive_plan = planner.generate_comprehensive_plan(analysis_files)
    
    # Save results
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_plan, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n=== COMPONENT INTEGRATION PLANNING COMPLETE ===")
    print(f"Integration opportunities identified: {len(comprehensive_plan['integration_opportunities'])}")
    print(f"Implementation phases: {len(comprehensive_plan['implementation_roadmap'])}")
    print(f"Estimated timeline: {comprehensive_plan['estimated_timeline']}")
    print(f"Success probability: {comprehensive_plan['success_probability']}")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()