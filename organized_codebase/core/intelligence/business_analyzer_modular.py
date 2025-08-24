"""
Modularized Business Analyzer
=============================

Main orchestrator for business rule analysis using modular components.
Replaces the original 1265-line business_analyzer.py.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging

from .business_base import BusinessAnalysisConfiguration
from .business_rule_extractor import BusinessRuleExtractor
from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.core.orchestration.workflows.business_workflow_analyzer import BusinessWorkflowAnalyzer
from .business_constraint_analyzer import BusinessConstraintAnalyzer


class BusinessAnalyzer:
    """
    Orchestrates business rule analysis across all components.
    
    This modularized version maintains all functionality of the original
    while being split into focused, maintainable components.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the business analyzer."""
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # Initialize configuration
        self.analysis_config = BusinessAnalysisConfiguration()
        
        # Initialize components
        self.rule_extractor = BusinessRuleExtractor(self.analysis_config)
        self.workflow_analyzer = BusinessWorkflowAnalyzer(self.analysis_config)
        self.constraint_analyzer = BusinessConstraintAnalyzer(self.analysis_config)
        
        # Results storage
        self.business_rules = []
        self.workflows = []
        self.domain_entities = []
        
        # Configuration
        self.root_path = None
        self.exclude_patterns = self.config.get('exclude_patterns', [
            '__pycache__', '.git', '.pytest_cache', 'venv', 'env',
            'build', 'dist', '*.egg-info', '.venv'
        ])
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the analyzer."""
        logger = logging.getLogger("business_analyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze(self, root_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive business rule analysis.
        
        Args:
            root_path: Optional root path to analyze. If None, uses current directory.
        """
        self.root_path = Path(root_path) if root_path else Path.cwd()
        self.logger.info(f"Starting business analysis for: {self.root_path}")
        
        # Get Python files
        python_files = self._get_python_files()
        self.logger.info(f"Found {len(python_files)} Python files to analyze")
        
        # Run analyses using components
        results = {
            "business_rules": self.rule_extractor.extract_business_rules(python_files),
            "validation_rules": self.rule_extractor.extract_validation_rules(python_files),
            "calculation_rules": self.rule_extractor.extract_calculation_rules(python_files),
            "authorization_rules": self.rule_extractor.extract_authorization_rules(python_files),
            "workflow_analysis": self.workflow_analyzer.analyze_workflows(python_files),
            "state_machines": self.workflow_analyzer.detect_state_machines(python_files),
            "domain_model": self.workflow_analyzer.extract_domain_model(python_files),
            "business_constraints": self.constraint_analyzer.extract_business_constraints(python_files),
            "decision_logic": self.constraint_analyzer.extract_decision_logic(python_files),
            "business_events": self.workflow_analyzer.extract_business_events(python_files),
            "compliance_rules": self.constraint_analyzer.extract_compliance_rules(python_files),
            "sla_rules": self.constraint_analyzer.extract_sla_rules(python_files),
            "pricing_rules": self.constraint_analyzer.extract_pricing_rules(python_files),
            "rule_dependencies": self._analyze_rule_dependencies(),
            "summary": self._generate_business_summary(results)
        }
        
        # Store extracted rules for dependency analysis
        self.business_rules = results["business_rules"].get("rules", [])
        self.workflows = results["workflow_analysis"].get("workflows", [])
        
        self.logger.info(f"Analysis complete: {len(self.business_rules)} business rules found")
        
        return results
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        if self.root_path.exists():
            for file_path in self.root_path.rglob("*.py"):
                # Skip excluded paths
                if not self._should_exclude(file_path):
                    python_files.append(file_path)
        return python_files
    
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if path should be excluded from analysis."""
        path_str = str(file_path)
        
        for pattern in self.exclude_patterns:
            if pattern in path_str:
                return True
        
        return False
    
    def _analyze_rule_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies between business rules"""
        dependencies = {
            "rule_graph": {},
            "dependency_chains": [],
            "circular_dependencies": [],
            "rule_conflicts": [],
            "rule_hierarchy": {},
            "execution_order": []
        }
        
        # Build dependency graph from extracted rules
        from collections import defaultdict
        rule_calls = defaultdict(set)
        
        for rule in self.business_rules:
            # Look for references to other rules
            for other_rule in self.business_rules:
                if rule != other_rule:
                    if (hasattr(other_rule, 'name') and hasattr(rule, 'conditions') and 
                        hasattr(rule, 'actions')):
                        if (other_rule.name in str(rule.conditions) or 
                            other_rule.name in str(rule.actions)):
                            rule_calls[rule.name].add(other_rule.name)
                            
        dependencies["rule_graph"] = {k: list(v) for k, v in rule_calls.items()}
        
        return dependencies
    
    def _generate_business_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of business rule analysis"""
        business_rules = results.get("business_rules", {}).get("rules", [])
        
        summary = {
            "total_rules": len(business_rules),
            "rule_categories": {},
            "domain_coverage": {},
            "complexity_assessment": {},
            "documentation_quality": {},
            "recommendations": [],
            "business_insights": []
        }
        
        # Count rules by category
        from collections import defaultdict
        categories = defaultdict(int)
        for rule in business_rules:
            if hasattr(rule, 'rule_type'):
                categories[rule.rule_type] += 1
        summary["rule_categories"] = dict(categories)
        
        # Assess domain coverage
        domains_covered = set()
        for rule in business_rules:
            if hasattr(rule, 'name') and hasattr(rule, 'description'):
                for domain, keywords in self.analysis_config.domain_keywords.items():
                    if any(keyword in rule.name.lower() or keyword in rule.description.lower() 
                          for keyword in keywords):
                        domains_covered.add(domain)
                        
        summary["domain_coverage"] = {
            "domains_identified": list(domains_covered),
            "coverage_percentage": (len(domains_covered) / 
                                  len(self.analysis_config.domain_keywords)) * 100 
                                  if self.analysis_config.domain_keywords else 0
        }
        
        # Assess complexity
        high_complexity_rules = []
        simple_rules = 0
        moderate_rules = 0
        total_conditions = 0
        
        for rule in business_rules:
            if hasattr(rule, 'conditions'):
                condition_count = len(rule.conditions)
                total_conditions += condition_count
                
                if condition_count <= 1:
                    simple_rules += 1
                elif condition_count <= 3:
                    moderate_rules += 1
                else:
                    high_complexity_rules.append(rule)
        
        summary["complexity_assessment"] = {
            "simple_rules": simple_rules,
            "moderate_rules": moderate_rules,
            "complex_rules": len(high_complexity_rules),
            "average_conditions": total_conditions / len(business_rules) if business_rules else 0
        }
        
        # Assess documentation
        documented_rules = []
        for rule in business_rules:
            if hasattr(rule, 'documentation') and rule.documentation:
                documented_rules.append(rule)
        
        summary["documentation_quality"] = {
            "documented_rules": len(documented_rules),
            "documentation_percentage": (len(documented_rules) / len(business_rules)) * 100 
                                       if business_rules else 0,
            "quality_score": self._calculate_documentation_quality_score(documented_rules)
        }
        
        # Generate recommendations
        if summary["documentation_quality"]["documentation_percentage"] < 50:
            summary["recommendations"].append({
                "priority": "high",
                "description": "Improve business rule documentation",
                "action": "Add clear documentation to business logic functions"
            })
            
        if len(high_complexity_rules) > len(business_rules) * 0.3:
            summary["recommendations"].append({
                "priority": "medium",
                "description": "Simplify complex business rules",
                "action": "Break down complex rules into smaller, composable rules"
            })
            
        # Generate business insights
        if domains_covered:
            summary["business_insights"].append({
                "insight": "Primary business domains",
                "domains": list(domains_covered)[:3],
                "description": f"System primarily handles {', '.join(list(domains_covered)[:3])} operations"
            })
            
        if self.workflows:
            summary["business_insights"].append({
                "insight": "Workflow complexity",
                "workflow_count": len(self.workflows),
                "description": f"System manages {len(self.workflows)} distinct business workflows"
            })
            
        return summary
    
    def _calculate_documentation_quality_score(self, documented_rules: List) -> float:
        """Calculate quality score for documentation"""
        if not documented_rules:
            return 0.0
            
        total_score = 0
        for rule in documented_rules:
            if hasattr(rule, 'documentation'):
                doc_length = len(rule.documentation)
                if doc_length > 100:
                    total_score += 1.0
                elif doc_length > 50:
                    total_score += 0.7
                elif doc_length > 20:
                    total_score += 0.4
                else:
                    total_score += 0.2
                    
        return (total_score / len(documented_rules)) * 100
    
    def get_critical_rules(self) -> List:
        """Get critical business rules requiring immediate attention."""
        return [
            rule for rule in self.business_rules 
            if hasattr(rule, 'priority') and rule.priority in ['critical', 'high']
        ]
    
    def get_rules_by_domain(self, domain: str) -> List:
        """Get business rules for a specific domain."""
        domain_rules = []
        domain_keywords = self.analysis_config.domain_keywords.get(domain, [])
        
        for rule in self.business_rules:
            if hasattr(rule, 'name') and hasattr(rule, 'description'):
                if any(keyword in rule.name.lower() or keyword in rule.description.lower() 
                      for keyword in domain_keywords):
                    domain_rules.append(rule)
                    
        return domain_rules
    
    def export_report(self, format: str = 'json') -> str:
        """Export business analysis report in specified format."""
        results = self.analyze()
        
        if format == 'json':
            import json
            return json.dumps(results, indent=2, default=str)
        elif format == 'markdown':
            return self._format_as_markdown(results)
        else:
            return str(results)
    
    def _format_as_markdown(self, results: Dict[str, Any]) -> str:
        """Format results as markdown."""
        summary = results.get('summary', {})
        
        md = f"""# Business Rule Analysis Report

## Summary
- **Total Business Rules**: {summary.get('total_rules', 0)}
- **Domain Coverage**: {summary.get('domain_coverage', {}).get('coverage_percentage', 0):.1f}%
- **Documentation Coverage**: {summary.get('documentation_quality', {}).get('documentation_percentage', 0):.1f}%

## Rule Categories
"""
        
        for category, count in summary.get('rule_categories', {}).items():
            md += f"- **{category.title()}**: {count} rules\n"
        
        md += "\n## Recommendations\n"
        for rec in summary.get('recommendations', []):
            md += f"- **{rec.get('priority', 'medium').title()}**: {rec.get('description', '')}\n"
        
        return md


# Export
__all__ = ['BusinessAnalyzer']