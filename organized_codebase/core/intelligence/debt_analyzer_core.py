"""
Technical Debt Analysis Module
Quantifies technical debt in developer-hours and provides remediation strategies
"""Core Analysis Module - Split from debt_analyzer.py"""


import ast
import re
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
import json
import math
import logging



        ]
        
        # Define quality gates
        prevention["quality_gates"] = [
            {
                "gate": "Code coverage > 80%",
                "enforcement": "Block PR merge",
                "prevents": "test debt accumulation"
            },
            {
                "gate": "Cyclomatic complexity < 10",
                "enforcement": "Require approval for exceptions",
                "prevents": "complexity debt"
            },
            {
                "gate": "No critical security issues",
                "enforcement": "Block deployment",
                "prevents": "security debt"
            }
        ]
        
        return prevention
    
    def _define_monitoring_metrics(self) -> Dict[str, Any]:
        """
        Define metrics to monitor technical debt over time
        """
        metrics = {
            "key_metrics": [],
            "dashboards": [],
            "alerts": [],
            "reporting_cadence": {},
            "success_criteria": {}
        }
        
        # Define key metrics
        metrics["key_metrics"] = [
            {
                "name": "Technical Debt Ratio",
                "formula": "Debt remediation cost / Development cost",
                "target": "< 15%",
                "current": self._calculate_debt_ratio()
            },
            {
                "name": "Debt Velocity",
                "formula": "New debt hours / Sprint",
                "target": "Decreasing trend",
                "current": "Not yet measured"
            },
            {
                "name": "Critical Issue Count",
                "formula": "Count of critical severity items",
                "target": "0",
                "current": len([i for i in self.debt_items if i.severity == "critical"])
            },
            {
                "name": "Code Coverage",
                "formula": "Tested lines / Total lines",
                "target": "> 80%",
                "current": self._get_current_coverage()
            },
            {
                "name": "Average Complexity",
                "formula": "Sum of cyclomatic complexity / Function count",
                "target": "< 5",
                "current": self._get_average_complexity()
            }
        ]
        
        # Define reporting cadence
        metrics["reporting_cadence"] = {
            "daily": ["New issues", "Build failures"],
            "weekly": ["Debt trend", "Remediation progress"],
            "monthly": ["Comprehensive debt report", "ROI analysis"],
            "quarterly": ["Strategic debt review", "Prevention effectiveness"]
        }
        
        # Define success criteria
        metrics["success_criteria"] = {
            "3_months": "50% reduction in critical issues",
            "6_months": "Debt ratio below 15%",
            "12_months": "90% automation of debt detection",
            "ongoing": "Debt velocity negative (paying down faster than creating)"
        }
        
        return metrics
    
    def _generate_debt_summary(self) -> Dict[str, Any]:
        """
        Generate executive summary of technical debt
        """
        total_hours = sum(item.estimated_hours for item in self.debt_items)
        hourly_rate = self.config.get("hourly_rate", 75)
        
        summary = {
            "executive_summary": {
                "total_debt_hours": total_hours,
                "total_debt_cost": total_hours * hourly_rate,
                "monthly_interest_cost": self._calculate_monthly_interest() * hourly_rate,
                "critical_issues": len([i for i in self.debt_items if i.severity == "critical"]),
                "recommended_action": self._get_recommended_action()
            },
            "debt_breakdown": {
                "by_severity": self._summarize_by_severity(),
                "by_type": self._summarize_by_type(),
                "by_risk": self._summarize_by_risk()
            },
            "business_impact": {
                "velocity_loss": self._calculate_velocity_impact()["current_velocity_reduction"],
                "quality_risk": self._assess_quality_risk(),
                "security_exposure": self._assess_security_exposure(),
                "talent_retention_risk": self._assess_talent_risk()
            },
            "remediation_summary": {
                "quick_wins_available": len(self._get_quick_wins()),
                "estimated_roi": self._calculate_simple_roi(),
                "payback_period": self._calculate_payback_period(),
                "recommended_investment": self._recommend_investment()
            },
            "next_steps": self._recommend_next_steps()
        }
        
        return summary
    
    # Helper methods
    def _calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
    
    def _detect_code_duplication(self, tree: ast.AST, file_path: Path) -> List[Dict]:
        """Detect duplicated code blocks - simplified implementation"""
        # In production, use more sophisticated algorithms
        return []
    
    def _assess_duplication_severity(self, dup: Dict) -> str:
        """Assess duplication severity - simplified implementation"""
        return "medium"
    
    def _is_poor_naming(self, name: str) -> bool:
        """Check if a name is poorly chosen"""
        if len(name) < 3 and name not in ['i', 'j', 'k', 'n', 'x', 'y']:
            return True
        if name in ['temp', 'tmp', 'var', 'val', 'data', 'obj']:
            return True
        return False
    
    def _lacks_error_handling(self, node: ast.FunctionDef) -> bool:
        """Check if function lacks error handling"""
        has_try = False
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_try = True
                break
        
        # Check if function performs risky operations
        risky_operations = False
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    if child.func.id in ['open', 'requests', 'urlopen']:
                        risky_operations = True
                        break
        
        return risky_operations and not has_try
    
    def _detect_dead_code(self, tree: ast.AST, file_path: Path) -> List[Dict]:
        """Detect dead code - simplified implementation"""
        return []
    
    def _analyze_coupling(self) -> List[Dict]:
        """Analyze coupling - simplified implementation"""
        return []
    
    def _detect_architectural_violations(self) -> List[Dict]:
        """Detect architectural violations - simplified implementation"""
        return []
    
    def _analyze_design_pattern_issues(self) -> List[Dict]:
        """Analyze design pattern issues - simplified implementation"""
        return []
    
    def _analyze_test_coverage_gaps(self) -> List[Dict]:
        """Analyze test coverage gaps - simplified implementation"""
        # Look for functions without corresponding test files
        gaps = []
        for file_path in self._get_python_files():
            if 'test_' not in file_path.name and file_path.name != '__init__.py':
                test_file = file_path.parent / f"test_{file_path.name}"
                if not test_file.exists():
                    gaps.append({
                        "function": file_path.name,
                        "location": str(file_path),
                        "criticality": "medium",
                        "complexity_factor": 1
                    })
        return gaps
    
    def _analyze_test_quality(self) -> List[Dict]:
        """Analyze test quality - simplified implementation"""
        return []
    
    def _detect_flaky_tests(self) -> List[Dict]:
        """Detect flaky tests - simplified implementation"""
        return []
    
    def _has_adequate_readme(self) -> bool:
        """Check if project has adequate README"""
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        for readme in readme_files:
            readme_path = self.root_path / readme
            if readme_path.exists():
                return readme_path.stat().st_size > 100  # At least 100 bytes
        return False
    
    def _check_outdated_dependencies(self) -> List[Dict]:
        """Check for outdated dependencies - simplified implementation"""
        return []
    
    def _assess_dependency_severity(self, dep: Dict) -> str:
        """Assess dependency severity - simplified implementation"""
        return "medium"
    
    def _check_deprecated_dependencies(self) -> List[Dict]:
        """Check for deprecated dependencies - simplified implementation"""
        return []
    
    def _scan_security_vulnerabilities(self) -> List[Dict]:
        """Scan for security vulnerabilities - simplified implementation"""
        return []
    
    def _detect_hardcoded_secrets(self) -> List[Dict]:
        """Detect hardcoded secrets - simplified implementation"""
        secrets = []
        for file_path in self._get_python_files():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Look for common secret patterns
                secret_patterns = [
                    (r'password\s*=\s*["\'][^"\']+["\']', 'password'),
                    (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key'),
                    (r'secret\s*=\s*["\'][^"\']+["\']', 'secret')
                ]
                
                for pattern, secret_type in secret_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        secrets.append({
                            "type": secret_type,
                            "location": str(file_path)
                        })
            except Exception as e:
                self.logger.warning(f"Could not scan {file_path} for secrets: {e}")
                
        return secrets
    
    def _analyze_algorithmic_complexity(self) -> List[Dict]:
        """Analyze algorithmic complexity - simplified implementation"""
        return []
    
    def _analyze_database_performance(self) -> List[Dict]:
        """Analyze database performance - simplified implementation"""
        return []
    
    def _get_debt_category(self, debt_type: str) -> str:
        """Map debt type to category"""
        categories = {
            "code_duplication": "maintainability",
            "complex_function": "maintainability",
            "poor_naming": "code_quality",
            "missing_tests": "testability",
            "missing_documentation": "documentation",
            "security_vulnerability": "security",
            "performance_issue": "performance",
            "outdated_dependency": "dependencies"
        }
        return categories.get(debt_type, "other")
    
    def _calculate_priority_score(self, item: DebtItem) -> float:
        """Calculate priority score for a debt item"""
        severity_weights = {
            "critical": 4,
            "high": 3,
            "medium": 2,
            "low": 1
        }
        
        severity_weight = severity_weights.get(item.severity, 1)
        
        # Priority = (severity * risk) / effort
        # Higher score = higher priority
        if item.estimated_hours > 0:
            return (severity_weight * item.risk_factor) / item.estimated_hours
        return 0
    
    def _estimate_duration(self, items: List[DebtItem]) -> int:
        """Estimate duration in weeks for fixing items"""
        total_hours = sum(item.estimated_hours for item in items)
        # Assuming 30 hours per week for debt reduction (75% of full time)
        return math.ceil(total_hours / 30) if total_hours > 0 else 0
    
    def _summarize_item(self, item: DebtItem) -> Dict:
        """Create summary of a debt item"""
        return {
            "type": item.type,
            "description": item.description,
            "location": item.location,
            "effort_hours": item.estimated_hours,
            "risk": item.risk_factor,
            "strategy": item.remediation_strategy
        }
    
    def _recommend_team_size(self, total_hours: float) -> int:
        """Recommend team size for debt remediation"""
        # Assuming 6-month target for remediation
        months = 6
        hours_per_developer_per_month = 120  # 75% allocation
        
        if total_hours > 0:
            required_developers = math.ceil(
                total_hours / (months * hours_per_developer_per_month)
            )
            # Cap at reasonable team size
            return min(required_developers, 5)
        return 1
    
    def _estimate_timeline(self, total_hours: float, team_size: int) -> Dict:
        """Estimate remediation timeline"""
        hours_per_developer_per_month = 120
        if team_size > 0 and total_hours > 0:
            months_required = math.ceil(
                total_hours / (team_size * hours_per_developer_per_month)
            )
        else:
            months_required = 0
        
        return {
            "months": months_required,
            "phases": math.ceil(months_required / 3) if months_required > 0 else 0,
            "milestones": self._generate_milestones(months_required)
        }
    
    def _generate_milestones(self, months: int) -> List[Dict]:
        """Generate milestone schedule"""
        milestones = []
        
        if months >= 1:
            milestones.append({
                "month": 1,
                "goal": "Critical security issues resolved",
                "success_metric": "Zero critical vulnerabilities"
            })
        
        if months >= 3:
            milestones.append({
                "month": 3,
                "goal": "High-priority debt reduced by 50%",
                "success_metric": "Velocity improvement measurable"
            })
        
        if months >= 6:
            milestones.append({
                "month": 6,
                "goal": "Technical debt ratio below 15%",
                "success_metric": "Sustainable debt level achieved"
            })
        
        return milestones
    
    def _calculate_monthly_interest(self) -> float:
        """Calculate total monthly interest in hours"""
        return sum(
            item.estimated_hours * item.interest_rate 
            for item in self.debt_items
        )
    
    def _suggest_automation(self, issue_type: str) -> str:
        """Suggest automation for issue type"""
        suggestions = {
            "poor_naming": "Set up linting rules for naming conventions",
            "missing_documentation": "Add pre-commit hooks to check for docstrings",
            "dead_code": "Use static analysis tools to detect unused code",
            "inconsistent_style": "Set up automatic code formatting with Black/Prettier"
        }
        return suggestions.get(issue_type, "Consider automation tools for this issue type")
    
    def _analyze_debt_patterns(self) -> Dict[str, bool]:
        """Analyze patterns in debt - simplified implementation"""
        test_debt_count = len([i for i in self.debt_items if i.type == "missing_tests"])
        complex_function_count = len([i for i in self.debt_items if i.type == "complex_function"])
        
        return {
            "missing_tests_prevalent": test_debt_count > 5,
            "complex_functions_common": complex_function_count > 3,
            "design_issues_prevalent": len([i for i in self.debt_items if "design" in i.type]) > 2
        }
    
    def _calculate_debt_ratio(self) -> str:
        """Calculate debt ratio - simplified implementation"""
        total_debt = sum(item.estimated_hours for item in self.debt_items)
        # Assume 1000 hours of total development
        ratio = (total_debt / 1000) * 100 if total_debt > 0 else 0
        return f"{ratio:.1f}%"
    
    def _get_current_coverage(self) -> str:
        """Get current test coverage - simplified implementation"""
        return "Not measured"
    
    def _get_average_complexity(self) -> str:
        """Get average complexity - simplified implementation"""
        return "Not measured"
    
    # Simplified placeholder methods for complex analysis functions
    def _estimate_morale_impact(self) -> Dict[str, str]:
        """Estimate morale impact - simplified implementation"""
        return {"developer_satisfaction": "declining", "turnover_risk": "medium"}
    
    def _calculate_delivery_impact(self) -> Dict[str, str]:
        """Calculate delivery impact - simplified implementation"""
        return {"release_frequency": "reduced", "feature_delivery": "delayed"}
    
    def _assess_innovation_impact(self) -> Dict[str, str]:
        """Assess innovation impact - simplified implementation"""
        return {"innovation_time": "reduced", "technical_exploration": "limited"}
    
    def _assess_operational_impact(self) -> Dict[str, str]:
        """Assess operational impact - simplified implementation"""
        return {"incident_frequency": "increased", "maintenance_overhead": "high"}
    
    def _get_recommended_action(self) -> str:
        """Get recommended action - simplified implementation"""
        critical_count = len([i for i in self.debt_items if i.severity == "critical"])
        if critical_count > 5:
            return "Immediate action required - critical debt threatening system stability"
        elif critical_count > 0:
            return "Address critical issues within 1 month, then systematic remediation"
        else:
            return "Gradual debt reduction with focus on prevention"
    
    def _summarize_by_severity(self) -> Dict[str, int]:
        """Summarize debt by severity"""
        summary = defaultdict(int)
        for item in self.debt_items:
            summary[item.severity] += 1
        return dict(summary)
    
    def _summarize_by_type(self) -> Dict[str, int]:
        """Summarize debt by type"""
        summary = defaultdict(int)
        for item in self.debt_items:
            summary[item.type] += 1
        return dict(summary)
    
    def _summarize_by_risk(self) -> Dict[str, int]:
        """Summarize debt by risk level"""
        summary = {"high": 0, "medium": 0, "low": 0}
        for item in self.debt_items:
            if item.risk_factor > 0.7:
                summary["high"] += 1
            elif item.risk_factor > 0.4:
                summary["medium"] += 1
            else:
                summary["low"] += 1
        return summary
    
    def _assess_quality_risk(self) -> str:
        """Assess quality risk - simplified implementation"""
        critical_count = len([i for i in self.debt_items if i.severity == "critical"])
        return "high" if critical_count > 3 else "medium" if critical_count > 1 else "low"
    
    def _assess_security_exposure(self) -> str:
        """Assess security exposure - simplified implementation"""
        security_issues = len([i for i in self.debt_items if "security" in i.type])
        return "critical" if security_issues > 2 else "medium" if security_issues > 0 else "low"
    
    def _assess_talent_risk(self) -> str:
        """Assess talent retention risk - simplified implementation"""
        total_debt = sum(item.estimated_hours for item in self.debt_items)
        return "high" if total_debt > 200 else "medium" if total_debt > 100 else "low"
    
    def _get_quick_wins(self) -> List[Dict]:
        """Get quick wins - simplified implementation"""
        return [item for item in self.debt_items if item.estimated_hours <= 2]
    
    def _calculate_simple_roi(self) -> str:
        """Calculate simple ROI - simplified implementation"""
        return "2.5x return within 12 months"
    
    def _calculate_payback_period(self) -> str:
        """Calculate payback period - simplified implementation"""
        return "4-6 months"
    
    def _recommend_investment(self) -> str:
        """Recommend investment - simplified implementation"""
        total_cost = sum(item.estimated_hours for item in self.debt_items) * 75
        return f"${total_cost:,.0f} over 6 months"
    
    def _recommend_next_steps(self) -> List[str]:
        """Recommend next steps - simplified implementation"""
        return [
            "Address critical security issues immediately",
            "Set up automated debt detection",
            "Create remediation roadmap",
            "Establish debt prevention processes"
        ]