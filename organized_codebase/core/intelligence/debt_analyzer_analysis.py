"""
Technical Debt Analysis Module - Split from debt_analyzer.py
Quantifies technical debt in developer-hours and provides remediation strategies
"""


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



        deprecated = self._check_deprecated_dependencies()
        for dep in deprecated:
            debt_item = DebtItem(
                type="deprecated_dependency",
                severity="high",
                location="requirements.txt",
                description=f"Deprecated package: {dep['name']}",
                estimated_hours=8.0,  # Migration typically takes longer
                interest_rate=self.interest_rates["maintainability"],
                risk_factor=0.9,
                remediation_strategy=f"Migrate to {dep.get('alternative', 'modern alternative')}",
                dependencies=dep.get("usage_locations", []),
                business_impact="Will eventually break, no support"
            )
            dep_debt.append(debt_item)
        
        return dep_debt
    
    def _analyze_security_debt(self) -> List[DebtItem]:
        """
        Analyze security-related technical debt
        """
        security_debt = []
        
        # Analyze security vulnerabilities
        vulnerabilities = self._scan_security_vulnerabilities()
        for vuln in vulnerabilities:
            debt_item = DebtItem(
                type="security_vulnerability",
                severity="critical",
                location=vuln["location"],
                description=f"{vuln['type']}: {vuln['description']}",
                estimated_hours=self.debt_cost_factors["security_vulnerability"],
                interest_rate=self.interest_rates["security"],
                risk_factor=1.0,  # Maximum risk
                remediation_strategy=vuln["remediation"],
                dependencies=[],
                business_impact="Exposes system to attacks"
            )
            security_debt.append(debt_item)
        
        # Check for hardcoded secrets
        secrets = self._detect_hardcoded_secrets()
        for secret in secrets:
            debt_item = DebtItem(
                type="hardcoded_secret",
                severity="critical",
                location=secret["location"],
                description=f"Hardcoded {secret['type']} detected",
                estimated_hours=2.0,
                interest_rate=self.interest_rates["security"],
                risk_factor=1.0,
                remediation_strategy="Move to environment variables or secret manager",
                dependencies=[],
                business_impact="Critical security breach risk"
            )
            security_debt.append(debt_item)
        
        return security_debt
    
    def _analyze_performance_debt(self) -> List[DebtItem]:
        """
        Analyze performance-related technical debt
        """
        perf_debt = []
        
        # Analyze algorithmic complexity
        complexity_issues = self._analyze_algorithmic_complexity()
        for issue in complexity_issues:
            debt_item = DebtItem(
                type="performance_issue",
                severity="high" if issue["complexity"] == "exponential" else "medium",
                location=issue["location"],
                description=f"O({issue['complexity']}) algorithm detected",
                estimated_hours=self.debt_cost_factors["performance_issue"],
                interest_rate=self.interest_rates["performance"],
                risk_factor=0.8,
                remediation_strategy=issue["optimization_suggestion"],
                dependencies=[],
                business_impact="Causes performance degradation at scale"
            )
            perf_debt.append(debt_item)
        
        # Analyze database performance issues
        db_issues = self._analyze_database_performance()
        for issue in db_issues:
            debt_item = DebtItem(
                type="database_performance",
                severity="high",
                location=issue["location"],
                description=issue["description"],
                estimated_hours=5.0,
                interest_rate=self.interest_rates["performance"],
                risk_factor=0.7,
                remediation_strategy=issue["solution"],
                dependencies=[],
                business_impact="Causes slow queries and timeouts"
            )
            perf_debt.append(debt_item)
        
        return perf_debt
    
    def _quantify_debt_in_hours(self) -> Dict[str, Any]:
        """
        Quantify all technical debt in developer-hours
        """
        quantification = {
            "total_debt_hours": 0,
            "debt_by_type": defaultdict(float),
            "debt_by_severity": defaultdict(float),
            "debt_distribution": {},
            "team_adjusted_hours": {},
            "monetary_cost": {}
        }
        
        # Calculate raw debt hours
        for item in self.debt_items:
            quantification["total_debt_hours"] += item.estimated_hours
            quantification["debt_by_type"][item.type] += item.estimated_hours
            quantification["debt_by_severity"][item.severity] += item.estimated_hours
        
        # Calculate team-adjusted hours
        for team_level in ["junior_developer", "mid_developer", "senior_developer"]:
            factor = self.productivity_factors[team_level]
            quantification["team_adjusted_hours"][team_level] = (
                quantification["total_debt_hours"] * factor
            )
        
        # Calculate monetary cost (assuming $75/hour average rate)
        hourly_rate = self.config.get("hourly_rate", 75)
        quantification["monetary_cost"] = {
            "total_cost": quantification["total_debt_hours"] * hourly_rate,
            "cost_by_type": {
                debt_type: hours * hourly_rate
                for debt_type, hours in quantification["debt_by_type"].items()
            }
        }
        
        # Calculate debt distribution percentages
        if quantification["total_debt_hours"] > 0:
            quantification["debt_distribution"] = {
                debt_type: (hours / quantification["total_debt_hours"]) * 100
                for debt_type, hours in quantification["debt_by_type"].items()
            }
        
        return quantification
    
    def _categorize_debt(self) -> Dict[str, Any]:
        """Categorize debt items - simplified implementation"""
        return {"categories": ["code", "design", "test", "documentation"]}
    
    def _calculate_debt_interest(self) -> Dict[str, Any]:
        """
        Calculate how technical debt grows over time
        """
        interest_calc = {
            "monthly_interest_hours": 0,
            "annual_interest_hours": 0,
            "compound_growth": [],
            "interest_by_category": defaultdict(float),
            "high_interest_items": []
        }
        
        # Calculate monthly interest
        for item in self.debt_items:
            monthly_interest = item.estimated_hours * item.interest_rate
            interest_calc["monthly_interest_hours"] += monthly_interest
            
            # Track by category
            category = self._get_debt_category(item.type)
            interest_calc["interest_by_category"][category] += monthly_interest
            
            # Identify high-interest items
            if item.interest_rate > 0.10:  # More than 10% monthly growth
                interest_calc["high_interest_items"].append({
                    "description": item.description,
                    "location": item.location,
                    "monthly_growth": monthly_interest,
                    "interest_rate": item.interest_rate
                })
        
        # Calculate annual interest (compounded monthly)
        interest_calc["annual_interest_hours"] = (
            interest_calc["monthly_interest_hours"] * 12
        )
        
        # Project compound growth over 12 months
        current_debt = sum(item.estimated_hours for item in self.debt_items)
        for month in range(1, 13):
            if current_debt > 0:
                monthly_debt = current_debt * (1 + (interest_calc["monthly_interest_hours"] / current_debt))
                interest_calc["compound_growth"].append({
                    "month": month,
                    "total_debt_hours": monthly_debt,
                    "added_debt": monthly_debt - current_debt
                })
                current_debt = monthly_debt
        
        return interest_calc
    
    def _analyze_debt_impact(self) -> Dict[str, Any]:
        """
        Analyze the impact of technical debt on development
        """
        impact_analysis = {
            "velocity_impact": self._calculate_velocity_impact(),
            "quality_impact": self._calculate_quality_impact(),
            "morale_impact": self._estimate_morale_impact(),
            "delivery_impact": self._calculate_delivery_impact(),
            "innovation_impact": self._assess_innovation_impact(),
            "operational_impact": self._assess_operational_impact()
        }
        
        return impact_analysis
    
    def _calculate_velocity_impact(self) -> Dict[str, Any]:
        """
        Calculate impact on team velocity
        """
        total_debt_hours = sum(item.estimated_hours for item in self.debt_items)
        
        # Estimate velocity reduction based on debt
        velocity_reduction = min(0.5, total_debt_hours / 1000)  # Max 50% reduction
        
        return {
            "current_velocity_reduction": f"{velocity_reduction * 100:.1f}%",
            "time_spent_on_workarounds": total_debt_hours * 0.1,  # 10% of debt time on workarounds
            "context_switching_overhead": total_debt_hours * 0.15,  # 15% overhead
            "debugging_time_increase": total_debt_hours * 0.2,  # 20% more debugging
            "estimated_sprint_impact": f"{velocity_reduction * 10:.1f} story points per sprint"
        }
    
    def _calculate_quality_impact(self) -> Dict[str, Any]:
        """
        Calculate impact on code quality
        """
        critical_items = [item for item in self.debt_items if item.severity == "critical"]
        high_items = [item for item in self.debt_items if item.severity == "high"]
        
        return {
            "bug_increase_factor": 1 + (len(critical_items) * 0.1 + len(high_items) * 0.05),
            "defect_escape_rate": min(0.3, len(critical_items) * 0.02),
            "code_review_time_increase": f"{len(self.debt_items) * 0.5:.1f} hours per week",
            "test_reliability_decrease": f"{len([i for i in self.debt_items if i.type == 'flaky_test']) * 5}%",
            "regression_risk": "high" if len(critical_items) > 5 else "medium" if len(high_items) > 10 else "low"
        }
    
    def _create_remediation_plan(self) -> Dict[str, Any]:
        """
        Create a prioritized remediation plan
        """
        plan = {
            "phases": [],
            "quick_wins": [],
            "critical_fixes": [],
            "long_term_improvements": [],
            "total_effort_hours": 0,
            "recommended_team_size": 0,
            "estimated_timeline": {}
        }
        
        # Sort debt items by priority (severity * risk_factor / estimated_hours)
        prioritized_items = sorted(
            self.debt_items,
            key=lambda x: self._calculate_priority_score(x),
            reverse=True
        )
        
        # Phase 1: Critical security and stability issues
        phase1_items = [
            item for item in prioritized_items
            if item.severity == "critical" or item.risk_factor > 0.9
        ]
        plan["phases"].append({
            "phase": 1,
            "name": "Critical Issues",
            "duration_weeks": self._estimate_duration(phase1_items),
            "items": [self._summarize_item(item) for item in phase1_items[:10]],
            "total_hours": sum(item.estimated_hours for item in phase1_items)
        })
        
        # Phase 2: High-impact improvements
        phase2_items = [
            item for item in prioritized_items
            if item.severity == "high" and item not in phase1_items
        ]
        plan["phases"].append({
            "phase": 2,
            "name": "High-Impact Improvements",
            "duration_weeks": self._estimate_duration(phase2_items),
            "items": [self._summarize_item(item) for item in phase2_items[:15]],
            "total_hours": sum(item.estimated_hours for item in phase2_items)
        })
        
        # Phase 3: Technical improvements
        phase3_items = [
            item for item in prioritized_items
            if item.severity == "medium" and item not in phase1_items + phase2_items
        ]
        plan["phases"].append({
            "phase": 3,
            "name": "Technical Improvements",
            "duration_weeks": self._estimate_duration(phase3_items),
            "items": [self._summarize_item(item) for item in phase3_items[:20]],
            "total_hours": sum(item.estimated_hours for item in phase3_items)
        })
        
        # Identify quick wins (low effort, good impact)
        plan["quick_wins"] = [
            self._summarize_item(item) for item in prioritized_items
            if item.estimated_hours < 2 and item.risk_factor > 0.5
        ][:10]
        
        # Calculate team recommendations
        total_hours = sum(item.estimated_hours for item in self.debt_items)
        plan["total_effort_hours"] = total_hours
        plan["recommended_team_size"] = self._recommend_team_size(total_hours)
        plan["estimated_timeline"] = self._estimate_timeline(total_hours, plan["recommended_team_size"])
        
        return plan
    
    def _calculate_remediation_roi(self) -> Dict[str, Any]:
        """
        Calculate return on investment for debt remediation
        """
        roi_analysis = {
            "cost_benefit_ratio": 0,
            "payback_period_months": 0,
            "savings_breakdown": {},
            "risk_reduction_value": 0,
            "productivity_gains": {},
            "quality_improvements": {}
        }
        
        # Calculate remediation cost
        total_remediation_hours = sum(item.estimated_hours for item in self.debt_items)
        hourly_rate = self.config.get("hourly_rate", 75)
        remediation_cost = total_remediation_hours * hourly_rate
        
        # Calculate ongoing cost of debt
        monthly_interest_hours = sum(
            item.estimated_hours * item.interest_rate for item in self.debt_items
        )
        monthly_debt_cost = monthly_interest_hours * hourly_rate
        
        # Calculate savings from remediation
        annual_savings = monthly_debt_cost * 12
        roi_analysis["savings_breakdown"] = {
            "reduced_maintenance": annual_savings * 0.4,
            "improved_velocity": annual_savings * 0.3,
            "fewer_incidents": annual_savings * 0.2,
            "reduced_onboarding": annual_savings * 0.1
        }
        
        # Calculate payback period
        if monthly_debt_cost > 0:
            roi_analysis["payback_period_months"] = remediation_cost / monthly_debt_cost
        
        # Calculate cost-benefit ratio
        total_annual_savings = sum(roi_analysis["savings_breakdown"].values())
        if remediation_cost > 0:
            roi_analysis["cost_benefit_ratio"] = total_annual_savings / remediation_cost
        
        # Calculate risk reduction value
        critical_items = [item for item in self.debt_items if item.severity == "critical"]
        roi_analysis["risk_reduction_value"] = len(critical_items) * 10000  # $10k per critical risk
        
        # Calculate productivity gains
        velocity_impact = self._calculate_velocity_impact()
        roi_analysis["productivity_gains"] = {
            "hours_saved_monthly": monthly_interest_hours,
            "velocity_improvement": velocity_impact["current_velocity_reduction"],
            "developer_satisfaction": "improved"
        }
        
        return roi_analysis
    
    def _analyze_debt_trends(self) -> Dict[str, Any]:
        """Analyze debt trends - simplified implementation"""
        return {"trend": "increasing", "projection": "concerning"}
    
    def _assess_team_capacity_impact(self) -> Dict[str, Any]:
        """Assess team capacity impact - simplified implementation"""
        return {"capacity_reduction": "15%", "burnout_risk": "medium"}
    
    def _assess_debt_risks(self) -> Dict[str, Any]:
        """Assess debt risks - simplified implementation"""
        return {"overall_risk": "high", "critical_areas": ["security", "performance"]}
    
    def _create_prioritization_matrix(self) -> Dict[str, Any]:
        """Create prioritization matrix - simplified implementation"""
        return {"high_priority": len([i for i in self.debt_items if i.severity == "critical"])}
    
    def _identify_quick_wins(self) -> Dict[str, Any]:
        """
        Identify quick wins for immediate impact
        """
        quick_wins = {
            "automation_opportunities": [],
            "simple_refactorings": [],
            "documentation_updates": [],
            "dependency_updates": [],
            "total_quick_win_hours": 0,
            "expected_impact": {}
        }
        
        for item in self.debt_items:
            if item.estimated_hours <= 2:  # Can be done in 2 hours or less
                if item.type in ["dead_code", "poor_naming", "inconsistent_style"]:
                    quick_wins["simple_refactorings"].append({
                        "description": item.description,
                        "location": item.location,
                        "effort_hours": item.estimated_hours,
                        "impact": "immediate"
                    })
                elif item.type == "missing_documentation":
                    quick_wins["documentation_updates"].append({
                        "description": item.description,
                        "location": item.location,
                        "effort_hours": item.estimated_hours,
                        "impact": "team_productivity"
                    })
                elif item.type == "outdated_dependency" and item.risk_factor < 0.5:
                    quick_wins["dependency_updates"].append({
                        "description": item.description,
                        "effort_hours": item.estimated_hours,
                        "impact": "security_and_features"
                    })
        
        # Identify automation opportunities
        repetitive_issues = defaultdict(int)
        for item in self.debt_items:
            repetitive_issues[item.type] += 1
        
        for issue_type, count in repetitive_issues.items():
            if count > 5:  # Repeated issue worth automating
                quick_wins["automation_opportunities"].append({
                    "type": issue_type,
                    "count": count,
                    "automation_suggestion": self._suggest_automation(issue_type),
                    "one_time_effort": 8,  # Hours to set up automation
                    "ongoing_savings": count * 0.5  # Hours saved per occurrence
                })
        
        # Calculate total quick win hours
        quick_wins["total_quick_win_hours"] = sum(
            item["effort_hours"] for category in 
            ["simple_refactorings", "documentation_updates", "dependency_updates"]
            for item in quick_wins[category]
        )
        
        # Estimate impact
        quick_wins["expected_impact"] = {
            "immediate_velocity_gain": "5-10%",
            "code_quality_improvement": "noticeable",
            "team_morale_boost": "positive",
            "risk_reduction": "moderate"
        }
        
        return quick_wins
    
    def _suggest_debt_prevention_measures(self) -> Dict[str, Any]:
        """
        Suggest measures to prevent future technical debt
        """
        prevention = {
            "process_improvements": [],
            "tooling_recommendations": [],
            "quality_gates": [],
            "training_needs": [],
            "architectural_guidelines": [],
            "monitoring_setup": []
        }
        
        # Analyze patterns in current debt
        debt_patterns = self._analyze_debt_patterns()
        
        # Suggest process improvements
        if debt_patterns.get("missing_tests_prevalent"):
            prevention["process_improvements"].append({
                "recommendation": "Implement TDD or mandatory test coverage",
                "rationale": "High amount of untested code detected",
                "expected_reduction": "70% reduction in test debt"
            })
        
        if debt_patterns.get("complex_functions_common"):
            prevention["process_improvements"].append({
                "recommendation": "Set complexity limits in code review",
                "rationale": "Many high-complexity functions found",
                "expected_reduction": "50% reduction in complexity debt"
            })
        
        # Recommend tooling
        prevention["tooling_recommendations"] = [
            {
                "tool": "Pre-commit hooks",
                "purpose": "Catch issues before commit",
                "prevents": ["poor_naming", "inconsistent_style", "missing_tests"]
            },
            {
                "tool": "Static analysis in CI",
                "purpose": "Automated code quality checks",
                "prevents": ["complex_function", "security_vulnerability", "dead_code"]
            },
            {
                "tool": "Dependency scanner",
                "purpose": "Track outdated and vulnerable dependencies",
                "prevents": ["outdated_dependency", "security_vulnerability"]
            }