"""
Debt Quantification and Prioritization Component
================================================

Quantifies technical debt in developer-hours and provides
prioritization strategies.
Part of modularized debt_analyzer system.
"""

import math
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .debt_base import (
    DebtItem, DebtMetrics, DebtCategory,
    DebtConfiguration, RemediationPlan, DebtTrend
)


class DebtQuantifier:
    """Quantifies and prioritizes technical debt."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize debt quantifier."""
        self.config = config or {}
        self.debt_config = DebtConfiguration()
        
        # Team configuration
        self.team_size = self.config.get('team_size', 5)
        self.hours_per_sprint = self.config.get('hours_per_sprint', 80)
        self.sprints_per_month = self.config.get('sprints_per_month', 2)
        
        # Financial factors
        self.developer_hourly_cost = self.config.get('developer_hourly_cost', 100)
        self.opportunity_cost_factor = self.config.get('opportunity_cost_factor', 1.5)
        
        # History tracking
        self.debt_trends = []
    
    def quantify_debt(self, debt_items: List[DebtItem]) -> DebtMetrics:
        """Quantify total technical debt in hours."""
        if not debt_items:
            return DebtMetrics(0, 0, 0, 0, 0, 0)
        
        # Calculate total debt hours
        total_hours = sum(item.estimated_hours for item in debt_items)
        
        # Calculate monthly interest
        monthly_interest = self._calculate_monthly_interest(debt_items)
        
        # Calculate debt ratio
        total_dev_hours = self.team_size * self.hours_per_sprint * self.sprints_per_month
        debt_ratio = total_hours / total_dev_hours if total_dev_hours > 0 else 0
        
        # Calculate break-even point
        break_even = self._calculate_break_even(total_hours, monthly_interest)
        
        # Calculate risk-adjusted cost
        risk_cost = self._calculate_risk_adjusted_cost(debt_items)
        
        # Calculate velocity impact
        velocity_impact = self._calculate_velocity_impact(debt_items)
        
        return DebtMetrics(
            total_debt_hours=total_hours,
            debt_ratio=debt_ratio,
            monthly_interest=monthly_interest,
            break_even_point=break_even,
            risk_adjusted_cost=risk_cost,
            team_velocity_impact=velocity_impact
        )
    
    def _calculate_monthly_interest(self, debt_items: List[DebtItem]) -> float:
        """Calculate how much debt grows per month."""
        monthly_growth = 0.0
        
        for item in debt_items:
            # Interest compounds on the principal
            monthly_growth += item.estimated_hours * item.interest_rate
        
        return monthly_growth
    
    def _calculate_break_even(self, total_hours: float, monthly_interest: float) -> int:
        """Calculate months until fixing debt pays off."""
        if monthly_interest <= 0:
            return 1  # Immediate payoff if no interest
        
        # Time saved per month after fixing
        time_saved_per_month = monthly_interest
        
        # Months to break even
        if time_saved_per_month > 0:
            return math.ceil(total_hours / time_saved_per_month)
        
        return 999  # Never breaks even
    
    def _calculate_risk_adjusted_cost(self, debt_items: List[DebtItem]) -> float:
        """Calculate risk-adjusted cost in hours."""
        total_risk_cost = 0.0
        
        for item in debt_items:
            base_cost = item.estimated_hours
            risk_multiplier = item.risk_factor
            
            # Add opportunity cost
            opportunity_cost = base_cost * self.opportunity_cost_factor
            
            # Total cost with risk
            total_risk_cost += base_cost * risk_multiplier + opportunity_cost
        
        return total_risk_cost
    
    def _calculate_velocity_impact(self, debt_items: List[DebtItem]) -> float:
        """Calculate impact on team velocity (0-1 scale)."""
        if not debt_items:
            return 0.0
        
        # Group by category
        category_impacts = defaultdict(float)
        
        for item in debt_items:
            category = item.category or DebtCategory.CODE_QUALITY
            
            # Different categories have different velocity impacts
            impact_factors = {
                DebtCategory.CODE_QUALITY: 0.15,
                DebtCategory.TESTING: 0.20,
                DebtCategory.DESIGN: 0.25,
                DebtCategory.DOCUMENTATION: 0.10,
                DebtCategory.INFRASTRUCTURE: 0.30,
                DebtCategory.DEPENDENCIES: 0.15,
                DebtCategory.SECURITY: 0.35,
                DebtCategory.PERFORMANCE: 0.25
            }
            
            factor = impact_factors.get(category, 0.15)
            category_impacts[category] += item.estimated_hours * factor
        
        # Calculate weighted impact
        total_hours = sum(item.estimated_hours for item in debt_items)
        if total_hours > 0:
            weighted_impact = sum(category_impacts.values()) / total_hours
            # Normalize to 0-1 scale
            return min(1.0, weighted_impact)
        
        return 0.0
    
    def prioritize_debt(self, debt_items: List[DebtItem]) -> List[RemediationPlan]:
        """Create prioritized remediation plan."""
        if not debt_items:
            return []
        
        # Calculate priority scores
        scored_items = []
        for item in debt_items:
            score = self._calculate_priority_score(item)
            scored_items.append((score, item))
        
        # Sort by priority (higher score = higher priority)
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        # Create remediation plans
        plans = []
        current_sprint = 1
        sprint_capacity = self.hours_per_sprint * self.team_size * 0.2  # 20% for debt
        current_sprint_hours = 0
        
        for score, item in scored_items:
            # Calculate ROI
            roi = self._calculate_roi(item)
            
            # Assign to sprint
            if current_sprint_hours + item.estimated_hours > sprint_capacity:
                current_sprint += 1
                current_sprint_hours = 0
            
            plan = RemediationPlan(
                debt_item=item,
                priority=len(plans) + 1,
                estimated_effort=item.estimated_hours,
                expected_roi=roi,
                dependencies=item.dependencies,
                target_sprint=current_sprint
            )
            
            plans.append(plan)
            current_sprint_hours += item.estimated_hours
        
        return plans
    
    def _calculate_priority_score(self, item: DebtItem) -> float:
        """Calculate priority score for debt item."""
        # Factors for prioritization
        severity_scores = {
            'critical': 10.0,
            'high': 7.0,
            'medium': 4.0,
            'low': 2.0,
            'trivial': 1.0
        }
        
        severity_score = severity_scores.get(item.severity, 3.0)
        
        # Consider interest rate (compound growth)
        interest_score = item.interest_rate * 10
        
        # Consider risk factor
        risk_score = item.risk_factor * 5
        
        # Consider effort (prefer quick wins)
        effort_score = 10 / (1 + item.estimated_hours)
        
        # Weighted combination
        total_score = (
            severity_score * 0.3 +
            interest_score * 0.3 +
            risk_score * 0.2 +
            effort_score * 0.2
        )
        
        return total_score
    
    def _calculate_roi(self, item: DebtItem) -> float:
        """Calculate return on investment for fixing debt."""
        # Time saved per month after fixing
        monthly_savings = item.estimated_hours * item.interest_rate
        
        # Payback period in months
        if monthly_savings > 0:
            payback_months = item.estimated_hours / monthly_savings
            
            # ROI over 12 months
            total_savings = monthly_savings * 12
            roi = (total_savings - item.estimated_hours) / item.estimated_hours
            
            return max(0, roi)
        
        return 0.0
    
    def track_trend(self, debt_items: List[DebtItem], metrics: DebtMetrics):
        """Track debt trends over time."""
        # Calculate category breakdown
        category_hours = defaultdict(float)
        for item in debt_items:
            category = item.category or DebtCategory.CODE_QUALITY
            category_hours[category] += item.estimated_hours
        
        # Create trend entry
        trend = DebtTrend(
            timestamp=datetime.now().isoformat(),
            total_debt_hours=metrics.total_debt_hours,
            debt_categories={k.value: v for k, v in category_hours.items()},
            velocity_impact=metrics.team_velocity_impact,
            quality_score=100 - (metrics.debt_ratio * 100)  # Inverse of debt ratio
        )
        
        self.debt_trends.append(trend)
        
        # Keep only last 12 months
        if len(self.debt_trends) > 12:
            self.debt_trends = self.debt_trends[-12:]
    
    def get_financial_impact(self, metrics: DebtMetrics) -> Dict[str, float]:
        """Calculate financial impact of technical debt."""
        # Direct cost of debt
        direct_cost = metrics.total_debt_hours * self.developer_hourly_cost
        
        # Monthly interest cost
        monthly_interest_cost = metrics.monthly_interest * self.developer_hourly_cost
        
        # Annual interest cost
        annual_interest_cost = monthly_interest_cost * 12
        
        # Velocity impact cost (lost productivity)
        monthly_dev_hours = self.team_size * self.hours_per_sprint * self.sprints_per_month
        lost_productivity_hours = monthly_dev_hours * metrics.team_velocity_impact
        monthly_productivity_cost = lost_productivity_hours * self.developer_hourly_cost
        
        return {
            'direct_cost': direct_cost,
            'monthly_interest_cost': monthly_interest_cost,
            'annual_interest_cost': annual_interest_cost,
            'monthly_productivity_loss': monthly_productivity_cost,
            'annual_productivity_loss': monthly_productivity_cost * 12,
            'total_annual_impact': annual_interest_cost + (monthly_productivity_cost * 12)
        }
    
    def generate_summary(self, 
                        debt_items: List[DebtItem],
                        metrics: DebtMetrics) -> Dict[str, Any]:
        """Generate executive summary of debt analysis."""
        financial_impact = self.get_financial_impact(metrics)
        
        # Group by category
        category_breakdown = defaultdict(lambda: {'count': 0, 'hours': 0})
        severity_breakdown = defaultdict(int)
        
        for item in debt_items:
            category = (item.category or DebtCategory.CODE_QUALITY).value
            category_breakdown[category]['count'] += 1
            category_breakdown[category]['hours'] += item.estimated_hours
            severity_breakdown[item.severity] += 1
        
        return {
            'total_debt_items': len(debt_items),
            'total_debt_hours': metrics.total_debt_hours,
            'debt_ratio_percent': metrics.debt_ratio * 100,
            'monthly_interest_hours': metrics.monthly_interest,
            'break_even_months': metrics.break_even_point,
            'velocity_impact_percent': metrics.team_velocity_impact * 100,
            'financial_impact': financial_impact,
            'category_breakdown': dict(category_breakdown),
            'severity_breakdown': dict(severity_breakdown),
            'recommendations': self._generate_recommendations(metrics)
        }
    
    def _generate_recommendations(self, metrics: DebtMetrics) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if metrics.debt_ratio > 0.3:
            recommendations.append("Critical: Debt exceeds 30% of capacity - immediate action required")
        
        if metrics.monthly_interest > 40:
            recommendations.append("High interest rate - prioritize high-growth debt items")
        
        if metrics.break_even_point < 3:
            recommendations.append("Quick wins available - debt will pay off within 3 months")
        
        if metrics.team_velocity_impact > 0.2:
            recommendations.append("Significant velocity impact - consider dedicated debt sprint")
        
        return recommendations


# Export
__all__ = ['DebtQuantifier']