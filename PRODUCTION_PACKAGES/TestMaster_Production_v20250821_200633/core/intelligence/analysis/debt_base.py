"""
Technical Debt Base Classes and Configuration
=============================================

Core data structures and configuration for technical debt analysis.
Part of modularized debt_analyzer system.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum


class DebtCategory(Enum):
    """Categories of technical debt."""
    CODE_QUALITY = "code_quality"
    DESIGN = "design"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    INFRASTRUCTURE = "infrastructure"
    DEPENDENCIES = "dependencies"
    SECURITY = "security"
    PERFORMANCE = "performance"


class DebtSeverity(Enum):
    """Severity levels for debt items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    TRIVIAL = "trivial"


@dataclass
class DebtItem:
    """Represents a single technical debt item."""
    type: str
    severity: str
    location: str
    description: str
    estimated_hours: float
    interest_rate: float  # How much this debt grows over time
    risk_factor: float
    remediation_strategy: str
    dependencies: List[str]
    business_impact: str
    category: Optional[DebtCategory] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': self.type,
            'severity': self.severity,
            'location': self.location,
            'description': self.description,
            'estimated_hours': self.estimated_hours,
            'interest_rate': self.interest_rate,
            'risk_factor': self.risk_factor,
            'remediation_strategy': self.remediation_strategy,
            'dependencies': self.dependencies,
            'business_impact': self.business_impact,
            'category': self.category.value if self.category else None
        }


@dataclass
class DebtMetrics:
    """Aggregate technical debt metrics."""
    total_debt_hours: float
    debt_ratio: float  # Debt hours / total development hours
    monthly_interest: float  # Additional hours added per month
    break_even_point: int  # Months until fixing debt pays off
    risk_adjusted_cost: float
    team_velocity_impact: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'total_debt_hours': self.total_debt_hours,
            'debt_ratio': self.debt_ratio,
            'monthly_interest': self.monthly_interest,
            'break_even_point': self.break_even_point,
            'risk_adjusted_cost': self.risk_adjusted_cost,
            'team_velocity_impact': self.team_velocity_impact
        }


class DebtConfiguration:
    """Configuration for debt analysis."""
    
    # Cost factors (hours to fix)
    DEBT_COST_FACTORS = {
        "code_duplication": 2.0,
        "missing_tests": 1.5,
        "complex_function": 4.0,
        "poor_naming": 0.5,
        "missing_documentation": 1.0,
        "deprecated_usage": 2.5,
        "security_vulnerability": 8.0,
        "performance_issue": 6.0,
        "architectural_violation": 12.0,
        "dead_code": 0.5,
        "inconsistent_style": 0.25,
        "tight_coupling": 5.0,
        "missing_error_handling": 3.0,
        "hardcoded_values": 1.0,
        "outdated_dependency": 4.0
    }
    
    # Interest rates (monthly growth factor)
    INTEREST_RATES = {
        "security": 0.15,
        "performance": 0.10,
        "maintainability": 0.08,
        "reliability": 0.12,
        "testability": 0.06,
        "documentation": 0.04,
        "code_quality": 0.05
    }
    
    # Team productivity factors
    PRODUCTIVITY_FACTORS = {
        "junior_developer": 1.5,
        "mid_developer": 1.0,
        "senior_developer": 0.7,
        "team_size_factor": 1.2,
        "context_switching": 1.3
    }
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        "low": 5,
        "medium": 10,
        "high": 20,
        "very_high": 30
    }
    
    # Risk factors
    RISK_MULTIPLIERS = {
        "critical_path": 2.0,
        "customer_facing": 1.8,
        "data_sensitive": 1.5,
        "high_traffic": 1.6,
        "legacy_system": 1.4
    }


@dataclass
class RemediationPlan:
    """Plan for addressing technical debt."""
    debt_item: DebtItem
    priority: int
    estimated_effort: float
    expected_roi: float
    dependencies: List[str]
    team_assignment: Optional[str] = None
    target_sprint: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'debt_item': self.debt_item.to_dict(),
            'priority': self.priority,
            'estimated_effort': self.estimated_effort,
            'expected_roi': self.expected_roi,
            'dependencies': self.dependencies,
            'team_assignment': self.team_assignment,
            'target_sprint': self.target_sprint
        }


@dataclass
class DebtTrend:
    """Tracks debt trends over time."""
    timestamp: str
    total_debt_hours: float
    debt_categories: Dict[str, float]
    velocity_impact: float
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp,
            'total_debt_hours': self.total_debt_hours,
            'debt_categories': self.debt_categories,
            'velocity_impact': self.velocity_impact,
            'quality_score': self.quality_score
        }


# Export all classes
__all__ = [
    'DebtCategory', 'DebtSeverity', 'DebtItem', 'DebtMetrics',
    'DebtConfiguration', 'RemediationPlan', 'DebtTrend'
]