"""
Advanced Test Analytics Engine for TestMaster
Test effectiveness scoring, coverage gap analysis, and risk-based prioritization
"""

import ast
import statistics
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import math


class AnalyticsMetric(Enum):
    """Analytics metrics"""
    EFFECTIVENESS = "effectiveness"
    REDUNDANCY = "redundancy"
    COVERAGE_GAP = "coverage_gap"
    RISK_SCORE = "risk_score"
    MAINTENANCE_BURDEN = "maintenance_burden"
    VALUE_SCORE = "value_score"


class TestCategory(Enum):
    """Test categorization"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    ACCEPTANCE = "acceptance"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class TestProfile:
    """Comprehensive test profile"""
    test_id: str
    category: TestCategory
    lines_of_code: int
    execution_time: float
    failure_rate: float
    coverage_contribution: float
    defects_found: int
    maintenance_cost: float
    last_modified: float
    dependencies: Set[str] = field(default_factory=set)


@dataclass
class CoverageGap:
    """Coverage gap analysis"""
    file_path: str
    uncovered_lines: List[int]
    uncovered_functions: List[str]
    risk_level: str
    gap_size: int
    suggested_tests: List[str]


@dataclass
class RedundancyGroup:
    """Group of redundant tests"""
    group_id: str
    test_ids: List[str]
    overlap_percentage: float
    recommendation: str
    potential_savings: float


@dataclass
class AnalyticsReport:
    """Complete analytics report"""
    overall_effectiveness: float
    total_tests_analyzed: int
    coverage_gaps: List[CoverageGap]
    redundant_tests: List[RedundancyGroup]
    high_value_tests: List[str]
    low_value_tests: List[str]
    recommendations: List[str]
    risk_assessment: Dict[str, Any]


class EffectivenessScorer:
    """Scores test effectiveness using multiple criteria"""
    
    def score_test_effectiveness(self, profile: TestProfile, 
                               codebase_metrics: Dict[str, Any]) -> float:
        """Calculate test effectiveness score (0-100)"""
        scores = {
            'defect_detection': self._score_defect_detection(profile),
            'coverage_contribution': self._score_coverage_contribution(profile),
            'execution_efficiency': self._score_execution_efficiency(profile),
            'maintenance_burden': self._score_maintenance_burden(profile),
            'reliability': self._score_reliability(profile)
        }
        
        weights = {
            'defect_detection': 0.30,
            'coverage_contribution': 0.25,
            'execution_efficiency': 0.20,
            'maintenance_burden': 0.15,
            'reliability': 0.10
        }
        
        weighted_score = sum(scores[metric] * weights[metric] 
                           for metric in scores)
        
        return round(weighted_score, 2)
    
    def _score_defect_detection(self, profile: TestProfile) -> float:
        """Score based on defect detection capability"""
        # Higher score for tests that find more defects
        if profile.defects_found == 0:
            return 20.0  # Base score for no known defects
        
        # Logarithmic scale for defect detection
        score = min(100, 40 + math.log10(profile.defects_found) * 30)
        return score
    
    def _score_coverage_contribution(self, profile: TestProfile) -> float:
        """Score based on coverage contribution"""
        # Direct mapping of coverage percentage
        return min(100, profile.coverage_contribution * 100)
    
    def _score_execution_efficiency(self, profile: TestProfile) -> float:
        """Score based on execution time efficiency"""
        # Faster tests get higher scores
        if profile.execution_time <= 0.1:  # Very fast
            return 100
        elif profile.execution_time <= 1.0:  # Fast
            return 80
        elif profile.execution_time <= 5.0:  # Moderate
            return 60
        elif profile.execution_time <= 30.0:  # Slow
            return 40
        else:  # Very slow
            return 20
    
    def _score_maintenance_burden(self, profile: TestProfile) -> float:
        """Score based on maintenance burden (inverted)"""
        # Lower maintenance cost = higher score
        if profile.maintenance_cost <= 1.0:
            return 100
        elif profile.maintenance_cost <= 3.0:
            return 80
        elif profile.maintenance_cost <= 6.0:
            return 60
        elif profile.maintenance_cost <= 10.0:
            return 40
        else:
            return 20
    
    def _score_reliability(self, profile: TestProfile) -> float:
        """Score based on test reliability"""
        # Lower failure rate = higher score
        reliability = 1.0 - profile.failure_rate
        return reliability * 100


class CoverageAnalyzer:
    """Analyzes coverage gaps and identifies testing opportunities"""
    
    def analyze_coverage_gaps(self, coverage_data: Dict[str, Any],
                             codebase_analysis: Dict[str, Any]) -> List[CoverageGap]:
        """Identify and analyze coverage gaps"""
        gaps = []
        
        for file_path, file_coverage in coverage_data.items():
            uncovered_lines = file_coverage.get('uncovered_lines', [])
            total_lines = file_coverage.get('total_lines', 0)
            
            if uncovered_lines and total_lines > 0:
                gap_percentage = len(uncovered_lines) / total_lines
                
                # Analyze gap characteristics
                gap = CoverageGap(
                    file_path=file_path,
                    uncovered_lines=uncovered_lines,
                    uncovered_functions=self._identify_uncovered_functions(
                        file_path, uncovered_lines, codebase_analysis
                    ),
                    risk_level=self._assess_gap_risk(gap_percentage, file_path),
                    gap_size=len(uncovered_lines),
                    suggested_tests=self._suggest_tests_for_gap(
                        file_path, uncovered_lines, codebase_analysis
                    )
                )
                gaps.append(gap)
        
        # Sort by risk level and gap size
        gaps.sort(key=lambda g: (
            {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[g.risk_level],
            g.gap_size
        ), reverse=True)
        
        return gaps
    
    def _identify_uncovered_functions(self, file_path: str, uncovered_lines: List[int],
                                    codebase_analysis: Dict[str, Any]) -> List[str]:
        """Identify functions with uncovered lines"""
        functions = codebase_analysis.get(file_path, {}).get('functions', [])
        uncovered_functions = []
        
        for func in functions:
            func_lines = range(func.get('start_line', 0), func.get('end_line', 0) + 1)
            if any(line in uncovered_lines for line in func_lines):
                uncovered_functions.append(func.get('name', 'unknown'))
        
        return uncovered_functions
    
    def _assess_gap_risk(self, gap_percentage: float, file_path: str) -> str:
        """Assess risk level of coverage gap"""
        # Critical files (main, core, api, etc.)
        critical_patterns = ['main', 'core', 'api', 'auth', 'security', 'payment']
        is_critical = any(pattern in file_path.lower() for pattern in critical_patterns)
        
        if is_critical:
            if gap_percentage > 0.3:
                return 'critical'
            elif gap_percentage > 0.1:
                return 'high'
            else:
                return 'medium'
        else:
            if gap_percentage > 0.5:
                return 'high'
            elif gap_percentage > 0.2:
                return 'medium'
            else:
                return 'low'
    
    def _suggest_tests_for_gap(self, file_path: str, uncovered_lines: List[int],
                              codebase_analysis: Dict[str, Any]) -> List[str]:
        """Suggest specific tests for coverage gaps"""
        suggestions = []
        
        # Analyze uncovered code patterns
        file_analysis = codebase_analysis.get(file_path, {})
        
        # Suggest based on uncovered functions
        uncovered_functions = self._identify_uncovered_functions(
            file_path, uncovered_lines, codebase_analysis
        )
        
        for func in uncovered_functions:
            suggestions.append(f"Unit test for function '{func}'")
        
        # Suggest based on code patterns
        if file_analysis.get('has_error_handling'):
            suggestions.append("Error handling test cases")
        if file_analysis.get('has_conditionals'):
            suggestions.append("Branch coverage tests")
        if file_analysis.get('has_loops'):
            suggestions.append("Loop boundary tests")
        
        return suggestions[:5]  # Limit to top 5 suggestions


class RedundancyDetector:
    """Detects redundant and overlapping tests"""
    
    def detect_redundant_tests(self, test_profiles: List[TestProfile]) -> List[RedundancyGroup]:
        """Detect groups of redundant tests"""
        redundant_groups = []
        analyzed = set()
        
        for i, test1 in enumerate(test_profiles):
            if test1.test_id in analyzed:
                continue
                
            similar_tests = [test1.test_id]
            
            for j, test2 in enumerate(test_profiles[i+1:], i+1):
                if test2.test_id in analyzed:
                    continue
                
                similarity = self._calculate_test_similarity(test1, test2)
                
                if similarity > 0.8:  # 80% similarity threshold
                    similar_tests.append(test2.test_id)
                    analyzed.add(test2.test_id)
            
            if len(similar_tests) > 1:
                overlap_percentage = self._calculate_overlap_percentage(
                    [t for t in test_profiles if t.test_id in similar_tests]
                )
                
                group = RedundancyGroup(
                    group_id=f"redundant_group_{len(redundant_groups)}",
                    test_ids=similar_tests,
                    overlap_percentage=overlap_percentage,
                    recommendation=self._generate_redundancy_recommendation(similar_tests),
                    potential_savings=self._calculate_potential_savings(
                        [t for t in test_profiles if t.test_id in similar_tests]
                    )
                )
                redundant_groups.append(group)
                analyzed.update(similar_tests)
        
        return redundant_groups
    
    def _calculate_test_similarity(self, test1: TestProfile, test2: TestProfile) -> float:
        """Calculate similarity between two tests"""
        # Multiple similarity factors
        factors = {
            'category': 1.0 if test1.category == test2.category else 0.0,
            'coverage_overlap': min(test1.coverage_contribution, test2.coverage_contribution) / 
                              max(test1.coverage_contribution, test2.coverage_contribution) 
                              if max(test1.coverage_contribution, test2.coverage_contribution) > 0 else 0,
            'execution_time': 1.0 - abs(test1.execution_time - test2.execution_time) / 
                            max(test1.execution_time, test2.execution_time, 1.0),
            'dependencies': len(test1.dependencies & test2.dependencies) / 
                          len(test1.dependencies | test2.dependencies) 
                          if test1.dependencies | test2.dependencies else 0
        }
        
        weights = {'category': 0.3, 'coverage_overlap': 0.4, 'execution_time': 0.2, 'dependencies': 0.1}
        
        similarity = sum(factors[factor] * weights[factor] for factor in factors)
        return similarity
    
    def _calculate_overlap_percentage(self, tests: List[TestProfile]) -> float:
        """Calculate coverage overlap percentage"""
        if len(tests) < 2:
            return 0.0
        
        # Simplified overlap calculation
        coverage_values = [t.coverage_contribution for t in tests]
        return min(coverage_values) / max(coverage_values) * 100 if max(coverage_values) > 0 else 0
    
    def _generate_redundancy_recommendation(self, test_ids: List[str]) -> str:
        """Generate recommendation for redundant tests"""
        if len(test_ids) == 2:
            return f"Consider merging tests {test_ids[0]} and {test_ids[1]}"
        else:
            return f"Consider consolidating {len(test_ids)} similar tests into a single comprehensive test"
    
    def _calculate_potential_savings(self, tests: List[TestProfile]) -> float:
        """Calculate potential savings from removing redundancy"""
        total_time = sum(t.execution_time for t in tests)
        total_maintenance = sum(t.maintenance_cost for t in tests)
        
        # Assume we can save 70% of execution time and 50% of maintenance
        time_savings = total_time * 0.7
        maintenance_savings = total_maintenance * 0.5
        
        return time_savings + maintenance_savings


class TestAnalytics:
    """Main test analytics engine"""
    
    def __init__(self):
        self.effectiveness_scorer = EffectivenessScorer()
        self.coverage_analyzer = CoverageAnalyzer()
        self.redundancy_detector = RedundancyDetector()
    
    def analyze_test_suite(self, test_profiles: List[TestProfile],
                          coverage_data: Dict[str, Any],
                          codebase_analysis: Dict[str, Any]) -> AnalyticsReport:
        """Perform comprehensive test suite analysis"""
        
        # Score effectiveness for all tests
        effectiveness_scores = {}
        for profile in test_profiles:
            effectiveness_scores[profile.test_id] = self.effectiveness_scorer.score_test_effectiveness(
                profile, codebase_analysis
            )
        
        # Analyze coverage gaps
        coverage_gaps = self.coverage_analyzer.analyze_coverage_gaps(
            coverage_data, codebase_analysis
        )
        
        # Detect redundant tests
        redundant_tests = self.redundancy_detector.detect_redundant_tests(test_profiles)
        
        # Identify high and low value tests
        sorted_tests = sorted(effectiveness_scores.items(), key=lambda x: x[1], reverse=True)
        high_value_tests = [test_id for test_id, score in sorted_tests if score >= 80]
        low_value_tests = [test_id for test_id, score in sorted_tests if score < 40]
        
        # Calculate overall effectiveness
        overall_effectiveness = statistics.mean(effectiveness_scores.values()) if effectiveness_scores else 0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            effectiveness_scores, coverage_gaps, redundant_tests
        )
        
        # Risk assessment
        risk_assessment = self._assess_testing_risks(
            test_profiles, coverage_gaps, effectiveness_scores
        )
        
        return AnalyticsReport(
            overall_effectiveness=overall_effectiveness,
            total_tests_analyzed=len(test_profiles),
            coverage_gaps=coverage_gaps,
            redundant_tests=redundant_tests,
            high_value_tests=high_value_tests,
            low_value_tests=low_value_tests,
            recommendations=recommendations,
            risk_assessment=risk_assessment
        )
    
    def _generate_recommendations(self, effectiveness_scores: Dict[str, float],
                                coverage_gaps: List[CoverageGap],
                                redundant_tests: List[RedundancyGroup]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Coverage recommendations
        critical_gaps = [g for g in coverage_gaps if g.risk_level == 'critical']
        if critical_gaps:
            recommendations.append(f"Address {len(critical_gaps)} critical coverage gaps immediately")
        
        # Redundancy recommendations
        if redundant_tests:
            total_savings = sum(g.potential_savings for g in redundant_tests)
            recommendations.append(f"Remove test redundancy to save {total_savings:.1f} hours")
        
        # Low value test recommendations
        low_value_count = sum(1 for score in effectiveness_scores.values() if score < 40)
        if low_value_count > 0:
            recommendations.append(f"Improve or remove {low_value_count} low-value tests")
        
        # High impact recommendations
        high_impact_gaps = [g for g in coverage_gaps if g.risk_level in ['critical', 'high']]
        if len(high_impact_gaps) > 5:
            recommendations.append("Prioritize testing strategy - too many high-risk gaps")
        
        return recommendations[:5]
    
    def _assess_testing_risks(self, test_profiles: List[TestProfile],
                            coverage_gaps: List[CoverageGap],
                            effectiveness_scores: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall testing risks"""
        
        # Calculate risk factors
        avg_effectiveness = statistics.mean(effectiveness_scores.values()) if effectiveness_scores else 0
        critical_gaps = len([g for g in coverage_gaps if g.risk_level == 'critical'])
        high_failure_rate_tests = len([t for t in test_profiles if t.failure_rate > 0.1])
        
        # Overall risk score
        risk_factors = {
            'low_effectiveness': max(0, 60 - avg_effectiveness) / 60,
            'coverage_gaps': min(critical_gaps / 10, 1.0),
            'flaky_tests': min(high_failure_rate_tests / len(test_profiles), 0.5) if test_profiles else 0
        }
        
        overall_risk = sum(risk_factors.values()) / len(risk_factors)
        
        return {
            'overall_risk_score': overall_risk * 100,
            'risk_level': 'high' if overall_risk > 0.7 else 'medium' if overall_risk > 0.4 else 'low',
            'risk_factors': risk_factors,
            'mitigation_priority': self._get_mitigation_priorities(risk_factors)
        }
    
    def _get_mitigation_priorities(self, risk_factors: Dict[str, float]) -> List[str]:
        """Get prioritized list of risk mitigations"""
        priorities = []
        
        sorted_risks = sorted(risk_factors.items(), key=lambda x: x[1], reverse=True)
        
        for risk_name, risk_value in sorted_risks:
            if risk_value > 0.3:
                if risk_name == 'low_effectiveness':
                    priorities.append("Improve test effectiveness through better assertions and coverage")
                elif risk_name == 'coverage_gaps':
                    priorities.append("Address critical coverage gaps in core functionality")
                elif risk_name == 'flaky_tests':
                    priorities.append("Stabilize flaky tests to improve reliability")
        
        return priorities