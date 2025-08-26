"""
Security Correlations - Advanced correlation algorithms for multi-layer security analysis

This module provides sophisticated correlation algorithms that identify relationships between:
- Security vulnerabilities and code quality
- Performance issues and security risks
- Code complexity and vulnerability density
- Compliance violations and architectural patterns
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of correlation analysis"""
    correlation_type: str
    correlation_strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    primary_metric: str
    secondary_metric: str
    primary_value: float
    secondary_value: float
    insight: str
    recommendations: List[str]
    supporting_evidence: Dict[str, Any]


class SecurityCorrelationAnalyzer:
    """
    Advanced correlation analyzer for security findings
    
    Identifies patterns and relationships between different security aspects
    to provide deeper insights and more accurate risk assessment.
    """
    
    def __init__(self):
        self.correlation_cache = {}
        self.correlation_thresholds = {
            'strong': 0.7,
            'moderate': 0.4,
            'weak': 0.2
        }
        
    def analyze_all_correlations(self, layer_results: Dict[str, Any]) -> Dict[str, CorrelationResult]:
        """
        Analyze all correlations between different security layers
        
        Args:
            layer_results: Results from all security analysis layers
            
        Returns:
            Dictionary of correlation results by type
        """
        correlations = {}
        
        # Quality-Security correlation
        if 'quality_analysis' in layer_results and 'realtime_analysis' in layer_results:
            correlations['quality_security'] = self.calculate_quality_security_correlation(
                layer_results['quality_analysis'],
                layer_results['realtime_analysis']
            )
        
        # Performance-Security correlation
        if 'performance_profile' in layer_results and 'realtime_analysis' in layer_results:
            correlations['performance_security'] = self.calculate_performance_security_correlation(
                layer_results['performance_profile'],
                layer_results['realtime_analysis']
            )
        
        # Complexity-Vulnerability correlation
        if 'classical_analysis' in layer_results and 'realtime_analysis' in layer_results:
            correlations['complexity_vulnerability'] = self.calculate_complexity_vulnerability_correlation(
                layer_results['classical_analysis'],
                layer_results['realtime_analysis']
            )
        
        # Compliance-Architecture correlation
        if 'compliance_results' in layer_results and 'classical_analysis' in layer_results:
            correlations['compliance_architecture'] = self.calculate_compliance_architecture_correlation(
                layer_results['compliance_results'],
                layer_results['classical_analysis']
            )
        
        # Cross-layer vulnerability correlation
        correlations['cross_layer'] = self.calculate_cross_layer_correlation(layer_results)
        
        return correlations
    
    def calculate_quality_security_correlation(
        self, 
        quality_data: Dict[str, Any],
        security_data: Dict[str, Any]
    ) -> CorrelationResult:
        """
        Calculate correlation between code quality and security vulnerabilities
        
        Theory: Poor code quality often leads to security vulnerabilities
        """
        try:
            # Extract metrics
            quality_score = quality_data.get('overall_score', 50.0)
            quality_issues = quality_data.get('metrics', [])
            security_alerts = security_data.get('alerts', [])
            
            # Count critical quality issues
            critical_quality = len([q for q in quality_issues 
                                   if q.get('severity') in ['error', 'critical']])
            
            # Count security vulnerabilities by severity
            critical_security = len([s for s in security_alerts 
                                    if s.get('threat_level') == 'critical'])
            high_security = len([s for s in security_alerts 
                               if s.get('threat_level') == 'high'])
            
            # Calculate correlation strength
            # Lower quality score correlates with more security issues
            quality_factor = (100 - quality_score) / 100
            security_factor = min((critical_security * 2 + high_security) / 10, 1.0)
            
            # Apply correlation algorithm
            correlation_strength = self._calculate_correlation_coefficient(
                quality_factor, security_factor, critical_quality
            )
            
            # Determine insight
            if correlation_strength > self.correlation_thresholds['strong']:
                insight = "Strong correlation: Poor code quality directly contributes to security vulnerabilities"
                recommendations = [
                    "Prioritize code quality improvements",
                    "Implement mandatory code reviews",
                    "Enable automated quality gates",
                    "Refactor high-complexity code sections"
                ]
            elif correlation_strength > self.correlation_thresholds['moderate']:
                insight = "Moderate correlation: Code quality issues increase security risk"
                recommendations = [
                    "Address critical quality issues",
                    "Improve test coverage",
                    "Standardize coding practices"
                ]
            else:
                insight = "Weak correlation: Quality and security issues appear independent"
                recommendations = [
                    "Continue monitoring both aspects",
                    "Investigate specific vulnerability sources"
                ]
            
            return CorrelationResult(
                correlation_type='quality_security',
                correlation_strength=correlation_strength,
                confidence=self._calculate_confidence(len(security_alerts), len(quality_issues)),
                primary_metric='quality_score',
                secondary_metric='security_vulnerabilities',
                primary_value=quality_score,
                secondary_value=len(security_alerts),
                insight=insight,
                recommendations=recommendations,
                supporting_evidence={
                    'critical_quality_issues': critical_quality,
                    'critical_security_issues': critical_security,
                    'quality_factor': quality_factor,
                    'security_factor': security_factor
                }
            )
            
        except Exception as e:
            logger.error(f"Quality-security correlation error: {e}")
            return self._create_error_correlation('quality_security')
    
    def calculate_performance_security_correlation(
        self,
        performance_data: Dict[str, Any],
        security_data: Dict[str, Any]
    ) -> CorrelationResult:
        """
        Calculate correlation between performance issues and security vulnerabilities
        
        Theory: Performance bottlenecks may indicate security problems (e.g., resource exhaustion attacks)
        """
        try:
            # Extract metrics
            execution_time = performance_data.get('duration', 1.0)
            memory_usage = performance_data.get('memory_usage', {}).get('delta_mb', 0)
            cpu_usage = performance_data.get('cpu_usage', {}).get('peak_percent', 0)
            security_alerts = security_data.get('alerts', [])
            
            # Identify performance anomalies
            perf_anomalies = 0
            if execution_time > 5.0:  # Slow execution
                perf_anomalies += 1
            if memory_usage > 500:  # High memory usage
                perf_anomalies += 1
            if cpu_usage > 80:  # High CPU usage
                perf_anomalies += 1
            
            # Check for specific security patterns
            dos_vulnerabilities = len([s for s in security_alerts 
                                      if 'dos' in s.get('title', '').lower() or
                                      'denial' in s.get('title', '').lower()])
            
            resource_vulnerabilities = len([s for s in security_alerts
                                          if 'resource' in s.get('title', '').lower() or
                                          'memory' in s.get('title', '').lower()])
            
            # Calculate correlation
            perf_factor = min(perf_anomalies / 3, 1.0)
            security_factor = min((dos_vulnerabilities + resource_vulnerabilities) / 5, 1.0)
            
            correlation_strength = self._calculate_weighted_correlation(
                perf_factor, security_factor, 
                weights={'performance': 0.4, 'security': 0.6}
            )
            
            # Add bonus correlation for specific patterns
            if perf_anomalies > 0 and (dos_vulnerabilities > 0 or resource_vulnerabilities > 0):
                correlation_strength = min(correlation_strength * 1.3, 1.0)
            
            # Generate insights
            if correlation_strength > self.correlation_thresholds['strong']:
                insight = "Strong correlation: Performance issues indicate potential security vulnerabilities"
                recommendations = [
                    "Investigate resource exhaustion vulnerabilities",
                    "Implement rate limiting and throttling",
                    "Add resource usage monitoring",
                    "Review algorithmic complexity"
                ]
            elif correlation_strength > self.correlation_thresholds['moderate']:
                insight = "Moderate correlation: Some performance patterns suggest security risks"
                recommendations = [
                    "Profile resource-intensive operations",
                    "Add timeout mechanisms",
                    "Monitor for anomalous resource usage"
                ]
            else:
                insight = "Weak correlation: Performance and security appear independent"
                recommendations = [
                    "Continue standard monitoring",
                    "Focus on individual optimizations"
                ]
            
            return CorrelationResult(
                correlation_type='performance_security',
                correlation_strength=correlation_strength,
                confidence=self._calculate_confidence(len(security_alerts), perf_anomalies),
                primary_metric='performance_anomalies',
                secondary_metric='security_vulnerabilities',
                primary_value=perf_anomalies,
                secondary_value=len(security_alerts),
                insight=insight,
                recommendations=recommendations,
                supporting_evidence={
                    'execution_time': execution_time,
                    'memory_usage_mb': memory_usage,
                    'cpu_usage_percent': cpu_usage,
                    'dos_vulnerabilities': dos_vulnerabilities,
                    'resource_vulnerabilities': resource_vulnerabilities
                }
            )
            
        except Exception as e:
            logger.error(f"Performance-security correlation error: {e}")
            return self._create_error_correlation('performance_security')
    
    def calculate_complexity_vulnerability_correlation(
        self,
        classical_data: Dict[str, Any],
        security_data: Dict[str, Any]
    ) -> CorrelationResult:
        """
        Calculate correlation between code complexity and vulnerability density
        
        Theory: Complex code is more likely to contain vulnerabilities
        """
        try:
            # Extract complexity metrics
            complexity_metrics = classical_data.get('complexity_metrics', {})
            cyclomatic = complexity_metrics.get('cyclomatic_complexity', 1)
            cognitive = complexity_metrics.get('cognitive_complexity', 1)
            nesting_depth = complexity_metrics.get('max_nesting_depth', 0)
            
            # Extract vulnerability data
            security_alerts = security_data.get('alerts', [])
            critical_vulns = len([s for s in security_alerts 
                                 if s.get('threat_level') == 'critical'])
            
            # Calculate complexity factor
            complexity_score = (cyclomatic / 10 + cognitive / 15 + nesting_depth / 5) / 3
            complexity_factor = min(complexity_score, 1.0)
            
            # Calculate vulnerability density
            loc = classical_data.get('metrics', {}).get('lines_of_code', 100)
            vulnerability_density = len(security_alerts) / max(loc / 1000, 0.1)  # Vulns per KLOC
            vuln_factor = min(vulnerability_density / 10, 1.0)
            
            # Calculate correlation with emphasis on critical vulnerabilities
            base_correlation = self._calculate_correlation_coefficient(
                complexity_factor, vuln_factor, critical_vulns
            )
            
            # Adjust for specific patterns
            if cyclomatic > 15 and critical_vulns > 0:
                correlation_strength = min(base_correlation * 1.2, 1.0)
            else:
                correlation_strength = base_correlation
            
            # Generate insights
            if correlation_strength > self.correlation_thresholds['strong']:
                insight = "Strong correlation: High complexity directly increases vulnerability risk"
                recommendations = [
                    "Refactor complex functions (cyclomatic > 10)",
                    "Reduce nesting depth",
                    "Split large functions into smaller units",
                    "Increase test coverage for complex code",
                    "Prioritize security reviews for complex modules"
                ]
            elif correlation_strength > self.correlation_thresholds['moderate']:
                insight = "Moderate correlation: Complexity contributes to security risk"
                recommendations = [
                    "Monitor complexity metrics",
                    "Review high-complexity functions",
                    "Implement complexity limits in CI/CD"
                ]
            else:
                insight = "Weak correlation: Vulnerabilities not strongly tied to complexity"
                recommendations = [
                    "Look for other vulnerability sources",
                    "Review security practices"
                ]
            
            return CorrelationResult(
                correlation_type='complexity_vulnerability',
                correlation_strength=correlation_strength,
                confidence=self._calculate_confidence(len(security_alerts), loc),
                primary_metric='code_complexity',
                secondary_metric='vulnerability_density',
                primary_value=complexity_score,
                secondary_value=vulnerability_density,
                insight=insight,
                recommendations=recommendations,
                supporting_evidence={
                    'cyclomatic_complexity': cyclomatic,
                    'cognitive_complexity': cognitive,
                    'max_nesting_depth': nesting_depth,
                    'vulnerabilities_per_kloc': vulnerability_density,
                    'critical_vulnerabilities': critical_vulns
                }
            )
            
        except Exception as e:
            logger.error(f"Complexity-vulnerability correlation error: {e}")
            return self._create_error_correlation('complexity_vulnerability')
    
    def calculate_compliance_architecture_correlation(
        self,
        compliance_data: Dict[str, Any],
        classical_data: Dict[str, Any]
    ) -> CorrelationResult:
        """
        Calculate correlation between compliance violations and architectural patterns
        
        Theory: Poor architecture often leads to compliance issues
        """
        try:
            # Extract compliance metrics
            compliance_violations = 0
            for standard, result in compliance_data.items():
                if not result.get('compliant', False):
                    compliance_violations += len(result.get('violations', []))
            
            # Extract architectural metrics
            architecture_metrics = classical_data.get('architecture_metrics', {})
            coupling = architecture_metrics.get('coupling', 0)
            cohesion = architecture_metrics.get('cohesion', 1)
            modularity = architecture_metrics.get('modularity_score', 0.5)
            
            # Calculate architecture quality factor
            arch_factor = (coupling / 10 + (1 - cohesion) + (1 - modularity)) / 3
            arch_factor = min(arch_factor, 1.0)
            
            # Calculate compliance factor
            compliance_factor = min(compliance_violations / 20, 1.0)
            
            # Calculate correlation
            correlation_strength = self._calculate_weighted_correlation(
                arch_factor, compliance_factor,
                weights={'architecture': 0.5, 'compliance': 0.5}
            )
            
            # Generate insights
            if correlation_strength > self.correlation_thresholds['strong']:
                insight = "Strong correlation: Architectural issues cause compliance violations"
                recommendations = [
                    "Redesign high-coupling modules",
                    "Implement security-by-design principles",
                    "Establish architectural compliance guidelines",
                    "Automate compliance checking in CI/CD"
                ]
            else:
                insight = "Moderate correlation: Some architectural patterns affect compliance"
                recommendations = [
                    "Review architectural decisions",
                    "Document compliance requirements"
                ]
            
            return CorrelationResult(
                correlation_type='compliance_architecture',
                correlation_strength=correlation_strength,
                confidence=0.75,
                primary_metric='architecture_quality',
                secondary_metric='compliance_violations',
                primary_value=1 - arch_factor,
                secondary_value=compliance_violations,
                insight=insight,
                recommendations=recommendations,
                supporting_evidence={
                    'coupling_score': coupling,
                    'cohesion_score': cohesion,
                    'modularity_score': modularity,
                    'total_violations': compliance_violations
                }
            )
            
        except Exception as e:
            logger.error(f"Compliance-architecture correlation error: {e}")
            return self._create_error_correlation('compliance_architecture')
    
    def calculate_cross_layer_correlation(self, layer_results: Dict[str, Any]) -> CorrelationResult:
        """
        Calculate correlation across all security layers for holistic insight
        """
        try:
            total_findings = 0
            layer_contributions = {}
            
            # Aggregate findings from all layers
            for layer_name, layer_data in layer_results.items():
                if isinstance(layer_data, dict):
                    findings = 0
                    if 'alerts' in layer_data:
                        findings += len(layer_data['alerts'])
                    if 'violations' in layer_data:
                        findings += len(layer_data['violations'])
                    if 'issues' in layer_data:
                        findings += len(layer_data['issues'])
                    
                    layer_contributions[layer_name] = findings
                    total_findings += findings
            
            # Calculate distribution entropy (measure of concentration)
            if total_findings > 0:
                entropy = 0
                for findings in layer_contributions.values():
                    if findings > 0:
                        prob = findings / total_findings
                        entropy -= prob * math.log2(prob)
                
                # Normalize entropy
                max_entropy = math.log2(len(layer_contributions)) if layer_contributions else 1
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            else:
                normalized_entropy = 0
            
            # High entropy means findings are distributed across layers (systemic issues)
            # Low entropy means findings are concentrated (specific issues)
            
            if normalized_entropy > 0.7:
                insight = "Systemic issues: Problems detected across multiple layers"
                recommendations = [
                    "Conduct comprehensive security review",
                    "Implement organization-wide security training",
                    "Review and update security policies"
                ]
            elif normalized_entropy > 0.4:
                insight = "Mixed issues: Some systemic patterns, some isolated problems"
                recommendations = [
                    "Focus on high-impact areas first",
                    "Implement targeted improvements"
                ]
            else:
                insight = "Isolated issues: Problems concentrated in specific areas"
                recommendations = [
                    "Target specific problem areas",
                    "Implement focused remediation"
                ]
            
            return CorrelationResult(
                correlation_type='cross_layer',
                correlation_strength=normalized_entropy,
                confidence=0.9,
                primary_metric='finding_distribution',
                secondary_metric='total_findings',
                primary_value=normalized_entropy,
                secondary_value=total_findings,
                insight=insight,
                recommendations=recommendations,
                supporting_evidence={
                    'layer_contributions': layer_contributions,
                    'distribution_entropy': entropy,
                    'normalized_entropy': normalized_entropy
                }
            )
            
        except Exception as e:
            logger.error(f"Cross-layer correlation error: {e}")
            return self._create_error_correlation('cross_layer')
    
    # Helper methods
    def _calculate_correlation_coefficient(self, factor1: float, factor2: float, 
                                          boost_factor: int = 0) -> float:
        """Calculate correlation coefficient with optional boost"""
        base_correlation = (factor1 + factor2) / 2
        boost = min(boost_factor * 0.1, 0.3)
        return min(base_correlation + boost, 1.0)
    
    def _calculate_weighted_correlation(self, factor1: float, factor2: float,
                                       weights: Dict[str, float]) -> float:
        """Calculate weighted correlation between factors"""
        weight_values = list(weights.values())
        return factor1 * weight_values[0] + factor2 * weight_values[1]
    
    def _calculate_confidence(self, sample_size1: int, sample_size2: int) -> float:
        """Calculate confidence based on sample sizes"""
        min_samples = min(sample_size1, sample_size2)
        if min_samples >= 50:
            return 0.95
        elif min_samples >= 20:
            return 0.85
        elif min_samples >= 10:
            return 0.75
        elif min_samples >= 5:
            return 0.65
        else:
            return 0.5
    
    def _create_error_correlation(self, correlation_type: str) -> CorrelationResult:
        """Create error correlation result"""
        return CorrelationResult(
            correlation_type=correlation_type,
            correlation_strength=0.0,
            confidence=0.0,
            primary_metric='error',
            secondary_metric='error',
            primary_value=0.0,
            secondary_value=0.0,
            insight='Correlation analysis failed',
            recommendations=['Review input data', 'Check layer results'],
            supporting_evidence={'error': True}
        )


def get_strongest_correlation(correlations: Dict[str, CorrelationResult]) -> Optional[CorrelationResult]:
    """Get the strongest correlation from a set of correlations"""
    if not correlations:
        return None
    
    return max(correlations.values(), key=lambda c: c.correlation_strength)


def summarize_correlations(correlations: Dict[str, CorrelationResult]) -> Dict[str, Any]:
    """Summarize correlation analysis results"""
    summary = {
        'total_correlations': len(correlations),
        'strong_correlations': [],
        'moderate_correlations': [],
        'weak_correlations': [],
        'average_strength': 0.0,
        'average_confidence': 0.0,
        'key_insights': [],
        'priority_recommendations': []
    }
    
    total_strength = 0.0
    total_confidence = 0.0
    
    for corr_type, correlation in correlations.items():
        total_strength += correlation.correlation_strength
        total_confidence += correlation.confidence
        
        if correlation.correlation_strength > 0.7:
            summary['strong_correlations'].append(corr_type)
            summary['key_insights'].append(correlation.insight)
            summary['priority_recommendations'].extend(correlation.recommendations[:2])
        elif correlation.correlation_strength > 0.4:
            summary['moderate_correlations'].append(corr_type)
        else:
            summary['weak_correlations'].append(corr_type)
    
    if correlations:
        summary['average_strength'] = total_strength / len(correlations)
        summary['average_confidence'] = total_confidence / len(correlations)
    
    # Remove duplicate recommendations
    summary['priority_recommendations'] = list(set(summary['priority_recommendations']))
    
    return summary