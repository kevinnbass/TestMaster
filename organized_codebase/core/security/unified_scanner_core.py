"""
Unified Scanner Core - Main scanner engine for comprehensive security analysis

This module provides the core scanning engine that coordinates all security layers,
manages scan execution, and generates comprehensive reports.
"""

import asyncio
import logging
import time
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from collections import deque
from dataclasses import asdict

from .security_scan_models import (
    ScanConfiguration, UnifiedScanResult, SecurityMetrics,
    create_unified_scan_result, create_scan_configuration,
    RiskLevel, RemediationAction
)
from .security_orchestrator import SecurityLayerOrchestrator, create_security_orchestrator
from .security_correlations import SecurityCorrelationAnalyzer, summarize_correlations

logger = logging.getLogger(__name__)


class UnifiedSecurityScanner:
    """
    Main unified security scanner combining all security analysis capabilities
    
    Provides comprehensive security scanning with multi-layer analysis,
    correlation detection, and intelligent remediation recommendations.
    """
    
    def __init__(self, config: Optional[ScanConfiguration] = None):
        """
        Initialize unified security scanner
        
        Args:
            config: Scan configuration (uses defaults if not provided)
        """
        self.config = config or create_scan_configuration(target_paths=['.'])
        self.orchestrator = create_security_orchestrator(
            max_workers=self.config.parallel_workers
        )
        self.correlation_analyzer = SecurityCorrelationAnalyzer()
        
        # Scan management
        self.scan_history = deque(maxlen=100)
        self.active_scans = {}
        self.scan_cache = {}
        
        # Statistics
        self.metrics = SecurityMetrics()
        
        logger.info("Unified Security Scanner initialized")
    
    async def scan_async(self, target_path: Optional[str] = None) -> UnifiedScanResult:
        """
        Perform asynchronous unified security scan
        
        Args:
            target_path: Specific path to scan (uses config paths if not provided)
            
        Returns:
            Comprehensive scan result with findings and correlations
        """
        scan_start = time.time()
        scan_id = self._generate_scan_id(target_path)
        
        logger.info(f"Starting unified security scan {scan_id}")
        
        try:
            # Determine scan targets
            targets = self._discover_targets(target_path)
            logger.info(f"Scanning {len(targets)} targets")
            
            # Mark scan as active
            self.active_scans[scan_id] = {
                'start_time': scan_start,
                'targets': targets,
                'status': 'running'
            }
            
            # Initialize result aggregator
            aggregated_results = {
                'vulnerabilities': [],
                'compliance_violations': [],
                'quality_issues': [],
                'performance_concerns': [],
                'enhanced_findings': [],
                'layer_results': {}
            }
            
            # Scan each target
            for target in targets:
                logger.info(f"Scanning target: {target}")
                
                # Check cache if enabled
                if self.config.cache_results:
                    cached_result = self._get_cached_result(target)
                    if cached_result:
                        logger.info(f"Using cached result for {target}")
                        aggregated_results['layer_results'][target] = cached_result
                        continue
                
                # Orchestrate scan across all layers
                layer_results = await self.orchestrator.orchestrate_scan(
                    target, self.config
                )
                aggregated_results['layer_results'][target] = layer_results
                
                # Cache result if enabled
                if self.config.cache_results:
                    self._cache_result(target, layer_results)
                
                # Aggregate findings
                self._aggregate_findings(aggregated_results, layer_results)
            
            # Generate unified result
            scan_duration = time.time() - scan_start
            unified_result = self._create_unified_result(
                scan_id, scan_start, targets, aggregated_results, scan_duration
            )
            
            # Update statistics
            self._update_statistics(unified_result)
            
            # Generate report if enabled
            if self.config.generate_report:
                await self._generate_report(unified_result)
            
            # Store in history
            self.scan_history.append(unified_result)
            
            # Mark scan as complete
            self.active_scans[scan_id]['status'] = 'completed'
            self.active_scans[scan_id]['end_time'] = time.time()
            
            logger.info(f"Scan {scan_id} completed in {scan_duration:.2f} seconds")
            return unified_result
            
        except Exception as e:
            logger.error(f"Scan error: {e}")
            self.metrics.failed_scans += 1
            
            if scan_id in self.active_scans:
                self.active_scans[scan_id]['status'] = 'failed'
                self.active_scans[scan_id]['error'] = str(e)
            
            raise
    
    def scan(self, target_path: Optional[str] = None) -> UnifiedScanResult:
        """
        Perform synchronous unified security scan
        
        Args:
            target_path: Specific path to scan
            
        Returns:
            Comprehensive scan result
        """
        return asyncio.run(self.scan_async(target_path))
    
    def _generate_scan_id(self, target_path: Optional[str]) -> str:
        """Generate unique scan ID"""
        timestamp = int(time.time())
        target_hash = hashlib.md5(
            str(target_path or 'all').encode()
        ).hexdigest()[:8]
        return f"unified_{timestamp}_{target_hash}"
    
    def _discover_targets(self, specific_target: Optional[str] = None) -> List[str]:
        """
        Discover scan targets based on configuration
        
        Args:
            specific_target: Specific target to scan
            
        Returns:
            List of target paths to scan
        """
        if specific_target:
            return [specific_target]
        
        targets = []
        for base_path in self.config.target_paths:
            base_path_obj = Path(base_path)
            
            if base_path_obj.is_file():
                targets.append(str(base_path_obj))
            elif base_path_obj.is_dir():
                for pattern in self.config.file_patterns:
                    for file_path in base_path_obj.rglob(pattern):
                        # Check exclusions
                        excluded = False
                        for exclude_pattern in self.config.exclude_patterns:
                            if exclude_pattern in str(file_path):
                                excluded = True
                                break
                        
                        if not excluded:
                            targets.append(str(file_path))
        
        return targets
    
    def _aggregate_findings(
        self,
        aggregated: Dict[str, Any],
        layer_results: Dict[str, Any]
    ):
        """
        Aggregate findings from layer results
        
        Args:
            aggregated: Aggregated results to update
            layer_results: Results from security layers
        """
        # Extract vulnerabilities
        if 'vulnerabilities' in layer_results:
            aggregated['vulnerabilities'].extend(layer_results['vulnerabilities'])
        
        # Extract compliance violations
        if 'compliance_violations' in layer_results:
            aggregated['compliance_violations'].extend(
                layer_results['compliance_violations']
            )
        
        # Extract quality issues
        if 'quality_issues' in layer_results:
            aggregated['quality_issues'].extend(layer_results['quality_issues'])
        
        # Extract performance concerns
        if 'performance_concerns' in layer_results:
            aggregated['performance_concerns'].extend(
                layer_results['performance_concerns']
            )
        
        # Extract enhanced findings from intelligence layer
        if 'intelligence' in layer_results:
            intelligence = layer_results['intelligence']
            if 'enhanced_findings' in intelligence:
                aggregated['enhanced_findings'].extend(
                    intelligence['enhanced_findings']
                )
    
    def _create_unified_result(
        self,
        scan_id: str,
        timestamp: float,
        targets: List[str],
        aggregated_results: Dict[str, Any],
        scan_duration: float
    ) -> UnifiedScanResult:
        """
        Create comprehensive unified scan result
        
        Args:
            scan_id: Unique scan identifier
            timestamp: Scan start timestamp
            targets: List of scanned targets
            aggregated_results: Aggregated findings
            scan_duration: Total scan duration
            
        Returns:
            Unified scan result with all findings and analysis
        """
        # Calculate risk metrics
        overall_risk = self._calculate_overall_risk(aggregated_results)
        risk_distribution = self._calculate_risk_distribution(aggregated_results)
        
        # Extract correlations
        all_correlations = self._extract_correlations(aggregated_results)
        correlation_summary = summarize_correlations(all_correlations)
        
        # Get strongest correlations
        security_quality_corr = all_correlations.get('quality_security', {})
        security_perf_corr = all_correlations.get('performance_security', {})
        complexity_vuln_corr = all_correlations.get('complexity_vulnerability', {})
        
        # Generate remediation plan
        remediation_plan = self._generate_remediation_plan(aggregated_results)
        
        # Calculate scan metrics
        scan_metrics = {
            'files_scanned': len(targets),
            'total_findings': self._count_total_findings(aggregated_results),
            'critical_findings': self._count_critical_findings(aggregated_results),
            'scan_duration': scan_duration,
            'avg_scan_time_per_file': scan_duration / max(len(targets), 1),
            'correlation_insights': len(correlation_summary.get('key_insights', []))
        }
        
        # Calculate coverage metrics
        coverage_metrics = self._calculate_coverage_metrics(aggregated_results)
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            aggregated_results, correlation_summary
        )
        
        # Extract risk trends if available
        risk_trends = self._extract_risk_trends(aggregated_results)
        
        # Create unified result
        return create_unified_scan_result(
            scan_id=scan_id,
            target_path=', '.join(targets[:3]) if len(targets) <= 3 
                       else f"{len(targets)} files",
            timestamp=timestamp,
            scan_duration=scan_duration,
            vulnerabilities=aggregated_results['vulnerabilities'],
            compliance_violations=aggregated_results['compliance_violations'],
            quality_issues=aggregated_results['quality_issues'],
            performance_concerns=aggregated_results['performance_concerns'],
            overall_risk_score=overall_risk,
            risk_distribution=risk_distribution,
            risk_trends=risk_trends,
            security_quality_correlation=security_quality_corr,
            security_performance_correlation=security_perf_corr,
            complexity_vulnerability_correlation=complexity_vuln_corr,
            enhanced_findings=aggregated_results['enhanced_findings'],
            remediation_plan=remediation_plan,
            scan_metrics=scan_metrics,
            coverage_metrics=coverage_metrics,
            confidence_score=confidence_score,
            layer_results=aggregated_results['layer_results']
        )
    
    def _calculate_overall_risk(self, results: Dict[str, Any]) -> float:
        """Calculate overall risk score from findings"""
        risk_components = []
        
        # Vulnerability risk (40% weight)
        vuln_count = len(results.get('vulnerabilities', []))
        critical_vulns = self._count_by_severity(
            results.get('vulnerabilities', []), 'critical'
        )
        vuln_risk = min(vuln_count * 10 + critical_vulns * 20, 100)
        risk_components.append(vuln_risk * 0.4)
        
        # Compliance risk (20% weight)
        compliance_count = len(results.get('compliance_violations', []))
        compliance_risk = min(compliance_count * 15, 100)
        risk_components.append(compliance_risk * 0.2)
        
        # Quality risk (20% weight)
        quality_count = len(results.get('quality_issues', []))
        quality_risk = min(quality_count * 5, 100)
        risk_components.append(quality_risk * 0.2)
        
        # Performance risk (20% weight)
        perf_count = len(results.get('performance_concerns', []))
        perf_risk = min(perf_count * 10, 100)
        risk_components.append(perf_risk * 0.2)
        
        return sum(risk_components)
    
    def _calculate_risk_distribution(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Calculate distribution of findings by risk level"""
        distribution = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }
        
        for vuln in results.get('vulnerabilities', []):
            level = vuln.get('threat_level', 'medium')
            if level in distribution:
                distribution[level] += 1
        
        return distribution
    
    def _extract_correlations(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all correlations from layer results"""
        correlations = {}
        
        for target, layer_results in results.get('layer_results', {}).items():
            if 'correlations' in layer_results:
                for corr_type, corr_data in layer_results['correlations'].items():
                    if corr_type not in correlations or \
                       corr_data.get('correlation_strength', 0) > \
                       correlations.get(corr_type, {}).get('correlation_strength', 0):
                        correlations[corr_type] = corr_data
        
        return correlations
    
    def _generate_remediation_plan(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate prioritized remediation plan"""
        plan = {
            'immediate_actions': [],
            'short_term_actions': [],
            'long_term_actions': [],
            'estimated_effort': 0,
            'priority_order': []
        }
        
        # Immediate actions for critical vulnerabilities
        critical_items = [
            v for v in results.get('vulnerabilities', [])
            if v.get('threat_level') == 'critical'
        ]
        
        for item in critical_items[:5]:
            action = {
                'action': f"Fix {item.get('title', 'critical vulnerability')}",
                'description': item.get('description', ''),
                'estimated_hours': 4
            }
            plan['immediate_actions'].append(action)
            plan['estimated_effort'] += 4
        
        # Short-term actions for high-risk issues
        high_items = [
            v for v in results.get('vulnerabilities', [])
            if v.get('threat_level') == 'high'
        ]
        
        for item in high_items[:10]:
            action = {
                'action': f"Address {item.get('title', 'high-risk issue')}",
                'description': item.get('description', ''),
                'estimated_hours': 2
            }
            plan['short_term_actions'].append(action)
            plan['estimated_effort'] += 2
        
        # Long-term improvements
        if results.get('quality_issues'):
            plan['long_term_actions'].append({
                'action': 'Improve code quality',
                'description': f"Address {len(results['quality_issues'])} quality issues",
                'estimated_hours': len(results['quality_issues']) * 0.5
            })
        
        # Set priority order
        plan['priority_order'] = (
            [a['action'] for a in plan['immediate_actions']] +
            [a['action'] for a in plan['short_term_actions'][:3]]
        )
        
        return plan
    
    def _calculate_coverage_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate scan coverage metrics"""
        layer_results = results.get('layer_results', {})
        total_layers = len(layer_results)
        
        if total_layers == 0:
            return {
                'security_coverage': 0.0,
                'compliance_coverage': 0.0,
                'quality_coverage': 0.0,
                'performance_coverage': 0.0,
                'overall_coverage': 0.0
            }
        
        # Count successful layer executions
        successful_layers = sum(
            1 for lr in layer_results.values()
            if lr and not lr.get('error')
        )
        
        return {
            'security_coverage': 1.0 if results.get('vulnerabilities') else 0.0,
            'compliance_coverage': 1.0 if results.get('compliance_violations') else 0.0,
            'quality_coverage': 1.0 if results.get('quality_issues') else 0.0,
            'performance_coverage': 1.0 if results.get('performance_concerns') else 0.0,
            'overall_coverage': successful_layers / total_layers
        }
    
    def _calculate_confidence_score(
        self,
        results: Dict[str, Any],
        correlation_summary: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for scan results"""
        confidence_factors = []
        
        # Factor 1: Coverage (30% weight)
        coverage = results.get('coverage_metrics', {}).get('overall_coverage', 0)
        confidence_factors.append(coverage * 0.3)
        
        # Factor 2: Correlation strength (30% weight)
        avg_correlation = correlation_summary.get('average_strength', 0)
        confidence_factors.append(avg_correlation * 0.3)
        
        # Factor 3: Sample size (20% weight)
        total_findings = self._count_total_findings(results)
        sample_confidence = min(total_findings / 50, 1.0)  # Max confidence at 50+ findings
        confidence_factors.append(sample_confidence * 0.2)
        
        # Factor 4: Correlation confidence (20% weight)
        corr_confidence = correlation_summary.get('average_confidence', 0.5)
        confidence_factors.append(corr_confidence * 0.2)
        
        return sum(confidence_factors)
    
    def _extract_risk_trends(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk trends from layer results"""
        # Look for trends in layer results
        for layer_results in results.get('layer_results', {}).values():
            if 'security_trends' in layer_results:
                return layer_results['security_trends']
        
        return {}
    
    def _count_total_findings(self, results: Dict[str, Any]) -> int:
        """Count total findings across all categories"""
        return (
            len(results.get('vulnerabilities', [])) +
            len(results.get('compliance_violations', [])) +
            len(results.get('quality_issues', [])) +
            len(results.get('performance_concerns', []))
        )
    
    def _count_critical_findings(self, results: Dict[str, Any]) -> int:
        """Count critical severity findings"""
        return self._count_by_severity(
            results.get('vulnerabilities', []), 'critical'
        )
    
    def _count_by_severity(self, findings: List[Dict], severity: str) -> int:
        """Count findings by severity level"""
        return len([
            f for f in findings
            if f.get('threat_level') == severity or f.get('severity') == severity
        ])
    
    def _update_statistics(self, result: UnifiedScanResult):
        """Update scanner statistics"""
        self.metrics.total_scans += 1
        self.metrics.successful_scans += 1
        self.metrics.total_vulnerabilities += len(result.vulnerabilities)
        self.metrics.total_scan_time += result.scan_duration
        
        # Update severity counts
        for vuln in result.vulnerabilities:
            level = vuln.get('threat_level', 'medium')
            if level == 'critical':
                self.metrics.critical_findings += 1
            elif level == 'high':
                self.metrics.high_findings += 1
            elif level == 'medium':
                self.metrics.medium_findings += 1
            elif level == 'low':
                self.metrics.low_findings += 1
        
        # Update averages
        if self.metrics.successful_scans > 0:
            self.metrics.avg_scan_time = (
                self.metrics.total_scan_time / self.metrics.successful_scans
            )
            self.metrics.avg_risk_score = (
                (self.metrics.avg_risk_score * (self.metrics.successful_scans - 1) +
                 result.overall_risk_score) / self.metrics.successful_scans
            )
            self.metrics.avg_confidence_score = (
                (self.metrics.avg_confidence_score * (self.metrics.successful_scans - 1) +
                 result.confidence_score) / self.metrics.successful_scans
            )
    
    def _get_cached_result(self, target: str) -> Optional[Dict[str, Any]]:
        """Get cached scan result if available and fresh"""
        cache_key = hashlib.md5(target.encode()).hexdigest()
        
        if cache_key in self.scan_cache:
            cached = self.scan_cache[cache_key]
            # Check if cache is fresh
            if time.time() - cached['timestamp'] < self.config.cache_ttl:
                self.metrics.cache_hit_rate += 1
                return cached['result']
        
        return None
    
    def _cache_result(self, target: str, result: Dict[str, Any]):
        """Cache scan result"""
        cache_key = hashlib.md5(target.encode()).hexdigest()
        self.scan_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
    
    async def _generate_report(self, result: UnifiedScanResult):
        """Generate security scan report"""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if self.config.report_format == 'json':
                await self._generate_json_report(result, output_dir, timestamp)
            elif self.config.report_format == 'html':
                await self._generate_html_report(result, output_dir, timestamp)
            elif self.config.report_format == 'markdown':
                await self._generate_markdown_report(result, output_dir, timestamp)
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
    
    async def _generate_json_report(
        self,
        result: UnifiedScanResult,
        output_dir: Path,
        timestamp: str
    ):
        """Generate JSON format report"""
        report_file = output_dir / f"security_report_{timestamp}.json"
        
        report_data = asdict(result)
        # Convert any non-serializable objects
        report_json = json.dumps(report_data, indent=2, default=str)
        
        with open(report_file, 'w') as f:
            f.write(report_json)
        
        logger.info(f"JSON report saved to: {report_file}")
    
    async def _generate_html_report(
        self,
        result: UnifiedScanResult,
        output_dir: Path,
        timestamp: str
    ):
        """Generate HTML format report"""
        # Simplified HTML report generation
        report_file = output_dir / f"security_report_{timestamp}.html"
        
        html_content = self._create_html_content(result)
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to: {report_file}")
    
    async def _generate_markdown_report(
        self,
        result: UnifiedScanResult,
        output_dir: Path,
        timestamp: str
    ):
        """Generate Markdown format report"""
        report_file = output_dir / f"security_report_{timestamp}.md"
        
        md_content = self._create_markdown_content(result)
        
        with open(report_file, 'w') as f:
            f.write(md_content)
        
        logger.info(f"Markdown report saved to: {report_file}")
    
    def _create_html_content(self, result: UnifiedScanResult) -> str:
        """Create HTML report content"""
        risk_class = 'critical' if result.overall_risk_score >= 80 else \
                    'high' if result.overall_risk_score >= 60 else 'medium'
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Security Scan Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #2c3e50; color: white; padding: 20px; }}
        .risk-score {{ font-size: 24px; font-weight: bold; }}
        .critical {{ color: #e74c3c; }}
        .high {{ color: #e67e22; }}
        .medium {{ color: #f39c12; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Unified Security Scan Report</h1>
        <p>Scan ID: {result.scan_id}</p>
        <p>Duration: {result.scan_duration:.2f} seconds</p>
    </div>
    
    <div class="section">
        <h2>Risk Assessment</h2>
        <p class="risk-score {risk_class}">
            Overall Risk: {result.overall_risk_score:.1f}/100
        </p>
        <p>Confidence: {result.confidence_score:.2%}</p>
    </div>
    
    <div class="section">
        <h2>Findings</h2>
        <ul>
            <li>Vulnerabilities: {len(result.vulnerabilities)}</li>
            <li>Compliance Violations: {len(result.compliance_violations)}</li>
            <li>Quality Issues: {len(result.quality_issues)}</li>
            <li>Performance Concerns: {len(result.performance_concerns)}</li>
        </ul>
    </div>
</body>
</html>"""
    
    def _create_markdown_content(self, result: UnifiedScanResult) -> str:
        """Create Markdown report content"""
        return f"""# Security Scan Report

## Scan Information
- **Scan ID**: {result.scan_id}
- **Duration**: {result.scan_duration:.2f} seconds
- **Target**: {result.target_path}

## Risk Assessment
- **Overall Risk Score**: {result.overall_risk_score:.1f}/100
- **Confidence Score**: {result.confidence_score:.2%}

## Findings Summary
- Vulnerabilities: {len(result.vulnerabilities)}
- Compliance Violations: {len(result.compliance_violations)}
- Quality Issues: {len(result.quality_issues)}
- Performance Concerns: {len(result.performance_concerns)}

## Risk Distribution
- Critical: {result.risk_distribution.get('critical', 0)}
- High: {result.risk_distribution.get('high', 0)}
- Medium: {result.risk_distribution.get('medium', 0)}
- Low: {result.risk_distribution.get('low', 0)}

## Remediation Plan
**Estimated Effort**: {result.remediation_plan.get('estimated_effort', 0):.1f} hours

### Immediate Actions
{chr(10).join(f"- {action['action']}" for action in result.remediation_plan.get('immediate_actions', [])[:5])}
"""
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scanner statistics"""
        return asdict(self.metrics)
    
    def get_active_scans(self) -> Dict[str, Any]:
        """Get information about active scans"""
        return self.active_scans.copy()
    
    async def shutdown(self):
        """Shutdown scanner and cleanup resources"""
        logger.info("Shutting down Unified Security Scanner")
        await self.orchestrator.shutdown()


# Factory function
def create_unified_scanner(config: Optional[ScanConfiguration] = None) -> UnifiedSecurityScanner:
    """
    Create and configure unified security scanner
    
    Args:
        config: Scanner configuration
        
    Returns:
        Configured scanner instance
    """
    return UnifiedSecurityScanner(config)