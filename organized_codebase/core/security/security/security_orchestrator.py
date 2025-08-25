"""
Security Layer Orchestrator - Coordinates all security analysis layers

This module orchestrates the execution of multiple security analysis layers in parallel,
managing dependencies, resource allocation, and result aggregation.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
import time

from .security_scan_models import (
    ScanConfiguration, LayerResult, SecurityFinding,
    ComplianceViolation, QualityIssue, PerformanceConcern
)
from .security_correlations import SecurityCorrelationAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class LayerConfiguration:
    """Configuration for individual security layer"""
    name: str
    enabled: bool
    timeout: int
    priority: int
    dependencies: List[str]
    retry_count: int = 1
    async_execution: bool = True


class SecurityLayerOrchestrator:
    """
    Orchestrates execution of all security analysis layers
    
    Manages parallel execution, dependency resolution, and result aggregation
    for comprehensive security scanning across multiple analysis dimensions.
    """
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize orchestrator with security layers
        
        Args:
            max_workers: Maximum parallel workers for layer execution
        """
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.correlation_analyzer = SecurityCorrelationAnalyzer()
        
        # Layer registry - would be populated with actual layer instances
        self.layers = {}
        self._initialize_mock_layers()
        
        # Execution tracking
        self.active_executions = {}
        self.execution_results = {}
        self.execution_errors = {}
        
        logger.info(f"Security orchestrator initialized with {max_workers} workers")
    
    def _initialize_mock_layers(self):
        """Initialize mock security layers for standalone operation"""
        # In production, these would be actual security analysis components
        self.layers = {
            'base_scanner': None,  # Would be UniversalSecurityScanner instance
            'compliance': None,    # Would be ComplianceFramework instance
            'realtime_monitor': None,  # Would be RealtimeSecurityMonitor instance
            'intelligence_agent': None,  # Would be SecurityIntelligenceAgent instance
            'quality_monitor': None,  # Would be CodeQualityMonitor instance
            'performance_profiler': None,  # Would be PerformanceProfiler instance
            'metrics_collector': None,  # Would be MetricsCollector instance
            'classical_integrator': None,  # Would be ClassicalAnalysisIntegrator instance
        }
    
    async def orchestrate_scan(
        self, 
        target_path: str,
        config: ScanConfiguration
    ) -> Dict[str, Any]:
        """
        Orchestrate security scan across all enabled layers
        
        Args:
            target_path: Path to scan target
            config: Scan configuration
            
        Returns:
            Aggregated results from all layers with correlations
        """
        scan_start = time.time()
        scan_id = f"orchestrated_{int(scan_start)}_{hash(target_path)}"
        
        logger.info(f"Starting orchestrated scan {scan_id} for {target_path}")
        
        try:
            # Determine which layers to execute based on configuration
            layers_to_execute = self._determine_execution_plan(config)
            
            # Execute layers in parallel where possible
            layer_results = await self._execute_layers(
                layers_to_execute, target_path, config
            )
            
            # Aggregate results from all layers
            aggregated_results = self._aggregate_layer_results(layer_results)
            
            # Perform correlation analysis
            correlations = self.correlation_analyzer.analyze_all_correlations(
                aggregated_results
            )
            aggregated_results['correlations'] = correlations
            
            # Generate intelligence insights if enabled
            if config.enable_intelligent_testing:
                intelligence = await self._generate_intelligence(
                    target_path, aggregated_results
                )
                aggregated_results['intelligence'] = intelligence
            
            # Calculate execution metrics
            scan_duration = time.time() - scan_start
            aggregated_results['scan_metrics'] = {
                'scan_id': scan_id,
                'duration': scan_duration,
                'layers_executed': len(layer_results),
                'layers_successful': sum(1 for r in layer_results.values() 
                                        if r.get('success', False)),
                'total_findings': self._count_total_findings(aggregated_results)
            }
            
            logger.info(f"Orchestrated scan {scan_id} completed in {scan_duration:.2f}s")
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Orchestration error for {target_path}: {e}")
            return {
                'error': str(e),
                'scan_id': scan_id,
                'partial_results': self.execution_results.get(scan_id, {})
            }
    
    def _determine_execution_plan(
        self, 
        config: ScanConfiguration
    ) -> List[LayerConfiguration]:
        """
        Determine which layers to execute based on configuration
        
        Args:
            config: Scan configuration
            
        Returns:
            List of layer configurations to execute
        """
        layers = []
        
        if config.enable_real_time_monitoring:
            layers.append(LayerConfiguration(
                name='realtime_monitor',
                enabled=True,
                timeout=30,
                priority=1,
                dependencies=[]
            ))
        
        if config.enable_classical_analysis:
            layers.append(LayerConfiguration(
                name='classical_integrator',
                enabled=True,
                timeout=20,
                priority=2,
                dependencies=[]
            ))
        
        if config.enable_quality_correlation:
            layers.append(LayerConfiguration(
                name='quality_monitor',
                enabled=True,
                timeout=15,
                priority=3,
                dependencies=[]
            ))
        
        if config.enable_performance_profiling:
            layers.append(LayerConfiguration(
                name='performance_profiler',
                enabled=True,
                timeout=10,
                priority=4,
                dependencies=[]
            ))
        
        if config.enable_compliance_checking:
            layers.append(LayerConfiguration(
                name='compliance',
                enabled=True,
                timeout=25,
                priority=2,
                dependencies=['classical_integrator']
            ))
        
        if config.enable_intelligent_testing:
            layers.append(LayerConfiguration(
                name='intelligence_agent',
                enabled=True,
                timeout=30,
                priority=5,
                dependencies=['realtime_monitor', 'classical_integrator']
            ))
        
        # Sort by priority
        layers.sort(key=lambda x: x.priority)
        
        return layers
    
    async def _execute_layers(
        self,
        layers: List[LayerConfiguration],
        target_path: str,
        config: ScanConfiguration
    ) -> Dict[str, Any]:
        """
        Execute security layers with dependency management
        
        Args:
            layers: Layer configurations to execute
            target_path: Scan target
            config: Scan configuration
            
        Returns:
            Results from all executed layers
        """
        results = {}
        completed_layers = set()
        
        # Execute layers respecting dependencies
        max_iterations = len(layers) * 2  # Prevent infinite loops
        iteration = 0
        
        while len(completed_layers) < len(layers) and iteration < max_iterations:
            iteration += 1
            tasks = []
            
            for layer_config in layers:
                if layer_config.name in completed_layers:
                    continue
                
                # Check if dependencies are satisfied
                if all(dep in completed_layers for dep in layer_config.dependencies):
                    # Create execution task
                    if layer_config.async_execution:
                        task = self._execute_layer_async(
                            layer_config, target_path, config
                        )
                    else:
                        task = asyncio.to_thread(
                            self._execute_layer_sync,
                            layer_config, target_path, config
                        )
                    tasks.append((layer_config.name, task))
            
            # Execute tasks in parallel
            if tasks:
                task_results = await asyncio.gather(
                    *[task for _, task in tasks],
                    return_exceptions=True
                )
                
                for (layer_name, _), result in zip(tasks, task_results):
                    if isinstance(result, Exception):
                        logger.error(f"Layer {layer_name} failed: {result}")
                        results[layer_name] = {'error': str(result), 'success': False}
                    else:
                        results[layer_name] = result
                    completed_layers.add(layer_name)
            
            # Small delay to prevent busy waiting
            if len(completed_layers) < len(layers):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _execute_layer_async(
        self,
        layer_config: LayerConfiguration,
        target_path: str,
        config: ScanConfiguration
    ) -> Dict[str, Any]:
        """
        Execute a security layer asynchronously
        
        Args:
            layer_config: Layer configuration
            target_path: Scan target
            config: Scan configuration
            
        Returns:
            Layer execution results
        """
        layer_name = layer_config.name
        logger.info(f"Executing layer: {layer_name}")
        
        try:
            # Simulate layer execution (would call actual layer in production)
            result = await self._simulate_layer_execution(
                layer_name, target_path, layer_config.timeout
            )
            
            return {
                'layer_name': layer_name,
                'success': True,
                'execution_time': result.get('execution_time', 0),
                'findings': result.get('findings', []),
                'metrics': result.get('metrics', {}),
                'raw_results': result
            }
            
        except asyncio.TimeoutError:
            logger.error(f"Layer {layer_name} timed out after {layer_config.timeout}s")
            return {
                'layer_name': layer_name,
                'success': False,
                'error': f'Timeout after {layer_config.timeout} seconds'
            }
        except Exception as e:
            logger.error(f"Layer {layer_name} error: {e}")
            return {
                'layer_name': layer_name,
                'success': False,
                'error': str(e)
            }
    
    def _execute_layer_sync(
        self,
        layer_config: LayerConfiguration,
        target_path: str,
        config: ScanConfiguration
    ) -> Dict[str, Any]:
        """
        Execute a security layer synchronously
        
        Args:
            layer_config: Layer configuration
            target_path: Scan target
            config: Scan configuration
            
        Returns:
            Layer execution results
        """
        # Wrapper for synchronous execution
        return asyncio.run(self._execute_layer_async(layer_config, target_path, config))
    
    async def _simulate_layer_execution(
        self,
        layer_name: str,
        target_path: str,
        timeout: int
    ) -> Dict[str, Any]:
        """
        Simulate layer execution for testing
        
        In production, this would call the actual security layer
        """
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Return simulated results based on layer type
        if layer_name == 'realtime_monitor':
            return {
                'execution_time': 0.5,
                'alerts': [
                    {
                        'title': 'Potential SQL Injection',
                        'threat_level': 'high',
                        'confidence': 0.85
                    }
                ],
                'metrics': {'alerts_generated': 1}
            }
        elif layer_name == 'quality_monitor':
            return {
                'execution_time': 0.3,
                'overall_score': 75.0,
                'metrics': [
                    {
                        'metric_name': 'complexity',
                        'value': 15,
                        'severity': 'medium'
                    }
                ]
            }
        elif layer_name == 'compliance':
            return {
                'execution_time': 0.4,
                'compliant': False,
                'violations': ['OWASP-A1', 'PCI-DSS-6.5.1']
            }
        else:
            return {
                'execution_time': 0.2,
                'findings': [],
                'metrics': {}
            }
    
    def _aggregate_layer_results(
        self,
        layer_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Aggregate results from all executed layers
        
        Args:
            layer_results: Results from individual layers
            
        Returns:
            Aggregated results
        """
        aggregated = {
            'vulnerabilities': [],
            'compliance_violations': [],
            'quality_issues': [],
            'performance_concerns': [],
            'layer_results': layer_results,
            'summary_metrics': {}
        }
        
        for layer_name, result in layer_results.items():
            if not result.get('success', False):
                continue
            
            # Extract findings based on layer type
            if 'alerts' in result:
                aggregated['vulnerabilities'].extend(result['alerts'])
            
            if 'violations' in result:
                for violation in result.get('violations', []):
                    aggregated['compliance_violations'].append({
                        'standard': violation,
                        'layer': layer_name
                    })
            
            if 'metrics' in result and layer_name == 'quality_monitor':
                for metric in result['metrics']:
                    if metric.get('severity') in ['high', 'critical']:
                        aggregated['quality_issues'].append(metric)
            
            if layer_name == 'performance_profiler' and 'metrics' in result:
                if result['metrics'].get('execution_time', 0) > 2.0:
                    aggregated['performance_concerns'].append({
                        'type': 'slow_execution',
                        'value': result['metrics']['execution_time']
                    })
        
        # Calculate summary metrics
        aggregated['summary_metrics'] = {
            'total_vulnerabilities': len(aggregated['vulnerabilities']),
            'total_compliance_violations': len(aggregated['compliance_violations']),
            'total_quality_issues': len(aggregated['quality_issues']),
            'total_performance_concerns': len(aggregated['performance_concerns']),
            'layers_executed': len(layer_results),
            'layers_successful': sum(1 for r in layer_results.values() 
                                    if r.get('success', False))
        }
        
        return aggregated
    
    async def _generate_intelligence(
        self,
        target_path: str,
        aggregated_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate security intelligence from aggregated results
        
        Args:
            target_path: Scan target
            aggregated_results: Aggregated layer results
            
        Returns:
            Intelligence insights and recommendations
        """
        intelligence = {
            'risk_assessment': self._assess_overall_risk(aggregated_results),
            'attack_surface': self._analyze_attack_surface(aggregated_results),
            'remediation_priority': self._prioritize_remediation(aggregated_results),
            'test_recommendations': self._generate_test_recommendations(aggregated_results)
        }
        
        return intelligence
    
    def _assess_overall_risk(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall security risk from aggregated results"""
        vuln_count = results['summary_metrics']['total_vulnerabilities']
        compliance_count = results['summary_metrics']['total_compliance_violations']
        
        # Simple risk calculation
        risk_score = min(100, vuln_count * 10 + compliance_count * 5)
        
        if risk_score >= 80:
            risk_level = 'CRITICAL'
        elif risk_score >= 60:
            risk_level = 'HIGH'
        elif risk_score >= 40:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'contributing_factors': {
                'vulnerabilities': vuln_count,
                'compliance_violations': compliance_count
            }
        }
    
    def _analyze_attack_surface(self, results: Dict[str, Any]) -> List[str]:
        """Analyze potential attack surface from findings"""
        attack_vectors = set()
        
        for vuln in results.get('vulnerabilities', []):
            if 'injection' in str(vuln).lower():
                attack_vectors.add('Input Validation')
            if 'auth' in str(vuln).lower():
                attack_vectors.add('Authentication')
            if 'crypto' in str(vuln).lower():
                attack_vectors.add('Cryptography')
        
        return list(attack_vectors)
    
    def _prioritize_remediation(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize remediation actions based on risk"""
        actions = []
        
        # High priority: Critical vulnerabilities
        critical_vulns = [v for v in results.get('vulnerabilities', [])
                         if v.get('threat_level') == 'critical']
        for vuln in critical_vulns[:3]:
            actions.append({
                'priority': 1,
                'action': f"Fix {vuln.get('title', 'vulnerability')}",
                'estimated_effort': 'High'
            })
        
        # Medium priority: Compliance violations
        for violation in results.get('compliance_violations', [])[:2]:
            actions.append({
                'priority': 2,
                'action': f"Address {violation.get('standard', 'compliance')} violation",
                'estimated_effort': 'Medium'
            })
        
        return actions
    
    def _generate_test_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate test recommendations based on findings"""
        recommendations = []
        
        if results['summary_metrics']['total_vulnerabilities'] > 0:
            recommendations.append("Add security test cases for identified vulnerabilities")
        
        if results['summary_metrics']['total_compliance_violations'] > 0:
            recommendations.append("Implement compliance validation tests")
        
        if results['summary_metrics']['total_quality_issues'] > 0:
            recommendations.append("Add code quality checks to CI/CD pipeline")
        
        return recommendations
    
    def _count_total_findings(self, results: Dict[str, Any]) -> int:
        """Count total findings across all categories"""
        return (
            results.get('summary_metrics', {}).get('total_vulnerabilities', 0) +
            results.get('summary_metrics', {}).get('total_compliance_violations', 0) +
            results.get('summary_metrics', {}).get('total_quality_issues', 0) +
            results.get('summary_metrics', {}).get('total_performance_concerns', 0)
        )
    
    async def shutdown(self):
        """Shutdown orchestrator and cleanup resources"""
        logger.info("Shutting down security orchestrator")
        self.executor.shutdown(wait=True)


# Factory function
def create_security_orchestrator(max_workers: int = 4) -> SecurityLayerOrchestrator:
    """
    Create and configure security orchestrator
    
    Args:
        max_workers: Maximum parallel workers
        
    Returns:
        Configured orchestrator instance
    """
    return SecurityLayerOrchestrator(max_workers=max_workers)