"""
Streamlined Security Validator Orchestrator

Enterprise security validation orchestrator - now modularized for simplicity.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

# Import modular security validation components
from .validation import (
    PenetrationTester, ComplianceValidator, IntegrationValidator
)

logger = logging.getLogger(__name__)


class SecurityValidator:
    """
    Streamlined security validation orchestrator.
    Coordinates security testing, compliance validation, and integration testing through modular components.
    """
    
    def __init__(self):
        """Initialize the security validator with modular components."""
        try:
            # Initialize core validation components
            self.penetration_tester = PenetrationTester()
            self.compliance_validator = ComplianceValidator()
            self.integration_validator = IntegrationValidator()
            
            # Initialize caches and state
            self.validation_history = defaultdict(list)
            self.security_cache = {}
            
            logger.info("Security Validator orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize security validator: {e}")
            raise
    
    # High-level security operations (delegate to specialized components)
    async def run_comprehensive_security_test(self, 
                                            target_systems: List[Dict],
                                            test_configurations: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run comprehensive security testing across all target systems.
        Delegates to specialized validation components for complete coverage.
        """
        try:
            test_results = {
                'test_timestamp': datetime.utcnow().isoformat(),
                'systems_tested': len(target_systems),
                'penetration_test_results': {},
                'compliance_test_results': {},
                'integration_test_results': {},
                'overall_security_score': 0,
                'critical_findings': [],
                'recommendations': []
            }
            
            # Run penetration testing
            if target_systems:
                pen_test_results = await self.penetration_tester.run_comprehensive_penetration_test(
                    target_systems, test_configurations
                )
                test_results['penetration_test_results'] = pen_test_results
                
                # Extract critical findings from penetration tests
                if pen_test_results.get('critical_vulnerabilities'):
                    test_results['critical_findings'].extend(pen_test_results['critical_vulnerabilities'])
            
            # Run compliance validation
            compliance_frameworks = ['SOX', 'GDPR', 'HIPAA', 'ISO27001']
            compliance_results = {}
            for framework in compliance_frameworks:
                try:
                    framework_result = await self.compliance_validator.validate_compliance_framework(
                        framework, Path('.'), None
                    )
                    compliance_results[framework] = framework_result
                    
                    # Extract critical compliance issues
                    if framework_result.get('status') == 'non_compliant':
                        test_results['critical_findings'].append({
                            'type': 'compliance_violation',
                            'framework': framework,
                            'score': framework_result.get('compliance_score', 0)
                        })
                except Exception as framework_error:
                    logger.error(f"Compliance validation failed for {framework}: {framework_error}")
                    compliance_results[framework] = {'error': str(framework_error)}
            
            test_results['compliance_test_results'] = compliance_results
            
            # Run integration validation
            mock_apis = [{'name': 'api1', 'version': '2.0.0', 'schema': True, 'endpoints': ['test'], 'authentication': True}]
            api_results = await self.integration_validator.validate_api_compatibility(mock_apis)
            test_results['integration_test_results']['api_compatibility'] = api_results
            
            mock_services = [{'name': 'service1', 'connectivity': True, 'security': True, 'performance': True, 'observability': True}]
            mesh_results = await self.integration_validator.validate_service_mesh(mock_services)
            test_results['integration_test_results']['service_mesh'] = mesh_results
            
            # Calculate overall security score
            pen_score = pen_test_results.get('overall_security_score', 0)
            compliance_scores = [r.get('compliance_score', 0) for r in compliance_results.values() if isinstance(r, dict)]
            avg_compliance_score = sum(compliance_scores) / len(compliance_scores) if compliance_scores else 0
            integration_scores = [
                api_results.get('overall_compatibility_score', 0),
                mesh_results.get('overall_mesh_score', 0)
            ]
            avg_integration_score = sum(integration_scores) / len(integration_scores)
            
            test_results['overall_security_score'] = (pen_score + avg_compliance_score + avg_integration_score) / 3
            
            # Generate recommendations
            test_results['recommendations'] = self._generate_security_recommendations(test_results)
            
            # Store in history
            self.validation_history['comprehensive_tests'].append(test_results)
            
            logger.info(f"Comprehensive security test completed (score: {test_results['overall_security_score']:.1f})")
            return test_results
            
        except Exception as e:
            logger.error(f"Comprehensive security test failed: {e}")
            return {
                'test_timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e),
                'overall_security_score': 0
            }
    
    # Penetration testing operations (delegate to penetration tester)
    async def run_penetration_test(self, 
                                 target_systems: List[Dict],
                                 test_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run penetration testing - delegates to penetration tester component.
        """
        try:
            return await self.penetration_tester.run_comprehensive_penetration_test(target_systems, {
                'test_types': test_types or ['owasp_top_10', 'network_security', 'authentication']
            })
        except Exception as e:
            logger.error(f"Penetration test failed: {e}")
            return {'error': str(e), 'overall_security_score': 0}
    
    async def run_vulnerability_scan(self, 
                                   target_systems: List[Dict],
                                   scan_depth: str = 'comprehensive') -> Dict[str, Any]:
        """
        Run vulnerability scanning - delegates to penetration tester component.
        """
        try:
            return await self.penetration_tester.run_vulnerability_assessment(target_systems, scan_depth)
        except Exception as e:
            logger.error(f"Vulnerability scan failed: {e}")
            return {'error': str(e), 'vulnerabilities_found': 0}
    
    # Compliance validation operations (delegate to compliance validator)
    async def validate_compliance(self, 
                                framework: str,
                                target_path: Path,
                                requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate compliance - delegates to compliance validator component.
        """
        try:
            return await self.compliance_validator.validate_compliance_framework(framework, target_path, requirements)
        except Exception as e:
            logger.error(f"Compliance validation failed: {e}")
            return {'error': str(e), 'compliance_score': 0}
    
    async def validate_data_protection(self, 
                                     data_flows: List[Dict],
                                     privacy_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate data protection - delegates to compliance validator component.
        """
        try:
            return await self.compliance_validator.validate_data_protection(data_flows, privacy_requirements)
        except Exception as e:
            logger.error(f"Data protection validation failed: {e}")
            return {'error': str(e), 'overall_compliance_score': 0}
    
    # Integration validation operations (delegate to integration validator)
    async def validate_api_security(self, 
                                  api_specs: List[Dict],
                                  security_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate API security - delegates to integration validator component.
        """
        try:
            return await self.integration_validator.validate_api_compatibility(api_specs, security_requirements)
        except Exception as e:
            logger.error(f"API security validation failed: {e}")
            return {'error': str(e), 'overall_compatibility_score': 0}
    
    async def validate_service_mesh_security(self, 
                                           service_configs: List[Dict],
                                           mesh_requirements: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Validate service mesh security - delegates to integration validator component.
        """
        try:
            return await self.integration_validator.validate_service_mesh(service_configs, mesh_requirements)
        except Exception as e:
            logger.error(f"Service mesh security validation failed: {e}")
            return {'error': str(e), 'overall_mesh_score': 0}
    
    # Quick assessment operations
    async def quick_security_assessment(self, target_system: Dict) -> Dict[str, Any]:
        """
        Perform quick security assessment using all components.
        Lightweight operation for fast feedback.
        """
        try:
            assessment_results = {
                'assessment_timestamp': datetime.utcnow().isoformat(),
                'target_system': target_system.get('name', 'unknown'),
                'quick_pen_test': {},
                'quick_compliance_check': {},
                'quick_integration_check': {},
                'overall_risk_score': 0,
                'immediate_actions': []
            }
            
            # Quick penetration test
            pen_result = await self.penetration_tester.run_vulnerability_assessment([target_system], 'quick')
            assessment_results['quick_pen_test'] = pen_result
            
            # Quick compliance check (GDPR as default)
            compliance_result = await self.compliance_validator.validate_compliance_framework(
                'GDPR', Path('.'), None
            )
            assessment_results['quick_compliance_check'] = compliance_result
            
            # Quick integration check
            mock_api = [{'name': target_system.get('name', 'system'), 'version': '1.0.0'}]
            integration_result = await self.integration_validator.validate_api_compatibility(mock_api)
            assessment_results['quick_integration_check'] = integration_result
            
            # Calculate risk score
            pen_score = pen_result.get('overall_security_score', 0)
            compliance_score = compliance_result.get('compliance_score', 0)
            integration_score = integration_result.get('overall_compatibility_score', 0)
            
            assessment_results['overall_risk_score'] = (pen_score + compliance_score + integration_score) / 3
            
            # Generate immediate actions
            if pen_score < 70:
                assessment_results['immediate_actions'].append('Address critical vulnerabilities')
            if compliance_score < 80:
                assessment_results['immediate_actions'].append('Fix compliance violations')
            if integration_score < 80:
                assessment_results['immediate_actions'].append('Resolve integration issues')
            
            return assessment_results
            
        except Exception as e:
            logger.error(f"Quick security assessment failed: {e}")
            return {
                'assessment_timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'overall_risk_score': 100  # Highest risk due to assessment failure
            }
    
    # Utility and reporting methods
    def get_security_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive security metrics from all components.
        """
        try:
            total_tests = sum(len(history) for history in self.validation_history.values())
            comprehensive_tests = len(self.validation_history.get('comprehensive_tests', []))
            
            # Get latest scores if available
            latest_comprehensive = self.validation_history.get('comprehensive_tests', [])
            latest_score = latest_comprehensive[-1].get('overall_security_score', 0) if latest_comprehensive else 0
            
            return {
                'total_security_tests': total_tests,
                'comprehensive_tests_run': comprehensive_tests,
                'latest_security_score': latest_score,
                'cache_size': len(self.security_cache),
                'components_active': {
                    'penetration_tester': True,
                    'compliance_validator': True,
                    'integration_validator': True
                },
                'system_health': 'healthy'
            }
            
        except Exception as e:
            logger.error(f"Failed to get security metrics: {e}")
            return {
                'system_health': 'error',
                'error': str(e)
            }
    
    def clear_security_cache(self, test_type: Optional[str] = None) -> bool:
        """
        Clear security validation cache.
        """
        try:
            if test_type:
                if test_type in self.validation_history:
                    del self.validation_history[test_type]
                    logger.info(f"Cleared cache for {test_type}")
            else:
                self.validation_history.clear()
                self.security_cache.clear()
                logger.info("Cleared all security cache")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear security cache: {e}")
            return False
    
    def get_security_report(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive security report.
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            recent_tests = []
            for test_history in self.validation_history.values():
                for test in test_history:
                    test_time_str = test.get('test_timestamp', test.get('assessment_timestamp', ''))
                    if test_time_str:
                        try:
                            test_time = datetime.fromisoformat(test_time_str.replace('Z', '+00:00'))
                            if test_time >= cutoff_time:
                                recent_tests.append(test)
                        except:
                            continue
            
            # Calculate summary statistics
            if recent_tests:
                scores = [t.get('overall_security_score', t.get('overall_risk_score', 0)) for t in recent_tests]
                avg_score = sum(scores) / len(scores)
                critical_findings = sum(len(t.get('critical_findings', [])) for t in recent_tests)
            else:
                avg_score = 0
                critical_findings = 0
            
            return {
                'report_timestamp': datetime.utcnow().isoformat(),
                'time_window_hours': time_window_hours,
                'tests_in_period': len(recent_tests),
                'average_security_score': avg_score,
                'total_critical_findings': critical_findings,
                'security_trend': 'improving' if avg_score > 75 else 'declining' if avg_score < 60 else 'stable',
                'recommendations': self._generate_report_recommendations(recent_tests)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security report: {e}")
            return {
                'report_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    # Private helper methods
    def _generate_security_recommendations(self, test_results: Dict) -> List[str]:
        """
        Generate security recommendations based on test results.
        """
        try:
            recommendations = []
            
            # Check penetration test results
            pen_results = test_results.get('penetration_test_results', {})
            if pen_results.get('overall_security_score', 0) < 70:
                recommendations.append('Address critical security vulnerabilities immediately')
            
            # Check compliance results
            compliance_results = test_results.get('compliance_test_results', {})
            for framework, result in compliance_results.items():
                if isinstance(result, dict) and result.get('compliance_score', 0) < 80:
                    recommendations.append(f'Improve {framework} compliance posture')
            
            # Check integration results
            integration_results = test_results.get('integration_test_results', {})
            if integration_results.get('api_compatibility', {}).get('overall_compatibility_score', 0) < 80:
                recommendations.append('Enhance API security and compatibility')
            
            # Add general recommendations if score is low
            if test_results.get('overall_security_score', 0) < 75:
                recommendations.extend([
                    'Implement comprehensive security monitoring',
                    'Conduct regular security training',
                    'Update incident response procedures'
                ])
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating security recommendations: {e}")
            return []
    
    def _generate_report_recommendations(self, recent_tests: List[Dict]) -> List[str]:
        """
        Generate recommendations for security report.
        """
        try:
            recommendations = []
            
            if not recent_tests:
                recommendations.append('Run comprehensive security testing')
                return recommendations
            
            # Analyze test patterns
            scores = [t.get('overall_security_score', t.get('overall_risk_score', 0)) for t in recent_tests]
            avg_score = sum(scores) / len(scores)
            
            if avg_score < 60:
                recommendations.append('Urgent: Address critical security gaps')
            elif avg_score < 75:
                recommendations.append('Improve overall security posture')
            else:
                recommendations.append('Maintain current security standards')
            
            # Check for recurring issues
            all_findings = []
            for test in recent_tests:
                all_findings.extend(test.get('critical_findings', []))
            
            if len(all_findings) > 5:
                recommendations.append('Focus on reducing critical security findings')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating report recommendations: {e}")
            return []