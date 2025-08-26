"""
Streamlined Security Intelligence Orchestrator

Enterprise security intelligence orchestrator - now modularized for simplicity.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

# Import modular security intelligence components
from .intelligence import (
    ThreatDetector, ThreatCategory, RiskLevel,
    IncidentResponder, IncidentSeverity, IncidentStatus
)

logger = logging.getLogger(__name__)


class SecurityIntelligence:
    """
    Streamlined security intelligence orchestrator.
    Coordinates threat detection, incident response, and security analytics through modular components.
    """
    
    def __init__(self):
        """Initialize the security intelligence orchestrator with modular components."""
        try:
            # Initialize core intelligence components
            self.threat_detector = ThreatDetector()
            self.incident_responder = IncidentResponder()
            
            # Initialize intelligence state
            self.intelligence_sessions = {}
            self.analytics_cache = {}
            self.intelligence_metrics = {
                'total_intelligence_operations': 0,
                'threats_detected': 0,
                'incidents_created': 0,
                'automated_responses': 0
            }
            
            logger.info("Security Intelligence orchestrator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize security intelligence: {e}")
            raise
    
    # High-level intelligence operations
    async def analyze_security_events(self, events: List[Dict[str, Any]], 
                                    analysis_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive security intelligence analysis on events.
        Coordinates threat detection and automated incident response.
        """
        try:
            analysis_start = datetime.utcnow()
            self.intelligence_metrics['total_intelligence_operations'] += 1
            
            analysis_results = {
                'analysis_timestamp': analysis_start.isoformat(),
                'events_analyzed': len(events),
                'threat_detection': {},
                'incident_response': {},
                'intelligence_insights': {},
                'automated_actions': [],
                'recommendations': []
            }
            
            # Perform threat detection
            threat_results = await self.threat_detector.detect_threats(events, analysis_config)
            analysis_results['threat_detection'] = threat_results
            
            detected_threats = threat_results.get('threats_detected', [])
            detected_anomalies = threat_results.get('anomalies_detected', [])
            
            # Update metrics
            self.intelligence_metrics['threats_detected'] += len(detected_threats)
            
            # Create incidents for high-severity threats
            incident_results = []
            for threat in detected_threats:
                if threat.get('severity') in ['critical', 'high']:
                    incident_data = self._threat_to_incident_data(threat, events)
                    incident_result = await self.incident_responder.create_incident(incident_data)
                    incident_results.append(incident_result)
                    
                    if incident_result.get('status') == 'created':
                        self.intelligence_metrics['incidents_created'] += 1
            
            analysis_results['incident_response'] = {
                'incidents_created': incident_results,
                'automated_responses': len([i for i in incident_results if i.get('initial_response_actions')])
            }
            
            # Generate intelligence insights
            insights = await self._generate_intelligence_insights(threat_results, incident_results)
            analysis_results['intelligence_insights'] = insights
            
            # Generate recommendations
            recommendations = self._generate_intelligence_recommendations(
                threat_results, incident_results, insights
            )
            analysis_results['recommendations'] = recommendations
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - analysis_start).total_seconds() * 1000
            analysis_results['processing_time_ms'] = processing_time
            
            logger.info(f"Security intelligence analysis completed: {len(detected_threats)} threats, {len(incident_results)} incidents")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Security intelligence analysis failed: {e}")
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e),
                'events_analyzed': len(events) if events else 0
            }
    
    # Threat intelligence operations (delegate to threat detector)
    async def get_threat_intelligence(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current threat intelligence - delegates to threat detector.
        """
        try:
            threat_category = ThreatCategory(category) if category else None
            return self.threat_detector.get_threat_intelligence(threat_category)
        except Exception as e:
            logger.error(f"Failed to get threat intelligence: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def analyze_threat_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze threat patterns - delegates to threat detector.
        """
        try:
            return await self.threat_detector.analyze_threat_patterns(time_window_hours)
        except Exception as e:
            logger.error(f"Failed to analyze threat patterns: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    async def predict_threat_likelihood(self, target_entities: List[str], 
                                      prediction_horizon_hours: int = 72) -> Dict[str, Any]:
        """
        Predict threat likelihood - delegates to threat detector.
        """
        try:
            return await self.threat_detector.predict_threat_likelihood(
                target_entities, prediction_horizon_hours
            )
        except Exception as e:
            logger.error(f"Failed to predict threat likelihood: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
    
    # Incident response operations (delegate to incident responder)
    async def create_security_incident(self, incident_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create security incident - delegates to incident responder.
        """
        try:
            return await self.incident_responder.create_incident(incident_data)
        except Exception as e:
            logger.error(f"Failed to create security incident: {e}")
            return {
                'status': 'creation_failed',
                'error': str(e)
            }
    
    async def update_incident_status(self, incident_id: str, new_status: str, 
                                   notes: Optional[str] = None) -> Dict[str, Any]:
        """
        Update incident status - delegates to incident responder.
        """
        try:
            return await self.incident_responder.update_incident_status(incident_id, new_status, notes)
        except Exception as e:
            logger.error(f"Failed to update incident status: {e}")
            return {
                'status': 'update_failed',
                'error': str(e)
            }
    
    async def execute_response_action(self, incident_id: str, action_type: str, 
                                    action_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute response action - delegates to incident responder.
        """
        try:
            return await self.incident_responder.execute_response_action(
                incident_id, action_type, action_params
            )
        except Exception as e:
            logger.error(f"Failed to execute response action: {e}")
            return {
                'status': 'execution_failed',
                'error': str(e)
            }
    
    def list_active_incidents(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List active incidents - delegates to incident responder.
        """
        try:
            return self.incident_responder.list_active_incidents(severity_filter)
        except Exception as e:
            logger.error(f"Failed to list active incidents: {e}")
            return []
    
    def get_incident_details(self, incident_id: str) -> Dict[str, Any]:
        """
        Get incident details - delegates to incident responder.
        """
        try:
            # Note: Making this async to match the interface
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're in an async context, we need to handle this differently
                # For now, return a simple response
                return {
                    'incident_id': incident_id,
                    'status': 'async_call_required',
                    'message': 'Use async method for detailed incident information'
                }
            else:
                return asyncio.run(self.incident_responder.get_incident_details(incident_id))
        except Exception as e:
            logger.error(f"Failed to get incident details: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # Analytics and reporting operations
    def get_security_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive security intelligence dashboard.
        """
        try:
            # Get metrics from all components
            threat_intelligence = self.threat_detector.get_threat_intelligence()
            response_metrics = self.incident_responder.get_response_metrics()
            
            # Calculate dashboard metrics
            active_incidents = len(self.incident_responder.active_incidents)
            critical_incidents = sum(
                1 for incident in self.incident_responder.active_incidents.values()
                if incident.severity == IncidentSeverity.CRITICAL
            )
            
            # Threat level assessment
            threat_indicators = len(self.threat_detector.threat_indicators)
            
            if critical_incidents > 0:
                overall_threat_level = "CRITICAL"
            elif active_incidents > 5:
                overall_threat_level = "HIGH"
            elif threat_indicators > 10:
                overall_threat_level = "MEDIUM"
            else:
                overall_threat_level = "LOW"
            
            return {
                'dashboard_timestamp': datetime.utcnow().isoformat(),
                'overall_threat_level': overall_threat_level,
                'summary_metrics': {
                    'active_incidents': active_incidents,
                    'critical_incidents': critical_incidents,
                    'threat_indicators': threat_indicators,
                    'total_intelligence_operations': self.intelligence_metrics['total_intelligence_operations'],
                    'threats_detected': self.intelligence_metrics['threats_detected'],
                    'automated_responses': self.intelligence_metrics['automated_responses']
                },
                'threat_intelligence_summary': {
                    'total_indicators': threat_intelligence.get('total_indicators', 0),
                    'recent_indicators_24h': threat_intelligence.get('recent_indicators_24h', 0),
                    'average_confidence': threat_intelligence.get('average_confidence', 0.0)
                },
                'incident_response_summary': {
                    'total_incidents': response_metrics.get('response_metrics', {}).get('total_incidents', 0),
                    'incidents_resolved': response_metrics.get('response_metrics', {}).get('incidents_resolved', 0),
                    'average_resolution_time': response_metrics.get('response_metrics', {}).get('average_resolution_time', 0.0)
                },
                'system_health': {
                    'threat_detection_enabled': self.threat_detector.detection_enabled,
                    'incident_automation_enabled': self.incident_responder.automation_enabled,
                    'components_active': {
                        'threat_detector': True,
                        'incident_responder': True
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate security dashboard: {e}")
            return {
                'dashboard_timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'overall_threat_level': 'UNKNOWN'
            }
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive intelligence metrics from all components.
        """
        try:
            # Get detection metrics
            detection_metrics = self.threat_detector.detection_metrics
            
            # Get response metrics
            response_metrics = self.incident_responder.get_response_metrics()
            
            return {
                'intelligence_metrics': self.intelligence_metrics,
                'threat_detection_metrics': detection_metrics,
                'incident_response_metrics': response_metrics,
                'cache_statistics': {
                    'analytics_cache_size': len(self.analytics_cache),
                    'active_sessions': len(self.intelligence_sessions)
                },
                'system_performance': {
                    'total_operations': self.intelligence_metrics['total_intelligence_operations'],
                    'automation_ratio': self._calculate_automation_ratio(),
                    'threat_detection_rate': self._calculate_threat_detection_rate()
                },
                'metrics_timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get intelligence metrics: {e}")
            return {
                'error': str(e),
                'metrics_timestamp': datetime.utcnow().isoformat()
            }
    
    async def generate_intelligence_report(self, report_type: str = "comprehensive", 
                                         time_period_hours: int = 24) -> Dict[str, Any]:
        """
        Generate comprehensive intelligence report.
        """
        try:
            report_start = datetime.utcnow()
            
            # Get threat patterns
            threat_patterns = await self.analyze_threat_patterns(time_period_hours)
            
            # Get incident statistics
            incident_stats = self._calculate_incident_statistics(time_period_hours)
            
            # Generate recommendations
            strategic_recommendations = self._generate_strategic_recommendations(
                threat_patterns, incident_stats
            )
            
            report = {
                'report_type': report_type,
                'report_timestamp': report_start.isoformat(),
                'time_period_hours': time_period_hours,
                'executive_summary': self._generate_executive_summary(threat_patterns, incident_stats),
                'threat_analysis': threat_patterns,
                'incident_statistics': incident_stats,
                'strategic_recommendations': strategic_recommendations,
                'next_review_date': (report_start + datetime.timedelta(hours=time_period_hours)).isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate intelligence report: {e}")
            return {
                'report_type': report_type,
                'report_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    # Private helper methods
    def _threat_to_incident_data(self, threat: Dict[str, Any], events: List[Dict]) -> Dict[str, Any]:
        """Convert threat detection to incident data."""
        try:
            # Extract affected systems from related events
            affected_systems = []
            for event in events:
                if event.get('event_id') == threat.get('source_event'):
                    affected_systems.extend(event.get('affected_systems', []))
            
            return {
                'title': f"Security Threat Detected: {threat.get('category', 'Unknown')}",
                'description': f"Threat ID: {threat.get('threat_id')} with confidence {threat.get('confidence', 0):.2f}",
                'severity': threat.get('severity', 'medium'),
                'affected_systems': list(set(affected_systems)),  # Remove duplicates
                'indicators': [threat.get('threat_id')],
                'metadata': {
                    'incident_type': threat.get('category', 'unknown'),
                    'threat_confidence': threat.get('confidence', 0.0),
                    'source_threat': threat.get('threat_id'),
                    'auto_created': True
                }
            }
        except Exception as e:
            logger.error(f"Threat to incident conversion failed: {e}")
            return {
                'title': 'Auto-generated Security Incident',
                'description': 'Incident created from threat detection',
                'severity': 'medium'
            }
    
    async def _generate_intelligence_insights(self, threat_results: Dict, 
                                            incident_results: List[Dict]) -> Dict[str, Any]:
        """Generate intelligence insights from analysis results."""
        try:
            insights = {
                'threat_trends': {},
                'attack_patterns': {},
                'risk_assessment': {},
                'prediction_accuracy': {}
            }
            
            # Analyze threat trends
            threats_detected = threat_results.get('threats_detected', [])
            if threats_detected:
                category_counts = defaultdict(int)
                for threat in threats_detected:
                    category_counts[threat.get('category', 'unknown')] += 1
                
                insights['threat_trends'] = {
                    'most_common_category': max(category_counts, key=category_counts.get) if category_counts else 'none',
                    'category_distribution': dict(category_counts),
                    'total_threats': len(threats_detected)
                }
            
            # Analyze incident response effectiveness
            if incident_results:
                successful_creations = sum(1 for i in incident_results if i.get('status') == 'created')
                insights['response_effectiveness'] = {
                    'incident_creation_rate': successful_creations / len(incident_results) if incident_results else 0,
                    'automated_response_rate': sum(1 for i in incident_results if i.get('initial_response_actions')) / len(incident_results) if incident_results else 0
                }
            
            return insights
            
        except Exception as e:
            logger.error(f"Intelligence insights generation failed: {e}")
            return {}
    
    def _generate_intelligence_recommendations(self, threat_results: Dict, 
                                             incident_results: List[Dict], 
                                             insights: Dict) -> List[str]:
        """Generate actionable intelligence recommendations."""
        try:
            recommendations = []
            
            # Threat-based recommendations
            threats_detected = threat_results.get('threats_detected', [])
            high_severity_threats = [t for t in threats_detected if t.get('severity') in ['critical', 'high']]
            
            if high_severity_threats:
                recommendations.append('Increase security monitoring due to high-severity threats detected')
            
            # Pattern-based recommendations
            threat_trends = insights.get('threat_trends', {})
            most_common = threat_trends.get('most_common_category')
            
            if most_common and most_common != 'none':
                recommendations.append(f'Focus defensive measures on {most_common} threat category')
            
            # Response effectiveness recommendations
            response_effectiveness = insights.get('response_effectiveness', {})
            if response_effectiveness.get('automated_response_rate', 0) < 0.5:
                recommendations.append('Consider increasing automation for incident response')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Intelligence recommendations generation failed: {e}")
            return []
    
    def _calculate_automation_ratio(self) -> float:
        """Calculate ratio of automated vs manual responses."""
        try:
            total_responses = (
                self.intelligence_metrics['automated_responses'] + 
                self.incident_responder.response_metrics.get('manual_responses', 0)
            )
            
            if total_responses == 0:
                return 0.0
            
            return self.intelligence_metrics['automated_responses'] / total_responses
            
        except Exception as e:
            logger.error(f"Automation ratio calculation failed: {e}")
            return 0.0
    
    def _calculate_threat_detection_rate(self) -> float:
        """Calculate threat detection rate."""
        try:
            total_operations = self.intelligence_metrics['total_intelligence_operations']
            threats_detected = self.intelligence_metrics['threats_detected']
            
            if total_operations == 0:
                return 0.0
            
            return threats_detected / total_operations
            
        except Exception as e:
            logger.error(f"Threat detection rate calculation failed: {e}")
            return 0.0
    
    def _calculate_incident_statistics(self, time_period_hours: int) -> Dict[str, Any]:
        """Calculate incident statistics for time period."""
        try:
            cutoff_time = datetime.utcnow() - datetime.timedelta(hours=time_period_hours)
            
            # Get incidents from the time period
            period_incidents = [
                incident for incident in self.incident_responder.active_incidents.values()
                if incident.created_at >= cutoff_time
            ]
            
            period_incidents.extend([
                incident for incident in self.incident_responder.incident_history
                if incident.created_at >= cutoff_time
            ])
            
            # Calculate statistics
            total_incidents = len(period_incidents)
            severity_distribution = defaultdict(int)
            
            for incident in period_incidents:
                severity_distribution[incident.severity.value] += 1
            
            return {
                'total_incidents': total_incidents,
                'severity_distribution': dict(severity_distribution),
                'time_period_hours': time_period_hours,
                'incidents_per_hour': total_incidents / time_period_hours if time_period_hours > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Incident statistics calculation failed: {e}")
            return {}
    
    def _generate_executive_summary(self, threat_patterns: Dict, incident_stats: Dict) -> str:
        """Generate executive summary of security intelligence."""
        try:
            threats_count = threat_patterns.get('threat_trends', {}).get('total_indicators', 0)
            incidents_count = incident_stats.get('total_incidents', 0)
            
            if threats_count == 0 and incidents_count == 0:
                return "Security posture remains stable with no significant threats or incidents detected."
            elif threats_count > 10 or incidents_count > 5:
                return f"Elevated threat activity detected with {threats_count} threat indicators and {incidents_count} security incidents requiring attention."
            else:
                return f"Normal security activity with {threats_count} threat indicators and {incidents_count} incidents under management."
                
        except Exception as e:
            logger.error(f"Executive summary generation failed: {e}")
            return "Security intelligence summary unavailable due to processing error."
    
    def _generate_strategic_recommendations(self, threat_patterns: Dict, 
                                          incident_stats: Dict) -> List[str]:
        """Generate strategic security recommendations."""
        try:
            recommendations = []
            
            # Based on threat patterns
            total_indicators = threat_patterns.get('threat_trends', {}).get('total_indicators', 0)
            if total_indicators > 20:
                recommendations.append('Consider enhancing threat detection capabilities')
            
            # Based on incident statistics
            incidents_per_hour = incident_stats.get('incidents_per_hour', 0)
            if incidents_per_hour > 0.5:
                recommendations.append('Evaluate incident response procedures for potential optimization')
            
            # General recommendations
            recommendations.extend([
                'Maintain regular security training for all personnel',
                'Review and update incident response playbooks quarterly',
                'Conduct threat hunting exercises monthly'
            ])
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Strategic recommendations generation failed: {e}")
            return []