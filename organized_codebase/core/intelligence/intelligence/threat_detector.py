"""
Focused Threat Detector

Handles threat detection, anomaly analysis, and risk assessment for security intelligence.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import numpy as np

logger = logging.getLogger(__name__)


class ThreatCategory(Enum):
    """Categories of security threats."""
    MALWARE = "malware"
    PHISHING = "phishing"
    INSIDER_THREAT = "insider_threat"
    ADVANCED_PERSISTENT_THREAT = "apt"
    DENIAL_OF_SERVICE = "dos"
    DATA_BREACH = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    LATERAL_MOVEMENT = "lateral_movement"
    COMMAND_INJECTION = "command_injection"
    ZERO_DAY = "zero_day"


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NEGLIGIBLE = "negligible"


@dataclass
class ThreatIndicator:
    """Security threat indicator with ML features."""
    indicator_id: str
    category: ThreatCategory
    severity: RiskLevel
    confidence: float
    source: str
    timestamp: datetime
    features: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetection:
    """Anomaly detection result."""
    anomaly_id: str
    detection_timestamp: datetime
    anomaly_type: str
    severity: RiskLevel
    confidence_score: float
    affected_entities: List[str]
    feature_vector: List[float]
    baseline_deviation: float
    context: Dict[str, Any] = field(default_factory=dict)


class ThreatDetector:
    """
    Focused threat detection engine.
    Handles ML-powered threat detection, anomaly analysis, and risk assessment.
    """
    
    def __init__(self):
        """Initialize threat detector with ML models and detection configurations."""
        try:
            # Threat detection configuration
            self.detection_enabled = True
            self.anomaly_threshold = 0.7
            self.risk_threshold = 0.8
            self.confidence_threshold = 0.6
            
            # Threat indicators storage
            self.threat_indicators = {}  # indicator_id -> ThreatIndicator
            self.threat_patterns = defaultdict(list)  # category -> List[patterns]
            self.anomaly_detections = {}  # anomaly_id -> AnomalyDetection
            
            # ML model state (mock - in real implementation would be actual ML models)
            self.ml_models = {
                'anomaly_detector': {'trained': True, 'accuracy': 0.92},
                'threat_classifier': {'trained': True, 'accuracy': 0.88},
                'risk_predictor': {'trained': True, 'accuracy': 0.85}
            }
            
            # Baseline profiles for anomaly detection
            self.baseline_profiles = {}
            self.feature_windows = defaultdict(lambda: deque(maxlen=1000))
            
            # Detection metrics
            self.detection_metrics = {
                'total_detections': 0,
                'true_positives': 0,
                'false_positives': 0,
                'threats_blocked': 0,
                'average_detection_time': 0.0
            }
            
            logger.info("Threat Detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize threat detector: {e}")
            raise
    
    async def detect_threats(self, security_events: List[Dict[str, Any]], 
                           detection_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Perform comprehensive threat detection on security events.
        
        Args:
            security_events: List of security events to analyze
            detection_config: Optional detection configuration
            
        Returns:
            Threat detection results with identified threats and risk assessments
        """
        try:
            detection_start = datetime.utcnow()
            self.detection_metrics['total_detections'] += 1
            
            detection_results = {
                'detection_timestamp': detection_start.isoformat(),
                'events_analyzed': len(security_events),
                'threats_detected': [],
                'anomalies_detected': [],
                'risk_assessment': {},
                'ml_predictions': {},
                'recommendations': []
            }
            
            if not security_events:
                return detection_results
            
            # Perform threat classification
            classified_threats = await self._classify_threats(security_events)
            detection_results['threats_detected'] = classified_threats
            
            # Perform anomaly detection
            anomalies = await self._detect_anomalies(security_events)
            detection_results['anomalies_detected'] = anomalies
            
            # Perform risk assessment
            risk_assessment = await self._assess_risk(classified_threats, anomalies)
            detection_results['risk_assessment'] = risk_assessment
            
            # Generate ML predictions
            ml_predictions = await self._generate_ml_predictions(security_events)
            detection_results['ml_predictions'] = ml_predictions
            
            # Generate recommendations
            recommendations = self._generate_threat_recommendations(
                classified_threats, anomalies, risk_assessment
            )
            detection_results['recommendations'] = recommendations
            
            # Update metrics
            detection_time = (datetime.utcnow() - detection_start).total_seconds() * 1000
            self._update_detection_metrics(detection_time, len(classified_threats))
            
            logger.info(f"Threat detection completed: {len(classified_threats)} threats, {len(anomalies)} anomalies")
            return detection_results
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return {
                'detection_timestamp': datetime.utcnow().isoformat(),
                'status': 'error',
                'error': str(e),
                'events_analyzed': len(security_events) if security_events else 0
            }
    
    async def analyze_threat_patterns(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Analyze threat patterns over a time window.
        
        Args:
            time_window_hours: Time window for pattern analysis
            
        Returns:
            Threat pattern analysis results
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=time_window_hours)
            
            pattern_analysis = {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'time_window_hours': time_window_hours,
                'threat_trends': {},
                'emerging_patterns': [],
                'attack_vectors': {},
                'geographical_distribution': {},
                'temporal_patterns': {}
            }
            
            # Analyze threat trends by category
            for category in ThreatCategory:
                category_threats = [
                    indicator for indicator in self.threat_indicators.values()
                    if (indicator.category == category and 
                        indicator.timestamp >= cutoff_time)
                ]
                
                pattern_analysis['threat_trends'][category.value] = {
                    'count': len(category_threats),
                    'severity_distribution': self._analyze_severity_distribution(category_threats),
                    'confidence_average': self._calculate_average_confidence(category_threats),
                    'trend_direction': self._analyze_trend_direction(category_threats)
                }
            
            # Detect emerging patterns
            emerging_patterns = await self._detect_emerging_patterns(cutoff_time)
            pattern_analysis['emerging_patterns'] = emerging_patterns
            
            # Analyze attack vectors
            attack_vectors = self._analyze_attack_vectors(cutoff_time)
            pattern_analysis['attack_vectors'] = attack_vectors
            
            # Analyze temporal patterns
            temporal_patterns = self._analyze_temporal_patterns(cutoff_time)
            pattern_analysis['temporal_patterns'] = temporal_patterns
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Threat pattern analysis failed: {e}")
            return {
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    async def predict_threat_likelihood(self, target_entities: List[str], 
                                      prediction_horizon_hours: int = 72) -> Dict[str, Any]:
        """
        Predict threat likelihood for specific entities.
        
        Args:
            target_entities: Entities to predict threats for
            prediction_horizon_hours: Prediction time horizon
            
        Returns:
            Threat likelihood predictions
        """
        try:
            predictions = {
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'prediction_horizon_hours': prediction_horizon_hours,
                'entity_predictions': {},
                'overall_threat_level': RiskLevel.LOW.value,
                'confidence_score': 0.0,
                'contributing_factors': []
            }
            
            total_risk_score = 0.0
            entity_count = len(target_entities)
            
            for entity in target_entities:
                entity_prediction = await self._predict_entity_threat(entity, prediction_horizon_hours)
                predictions['entity_predictions'][entity] = entity_prediction
                total_risk_score += entity_prediction.get('risk_score', 0.0)
            
            # Calculate overall threat level
            if entity_count > 0:
                avg_risk_score = total_risk_score / entity_count
                predictions['overall_threat_level'] = self._risk_score_to_level(avg_risk_score).value
                predictions['confidence_score'] = min(avg_risk_score, 1.0)
            
            # Identify contributing factors
            contributing_factors = self._identify_contributing_factors(predictions['entity_predictions'])
            predictions['contributing_factors'] = contributing_factors
            
            return predictions
            
        except Exception as e:
            logger.error(f"Threat prediction failed: {e}")
            return {
                'prediction_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    def get_threat_intelligence(self, threat_category: Optional[ThreatCategory] = None) -> Dict[str, Any]:
        """
        Get current threat intelligence summary.
        
        Args:
            threat_category: Optional category filter
            
        Returns:
            Threat intelligence summary
        """
        try:
            # Filter indicators
            if threat_category:
                indicators = [
                    indicator for indicator in self.threat_indicators.values()
                    if indicator.category == threat_category
                ]
            else:
                indicators = list(self.threat_indicators.values())
            
            # Calculate statistics
            total_indicators = len(indicators)
            
            if total_indicators == 0:
                return {
                    'intelligence_timestamp': datetime.utcnow().isoformat(),
                    'total_indicators': 0,
                    'threat_category': threat_category.value if threat_category else 'all',
                    'status': 'no_data'
                }
            
            # Severity distribution
            severity_dist = defaultdict(int)
            confidence_scores = []
            
            for indicator in indicators:
                severity_dist[indicator.severity.value] += 1
                confidence_scores.append(indicator.confidence)
            
            # Recent activity (last 24 hours)
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            recent_indicators = [i for i in indicators if i.timestamp >= recent_cutoff]
            
            return {
                'intelligence_timestamp': datetime.utcnow().isoformat(),
                'threat_category': threat_category.value if threat_category else 'all',
                'total_indicators': total_indicators,
                'recent_indicators_24h': len(recent_indicators),
                'severity_distribution': dict(severity_dist),
                'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
                'confidence_range': {
                    'min': min(confidence_scores) if confidence_scores else 0.0,
                    'max': max(confidence_scores) if confidence_scores else 0.0
                },
                'most_common_sources': self._get_most_common_sources(indicators),
                'detection_metrics': self.detection_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get threat intelligence: {e}")
            return {
                'intelligence_timestamp': datetime.utcnow().isoformat(),
                'error': str(e)
            }
    
    # Private helper methods
    async def _classify_threats(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify threats using ML models."""
        try:
            classified_threats = []
            
            for event in events:
                # Extract features from event
                features = self._extract_threat_features(event)
                
                # Mock ML classification - in real implementation would use actual ML model
                threat_probability = np.random.uniform(0.1, 0.9)
                
                if threat_probability > self.confidence_threshold:
                    # Determine threat category based on features
                    threat_category = self._determine_threat_category(features)
                    severity = self._calculate_threat_severity(threat_probability, features)
                    
                    threat = {
                        'threat_id': f"threat_{datetime.utcnow().timestamp()}",
                        'category': threat_category.value,
                        'severity': severity.value,
                        'confidence': threat_probability,
                        'source_event': event.get('event_id', 'unknown'),
                        'features': features,
                        'detection_timestamp': datetime.utcnow().isoformat()
                    }
                    
                    classified_threats.append(threat)
                    
                    # Store as threat indicator
                    indicator = ThreatIndicator(
                        indicator_id=threat['threat_id'],
                        category=threat_category,
                        severity=severity,
                        confidence=threat_probability,
                        source=event.get('source', 'unknown'),
                        timestamp=datetime.utcnow(),
                        features=features
                    )
                    self.threat_indicators[threat['threat_id']] = indicator
            
            return classified_threats
            
        except Exception as e:
            logger.error(f"Threat classification failed: {e}")
            return []
    
    async def _detect_anomalies(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalies in security events."""
        try:
            anomalies = []
            
            for event in events:
                # Extract features for anomaly detection
                feature_vector = self._extract_anomaly_features(event)
                
                # Calculate deviation from baseline
                baseline_deviation = self._calculate_baseline_deviation(feature_vector)
                
                if baseline_deviation > self.anomaly_threshold:
                    anomaly_severity = self._calculate_anomaly_severity(baseline_deviation)
                    
                    anomaly = {
                        'anomaly_id': f"anomaly_{datetime.utcnow().timestamp()}",
                        'detection_timestamp': datetime.utcnow().isoformat(),
                        'anomaly_type': event.get('event_type', 'unknown'),
                        'severity': anomaly_severity.value,
                        'confidence_score': min(baseline_deviation, 1.0),
                        'affected_entities': [event.get('source', 'unknown')],
                        'baseline_deviation': baseline_deviation,
                        'feature_vector': feature_vector
                    }
                    
                    anomalies.append(anomaly)
                    
                    # Store anomaly detection
                    detection = AnomalyDetection(
                        anomaly_id=anomaly['anomaly_id'],
                        detection_timestamp=datetime.utcnow(),
                        anomaly_type=anomaly['anomaly_type'],
                        severity=anomaly_severity,
                        confidence_score=anomaly['confidence_score'],
                        affected_entities=anomaly['affected_entities'],
                        feature_vector=feature_vector,
                        baseline_deviation=baseline_deviation
                    )
                    self.anomaly_detections[anomaly['anomaly_id']] = detection
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _assess_risk(self, threats: List[Dict], anomalies: List[Dict]) -> Dict[str, Any]:
        """Assess overall risk based on threats and anomalies."""
        try:
            # Calculate threat risk
            threat_risk = 0.0
            if threats:
                threat_scores = [self._severity_to_score(t['severity']) for t in threats]
                threat_risk = max(threat_scores) if threat_scores else 0.0
            
            # Calculate anomaly risk
            anomaly_risk = 0.0
            if anomalies:
                anomaly_scores = [a['confidence_score'] for a in anomalies]
                anomaly_risk = max(anomaly_scores) if anomaly_scores else 0.0
            
            # Combined risk assessment
            combined_risk = max(threat_risk, anomaly_risk)
            risk_level = self._risk_score_to_level(combined_risk)
            
            return {
                'overall_risk_level': risk_level.value,
                'overall_risk_score': combined_risk,
                'threat_risk_score': threat_risk,
                'anomaly_risk_score': anomaly_risk,
                'risk_factors': {
                    'threat_count': len(threats),
                    'anomaly_count': len(anomalies),
                    'highest_threat_severity': max([self._severity_to_score(t['severity']) for t in threats]) if threats else 0.0,
                    'highest_anomaly_confidence': max([a['confidence_score'] for a in anomalies]) if anomalies else 0.0
                }
            }
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {
                'overall_risk_level': RiskLevel.MEDIUM.value,
                'error': str(e)
            }
    
    async def _generate_ml_predictions(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate ML-based predictions."""
        try:
            predictions = {
                'prediction_accuracy': {},
                'model_confidence': {},
                'predicted_attack_types': [],
                'prediction_timeline': {}
            }
            
            # Mock ML predictions - in real implementation would use actual models
            for model_name, model_info in self.ml_models.items():
                predictions['prediction_accuracy'][model_name] = model_info['accuracy']
                predictions['model_confidence'][model_name] = np.random.uniform(0.7, 0.95)
            
            # Predict likely attack types
            attack_probabilities = {}
            for category in ThreatCategory:
                attack_probabilities[category.value] = np.random.uniform(0.1, 0.8)
            
            # Sort by probability and take top 3
            sorted_attacks = sorted(attack_probabilities.items(), key=lambda x: x[1], reverse=True)
            predictions['predicted_attack_types'] = sorted_attacks[:3]
            
            return predictions
            
        except Exception as e:
            logger.error(f"ML prediction generation failed: {e}")
            return {}
    
    def _extract_threat_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features for threat classification."""
        try:
            return {
                'event_type': event.get('event_type', 'unknown'),
                'source_ip': event.get('source_ip', ''),
                'destination_ip': event.get('destination_ip', ''),
                'payload_size': event.get('payload_size', 0),
                'timestamp_hour': datetime.now().hour,
                'is_weekend': datetime.now().weekday() >= 5,
                'protocol': event.get('protocol', 'unknown')
            }
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _determine_threat_category(self, features: Dict[str, Any]) -> ThreatCategory:
        """Determine threat category based on features."""
        try:
            # Simple rule-based categorization - in real implementation would use ML
            event_type = features.get('event_type', '').lower()
            
            if 'malware' in event_type or 'virus' in event_type:
                return ThreatCategory.MALWARE
            elif 'phish' in event_type:
                return ThreatCategory.PHISHING
            elif 'dos' in event_type or 'ddos' in event_type:
                return ThreatCategory.DENIAL_OF_SERVICE
            elif 'injection' in event_type:
                return ThreatCategory.COMMAND_INJECTION
            else:
                return ThreatCategory.INSIDER_THREAT
                
        except Exception as e:
            logger.error(f"Threat category determination failed: {e}")
            return ThreatCategory.INSIDER_THREAT
    
    def _calculate_threat_severity(self, probability: float, features: Dict[str, Any]) -> RiskLevel:
        """Calculate threat severity."""
        try:
            if probability >= 0.9:
                return RiskLevel.CRITICAL
            elif probability >= 0.7:
                return RiskLevel.HIGH
            elif probability >= 0.5:
                return RiskLevel.MEDIUM
            elif probability >= 0.3:
                return RiskLevel.LOW
            else:
                return RiskLevel.NEGLIGIBLE
        except Exception as e:
            logger.error(f"Threat severity calculation failed: {e}")
            return RiskLevel.MEDIUM
    
    def _severity_to_score(self, severity: str) -> float:
        """Convert severity to numerical score."""
        severity_scores = {
            'critical': 1.0,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4,
            'negligible': 0.2
        }
        return severity_scores.get(severity, 0.5)
    
    def _risk_score_to_level(self, score: float) -> RiskLevel:
        """Convert risk score to risk level."""
        if score >= 0.9:
            return RiskLevel.CRITICAL
        elif score >= 0.7:
            return RiskLevel.HIGH
        elif score >= 0.5:
            return RiskLevel.MEDIUM
        elif score >= 0.3:
            return RiskLevel.LOW
        else:
            return RiskLevel.NEGLIGIBLE
    
    def _extract_anomaly_features(self, event: Dict[str, Any]) -> List[float]:
        """Extract numerical features for anomaly detection."""
        try:
            return [
                float(event.get('payload_size', 0)),
                float(event.get('timestamp', datetime.now().timestamp())),
                float(len(event.get('source_ip', ''))),
                float(event.get('response_time', 0)),
                float(event.get('error_count', 0))
            ]
        except Exception as e:
            logger.error(f"Anomaly feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0]
    
    def _calculate_baseline_deviation(self, feature_vector: List[float]) -> float:
        """Calculate deviation from baseline."""
        try:
            # Mock baseline calculation - in real implementation would use statistical models
            if not feature_vector:
                return 0.0
            
            # Simple z-score based deviation
            mean_value = statistics.mean(feature_vector) if feature_vector else 0.0
            return min(abs(mean_value - 50.0) / 100.0, 1.0)  # Normalize to 0-1
            
        except Exception as e:
            logger.error(f"Baseline deviation calculation failed: {e}")
            return 0.0
    
    def _calculate_anomaly_severity(self, deviation: float) -> RiskLevel:
        """Calculate anomaly severity based on deviation."""
        return self._risk_score_to_level(deviation)
    
    def _generate_threat_recommendations(self, threats: List[Dict], 
                                       anomalies: List[Dict], 
                                       risk_assessment: Dict) -> List[str]:
        """Generate threat mitigation recommendations."""
        try:
            recommendations = []
            
            # Threat-based recommendations
            if threats:
                high_severity_threats = [t for t in threats if t['severity'] in ['critical', 'high']]
                if high_severity_threats:
                    recommendations.append('Immediate threat response required for high-severity threats')
                    
            # Anomaly-based recommendations
            if anomalies:
                high_confidence_anomalies = [a for a in anomalies if a['confidence_score'] > 0.8]
                if high_confidence_anomalies:
                    recommendations.append('Investigate high-confidence anomalies immediately')
            
            # Risk-based recommendations
            overall_risk = risk_assessment.get('overall_risk_level', 'low')
            if overall_risk in ['critical', 'high']:
                recommendations.append('Activate incident response procedures')
                recommendations.append('Increase security monitoring frequency')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def _update_detection_metrics(self, detection_time: float, threats_found: int) -> None:
        """Update detection performance metrics."""
        try:
            # Update average detection time
            total_detections = self.detection_metrics['total_detections']
            current_avg = self.detection_metrics['average_detection_time']
            
            if total_detections > 1:
                self.detection_metrics['average_detection_time'] = (
                    (current_avg * (total_detections - 1) + detection_time) / total_detections
                )
            else:
                self.detection_metrics['average_detection_time'] = detection_time
            
            # Update threat counts (simplified)
            self.detection_metrics['threats_blocked'] += threats_found
            
        except Exception as e:
            logger.error(f"Metrics update failed: {e}")
    
    async def _detect_emerging_patterns(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Detect emerging threat patterns."""
        try:
            # Mock pattern detection - in real implementation would use advanced ML
            return [
                {
                    'pattern_type': 'coordinated_attack',
                    'confidence': 0.85,
                    'first_observed': cutoff_time.isoformat(),
                    'affected_systems': ['system1', 'system2']
                }
            ]
        except Exception as e:
            logger.error(f"Emerging pattern detection failed: {e}")
            return []
    
    async def _predict_entity_threat(self, entity: str, horizon_hours: int) -> Dict[str, Any]:
        """Predict threat likelihood for specific entity."""
        try:
            # Mock prediction - in real implementation would use ML models
            risk_score = np.random.uniform(0.2, 0.9)
            
            return {
                'entity': entity,
                'risk_score': risk_score,
                'risk_level': self._risk_score_to_level(risk_score).value,
                'prediction_confidence': 0.8,
                'contributing_factors': ['unusual_activity', 'external_threat_intel']
            }
        except Exception as e:
            logger.error(f"Entity threat prediction failed: {e}")
            return {'entity': entity, 'risk_score': 0.5, 'error': str(e)}
    
    def _analyze_severity_distribution(self, threats: List[ThreatIndicator]) -> Dict[str, int]:
        """Analyze severity distribution of threats."""
        try:
            distribution = defaultdict(int)
            for threat in threats:
                distribution[threat.severity.value] += 1
            return dict(distribution)
        except Exception as e:
            logger.error(f"Severity distribution analysis failed: {e}")
            return {}
    
    def _calculate_average_confidence(self, threats: List[ThreatIndicator]) -> float:
        """Calculate average confidence of threats."""
        try:
            if not threats:
                return 0.0
            return statistics.mean([threat.confidence for threat in threats])
        except Exception as e:
            logger.error(f"Average confidence calculation failed: {e}")
            return 0.0
    
    def _analyze_trend_direction(self, threats: List[ThreatIndicator]) -> str:
        """Analyze trend direction for threats."""
        try:
            if len(threats) < 2:
                return 'insufficient_data'
            
            # Simple trend analysis based on count over time
            recent_count = len([t for t in threats[-10:]])  # Last 10
            older_count = len([t for t in threats[:-10]])   # Everything else
            
            if recent_count > older_count * 1.2:
                return 'increasing'
            elif recent_count < older_count * 0.8:
                return 'decreasing'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return 'unknown'
    
    def _analyze_attack_vectors(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze attack vectors."""
        try:
            # Mock analysis - in real implementation would analyze actual attack vectors
            return {
                'most_common_vectors': ['email', 'web', 'network'],
                'vector_distribution': {'email': 45, 'web': 30, 'network': 25},
                'emerging_vectors': ['supply_chain']
            }
        except Exception as e:
            logger.error(f"Attack vector analysis failed: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, cutoff_time: datetime) -> Dict[str, Any]:
        """Analyze temporal patterns in threats."""
        try:
            # Mock temporal analysis
            return {
                'peak_hours': [9, 14, 22],
                'peak_days': ['Monday', 'Friday'],
                'attack_frequency': 'increasing',
                'seasonal_trends': 'Q4_spike'
            }
        except Exception as e:
            logger.error(f"Temporal pattern analysis failed: {e}")
            return {}
    
    def _identify_contributing_factors(self, entity_predictions: Dict[str, Any]) -> List[str]:
        """Identify factors contributing to threat predictions."""
        try:
            factors = set()
            for entity, prediction in entity_predictions.items():
                factors.update(prediction.get('contributing_factors', []))
            return list(factors)
        except Exception as e:
            logger.error(f"Contributing factors identification failed: {e}")
            return []
    
    def _get_most_common_sources(self, indicators: List[ThreatIndicator]) -> List[str]:
        """Get most common threat sources."""
        try:
            source_counts = defaultdict(int)
            for indicator in indicators:
                source_counts[indicator.source] += 1
            
            # Return top 5 sources
            sorted_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)
            return [source for source, count in sorted_sources[:5]]
        except Exception as e:
            logger.error(f"Source analysis failed: {e}")
            return []