"""
ML Security Guardian - TestMaster Advanced ML
Advanced security monitoring and threat detection with ML-driven analysis
Enterprise ML Module #3/8 for comprehensive system intelligence
"""

import asyncio
import hashlib
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from threading import Event, Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import json
import re
import ipaddress

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


class ThreatLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class SecurityEventType(Enum):
    AUTHENTICATION_FAILURE = "auth_failure"
    AUTHORIZATION_VIOLATION = "auth_violation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    ANOMALOUS_TRAFFIC = "anomalous_traffic"
    MALWARE_DETECTED = "malware_detected"
    DATA_EXFILTRATION = "data_exfiltration"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    SQL_INJECTION = "sql_injection"
    XSS_ATTEMPT = "xss_attempt"
    DOS_ATTACK = "dos_attack"
    INTRUSION_ATTEMPT = "intrusion_attempt"


class ResponseAction(Enum):
    LOG_ONLY = "log_only"
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    RATE_LIMIT = "rate_limit"
    QUARANTINE = "quarantine"
    ESCALATE = "escalate"
    AUTO_REMEDIATE = "auto_remediate"


@dataclass
class SecurityEvent:
    """ML-enhanced security event with intelligent analysis"""
    
    event_id: str
    timestamp: datetime
    event_type: SecurityEventType
    threat_level: ThreatLevel
    source_ip: str
    target: str
    description: str
    raw_data: Dict[str, Any] = field(default_factory=dict)
    
    # ML Enhancement Fields
    anomaly_score: float = 0.0
    threat_probability: float = 0.0
    attack_vector_prediction: List[str] = field(default_factory=list)
    ml_confidence: float = 0.0
    
    # Context and attribution
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    geolocation: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Response tracking
    response_actions: List[ResponseAction] = field(default_factory=list)
    auto_remediated: bool = False
    investigation_status: str = "pending"


@dataclass
class ThreatPattern:
    """ML-identified threat pattern"""
    
    pattern_id: str
    pattern_name: str
    event_types: List[SecurityEventType]
    time_window: int  # seconds
    min_occurrences: int
    severity_multiplier: float = 1.0
    
    # ML Enhancement
    ml_detected: bool = True
    confidence_score: float = 0.0
    false_positive_rate: float = 0.0
    pattern_features: Dict[str, Any] = field(default_factory=dict)
    
    # Pattern statistics
    detected_count: int = 0
    last_detected: Optional[datetime] = None
    effectiveness_score: float = 0.0


@dataclass
class SecurityMetrics:
    """Comprehensive security metrics with ML insights"""
    
    timestamp: datetime
    total_events: int
    threat_events: int
    blocked_attempts: int
    false_positives: int
    
    # Threat distribution
    threat_by_level: Dict[ThreatLevel, int] = field(default_factory=dict)
    threat_by_type: Dict[SecurityEventType, int] = field(default_factory=dict)
    
    # ML Performance
    ml_accuracy: float = 0.0
    ml_precision: float = 0.0
    ml_recall: float = 0.0
    anomaly_detection_rate: float = 0.0
    
    # Response effectiveness
    response_time_avg: float = 0.0
    auto_remediation_rate: float = 0.0
    incident_resolution_time: float = 0.0


class MLSecurityGuardian:
    """
    ML-enhanced security monitoring and threat detection system
    """
    
    def __init__(self,
                 enable_ml_detection: bool = True,
                 monitoring_interval: int = 30,
                 auto_remediation: bool = True,
                 threat_threshold: float = 0.7):
        """Initialize ML security guardian"""
        
        self.enable_ml_detection = enable_ml_detection
        self.monitoring_interval = monitoring_interval
        self.auto_remediation = auto_remediation
        self.threat_threshold = threat_threshold
        
        # ML Models for Security Intelligence
        self.threat_classifier: Optional[RandomForestClassifier] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.attack_predictor: Optional[LogisticRegression] = None
        self.behavior_clusterer: Optional[DBSCAN] = None
        
        # ML Feature Processing
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.security_feature_history: deque = deque(maxlen=5000)
        
        # Security Event Management
        self.security_events: deque = deque(maxlen=10000)
        self.active_incidents: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        self.rate_limited_ips: Dict[str, Dict[str, Any]] = {}
        
        # Threat Intelligence
        self.threat_patterns: Dict[str, ThreatPattern] = {}
        self.known_malicious_ips: set = set()
        self.suspicious_user_agents: List[str] = []
        self.attack_signatures: Dict[str, List[str]] = {}
        
        # ML Insights and Predictions
        self.ml_threat_predictions: Dict[str, Dict[str, float]] = {}
        self.behavioral_anomalies: List[Dict[str, Any]] = []
        self.security_insights: List[Dict[str, Any]] = []
        
        # Configuration
        self.ip_block_duration = 3600  # 1 hour
        self.rate_limit_threshold = 100  # requests per minute
        self.anomaly_threshold = -0.5
        self.pattern_match_threshold = 0.8
        
        # Statistics and Metrics
        self.security_metrics: deque = deque(maxlen=1000)
        self.security_stats = {
            'events_processed': 0,
            'threats_detected': 0,
            'attacks_blocked': 0,
            'false_positives': 0,
            'ml_detections': 0,
            'auto_remediations': 0,
            'start_time': datetime.now()
        }
        
        # Synchronization
        self.security_lock = RLock()
        self.ml_lock = Lock()
        self.shutdown_event = Event()
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize ML models and threat patterns
        if enable_ml_detection:
            self._initialize_ml_models()
            asyncio.create_task(self._ml_security_loop())
        
        self._initialize_threat_patterns()
        asyncio.create_task(self._security_monitoring_loop())
    
    def _initialize_ml_models(self):
        \"\"\"Initialize ML models for security intelligence\"\"\"
        
        try:
            # Threat classification model
            self.threat_classifier = RandomForestClassifier(
                n_estimators=150,
                max_depth=15,
                random_state=42,
                class_weight='balanced'
            )
            
            # Anomaly detection for unusual behavior
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            # Attack vector prediction
            self.attack_predictor = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
            
            # Behavioral clustering
            self.behavior_clusterer = DBSCAN(
                eps=0.3,
                min_samples=5,
                metric='euclidean'
            )
            
            self.logger.info("Security ML models initialized")
            
        except Exception as e:
            self.logger.error(f"Security ML model initialization failed: {e}")
            self.enable_ml_detection = False
    
    def _initialize_threat_patterns(self):
        \"\"\"Initialize known threat patterns\"\"\"
        
        # Brute force attack pattern
        brute_force = ThreatPattern(
            pattern_id="brute_force",
            pattern_name="Brute Force Attack",
            event_types=[SecurityEventType.AUTHENTICATION_FAILURE],
            time_window=300,  # 5 minutes
            min_occurrences=10,
            severity_multiplier=1.5
        )
        
        # SQL injection pattern
        sql_injection = ThreatPattern(
            pattern_id="sql_injection",
            pattern_name="SQL Injection Attack",
            event_types=[SecurityEventType.SQL_INJECTION],
            time_window=60,  # 1 minute
            min_occurrences=3,
            severity_multiplier=2.0
        )
        
        # DDoS attack pattern
        ddos_attack = ThreatPattern(
            pattern_id="ddos_attack",
            pattern_name="Distributed Denial of Service",
            event_types=[SecurityEventType.DOS_ATTACK, SecurityEventType.ANOMALOUS_TRAFFIC],
            time_window=180,  # 3 minutes
            min_occurrences=50,
            severity_multiplier=2.5
        )
        
        # Data exfiltration pattern
        data_exfiltration = ThreatPattern(
            pattern_id="data_exfiltration",
            pattern_name="Data Exfiltration Attempt",
            event_types=[SecurityEventType.DATA_EXFILTRATION, SecurityEventType.SUSPICIOUS_ACTIVITY],
            time_window=600,  # 10 minutes
            min_occurrences=5,
            severity_multiplier=3.0
        )
        
        self.threat_patterns = {
            pattern.pattern_id: pattern
            for pattern in [brute_force, sql_injection, ddos_attack, data_exfiltration]
        }
        
        # Initialize attack signatures
        self.attack_signatures = {
            'sql_injection': [
                r"(union\s+select|drop\s+table|exec\s+xp_)",
                r"(\'\s*or\s*\'\s*=\s*\'|\'\s*or\s*1\s*=\s*1)",
                r"(insert\s+into|delete\s+from|update\s+.+set)"
            ],
            'xss': [
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"on\w+\s*="
            ],
            'path_traversal': [
                r"\.\./",
                r"%2e%2e%2f",
                r"..\\\\",
                r"%2e%2e%5c"
            ]
        }
    
    async def process_security_event(self,
                                   event_type: SecurityEventType,
                                   source_ip: str,
                                   target: str,
                                   description: str,
                                   raw_data: Dict[str, Any] = None,
                                   user_id: str = None,
                                   session_id: str = None) -> SecurityEvent:
        \"\"\"Process and analyze security event with ML enhancement\"\"\"
        
        try:
            # Create security event
            event = SecurityEvent(
                event_id=f"sec_{int(time.time() * 1000)}_{hash(source_ip) % 10000}",
                timestamp=datetime.now(),
                event_type=event_type,
                threat_level=ThreatLevel.LOW,  # Initial assessment
                source_ip=source_ip,
                target=target,
                description=description,
                raw_data=raw_data or {},
                user_id=user_id,
                session_id=session_id
            )
            
            # ML Enhancement
            if self.enable_ml_detection:
                await self._enhance_event_with_ml(event)
            
            # Pattern matching
            await self._check_threat_patterns(event)
            
            # Determine threat level
            event.threat_level = await self._assess_threat_level(event)
            
            # Store event
            with self.security_lock:
                self.security_events.append(event)
                self.security_stats['events_processed'] += 1
                
                if event.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
                    self.security_stats['threats_detected'] += 1
            
            # Determine response actions
            response_actions = await self._determine_response_actions(event)
            event.response_actions = response_actions
            
            # Execute response actions
            if response_actions:
                await self._execute_response_actions(event, response_actions)
            
            self.logger.info(f"Security event processed: {event.event_type.value} from {source_ip}")
            
            return event
            
        except Exception as e:
            self.logger.error(f"Security event processing failed: {e}")
            return None
    
    async def _enhance_event_with_ml(self, event: SecurityEvent):
        \"\"\"Enhance security event with ML analysis\"\"\"
        
        try:
            with self.ml_lock:
                # Extract features for ML analysis
                features = await self._extract_security_features(event)
                
                # Anomaly detection
                if self.anomaly_detector and len(self.security_feature_history) >= 50:
                    anomaly_score = self.anomaly_detector.decision_function([features])[0]
                    event.anomaly_score = float(anomaly_score)
                    
                    if anomaly_score < self.anomaly_threshold:
                        event.ml_confidence += 0.3
                        self.security_stats['ml_detections'] += 1
                
                # Threat classification
                if self.threat_classifier and len(self.security_feature_history) >= 100:
                    threat_prob = await self._predict_threat_probability(features)
                    event.threat_probability = threat_prob
                    
                    if threat_prob > self.threat_threshold:
                        event.ml_confidence += 0.4
                
                # Attack vector prediction
                if self.attack_predictor:
                    attack_vectors = await self._predict_attack_vectors(features)
                    event.attack_vector_prediction = attack_vectors
                    event.ml_confidence += 0.3
                
                # Store features for model training
                self.security_feature_history.append(features)
                
        except Exception as e:
            self.logger.error(f"ML security enhancement failed: {e}")
    
    def _extract_security_features(self, event: SecurityEvent) -> np.ndarray:
        \"\"\"Extract ML features from security event\"\"\"
        
        try:
            # IP address features
            ip_features = await self._extract_ip_features(event.source_ip)
            
            # Temporal features
            hour = event.timestamp.hour
            day_of_week = event.timestamp.weekday()
            
            # Event type encoding
            event_type_encoded = list(SecurityEventType).index(event.event_type)
            
            # Text analysis features
            description_length = len(event.description)
            contains_suspicious_keywords = int(await self._contains_suspicious_keywords(event.description))
            
            # User and session features
            user_id_hash = hash(event.user_id or "") % 1000
            session_features = await self._extract_session_features(event.session_id)
            
            # Historical behavior features
            ip_history_features = await self._extract_ip_history_features(event.source_ip)
            
            # Combine all features
            features = np.array([
                event_type_encoded,
                hour / 24.0,
                day_of_week / 7.0,
                description_length / 1000.0,  # Normalize
                contains_suspicious_keywords,
                user_id_hash / 1000.0,
                *ip_features,
                *session_features,
                *ip_history_features
            ])
            
            return features.astype(np.float64)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.zeros(15)  # Return default feature vector
    
    async def _extract_ip_features(self, ip_str: str) -> List[float]:
        \"\"\"Extract features from IP address\"\"\"
        
        try:
            # IP address analysis
            is_private = int(ipaddress.ip_address(ip_str).is_private)
            is_known_malicious = int(ip_str in self.known_malicious_ips)
            
            # Geolocation features (simplified)
            ip_hash = hash(ip_str) % 1000
            geo_region = ip_hash / 1000.0  # Simplified geographic encoding
            
            return [is_private, is_known_malicious, geo_region]
            
        except Exception as e:
            return [0.0, 0.0, 0.5]  # Default values
    
    async def _check_threat_patterns(self, event: SecurityEvent):
        \"\"\"Check event against known threat patterns\"\"\"
        
        try:
            for pattern_id, pattern in self.threat_patterns.items():
                if event.event_type in pattern.event_types:
                    # Check for pattern match in recent events
                    recent_events = [
                        e for e in list(self.security_events)[-100:]
                        if (datetime.now() - e.timestamp).total_seconds() <= pattern.time_window
                        and e.source_ip == event.source_ip
                        and e.event_type in pattern.event_types
                    ]
                    
                    if len(recent_events) >= pattern.min_occurrences:
                        # Pattern detected!
                        pattern.detected_count += 1
                        pattern.last_detected = datetime.now()
                        
                        # Increase threat level
                        current_level_value = event.threat_level.value
                        new_level_value = min(5, int(current_level_value * pattern.severity_multiplier))
                        event.threat_level = ThreatLevel(new_level_value)
                        
                        self.logger.warning(f"Threat pattern detected: {pattern.pattern_name} from {event.source_ip}")
                        
                        # Create incident if critical
                        if event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
                            await self._create_security_incident(event, pattern)
        
        except Exception as e:
            self.logger.error(f"Threat pattern checking failed: {e}")
    
    async def _assess_threat_level(self, event: SecurityEvent) -> ThreatLevel:
        \"\"\"Assess overall threat level using ML and rule-based analysis\"\"\"
        
        try:
            base_level = event.threat_level.value
            
            # ML confidence factor
            if event.ml_confidence > 0.8:
                base_level = min(5, base_level + 1)
            elif event.ml_confidence > 0.6:
                base_level = min(5, base_level + 0.5)
            
            # Anomaly score factor
            if event.anomaly_score < -0.8:
                base_level = min(5, base_level + 2)
            elif event.anomaly_score < self.anomaly_threshold:
                base_level = min(5, base_level + 1)
            
            # Threat probability factor
            if event.threat_probability > 0.9:
                base_level = min(5, base_level + 2)
            elif event.threat_probability > self.threat_threshold:
                base_level = min(5, base_level + 1)
            
            # Known malicious IP factor
            if event.source_ip in self.known_malicious_ips:
                base_level = min(5, base_level + 2)
            
            # Attack vector prediction factor
            if len(event.attack_vector_prediction) > 2:
                base_level = min(5, base_level + 1)
            
            return ThreatLevel(int(base_level))
            
        except Exception as e:
            self.logger.error(f"Threat level assessment failed: {e}")
            return ThreatLevel.MEDIUM
    
    async def _determine_response_actions(self, event: SecurityEvent) -> List[ResponseAction]:
        \"\"\"Determine appropriate response actions based on threat level\"\"\"
        
        actions = []
        
        try:
            # Always log
            actions.append(ResponseAction.LOG_ONLY)
            
            if event.threat_level == ThreatLevel.LOW:
                # Low threats: just monitor
                pass
            
            elif event.threat_level == ThreatLevel.MEDIUM:
                # Medium threats: alert and rate limit
                actions.extend([ResponseAction.ALERT, ResponseAction.RATE_LIMIT])
            
            elif event.threat_level == ThreatLevel.HIGH:
                # High threats: block and alert
                actions.extend([ResponseAction.ALERT, ResponseAction.BLOCK_IP])
                
                if self.auto_remediation:
                    actions.append(ResponseAction.AUTO_REMEDIATE)
            
            elif event.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]:
                # Critical/Emergency: full response
                actions.extend([
                    ResponseAction.ALERT,
                    ResponseAction.BLOCK_IP,
                    ResponseAction.ESCALATE,
                    ResponseAction.QUARANTINE
                ])
                
                if self.auto_remediation:
                    actions.append(ResponseAction.AUTO_REMEDIATE)
            
            return actions
            
        except Exception as e:
            self.logger.error(f"Response action determination failed: {e}")
            return [ResponseAction.LOG_ONLY]
    
    async def _execute_response_actions(self, event: SecurityEvent, actions: List[ResponseAction]):
        \"\"\"Execute security response actions\"\"\"
        
        try:
            for action in actions:
                if action == ResponseAction.LOG_ONLY:
                    self.logger.info(f"Security event logged: {event.event_id}")
                
                elif action == ResponseAction.ALERT:
                    await self._send_security_alert(event)
                
                elif action == ResponseAction.BLOCK_IP:
                    await self._block_ip_address(event.source_ip)
                    self.security_stats['attacks_blocked'] += 1
                
                elif action == ResponseAction.RATE_LIMIT:
                    await self._apply_rate_limiting(event.source_ip)
                
                elif action == ResponseAction.QUARANTINE:
                    await self._quarantine_session(event.session_id)
                
                elif action == ResponseAction.ESCALATE:
                    await self._escalate_incident(event)
                
                elif action == ResponseAction.AUTO_REMEDIATE:
                    await self._auto_remediate_threat(event)
                    event.auto_remediated = True
                    self.security_stats['auto_remediations'] += 1
            
        except Exception as e:
            self.logger.error(f"Response action execution failed: {e}")
    
    async def _security_monitoring_loop(self):
        \"\"\"Main security monitoring loop\"\"\"
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.monitoring_interval)
                
                # Update security metrics
                await self._update_security_metrics()
                
                # Check for expired IP blocks
                await self._cleanup_expired_blocks()
                
                # Analyze recent security trends
                await self._analyze_security_trends()
                
                # Update threat intelligence
                await self._update_threat_intelligence()
                
            except Exception as e:
                self.logger.error(f"Security monitoring loop error: {e}")
                await asyncio.sleep(5)
    
    async def _ml_security_loop(self):
        \"\"\"ML security analysis and model training loop\"\"\"
        
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                
                if len(self.security_feature_history) >= 200:
                    # Retrain ML models
                    await self._retrain_security_models()
                    
                    # Update threat predictions
                    await self._update_threat_predictions()
                    
                    # Generate security insights
                    await self._generate_security_insights()
                
            except Exception as e:
                self.logger.error(f"ML security loop error: {e}")
                await asyncio.sleep(30)
    
    def get_security_dashboard(self) -> Dict[str, Any]:
        \"\"\"Get comprehensive security dashboard\"\"\"
        
        # Recent threat summary
        recent_events = list(self.security_events)[-100:]
        threat_distribution = defaultdict(int)
        for event in recent_events:
            threat_distribution[event.threat_level] += 1
        
        # Active incidents
        active_incidents = len(self.active_incidents)
        
        # Blocked IPs count
        active_blocks = len([ip for ip, block_time in self.blocked_ips.items() 
                           if datetime.now() - block_time < timedelta(seconds=self.ip_block_duration)])
        
        return {
            'security_overview': {
                'total_events_24h': len([e for e in recent_events 
                                       if (datetime.now() - e.timestamp) < timedelta(hours=24)]),
                'threats_detected_24h': len([e for e in recent_events 
                                           if e.threat_level.value >= 3 and 
                                           (datetime.now() - e.timestamp) < timedelta(hours=24)]),
                'active_incidents': active_incidents,
                'blocked_ips': active_blocks,
                'ml_detection_rate': self.security_stats['ml_detections'] / max(1, self.security_stats['events_processed'])
            },
            'threat_distribution': {level.name: count for level, count in threat_distribution.items()},
            'statistics': self.security_stats.copy(),
            'ml_status': {
                'ml_detection_enabled': self.enable_ml_detection,
                'feature_history_size': len(self.security_feature_history),
                'threat_predictions': len(self.ml_threat_predictions),
                'behavioral_anomalies': len(self.behavioral_anomalies)
            },
            'recent_insights': self.security_insights[-5:] if self.security_insights else []
        }
    
    async def shutdown(self):
        \"\"\"Graceful shutdown of security guardian\"\"\"
        
        self.logger.info("Shutting down ML security guardian...")
        self.shutdown_event.set()
        await asyncio.sleep(1)
        self.logger.info("ML security guardian shutdown complete")