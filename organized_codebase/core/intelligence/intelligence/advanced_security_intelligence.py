"""
Advanced Security Intelligence System
====================================

Enterprise-grade security intelligence with threat detection, integrity verification,
access monitoring, and real-time security analytics. Extracted from archive
security and integrity verification components.

Provides comprehensive security monitoring and threat response capabilities.
"""

import hashlib
import hmac
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
import os
import uuid

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security monitoring levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats"""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_BREACH = "data_breach"
    INTEGRITY_VIOLATION = "integrity_violation"
    INJECTION_ATTACK = "injection_attack"
    BRUTE_FORCE = "brute_force"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    MALICIOUS_PAYLOAD = "malicious_payload"


class AccessAction(Enum):
    """Types of access actions"""
    LOGIN = "login"
    LOGOUT = "logout"
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str = field(default_factory=lambda: f"sec_{uuid.uuid4().hex[:8]}")
    event_type: ThreatType = ThreatType.ANOMALOUS_BEHAVIOR
    severity: SecurityLevel = SecurityLevel.LOW
    timestamp: datetime = field(default_factory=datetime.now)
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[AccessAction] = None
    details: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    resolved: bool = False
    response_actions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'severity': self.severity.value,
            'timestamp': self.timestamp.isoformat(),
            'source_ip': self.source_ip,
            'user_id': self.user_id,
            'resource': self.resource,
            'action': self.action.value if self.action else None,
            'details': self.details,
            'risk_score': self.risk_score,
            'resolved': self.resolved,
            'response_actions': self.response_actions
        }


@dataclass
class IntegrityRecord:
    """Data integrity verification record"""
    record_id: str = field(default_factory=lambda: f"int_{uuid.uuid4().hex[:8]}")
    data_hash: str = ""
    chain_hash: str = ""
    signature: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    sequence_number: int = 0
    verification_status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPattern:
    """User access pattern analysis"""
    user_id: str
    typical_hours: Set[int] = field(default_factory=set)
    typical_resources: Set[str] = field(default_factory=set)
    typical_actions: Set[AccessAction] = field(default_factory=set)
    typical_ips: Set[str] = field(default_factory=set)
    access_frequency: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedSecurityIntelligence:
    """Enterprise security intelligence system with threat detection and response"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Security configuration
        self.security_level = SecurityLevel(self.config.get('security_level', 'medium'))
        self.enable_real_time_monitoring = self.config.get('enable_real_time', True)
        self.threat_threshold = self.config.get('threat_threshold', 5.0)
        self.max_login_attempts = self.config.get('max_login_attempts', 5)
        
        # Security data storage
        self.security_events: deque = deque(maxlen=10000)
        self.integrity_records: Dict[str, IntegrityRecord] = {}
        self.access_patterns: Dict[str, AccessPattern] = {}
        self.active_threats: Dict[str, SecurityEvent] = {}
        
        # Threat detection
        self.failed_logins: Dict[str, List[datetime]] = defaultdict(list)
        self.suspicious_ips: Set[str] = set()
        self.blacklisted_patterns: List[str] = []
        self.access_history: deque = deque(maxlen=1000)
        
        # Chain verification
        self.chain_sequence = 0
        self.last_chain_hash = "genesis"
        self.secret_key = self.config.get('secret_key', 'default_secret_key_change_in_production')
        
        # Performance tracking
        self.security_stats = {
            'events_processed': 0,
            'threats_detected': 0,
            'integrity_violations': 0,
            'blocked_attempts': 0,
            'false_positives': 0
        }
        
        # Background monitoring
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Initialize components
        self._initialize_security_db()
        self._load_threat_patterns()
        
        # Start monitoring
        if self.enable_real_time_monitoring:
            self.monitoring_thread.start()
            self.analysis_thread.start()
        
        self.logger.info("Advanced Security Intelligence initialized")
    
    def _initialize_security_db(self):
        """Initialize security database"""
        try:
            db_path = self.config.get('security_db_path', 'data/security.db')
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS security_events (
                        event_id TEXT PRIMARY KEY,
                        event_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source_ip TEXT,
                        user_id TEXT,
                        resource TEXT,
                        details TEXT,
                        risk_score REAL,
                        resolved BOOLEAN
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS integrity_records (
                        record_id TEXT PRIMARY KEY,
                        data_hash TEXT NOT NULL,
                        chain_hash TEXT NOT NULL,
                        signature TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        sequence_number INTEGER,
                        verification_status TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Security database initialization failed: {e}")
    
    def _load_threat_patterns(self):
        """Load known threat patterns"""
        # Common malicious patterns
        self.blacklisted_patterns = [
            r'<script[^>]*>.*?</script>',  # XSS
            r'union\s+select',             # SQL injection
            r'\.\./',                      # Path traversal
            r'eval\s*\(',                  # Code injection
            r'exec\s*\(',                  # Command injection
            r'<iframe[^>]*>',              # Iframe injection
            r'javascript:',                # JavaScript protocol
            r'vbscript:',                  # VBScript protocol
        ]
    
    def log_access_attempt(self, user_id: str, resource: str, action: AccessAction,
                          source_ip: str, success: bool = True,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Log access attempt and perform security analysis
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Type of action
            source_ip: Source IP address
            success: Whether access was successful
            metadata: Additional metadata
            
        Returns:
            Event ID if threat detected, None otherwise
        """
        with self.lock:
            timestamp = datetime.now()
            
            # Update access history
            access_record = {
                'user_id': user_id,
                'resource': resource,
                'action': action.value,
                'source_ip': source_ip,
                'timestamp': timestamp,
                'success': success,
                'metadata': metadata or {}
            }
            self.access_history.append(access_record)
            
            # Update user access patterns
            self._update_access_pattern(user_id, resource, action, source_ip, timestamp)
            
            # Check for threats
            threat_event = self._analyze_access_attempt(access_record)
            
            if not success:
                # Track failed attempts
                self.failed_logins[user_id].append(timestamp)
                
                # Clean old failed attempts
                cutoff_time = timestamp - timedelta(hours=1)
                self.failed_logins[user_id] = [
                    attempt for attempt in self.failed_logins[user_id]
                    if attempt > cutoff_time
                ]
                
                # Check for brute force
                if len(self.failed_logins[user_id]) >= self.max_login_attempts:
                    threat_event = self._create_security_event(
                        ThreatType.BRUTE_FORCE,
                        SecurityLevel.HIGH,
                        source_ip=source_ip,
                        user_id=user_id,
                        resource=resource,
                        action=action,
                        details={
                            'failed_attempts': len(self.failed_logins[user_id]),
                            'time_window': '1 hour'
                        }
                    )
            
            # Update statistics
            self.security_stats['events_processed'] += 1
            
            return threat_event.event_id if threat_event else None
    
    def verify_data_integrity(self, data: Dict[str, Any], 
                            expected_hash: Optional[str] = None) -> Tuple[bool, str]:
        """
        Verify data integrity with chain validation
        
        Args:
            data: Data to verify
            expected_hash: Expected hash value
            
        Returns:
            Tuple of (is_valid, record_id)
        """
        with self.lock:
            try:
                # Generate data hash
                data_str = json.dumps(data, sort_keys=True)
                data_hash = hashlib.sha256(data_str.encode()).hexdigest()
                
                # Check against expected hash
                if expected_hash and data_hash != expected_hash:
                    self._create_security_event(
                        ThreatType.INTEGRITY_VIOLATION,
                        SecurityLevel.HIGH,
                        details={
                            'expected_hash': expected_hash,
                            'actual_hash': data_hash,
                            'violation_type': 'hash_mismatch'
                        }
                    )
                    return False, ""
                
                # Generate chain hash
                self.chain_sequence += 1
                chain_data = f"{self.last_chain_hash}:{data_hash}:{self.chain_sequence}"
                chain_hash = hashlib.sha256(chain_data.encode()).hexdigest()
                
                # Generate signature
                signature = hmac.new(
                    self.secret_key.encode(),
                    chain_hash.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                # Create integrity record
                record = IntegrityRecord(
                    data_hash=data_hash,
                    chain_hash=chain_hash,
                    signature=signature,
                    sequence_number=self.chain_sequence,
                    verification_status="verified"
                )
                
                self.integrity_records[record.record_id] = record
                self.last_chain_hash = chain_hash
                
                self.logger.debug(f"Data integrity verified: {record.record_id}")
                return True, record.record_id
                
            except Exception as e:
                self.logger.error(f"Integrity verification failed: {e}")
                
                self._create_security_event(
                    ThreatType.INTEGRITY_VIOLATION,
                    SecurityLevel.CRITICAL,
                    details={'error': str(e), 'verification_failed': True}
                )
                
                return False, ""
    
    def scan_for_threats(self, content: str, content_type: str = "text") -> List[Dict[str, Any]]:
        """
        Scan content for security threats
        
        Args:
            content: Content to scan
            content_type: Type of content
            
        Returns:
            List of detected threats
        """
        threats = []
        
        try:
            # Check against blacklisted patterns
            import re
            
            for pattern in self.blacklisted_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    threats.append({
                        'threat_type': ThreatType.MALICIOUS_PAYLOAD.value,
                        'pattern': pattern,
                        'matches': matches,
                        'severity': SecurityLevel.HIGH.value,
                        'content_type': content_type
                    })
            
            # Check for suspicious file extensions
            if content_type == "filename":
                suspicious_extensions = ['.exe', '.bat', '.cmd', '.scr', '.pif', '.com']
                for ext in suspicious_extensions:
                    if content.lower().endswith(ext):
                        threats.append({
                            'threat_type': ThreatType.MALICIOUS_PAYLOAD.value,
                            'pattern': f'suspicious_extension_{ext}',
                            'severity': SecurityLevel.MEDIUM.value,
                            'content_type': content_type
                        })
            
            # Check for excessively long content (potential buffer overflow)
            if len(content) > 10000:  # 10KB threshold
                threats.append({
                    'threat_type': ThreatType.ANOMALOUS_BEHAVIOR.value,
                    'pattern': 'excessive_length',
                    'severity': SecurityLevel.MEDIUM.value,
                    'content_length': len(content)
                })
            
            # Create security events for detected threats
            if threats:
                self._create_security_event(
                    ThreatType.MALICIOUS_PAYLOAD,
                    SecurityLevel.HIGH,
                    details={
                        'threats_detected': len(threats),
                        'content_type': content_type,
                        'threat_details': threats
                    }
                )
            
        except Exception as e:
            self.logger.error(f"Threat scanning failed: {e}")
        
        return threats
    
    def _analyze_access_attempt(self, access_record: Dict[str, Any]) -> Optional[SecurityEvent]:
        """Analyze access attempt for threats"""
        user_id = access_record['user_id']
        source_ip = access_record['source_ip']
        resource = access_record['resource']
        action = AccessAction(access_record['action'])
        timestamp = access_record['timestamp']
        
        # Check against known patterns
        if user_id in self.access_patterns:
            pattern = self.access_patterns[user_id]
            risk_score = 0.0
            anomalies = []
            
            # Check time anomaly
            current_hour = timestamp.hour
            if current_hour not in pattern.typical_hours and len(pattern.typical_hours) > 0:
                risk_score += 2.0
                anomalies.append("unusual_time")
            
            # Check IP anomaly
            if source_ip not in pattern.typical_ips and len(pattern.typical_ips) > 0:
                risk_score += 3.0
                anomalies.append("unusual_ip")
            
            # Check resource anomaly
            if resource not in pattern.typical_resources and len(pattern.typical_resources) > 0:
                risk_score += 1.5
                anomalies.append("unusual_resource")
            
            # Check action anomaly
            if action not in pattern.typical_actions and len(pattern.typical_actions) > 0:
                risk_score += 1.0
                anomalies.append("unusual_action")
            
            # Check if IP is suspicious
            if source_ip in self.suspicious_ips:
                risk_score += 4.0
                anomalies.append("suspicious_ip")
            
            # Create threat if risk score is high
            if risk_score >= self.threat_threshold:
                severity = SecurityLevel.CRITICAL if risk_score >= 8.0 else SecurityLevel.HIGH
                
                return self._create_security_event(
                    ThreatType.ANOMALOUS_BEHAVIOR,
                    severity,
                    source_ip=source_ip,
                    user_id=user_id,
                    resource=resource,
                    action=action,
                    details={
                        'risk_score': risk_score,
                        'anomalies': anomalies,
                        'analysis_type': 'access_pattern'
                    }
                )
        
        return None
    
    def _update_access_pattern(self, user_id: str, resource: str, action: AccessAction,
                              source_ip: str, timestamp: datetime):
        """Update user access pattern"""
        if user_id not in self.access_patterns:
            self.access_patterns[user_id] = AccessPattern(user_id=user_id)
        
        pattern = self.access_patterns[user_id]
        pattern.typical_hours.add(timestamp.hour)
        pattern.typical_resources.add(resource)
        pattern.typical_actions.add(action)
        pattern.typical_ips.add(source_ip)
        pattern.last_updated = timestamp
        
        # Limit set sizes to prevent memory bloat
        if len(pattern.typical_ips) > 10:
            # Keep only most recent IPs
            pattern.typical_ips = set(list(pattern.typical_ips)[-5:])
        
        if len(pattern.typical_resources) > 50:
            pattern.typical_resources = set(list(pattern.typical_resources)[-25:])
    
    def _create_security_event(self, threat_type: ThreatType, severity: SecurityLevel,
                              source_ip: Optional[str] = None, user_id: Optional[str] = None,
                              resource: Optional[str] = None, action: Optional[AccessAction] = None,
                              details: Optional[Dict[str, Any]] = None) -> SecurityEvent:
        """Create and log security event"""
        event = SecurityEvent(
            event_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            resource=resource,
            action=action,
            details=details or {}
        )
        
        # Calculate risk score
        event.risk_score = self._calculate_risk_score(event)
        
        # Store event
        self.security_events.append(event)
        self.active_threats[event.event_id] = event
        
        # Update statistics
        self.security_stats['threats_detected'] += 1
        
        # Log based on severity
        if severity == SecurityLevel.CRITICAL:
            self.logger.critical(f"CRITICAL SECURITY THREAT: {threat_type.value} - {event.event_id}")
        elif severity == SecurityLevel.HIGH:
            self.logger.error(f"Security threat detected: {threat_type.value} - {event.event_id}")
        else:
            self.logger.warning(f"Security event: {threat_type.value} - {event.event_id}")
        
        return event
    
    def _calculate_risk_score(self, event: SecurityEvent) -> float:
        """Calculate risk score for security event"""
        base_score = {
            SecurityLevel.LOW: 1.0,
            SecurityLevel.MEDIUM: 3.0,
            SecurityLevel.HIGH: 6.0,
            SecurityLevel.CRITICAL: 9.0
        }[event.severity]
        
        # Adjust based on threat type
        threat_multipliers = {
            ThreatType.DATA_BREACH: 2.0,
            ThreatType.INTEGRITY_VIOLATION: 1.8,
            ThreatType.PRIVILEGE_ESCALATION: 1.6,
            ThreatType.INJECTION_ATTACK: 1.5,
            ThreatType.BRUTE_FORCE: 1.3,
            ThreatType.UNAUTHORIZED_ACCESS: 1.2,
            ThreatType.MALICIOUS_PAYLOAD: 1.4,
            ThreatType.ANOMALOUS_BEHAVIOR: 1.0
        }
        
        multiplier = threat_multipliers.get(event.event_type, 1.0)
        
        # Additional factors from details
        additional_score = 0.0
        if 'risk_score' in event.details:
            additional_score = event.details['risk_score']
        
        return min(10.0, base_score * multiplier + additional_score)
    
    def _monitoring_loop(self):
        """Background security monitoring loop"""
        while self.is_monitoring:
            try:
                # Check for expired events
                self._cleanup_expired_events()
                
                # Update suspicious IP list
                self._update_suspicious_ips()
                
                # Generate security alerts
                self._generate_security_alerts()
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Security monitoring loop error: {e}")
                time.sleep(30)
    
    def _analysis_loop(self):
        """Background security analysis loop"""
        while self.is_monitoring:
            try:
                # Analyze access patterns
                self._analyze_access_patterns()
                
                # Update threat intelligence
                self._update_threat_intelligence()
                
                time.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Security analysis loop error: {e}")
                time.sleep(60)
    
    def _cleanup_expired_events(self):
        """Clean up old security events"""
        cutoff_time = datetime.now() - timedelta(days=30)
        
        # Remove old events from active threats
        expired_events = [
            event_id for event_id, event in self.active_threats.items()
            if event.timestamp < cutoff_time
        ]
        
        for event_id in expired_events:
            del self.active_threats[event_id]
    
    def _update_suspicious_ips(self):
        """Update suspicious IP list based on recent events"""
        # Find IPs with multiple high-severity events
        ip_threat_counts = defaultdict(int)
        
        recent_cutoff = datetime.now() - timedelta(hours=24)
        
        for event in self.security_events:
            if (event.timestamp > recent_cutoff and 
                event.source_ip and 
                event.severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]):
                ip_threat_counts[event.source_ip] += 1
        
        # Add IPs with 3+ threats to suspicious list
        for ip, count in ip_threat_counts.items():
            if count >= 3:
                self.suspicious_ips.add(ip)
        
        # Remove old suspicious IPs (7 days)
        # In a real implementation, this would be more sophisticated
    
    def _generate_security_alerts(self):
        """Generate security alerts for high-priority events"""
        critical_events = [
            event for event in self.active_threats.values()
            if (event.severity == SecurityLevel.CRITICAL and 
                not event.resolved and
                (datetime.now() - event.timestamp).total_seconds() < 3600)  # Last hour
        ]
        
        if critical_events:
            self.logger.critical(f"SECURITY ALERT: {len(critical_events)} critical threats active")
    
    def _analyze_access_patterns(self):
        """Analyze access patterns for anomalies"""
        # This is a simplified implementation
        # Real implementation would use more sophisticated ML techniques
        pass
    
    def _update_threat_intelligence(self):
        """Update threat intelligence data"""
        # This would integrate with external threat intelligence feeds
        # For now, just update internal statistics
        self.security_stats['integrity_violations'] = sum(
            1 for event in self.security_events
            if event.event_type == ThreatType.INTEGRITY_VIOLATION
        )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        with self.lock:
            active_threats_by_severity = defaultdict(int)
            for event in self.active_threats.values():
                if not event.resolved:
                    active_threats_by_severity[event.severity.value] += 1
            
            return {
                'security_level': self.security_level.value,
                'monitoring_active': self.is_monitoring,
                'statistics': self.security_stats.copy(),
                'active_threats': {
                    'total': len(self.active_threats),
                    'by_severity': dict(active_threats_by_severity)
                },
                'suspicious_ips': len(self.suspicious_ips),
                'integrity_records': len(self.integrity_records),
                'access_patterns': len(self.access_patterns),
                'chain_sequence': self.chain_sequence,
                'last_updated': datetime.now().isoformat()
            }
    
    def get_threat_events(self, severity: Optional[SecurityLevel] = None,
                         hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent threat events"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        events = [
            event.to_dict() for event in self.security_events
            if (event.timestamp > cutoff_time and
                (severity is None or event.severity == severity))
        ]
        
        return sorted(events, key=lambda x: x['timestamp'], reverse=True)
    
    def resolve_threat(self, event_id: str, resolution_notes: str = "") -> bool:
        """Mark threat as resolved"""
        with self.lock:
            if event_id in self.active_threats:
                event = self.active_threats[event_id]
                event.resolved = True
                event.response_actions.append(f"Resolved: {resolution_notes}")
                
                self.logger.info(f"Threat {event_id} resolved: {resolution_notes}")
                return True
            
            return False
    
    def shutdown(self):
        """Gracefully shutdown security system"""
        self.logger.info("Shutting down Advanced Security Intelligence")
        self.is_monitoring = False
        
        # Wait for threads to complete
        for thread in [self.monitoring_thread, self.analysis_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)
        
        self.logger.info("Advanced Security Intelligence shutdown complete")


# Global security intelligence instance
advanced_security_intelligence = AdvancedSecurityIntelligence()

# Export
__all__ = [
    'SecurityLevel', 'ThreatType', 'AccessAction',
    'SecurityEvent', 'IntegrityRecord', 'AccessPattern',
    'AdvancedSecurityIntelligence', 'advanced_security_intelligence'
]