"""
MetaGPT Derived Enterprise Audit Logging and Compliance Module
Extracted from MetaGPT repository patterns for enterprise security
Enhanced for comprehensive audit trails and compliance tracking
"""

import json
import time
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from .error_handler import SecurityError, security_error_handler


class AuditEventType(Enum):
    """Audit event types based on MetaGPT patterns"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGE = "permission_change"
    DATA_ACCESS = "data_access"
    DATA_MODIFICATION = "data_modification"
    SYSTEM_CONFIGURATION = "system_configuration"
    SECURITY_VIOLATION = "security_violation"
    COMPLIANCE_CHECK = "compliance_check"
    BACKUP_OPERATION = "backup_operation"
    EXPORT_OPERATION = "export_operation"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    SOC2 = "soc2"
    GDPR = "gdpr"
    HIPAA = "hipaa"
    ISO27001 = "iso27001"
    PCI_DSS = "pci_dss"
    CUSTOM = "custom"


@dataclass
class AuditEvent:
    """Enterprise audit event with comprehensive metadata"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: AuditEventType = AuditEventType.DATA_ACCESS
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: str = ""
    session_id: Optional[str] = None
    resource: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool = True
    risk_level: str = "low"
    compliance_tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        # Add compliance tags based on event type
        if self.event_type == AuditEventType.DATA_ACCESS:
            self.compliance_tags.update(["gdpr", "soc2"])
        elif self.event_type == AuditEventType.SECURITY_VIOLATION:
            self.compliance_tags.update(["soc2", "iso27001"])
        elif self.event_type == AuditEventType.PERMISSION_CHANGE:
            self.compliance_tags.update(["soc2", "iso27001", "pci_dss"])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert audit event to dictionary for storage"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'session_id': self.session_id,
            'resource': self.resource,
            'action': self.action,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'success': self.success,
            'risk_level': self.risk_level,
            'compliance_tags': list(self.compliance_tags)
        }


@dataclass
class ComplianceRule:
    """Compliance rule definition"""
    rule_id: str
    framework: ComplianceFramework
    title: str
    description: str
    required_events: List[AuditEventType]
    retention_days: int = 365
    alert_threshold: int = 5
    severity: str = "medium"
    
    def matches_event(self, event: AuditEvent) -> bool:
        """Check if event matches this compliance rule"""
        return (
            event.event_type in self.required_events and
            self.framework.value in event.compliance_tags
        )


class AuditStorage:
    """Secure audit log storage with integrity protection"""
    
    def __init__(self, storage_path: str = "audit_logs"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.current_log_file = None
        self.log_rotation_size = 50 * 1024 * 1024  # 50MB
        self.logger = logging.getLogger(__name__)
        
        # Create hash chain for integrity
        self.hash_chain: List[str] = []
        self._load_hash_chain()
    
    def store_event(self, event: AuditEvent) -> bool:
        """Store audit event with integrity protection"""
        try:
            # Get current log file
            log_file = self._get_current_log_file()
            
            # Create event record with hash chain
            event_data = event.to_dict()
            previous_hash = self.hash_chain[-1] if self.hash_chain else "genesis"
            event_hash = self._calculate_event_hash(event_data, previous_hash)
            
            record = {
                'event': event_data,
                'hash': event_hash,
                'previous_hash': previous_hash
            }
            
            # Write to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
            
            # Update hash chain
            self.hash_chain.append(event_hash)
            self._save_hash_chain()
            
            # Check for rotation
            self._check_log_rotation(log_file)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store audit event: {e}")
            return False
    
    def retrieve_events(self, start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None,
                       event_types: Optional[List[AuditEventType]] = None,
                       user_id: Optional[str] = None) -> List[AuditEvent]:
        """Retrieve audit events with filtering"""
        events = []
        
        try:
            # Get all log files in time range
            log_files = self._get_log_files_in_range(start_time, end_time)
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            event_data = record['event']
                            
                            # Parse timestamp
                            event_time = datetime.fromisoformat(event_data['timestamp'])
                            
                            # Apply filters
                            if start_time and event_time < start_time:
                                continue
                            if end_time and event_time > end_time:
                                continue
                            if event_types and AuditEventType(event_data['event_type']) not in event_types:
                                continue
                            if user_id and event_data['user_id'] != user_id:
                                continue
                            
                            # Create AuditEvent object
                            event = AuditEvent(
                                event_id=event_data['event_id'],
                                event_type=AuditEventType(event_data['event_type']),
                                timestamp=event_time,
                                user_id=event_data['user_id'],
                                session_id=event_data['session_id'],
                                resource=event_data['resource'],
                                action=event_data['action'],
                                details=event_data['details'],
                                ip_address=event_data['ip_address'],
                                user_agent=event_data['user_agent'],
                                success=event_data['success'],
                                risk_level=event_data['risk_level'],
                                compliance_tags=set(event_data['compliance_tags'])
                            )
                            
                            events.append(event)
                            
                        except (json.JSONDecodeError, KeyError, ValueError) as e:
                            self.logger.warning(f"Invalid audit record: {e}")
                            continue
            
            # Sort by timestamp
            events.sort(key=lambda x: x.timestamp)
            return events
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve audit events: {e}")
            return []
    
    def verify_integrity(self) -> bool:
        """Verify audit log integrity using hash chain"""
        try:
            # Get all log files
            log_files = sorted(self.storage_path.glob("audit_*.jsonl"))
            
            expected_hash = "genesis"
            
            for log_file in log_files:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            
                            # Verify hash chain
                            if record['previous_hash'] != expected_hash:
                                self.logger.error(f"Hash chain broken at {record['event']['event_id']}")
                                return False
                            
                            # Verify event hash
                            calculated_hash = self._calculate_event_hash(
                                record['event'], 
                                record['previous_hash']
                            )
                            
                            if record['hash'] != calculated_hash:
                                self.logger.error(f"Event hash mismatch at {record['event']['event_id']}")
                                return False
                            
                            expected_hash = record['hash']
                            
                        except (json.JSONDecodeError, KeyError) as e:
                            self.logger.error(f"Invalid record format: {e}")
                            return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return False
    
    def _get_current_log_file(self) -> Path:
        """Get current log file path"""
        if not self.current_log_file:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H")
            self.current_log_file = self.storage_path / f"audit_{timestamp}.jsonl"
        
        return self.current_log_file
    
    def _check_log_rotation(self, log_file: Path):
        """Check if log rotation is needed"""
        if log_file.stat().st_size > self.log_rotation_size:
            self.current_log_file = None  # Force new file creation
    
    def _calculate_event_hash(self, event_data: Dict[str, Any], previous_hash: str) -> str:
        """Calculate hash for event integrity"""
        hash_input = json.dumps(event_data, sort_keys=True) + previous_hash
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def _load_hash_chain(self):
        """Load hash chain from storage"""
        chain_file = self.storage_path / "hash_chain.json"
        if chain_file.exists():
            try:
                with open(chain_file, 'r') as f:
                    self.hash_chain = json.load(f)
            except Exception:
                self.hash_chain = []
    
    def _save_hash_chain(self):
        """Save hash chain to storage"""
        chain_file = self.storage_path / "hash_chain.json"
        try:
            # Keep only recent hashes (last 10000)
            if len(self.hash_chain) > 10000:
                self.hash_chain = self.hash_chain[-10000:]
            
            with open(chain_file, 'w') as f:
                json.dump(self.hash_chain, f)
        except Exception as e:
            self.logger.error(f"Failed to save hash chain: {e}")
    
    def _get_log_files_in_range(self, start_time: Optional[datetime], 
                               end_time: Optional[datetime]) -> List[Path]:
        """Get log files that might contain events in time range"""
        all_files = sorted(self.storage_path.glob("audit_*.jsonl"))
        
        if not start_time and not end_time:
            return all_files
        
        # For now, return all files (more sophisticated filtering could be added)
        return all_files


class EnterpriseAuditManager:
    """Enterprise audit logging and compliance management"""
    
    def __init__(self, storage_path: str = "audit_logs"):
        self.storage = AuditStorage(storage_path)
        self.compliance_rules: List[ComplianceRule] = []
        self.logger = logging.getLogger(__name__)
        
        # Load default compliance rules
        self._load_default_compliance_rules()
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log audit event and check compliance"""
        try:
            # Store the event
            success = self.storage.store_event(event)
            
            if success:
                # Check compliance rules
                self._check_compliance_violations(event)
                self.logger.info(f"Audit event logged: {event.event_id}")
            else:
                error = SecurityError(f"Failed to log audit event: {event.event_id}", "AUDIT_001")
                security_error_handler.handle_error(error)
            
            return success
            
        except Exception as e:
            error = SecurityError(f"Audit logging failed: {str(e)}", "AUDIT_002")
            security_error_handler.handle_error(error)
            return False
    
    def generate_compliance_report(self, framework: ComplianceFramework,
                                 start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for specific framework"""
        try:
            # Get events for time period
            events = self.storage.retrieve_events(start_date, end_date)
            
            # Filter events by compliance framework
            relevant_events = [
                event for event in events
                if framework.value in event.compliance_tags
            ]
            
            # Analyze compliance
            report = {
                'framework': framework.value,
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'total_events': len(relevant_events),
                'event_breakdown': {},
                'compliance_violations': [],
                'risk_analysis': self._analyze_risk_levels(relevant_events),
                'recommendations': []
            }
            
            # Event type breakdown
            for event in relevant_events:
                event_type = event.event_type.value
                if event_type not in report['event_breakdown']:
                    report['event_breakdown'][event_type] = 0
                report['event_breakdown'][event_type] += 1
            
            # Check for violations
            violations = self._detect_compliance_violations(relevant_events, framework)
            report['compliance_violations'] = violations
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(relevant_events, framework)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate compliance report: {e}")
            return {}
    
    def add_compliance_rule(self, rule: ComplianceRule):
        """Add custom compliance rule"""
        self.compliance_rules.append(rule)
        self.logger.info(f"Added compliance rule: {rule.rule_id}")
    
    def verify_audit_integrity(self) -> bool:
        """Verify integrity of audit logs"""
        return self.storage.verify_integrity()
    
    def _load_default_compliance_rules(self):
        """Load default compliance rules based on common frameworks"""
        # SOC2 rules
        self.compliance_rules.append(ComplianceRule(
            rule_id="SOC2_ACCESS_001",
            framework=ComplianceFramework.SOC2,
            title="User Access Monitoring",
            description="Monitor all user access events",
            required_events=[AuditEventType.USER_LOGIN, AuditEventType.DATA_ACCESS],
            retention_days=365
        ))
        
        # GDPR rules
        self.compliance_rules.append(ComplianceRule(
            rule_id="GDPR_DATA_001",
            framework=ComplianceFramework.GDPR,
            title="Data Processing Audit",
            description="Audit all data processing activities",
            required_events=[AuditEventType.DATA_ACCESS, AuditEventType.DATA_MODIFICATION],
            retention_days=1095  # 3 years
        ))
        
        # ISO27001 rules
        self.compliance_rules.append(ComplianceRule(
            rule_id="ISO27001_SEC_001",
            framework=ComplianceFramework.ISO27001,
            title="Security Event Monitoring",
            description="Monitor security-related events",
            required_events=[AuditEventType.SECURITY_VIOLATION, AuditEventType.PERMISSION_CHANGE],
            retention_days=730  # 2 years
        ))
    
    def _check_compliance_violations(self, event: AuditEvent):
        """Check if event indicates compliance violations"""
        for rule in self.compliance_rules:
            if rule.matches_event(event):
                # Check for potential violations based on event details
                if not event.success and event.risk_level == "high":
                    self.logger.warning(f"Potential compliance violation for rule {rule.rule_id}")
    
    def _detect_compliance_violations(self, events: List[AuditEvent], 
                                    framework: ComplianceFramework) -> List[Dict[str, Any]]:
        """Detect compliance violations in event set"""
        violations = []
        
        # Check for excessive failed logins
        failed_logins = [e for e in events if e.event_type == AuditEventType.USER_LOGIN and not e.success]
        if len(failed_logins) > 50:  # Threshold
            violations.append({
                'type': 'excessive_failed_logins',
                'count': len(failed_logins),
                'severity': 'high'
            })
        
        # Check for unusual access patterns
        high_risk_events = [e for e in events if e.risk_level == "high"]
        if len(high_risk_events) > 10:
            violations.append({
                'type': 'high_risk_activity',
                'count': len(high_risk_events),
                'severity': 'medium'
            })
        
        return violations
    
    def _analyze_risk_levels(self, events: List[AuditEvent]) -> Dict[str, int]:
        """Analyze risk levels in events"""
        risk_counts = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        for event in events:
            risk_level = event.risk_level
            if risk_level in risk_counts:
                risk_counts[risk_level] += 1
        
        return risk_counts
    
    def _generate_recommendations(self, events: List[AuditEvent], 
                                framework: ComplianceFramework) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        # Check event frequency
        if len(events) < 100:
            recommendations.append("Consider increasing audit event granularity")
        
        # Check for security events
        security_events = [e for e in events if e.event_type == AuditEventType.SECURITY_VIOLATION]
        if len(security_events) > 5:
            recommendations.append("Review security policies and implement additional controls")
        
        # Framework-specific recommendations
        if framework == ComplianceFramework.GDPR:
            data_events = [e for e in events if 'personal_data' in e.details]
            if data_events:
                recommendations.append("Ensure personal data processing has legal basis")
        
        return recommendations


# Global audit manager
enterprise_audit = EnterpriseAuditManager()


def log_audit_event(event_type: AuditEventType, user_id: str, resource: str, 
                   action: str, success: bool = True, details: Optional[Dict[str, Any]] = None) -> bool:
    """Convenience function to log audit event"""
    event = AuditEvent(
        event_type=event_type,
        user_id=user_id,
        resource=resource,
        action=action,
        success=success,
        details=details or {}
    )
    
    return enterprise_audit.log_event(event)