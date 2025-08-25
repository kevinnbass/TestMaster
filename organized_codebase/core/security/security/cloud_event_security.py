"""
AutoGen Derived Cloud Event Security Module
Extracted from AutoGen CloudEvent protobuf patterns and event security
Enhanced for secure distributed event processing
"""

import uuid
import json
import time
import hmac
import base64
import hashlib
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, ValidationError, security_error_handler


class EventSecurityLevel(Enum):
    """Cloud event security levels"""
    PUBLIC = "public"
    INTERNAL = "internal" 
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class EventProcessingStatus(Enum):
    """Event processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class CloudEventAttributeValue:
    """Cloud event attribute value based on AutoGen protobuf patterns"""
    value_type: str  # boolean, integer, string, bytes, uri, uri_ref, timestamp
    value: Any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            'value_type': self.value_type,
            'value': self.value if isinstance(self.value, (str, int, bool)) else str(self.value)
        }


@dataclass
class SecureCloudEvent:
    """Secure cloud event structure based on AutoGen CloudEvent protobuf"""
    # Required CloudEvent attributes
    event_id: str
    source: str  # URI-reference
    spec_version: str = "1.0"
    event_type: str = "default"
    
    # Optional & Extension attributes
    attributes: Dict[str, CloudEventAttributeValue] = field(default_factory=dict)
    
    # Event data (one of: binary, text, or structured)
    binary_data: Optional[bytes] = None
    text_data: Optional[str] = None
    structured_data: Optional[Dict[str, Any]] = None
    
    # Security extensions
    security_level: EventSecurityLevel = EventSecurityLevel.INTERNAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    encryption_key_id: Optional[str] = None
    ttl_seconds: int = 3600  # 1 hour default
    
    # Processing metadata
    processing_status: EventProcessingStatus = EventProcessingStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        # Validate required fields
        if not self.source:
            raise ValidationError("Cloud event source is required")
        
        # Add standard attributes
        self.attributes.update({
            'time': CloudEventAttributeValue('timestamp', self.timestamp.isoformat()),
            'security_level': CloudEventAttributeValue('string', self.security_level.value)
        })
    
    @property
    def is_expired(self) -> bool:
        """Check if event has expired"""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    @property
    def data_content_type(self) -> Optional[str]:
        """Get data content type"""
        if self.binary_data:
            return "application/octet-stream"
        elif self.text_data:
            return "text/plain"
        elif self.structured_data:
            return "application/json"
        return None
    
    def get_data(self) -> Optional[Union[bytes, str, Dict[str, Any]]]:
        """Get event data in appropriate format"""
        if self.binary_data:
            return self.binary_data
        elif self.text_data:
            return self.text_data
        elif self.structured_data:
            return self.structured_data
        return None
    
    def add_processing_step(self, step: str, details: Dict[str, Any] = None):
        """Add processing step to history"""
        self.processing_history.append({
            'step': step,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.event_id,
            'source': self.source,
            'specversion': self.spec_version,
            'type': self.event_type,
            'attributes': {k: v.to_dict() for k, v in self.attributes.items()},
            'data_content_type': self.data_content_type,
            'data': self.get_data(),
            'security_level': self.security_level.value,
            'timestamp': self.timestamp.isoformat(),
            'signature': self.signature,
            'ttl_seconds': self.ttl_seconds,
            'processing_status': self.processing_status.value,
            'validation_errors': self.validation_errors
        }


class CloudEventValidator:
    """Cloud event security validator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_event_size = 1024 * 1024  # 1MB
        self.max_attributes = 100
        self.allowed_sources_pattern = r'^[a-zA-Z][a-zA-Z0-9\-\.]*(/[a-zA-Z0-9\-\.]*)*$'
        self.allowed_types_pattern = r'^[a-zA-Z][a-zA-Z0-9\-\.]*$'
        
        # Security patterns to detect in event data
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'eval\s*\(',
            r'exec\s*\(',
            r'system\s*\(',
            r'shell_exec\s*\('
        ]
    
    def validate_event(self, event: SecureCloudEvent) -> Tuple[bool, List[str]]:
        """Validate cloud event for security and compliance"""
        try:
            errors = []
            
            # Basic structure validation
            if not event.event_id:
                errors.append("Event ID is required")
            
            if not event.source:
                errors.append("Event source is required")
            
            # Validate source format
            import re
            if not re.match(self.allowed_sources_pattern, event.source):
                errors.append(f"Invalid source format: {event.source}")
            
            # Validate event type format  
            if not re.match(self.allowed_types_pattern, event.event_type):
                errors.append(f"Invalid event type format: {event.event_type}")
            
            # Check event age
            if event.is_expired:
                errors.append(f"Event has expired (age: {(datetime.utcnow() - event.timestamp).total_seconds()} seconds)")
            
            # Validate attributes
            if len(event.attributes) > self.max_attributes:
                errors.append(f"Too many attributes: {len(event.attributes)} (max: {self.max_attributes})")
            
            # Check event size
            event_size = self._calculate_event_size(event)
            if event_size > self.max_event_size:
                errors.append(f"Event too large: {event_size} bytes (max: {self.max_event_size})")
            
            # Validate data content
            data_errors = self._validate_event_data(event)
            errors.extend(data_errors)
            
            # Check for suspicious patterns in text data
            if event.text_data:
                for pattern in self.dangerous_patterns:
                    if re.search(pattern, event.text_data, re.IGNORECASE):
                        errors.append(f"Suspicious pattern detected in text data: {pattern}")
            
            # Validate structured data
            if event.structured_data:
                struct_errors = self._validate_structured_data(event.structured_data)
                errors.extend(struct_errors)
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Event validation error: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _calculate_event_size(self, event: SecureCloudEvent) -> int:
        """Calculate approximate event size"""
        size = 0
        size += len(json.dumps(event.to_dict()).encode('utf-8'))
        if event.binary_data:
            size += len(event.binary_data)
        return size
    
    def _validate_event_data(self, event: SecureCloudEvent) -> List[str]:
        """Validate event data content"""
        errors = []
        
        # Check that only one data type is set
        data_types = sum([
            event.binary_data is not None,
            event.text_data is not None,
            event.structured_data is not None
        ])
        
        if data_types > 1:
            errors.append("Event can only contain one type of data (binary, text, or structured)")
        
        # Validate text data
        if event.text_data:
            try:
                event.text_data.encode('utf-8')
            except UnicodeEncodeError:
                errors.append("Text data contains invalid UTF-8 characters")
        
        # Validate binary data
        if event.binary_data:
            if not isinstance(event.binary_data, bytes):
                errors.append("Binary data must be bytes type")
        
        return errors
    
    def _validate_structured_data(self, data: Dict[str, Any], max_depth: int = 10, current_depth: int = 0) -> List[str]:
        """Validate structured data recursively"""
        errors = []
        
        if current_depth > max_depth:
            errors.append(f"Structured data too deeply nested (max depth: {max_depth})")
            return errors
        
        try:
            # Check if data is JSON serializable
            json.dumps(data)
            
            # Recursively check nested structures
            for key, value in data.items():
                if isinstance(value, dict):
                    nested_errors = self._validate_structured_data(value, max_depth, current_depth + 1)
                    errors.extend(nested_errors)
                elif isinstance(value, str):
                    # Check for dangerous patterns in string values
                    for pattern in self.dangerous_patterns:
                        import re
                        if re.search(pattern, value, re.IGNORECASE):
                            errors.append(f"Suspicious pattern in structured data field '{key}': {pattern}")
            
        except (TypeError, ValueError) as e:
            errors.append(f"Structured data is not JSON serializable: {str(e)}")
        
        return errors


class EventSignatureManager:
    """Event signature and integrity management"""
    
    def __init__(self, signing_key: Optional[str] = None):
        self.signing_key = signing_key or "default_signing_key"
        self.logger = logging.getLogger(__name__)
    
    def sign_event(self, event: SecureCloudEvent) -> str:
        """Create HMAC signature for event"""
        try:
            # Create canonical representation for signing
            canonical_data = {
                'id': event.event_id,
                'source': event.source,
                'specversion': event.spec_version,
                'type': event.event_type,
                'timestamp': event.timestamp.isoformat(),
                'data': event.get_data()
            }
            
            canonical_string = json.dumps(canonical_data, sort_keys=True, separators=(',', ':'))
            signature = hmac.new(
                self.signing_key.encode('utf-8'),
                canonical_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            event.signature = signature
            return signature
            
        except Exception as e:
            error = SecurityError(f"Event signing failed: {str(e)}", "EVENT_SIGN_001")
            security_error_handler.handle_error(error)
            raise error
    
    def verify_signature(self, event: SecureCloudEvent) -> bool:
        """Verify event signature"""
        try:
            if not event.signature:
                return False
            
            original_signature = event.signature
            event.signature = None  # Temporarily remove for verification
            
            expected_signature = self.sign_event(event)
            event.signature = original_signature  # Restore original
            
            return hmac.compare_digest(original_signature, expected_signature)
            
        except Exception as e:
            self.logger.error(f"Signature verification error: {e}")
            return False


class CloudEventSecurityManager:
    """Central cloud event security management"""
    
    def __init__(self, signing_key: Optional[str] = None):
        self.validator = CloudEventValidator()
        self.signature_manager = EventSignatureManager(signing_key)
        self.event_store: Dict[str, SecureCloudEvent] = {}
        self.processed_events: List[str] = []  # Event IDs
        self.rejected_events: List[Tuple[str, str, datetime]] = []  # (event_id, reason, timestamp)
        self.max_stored_events = 10000
        self.logger = logging.getLogger(__name__)
        
        # Security policies by level
        self.security_policies = {
            EventSecurityLevel.PUBLIC: {
                'require_signature': False,
                'max_ttl_seconds': 86400,  # 24 hours
                'audit_required': False
            },
            EventSecurityLevel.INTERNAL: {
                'require_signature': True,
                'max_ttl_seconds': 3600,   # 1 hour
                'audit_required': True
            },
            EventSecurityLevel.CONFIDENTIAL: {
                'require_signature': True,
                'max_ttl_seconds': 1800,   # 30 minutes
                'audit_required': True
            },
            EventSecurityLevel.RESTRICTED: {
                'require_signature': True,
                'max_ttl_seconds': 300,    # 5 minutes
                'audit_required': True
            }
        }
    
    def process_event(self, event: SecureCloudEvent, 
                     auto_sign: bool = True) -> Tuple[bool, Optional[str]]:
        """Process cloud event with security validation"""
        try:
            event.processing_status = EventProcessingStatus.PROCESSING
            event.add_processing_step('processing_started')
            
            # Apply security policy
            policy = self.security_policies.get(event.security_level)
            if policy:
                # Check TTL limits
                if event.ttl_seconds > policy['max_ttl_seconds']:
                    event.ttl_seconds = policy['max_ttl_seconds']
                
                # Sign if required and not already signed
                if policy['require_signature'] and not event.signature and auto_sign:
                    self.signature_manager.sign_event(event)
            
            # Validate event
            is_valid, errors = self.validator.validate_event(event)
            if not is_valid:
                event.processing_status = EventProcessingStatus.REJECTED
                event.validation_errors = errors
                event.add_processing_step('validation_failed', {'errors': errors})
                
                # Store rejected event for audit
                self.rejected_events.append((event.event_id, str(errors), datetime.utcnow()))
                
                self.logger.warning(f"Event rejected: {event.event_id} - {errors}")
                return False, f"Event validation failed: {errors}"
            
            # Verify signature if present
            if event.signature and not self.signature_manager.verify_signature(event):
                event.processing_status = EventProcessingStatus.REJECTED
                event.add_processing_step('signature_verification_failed')
                
                self.rejected_events.append((event.event_id, "Invalid signature", datetime.utcnow()))
                self.logger.warning(f"Event signature verification failed: {event.event_id}")
                return False, "Event signature verification failed"
            
            # Store processed event
            self.event_store[event.event_id] = event
            self.processed_events.append(event.event_id)
            
            # Maintain store size
            if len(self.event_store) > self.max_stored_events:
                oldest_events = list(self.event_store.keys())[:100]
                for event_id in oldest_events:
                    del self.event_store[event_id]
            
            # Update status
            event.processing_status = EventProcessingStatus.COMPLETED
            event.add_processing_step('processing_completed')
            
            self.logger.info(f"Event processed successfully: {event.event_id}")
            return True, None
            
        except Exception as e:
            event.processing_status = EventProcessingStatus.FAILED
            event.add_processing_step('processing_error', {'error': str(e)})
            
            error = SecurityError(f"Event processing failed: {str(e)}", "EVENT_PROC_001")
            security_error_handler.handle_error(error)
            return False, str(error)
    
    def create_secure_event(self, source: str, event_type: str,
                           data: Optional[Union[str, bytes, Dict[str, Any]]] = None,
                           security_level: EventSecurityLevel = EventSecurityLevel.INTERNAL,
                           attributes: Dict[str, Any] = None) -> SecureCloudEvent:
        """Create a new secure cloud event"""
        try:
            event = SecureCloudEvent(
                event_id=str(uuid.uuid4()),
                source=source,
                event_type=event_type,
                security_level=security_level
            )
            
            # Set data based on type
            if isinstance(data, bytes):
                event.binary_data = data
            elif isinstance(data, str):
                event.text_data = data
            elif isinstance(data, dict):
                event.structured_data = data
            
            # Add custom attributes
            if attributes:
                for key, value in attributes.items():
                    if isinstance(value, (str, int, bool)):
                        attr_type = type(value).__name__
                        event.attributes[key] = CloudEventAttributeValue(attr_type, value)
            
            event.add_processing_step('event_created')
            return event
            
        except Exception as e:
            error = SecurityError(f"Event creation failed: {str(e)}", "EVENT_CREATE_001")
            security_error_handler.handle_error(error)
            raise error
    
    def get_event_stats(self) -> Dict[str, Any]:
        """Get event processing statistics"""
        total_processed = len(self.processed_events)
        total_rejected = len(self.rejected_events)
        total_stored = len(self.event_store)
        
        # Count by security level
        level_counts = {}
        for event in self.event_store.values():
            level = event.security_level.value
            level_counts[level] = level_counts.get(level, 0) + 1
        
        # Recent activity (last hour)
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_processed = sum(1 for event in self.event_store.values() if event.timestamp > hour_ago)
        recent_rejected = sum(1 for _, _, timestamp in self.rejected_events if timestamp > hour_ago)
        
        return {
            'total_processed': total_processed,
            'total_rejected': total_rejected,
            'total_stored': total_stored,
            'security_level_distribution': level_counts,
            'recent_processed_1h': recent_processed,
            'recent_rejected_1h': recent_rejected,
            'policies_configured': len(self.security_policies)
        }


# Global cloud event security manager
cloud_event_security = CloudEventSecurityManager()


def create_secure_cloud_event(source: str, event_type: str, 
                             data: Any = None,
                             security_level: EventSecurityLevel = EventSecurityLevel.INTERNAL) -> SecureCloudEvent:
    """Convenience function to create secure cloud event"""
    return cloud_event_security.create_secure_event(source, event_type, data, security_level)


def process_cloud_event(event: SecureCloudEvent) -> Tuple[bool, Optional[str]]:
    """Convenience function to process cloud event"""
    return cloud_event_security.process_event(event)