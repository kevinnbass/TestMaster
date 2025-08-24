"""
AutoGen Derived Identity Validation System
Extracted from AutoGen AgentId validation patterns and identity management
Enhanced for comprehensive agent identity security
"""

import re
import uuid
import hashlib
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, AuthenticationError, security_error_handler


class IdentityStatus(Enum):
    """Identity validation status"""
    VALID = "valid"
    INVALID = "invalid" 
    PENDING = "pending"
    REVOKED = "revoked"
    EXPIRED = "expired"


@dataclass
class IdentityValidationRule:
    """Identity validation rule configuration"""
    name: str
    pattern: str
    description: str
    is_required: bool = True
    error_message: str = ""


@dataclass  
class ValidatedIdentity:
    """Validated identity structure based on AutoGen patterns"""
    identity_type: str
    identity_key: str
    validation_status: IdentityStatus = IdentityStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    validated_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[str] = field(default_factory=list)
    trust_score: float = 0.0
    
    @property
    def full_identity(self) -> str:
        """Get full identity in type/key format"""
        return f"{self.identity_type}/{self.identity_key}"
    
    @property
    def is_expired(self) -> bool:
        """Check if identity has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def is_valid(self) -> bool:
        """Check if identity is currently valid"""
        return (
            self.validation_status == IdentityStatus.VALID and
            not self.is_expired and
            len(self.validation_errors) == 0
        )


class IdentityValidator:
    """Core identity validation engine based on AutoGen patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # AutoGen-based validation rules
        self.validation_rules = {
            'type_format': IdentityValidationRule(
                name='type_format',
                pattern=r'^[a-zA-Z_][a-zA-Z0-9_]*$',
                description='Type must be alphanumeric (a-z, 0-9, _) and cannot start with a number',
                error_message='Invalid identity type format'
            ),
            'key_format': IdentityValidationRule(
                name='key_format', 
                pattern=r'^[\x20-\x7E]+$',  # ASCII 32-126
                description='Key must only contain ASCII characters 32-126',
                error_message='Invalid identity key format'
            ),
            'type_length': IdentityValidationRule(
                name='type_length',
                pattern=r'^.{1,64}$',
                description='Type must be 1-64 characters',
                error_message='Identity type length invalid'
            ),
            'key_length': IdentityValidationRule(
                name='key_length',
                pattern=r'^.{1,256}$',
                description='Key must be 1-256 characters',
                error_message='Identity key length invalid'
            )
        }
        
        # Blocked patterns for security
        self.blocked_patterns = [
            r'\.\./',           # Path traversal
            r'<script[^>]*>',   # XSS
            r'javascript:',     # JS injection
            r'eval\s*\(',       # Code injection
            r'system\s*\(',     # System calls
            r'__import__',      # Python imports
        ]
        
        # Reserved type names that shouldn't be used
        self.reserved_types = {
            'system', 'admin', 'root', 'service', 'internal',
            'security', 'auth', 'authentication', 'authorization'
        }
    
    def validate_identity(self, identity_type: str, identity_key: str, 
                         metadata: Dict[str, Any] = None) -> ValidatedIdentity:
        """Validate an identity based on AutoGen patterns"""
        identity = ValidatedIdentity(
            identity_type=identity_type,
            identity_key=identity_key,
            metadata=metadata or {},
            validation_status=IdentityStatus.PENDING
        )
        
        try:
            errors = []
            
            # Apply validation rules
            for rule_name, rule in self.validation_rules.items():
                if rule_name.startswith('type_'):
                    value = identity_type
                elif rule_name.startswith('key_'):
                    value = identity_key
                else:
                    continue
                
                if not re.match(rule.pattern, value):
                    errors.append(f"{rule.error_message}: {rule.description}")
            
            # Check for blocked patterns
            combined_value = f"{identity_type}/{identity_key}"
            for pattern in self.blocked_patterns:
                if re.search(pattern, combined_value, re.IGNORECASE):
                    errors.append(f"Identity contains blocked pattern: {pattern}")
            
            # Check for reserved type names
            if identity_type.lower() in self.reserved_types:
                errors.append(f"Identity type '{identity_type}' is reserved")
            
            # Check for suspicious characters
            if self._contains_suspicious_chars(combined_value):
                errors.append("Identity contains suspicious characters")
            
            # Update identity with validation results
            identity.validation_errors = errors
            
            if len(errors) == 0:
                identity.validation_status = IdentityStatus.VALID
                identity.validated_at = datetime.utcnow()
                identity.trust_score = self._calculate_trust_score(identity)
                self.logger.info(f"Identity validated successfully: {identity.full_identity}")
            else:
                identity.validation_status = IdentityStatus.INVALID
                self.logger.warning(f"Identity validation failed: {identity.full_identity} - {errors}")
            
            return identity
            
        except Exception as e:
            error = SecurityError(f"Identity validation error: {str(e)}", "ID_VAL_001")
            security_error_handler.handle_error(error)
            identity.validation_status = IdentityStatus.INVALID
            identity.validation_errors.append(f"Validation exception: {str(e)}")
            return identity
    
    def _contains_suspicious_chars(self, value: str) -> bool:
        """Check for suspicious character sequences"""
        suspicious_sequences = [
            '\x00',     # Null bytes
            '\r\n',     # CRLF injection
            '\n\r',     # Reverse CRLF
            '../../',   # Path traversal
            '%00',      # Null byte encoding
            '%2e%2e',   # Encoded dots
        ]
        
        for seq in suspicious_sequences:
            if seq in value:
                return True
        return False
    
    def _calculate_trust_score(self, identity: ValidatedIdentity) -> float:
        """Calculate trust score for identity"""
        score = 100.0
        
        # Deduct for validation errors
        score -= len(identity.validation_errors) * 20
        
        # Check identity age (newer identities have lower initial trust)
        age_hours = (datetime.utcnow() - identity.created_at).total_seconds() / 3600
        if age_hours < 1:
            score -= 10  # New identities get slight penalty
        
        # Check for metadata completeness
        if len(identity.metadata) == 0:
            score -= 5
        
        # Check identity key complexity
        if len(identity.identity_key) < 8:
            score -= 10
        
        # Normalize to 0-100 range
        return max(0.0, min(100.0, score))


class IdentityRegistry:
    """Identity registry for tracking and managing validated identities"""
    
    def __init__(self):
        self.validator = IdentityValidator()
        self.identities: Dict[str, ValidatedIdentity] = {}
        self.identity_history: List[Tuple[str, str, datetime]] = []  # (action, identity, timestamp)
        self.trust_threshold = 50.0
        self.logger = logging.getLogger(__name__)
    
    def register_identity(self, identity_type: str, identity_key: str, 
                         metadata: Dict[str, Any] = None, 
                         expires_in_hours: Optional[int] = None) -> ValidatedIdentity:
        """Register and validate a new identity"""
        try:
            # Validate the identity
            identity = self.validator.validate_identity(identity_type, identity_key, metadata)
            
            # Set expiration if specified
            if expires_in_hours:
                identity.expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            # Store the identity
            self.identities[identity.full_identity] = identity
            
            # Log the registration
            self.identity_history.append(('register', identity.full_identity, datetime.utcnow()))
            
            # Keep history to reasonable size
            if len(self.identity_history) > 10000:
                self.identity_history = self.identity_history[-5000:]
            
            self.logger.info(f"Identity registered: {identity.full_identity} (Status: {identity.validation_status.value})")
            return identity
            
        except Exception as e:
            error = SecurityError(f"Identity registration failed: {str(e)}", "ID_REG_001")
            security_error_handler.handle_error(error)
            raise error
    
    def get_identity(self, identity_id: str) -> Optional[ValidatedIdentity]:
        """Retrieve an identity from the registry"""
        identity = self.identities.get(identity_id)
        if identity and identity.is_expired:
            # Mark expired identity
            identity.validation_status = IdentityStatus.EXPIRED
            self.logger.warning(f"Identity expired: {identity_id}")
        return identity
    
    def revoke_identity(self, identity_id: str, reason: str = "") -> bool:
        """Revoke an identity"""
        try:
            if identity_id in self.identities:
                identity = self.identities[identity_id]
                identity.validation_status = IdentityStatus.REVOKED
                identity.metadata['revocation_reason'] = reason
                identity.metadata['revoked_at'] = datetime.utcnow().isoformat()
                
                # Log the revocation
                self.identity_history.append(('revoke', identity_id, datetime.utcnow()))
                
                self.logger.warning(f"Identity revoked: {identity_id} - Reason: {reason}")
                return True
            
            return False
            
        except Exception as e:
            error = SecurityError(f"Identity revocation failed: {str(e)}", "ID_REV_001")
            security_error_handler.handle_error(error)
            return False
    
    def is_identity_trusted(self, identity_id: str) -> bool:
        """Check if an identity meets trust threshold"""
        identity = self.get_identity(identity_id)
        if not identity:
            return False
        
        return (
            identity.is_valid and
            identity.trust_score >= self.trust_threshold
        )
    
    def cleanup_expired_identities(self) -> int:
        """Clean up expired identities"""
        expired_count = 0
        current_time = datetime.utcnow()
        expired_ids = []
        
        for identity_id, identity in self.identities.items():
            if identity.is_expired:
                expired_ids.append(identity_id)
        
        for identity_id in expired_ids:
            del self.identities[identity_id]
            expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired identities")
        
        return expired_count
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get identity registry statistics"""
        total_identities = len(self.identities)
        
        # Count by status
        status_counts = {}
        trust_scores = []
        
        for identity in self.identities.values():
            status = identity.validation_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            trust_scores.append(identity.trust_score)
        
        # Calculate trust statistics
        avg_trust = sum(trust_scores) / len(trust_scores) if trust_scores else 0
        trusted_count = sum(1 for score in trust_scores if score >= self.trust_threshold)
        
        # Recent activity
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_activity = [
            entry for entry in self.identity_history
            if entry[2] > hour_ago
        ]
        
        return {
            'total_identities': total_identities,
            'status_breakdown': status_counts,
            'trusted_identities': trusted_count,
            'average_trust_score': round(avg_trust, 2),
            'trust_threshold': self.trust_threshold,
            'recent_activity_1h': len(recent_activity),
            'total_history_entries': len(self.identity_history)
        }


# Global identity registry
identity_registry = IdentityRegistry()


def validate_agent_identity(agent_type: str, agent_key: str, 
                          metadata: Dict[str, Any] = None) -> ValidatedIdentity:
    """Convenience function to validate an agent identity"""
    return identity_registry.register_identity(agent_type, agent_key, metadata)


def is_trusted_identity(identity_id: str) -> bool:
    """Convenience function to check if identity is trusted"""
    return identity_registry.is_identity_trusted(identity_id)