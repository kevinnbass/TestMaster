"""
Swarms Derived Distributed Key Management and API Security Module
Extracted from Swarms key generation patterns for secure distributed authentication
Enhanced for cryptographic security and distributed key management
"""

import uuid
import time
import json
import hashlib
import hmac
import secrets
import string
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from .error_handler import SecurityError, security_error_handler


class KeyType(Enum):
    """Types of cryptographic keys supported"""
    API_KEY = "api_key"
    ACCESS_TOKEN = "access_token"
    REFRESH_TOKEN = "refresh_token"
    SYMMETRIC_KEY = "symmetric_key"
    PUBLIC_KEY = "public_key"
    PRIVATE_KEY = "private_key"
    MASTER_KEY = "master_key"


class KeyStatus(Enum):
    """Key lifecycle status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"
    COMPROMISED = "compromised"


class SecurityLevel(Enum):
    """Security levels for key management"""
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CryptoKey:
    """Secure cryptographic key with metadata"""
    key_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    key_type: KeyType = KeyType.API_KEY
    key_value: Optional[str] = None
    public_component: Optional[str] = None
    private_component: Optional[bytes] = None
    owner_id: str = ""
    purpose: str = ""
    security_level: SecurityLevel = SecurityLevel.STANDARD
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    use_count: int = 0
    max_uses: Optional[int] = None
    allowed_operations: Set[str] = field(default_factory=set)
    status: KeyStatus = KeyStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.owner_id:
            raise SecurityError("Owner ID is required", "CRYPTO_KEY_001")
        
        # Set default expiration based on key type
        if not self.expires_at:
            if self.key_type == KeyType.ACCESS_TOKEN:
                self.expires_at = self.created_at + timedelta(hours=1)
            elif self.key_type == KeyType.REFRESH_TOKEN:
                self.expires_at = self.created_at + timedelta(days=30)
            elif self.key_type == KeyType.API_KEY:
                self.expires_at = self.created_at + timedelta(days=365)
            else:
                self.expires_at = self.created_at + timedelta(days=90)
    
    @property
    def is_expired(self) -> bool:
        """Check if key has expired"""
        return self.expires_at and datetime.utcnow() > self.expires_at
    
    @property
    def is_active(self) -> bool:
        """Check if key is active and usable"""
        return (
            self.status == KeyStatus.ACTIVE and
            not self.is_expired and
            (self.max_uses is None or self.use_count < self.max_uses)
        )
    
    @property
    def days_until_expiry(self) -> Optional[int]:
        """Get days until key expiry"""
        if not self.expires_at:
            return None
        
        delta = self.expires_at - datetime.utcnow()
        return max(0, delta.days)
    
    def calculate_key_hash(self) -> str:
        """Calculate hash of key for integrity verification"""
        key_data = {
            'key_id': self.key_id,
            'key_type': self.key_type.value,
            'owner_id': self.owner_id,
            'created_at': self.created_at.isoformat(),
            'security_level': self.security_level.value
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


@dataclass
class KeyPolicy:
    """Security policy for key management"""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    policy_name: str = ""
    key_types: Set[KeyType] = field(default_factory=set)
    min_key_length: int = 32
    max_key_lifetime: int = 365  # days
    rotation_interval: int = 90  # days
    max_uses_per_key: Optional[int] = None
    allowed_operations: Set[str] = field(default_factory=set)
    required_security_level: SecurityLevel = SecurityLevel.STANDARD
    audit_required: bool = True
    
    def validates_key(self, key: CryptoKey) -> Tuple[bool, str]:
        """Validate if key complies with policy"""
        if key.key_type not in self.key_types and self.key_types:
            return False, f"Key type {key.key_type.value} not allowed by policy"
        
        if key.security_level.value < self.required_security_level.value:
            return False, f"Security level {key.security_level.value} below required {self.required_security_level.value}"
        
        if key.key_value and len(key.key_value) < self.min_key_length:
            return False, f"Key length {len(key.key_value)} below minimum {self.min_key_length}"
        
        if key.days_until_expiry and key.days_until_expiry > self.max_key_lifetime:
            return False, f"Key lifetime exceeds maximum {self.max_key_lifetime} days"
        
        return True, "Policy validation passed"


class CryptographicOperations:
    """Cryptographic operations for key management"""
    
    @staticmethod
    def generate_api_key(prefix: str = "sk-", length: int = 32) -> str:
        """Generate cryptographically secure API key"""
        if length < 16:
            raise SecurityError("API key length must be at least 16 characters", "CRYPTO_OP_001")
        
        alphabet = string.ascii_letters + string.digits
        random_part = "".join(secrets.choice(alphabet) for _ in range(length))
        return f"{prefix}{random_part}"
    
    @staticmethod
    def generate_token(length: int = 48) -> str:
        """Generate cryptographically secure token"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_symmetric_key() -> bytes:
        """Generate symmetric encryption key"""
        return Fernet.generate_key()
    
    @staticmethod
    def generate_rsa_keypair(key_size: int = 2048) -> Tuple[bytes, bytes]:
        """Generate RSA public/private key pair"""
        if key_size < 2048:
            raise SecurityError("RSA key size must be at least 2048 bits", "CRYPTO_OP_002")
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return public_pem, private_pem
    
    @staticmethod
    def derive_key_from_password(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """Derive encryption key from password using PBKDF2"""
        if not salt:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = kdf.derive(password.encode())
        return key, salt
    
    @staticmethod
    def create_signature(message: str, private_key_pem: bytes) -> bytes:
        """Create digital signature using RSA private key"""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem, password=None
            )
            
            signature = private_key.sign(
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return signature
            
        except Exception as e:
            raise SecurityError(f"Signature creation failed: {str(e)}", "CRYPTO_OP_003")
    
    @staticmethod
    def verify_signature(message: str, signature: bytes, public_key_pem: bytes) -> bool:
        """Verify digital signature using RSA public key"""
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            
            public_key.verify(
                signature,
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            
            return True
            
        except Exception:
            return False


class KeyRotationManager:
    """Automated key rotation and lifecycle management"""
    
    def __init__(self, check_interval: int = 3600):  # 1 hour
        self.check_interval = check_interval
        self.rotation_policies: Dict[str, Dict[str, Any]] = {}
        self.rotation_active = False
        self.rotation_thread: Optional[threading.Thread] = None
        self.logger = logging.getLogger(__name__)
    
    def add_rotation_policy(self, key_type: KeyType, rotation_days: int, 
                          warning_days: int = 7) -> str:
        """Add automatic rotation policy for key type"""
        policy_id = str(uuid.uuid4())
        
        self.rotation_policies[policy_id] = {
            'key_type': key_type,
            'rotation_days': rotation_days,
            'warning_days': warning_days,
            'last_check': datetime.utcnow()
        }
        
        return policy_id
    
    def start_rotation_monitoring(self):
        """Start background key rotation monitoring"""
        if not self.rotation_active:
            self.rotation_active = True
            self.rotation_thread = threading.Thread(target=self._rotation_monitor, daemon=True)
            self.rotation_thread.start()
            self.logger.info("Key rotation monitoring started")
    
    def stop_rotation_monitoring(self):
        """Stop background key rotation monitoring"""
        self.rotation_active = False
        if self.rotation_thread:
            self.rotation_thread.join(timeout=5)
        self.logger.info("Key rotation monitoring stopped")
    
    def _rotation_monitor(self):
        """Background key rotation monitoring loop"""
        while self.rotation_active:
            try:
                for policy_id, policy in self.rotation_policies.items():
                    self._check_rotation_policy(policy_id, policy)
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Rotation monitoring error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _check_rotation_policy(self, policy_id: str, policy: Dict[str, Any]):
        """Check if keys need rotation according to policy"""
        try:
            # This would integrate with the key manager to check keys
            # For now, just log the check
            self.logger.debug(f"Checking rotation policy {policy_id}")
            policy['last_check'] = datetime.utcnow()
            
        except Exception as e:
            self.logger.error(f"Policy check failed for {policy_id}: {e}")


class DistributedKeyManagementSecurityManager:
    """Secure distributed key management system"""
    
    def __init__(self):
        self.keys: Dict[str, CryptoKey] = {}
        self.policies: Dict[str, KeyPolicy] = {}
        self.key_usage_log: List[Dict[str, Any]] = []
        self.revoked_keys: Set[str] = set()
        
        # Security components
        self.crypto_ops = CryptographicOperations()
        self.rotation_manager = KeyRotationManager()
        self.key_lock = threading.RLock()
        
        # Security settings
        self.max_keys_per_owner = 100
        self.audit_all_operations = True
        self.key_access_rate_limit = 1000  # per hour
        self.rate_tracking: Dict[str, List[datetime]] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Start key rotation monitoring
        self.rotation_manager.start_rotation_monitoring()
    
    def create_key(self, key_type: KeyType, owner_id: str, purpose: str = "",
                  security_level: SecurityLevel = SecurityLevel.STANDARD,
                  custom_params: Optional[Dict[str, Any]] = None) -> Optional[CryptoKey]:
        """Create new cryptographic key with security validation"""
        try:
            with self.key_lock:
                # Check owner key limit
                owner_keys = [k for k in self.keys.values() if k.owner_id == owner_id]
                if len(owner_keys) >= self.max_keys_per_owner:
                    raise SecurityError(f"Key limit exceeded for owner {owner_id}", "KEY_CREATE_001")
                
                # Generate key based on type
                key_value = None
                public_component = None
                private_component = None
                
                if key_type == KeyType.API_KEY:
                    prefix = custom_params.get('prefix', 'sk-') if custom_params else 'sk-'
                    length = custom_params.get('length', 32) if custom_params else 32
                    key_value = self.crypto_ops.generate_api_key(prefix, length)
                
                elif key_type in [KeyType.ACCESS_TOKEN, KeyType.REFRESH_TOKEN]:
                    length = custom_params.get('length', 48) if custom_params else 48
                    key_value = self.crypto_ops.generate_token(length)
                
                elif key_type == KeyType.SYMMETRIC_KEY:
                    key_bytes = self.crypto_ops.generate_symmetric_key()
                    key_value = base64.b64encode(key_bytes).decode()
                
                elif key_type in [KeyType.PUBLIC_KEY, KeyType.PRIVATE_KEY]:
                    key_size = custom_params.get('key_size', 2048) if custom_params else 2048
                    public_pem, private_pem = self.crypto_ops.generate_rsa_keypair(key_size)
                    
                    public_component = public_pem.decode()
                    private_component = private_pem
                    
                    if key_type == KeyType.PUBLIC_KEY:
                        key_value = public_component
                    else:
                        key_value = "PRIVATE_KEY_STORED_SECURELY"
                
                else:
                    key_value = self.crypto_ops.generate_token(64)
                
                # Create key object
                key = CryptoKey(
                    key_type=key_type,
                    key_value=key_value,
                    public_component=public_component,
                    private_component=private_component,
                    owner_id=owner_id,
                    purpose=purpose,
                    security_level=security_level
                )
                
                # Validate against policies
                for policy in self.policies.values():
                    valid, message = policy.validates_key(key)
                    if not valid:
                        raise SecurityError(f"Key policy violation: {message}", "KEY_POLICY_001")
                
                # Store key
                self.keys[key.key_id] = key
                
                # Log key creation
                self._audit_operation("KEY_CREATED", key.key_id, owner_id, {
                    'key_type': key_type.value,
                    'security_level': security_level.value,
                    'purpose': purpose
                })
                
                self.logger.info(f"Key created: {key.key_id} for {owner_id}")
                return key
                
        except Exception as e:
            error = SecurityError(f"Key creation failed: {str(e)}", "KEY_CREATE_FAIL_001")
            security_error_handler.handle_error(error)
            return None
    
    def use_key(self, key_id: str, operation: str, requester_id: str) -> Tuple[bool, Optional[CryptoKey]]:
        """Use key for cryptographic operation with validation"""
        try:
            with self.key_lock:
                # Check rate limiting
                if not self._check_rate_limit(requester_id):
                    raise SecurityError("Rate limit exceeded", "KEY_USE_001")
                
                # Validate key exists and is active
                if key_id not in self.keys:
                    raise SecurityError("Key not found", "KEY_USE_002")
                
                if key_id in self.revoked_keys:
                    raise SecurityError("Key has been revoked", "KEY_USE_003")
                
                key = self.keys[key_id]
                
                if not key.is_active:
                    raise SecurityError("Key is not active", "KEY_USE_004")
                
                # Check if requester is authorized
                if key.owner_id != requester_id:
                    raise SecurityError("Unauthorized key access", "KEY_USE_005")
                
                # Check operation permissions
                if key.allowed_operations and operation not in key.allowed_operations:
                    raise SecurityError(f"Operation '{operation}' not allowed for key", "KEY_USE_006")
                
                # Update key usage
                key.last_used = datetime.utcnow()
                key.use_count += 1
                
                # Update rate tracking
                self._update_rate_tracking(requester_id)
                
                # Log key usage
                self._audit_operation("KEY_USED", key_id, requester_id, {
                    'operation': operation,
                    'use_count': key.use_count
                })
                
                return True, key
                
        except Exception as e:
            self._audit_operation("KEY_USE_FAILED", key_id, requester_id, {
                'operation': operation,
                'error': str(e)
            })
            
            error = SecurityError(f"Key usage failed: {str(e)}", "KEY_USE_FAIL_001")
            security_error_handler.handle_error(error)
            return False, None
    
    def revoke_key(self, key_id: str, requester_id: str, reason: str = "") -> bool:
        """Revoke key with security validation"""
        try:
            with self.key_lock:
                if key_id not in self.keys:
                    raise SecurityError("Key not found", "KEY_REVOKE_001")
                
                key = self.keys[key_id]
                
                # Only owner or admin can revoke key
                if key.owner_id != requester_id:
                    raise SecurityError("Unauthorized key revocation", "KEY_REVOKE_002")
                
                # Update key status
                key.status = KeyStatus.REVOKED
                self.revoked_keys.add(key_id)
                
                # Log revocation
                self._audit_operation("KEY_REVOKED", key_id, requester_id, {
                    'reason': reason,
                    'revoked_at': datetime.utcnow().isoformat()
                })
                
                self.logger.info(f"Key revoked: {key_id} by {requester_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Key revocation failed: {str(e)}", "KEY_REVOKE_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def rotate_key(self, key_id: str, requester_id: str) -> Optional[CryptoKey]:
        """Rotate key by creating new one and revoking old"""
        try:
            with self.key_lock:
                if key_id not in self.keys:
                    raise SecurityError("Key not found", "KEY_ROTATE_001")
                
                old_key = self.keys[key_id]
                
                # Only owner can rotate key
                if old_key.owner_id != requester_id:
                    raise SecurityError("Unauthorized key rotation", "KEY_ROTATE_002")
                
                # Create new key with same parameters
                new_key = self.create_key(
                    key_type=old_key.key_type,
                    owner_id=old_key.owner_id,
                    purpose=old_key.purpose,
                    security_level=old_key.security_level
                )
                
                if new_key:
                    # Revoke old key
                    self.revoke_key(key_id, requester_id, "Key rotated")
                    
                    # Log rotation
                    self._audit_operation("KEY_ROTATED", key_id, requester_id, {
                        'old_key_id': key_id,
                        'new_key_id': new_key.key_id
                    })
                    
                    self.logger.info(f"Key rotated: {key_id} -> {new_key.key_id}")
                    return new_key
                
                return None
                
        except Exception as e:
            error = SecurityError(f"Key rotation failed: {str(e)}", "KEY_ROTATE_FAIL_001")
            security_error_handler.handle_error(error)
            return None
    
    def get_key_stats(self) -> Dict[str, Any]:
        """Get key management statistics"""
        with self.key_lock:
            status_counts = {}
            type_counts = {}
            security_level_counts = {}
            
            for key in self.keys.values():
                # Status breakdown
                status = key.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
                # Type breakdown
                key_type = key.key_type.value
                type_counts[key_type] = type_counts.get(key_type, 0) + 1
                
                # Security level breakdown
                level = key.security_level.value
                security_level_counts[level] = security_level_counts.get(level, 0) + 1
            
            # Keys expiring soon
            soon_threshold = datetime.utcnow() + timedelta(days=7)
            expiring_soon = sum(
                1 for key in self.keys.values()
                if key.expires_at and key.expires_at < soon_threshold
            )
            
            return {
                'total_keys': len(self.keys),
                'active_keys': status_counts.get('active', 0),
                'revoked_keys': len(self.revoked_keys),
                'status_breakdown': status_counts,
                'type_breakdown': type_counts,
                'security_level_breakdown': security_level_counts,
                'keys_expiring_soon': expiring_soon,
                'total_usage': sum(k.use_count for k in self.keys.values()),
                'audit_entries': len(self.key_usage_log)
            }
    
    def _check_rate_limit(self, requester_id: str) -> bool:
        """Check if requester is within rate limits"""
        if requester_id not in self.rate_tracking:
            self.rate_tracking[requester_id] = []
        
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        # Clean old entries
        self.rate_tracking[requester_id] = [
            timestamp for timestamp in self.rate_tracking[requester_id]
            if timestamp > hour_ago
        ]
        
        # Check rate limit
        return len(self.rate_tracking[requester_id]) < self.key_access_rate_limit
    
    def _update_rate_tracking(self, requester_id: str):
        """Update rate tracking for requester"""
        if requester_id not in self.rate_tracking:
            self.rate_tracking[requester_id] = []
        
        self.rate_tracking[requester_id].append(datetime.utcnow())
    
    def _audit_operation(self, operation: str, key_id: str, requester_id: str, details: Dict[str, Any]):
        """Log operation for audit trail"""
        if self.audit_all_operations:
            audit_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'operation': operation,
                'key_id': key_id,
                'requester_id': requester_id,
                'details': details
            }
            
            self.key_usage_log.append(audit_entry)
            
            # Keep only recent audit entries (last 10000)
            if len(self.key_usage_log) > 10000:
                self.key_usage_log = self.key_usage_log[-10000:]


# Global distributed key management system
distributed_key_management_security = DistributedKeyManagementSecurityManager()


def generate_secure_api_key(owner_id: str, purpose: str = "", prefix: str = "sk-") -> Optional[str]:
    """Convenience function to generate secure API key"""
    try:
        key = distributed_key_management_security.create_key(
            key_type=KeyType.API_KEY,
            owner_id=owner_id,
            purpose=purpose,
            custom_params={'prefix': prefix}
        )
        
        return key.key_value if key else None
        
    except Exception as e:
        logging.getLogger(__name__).error(f"API key generation failed: {e}")
        return None


def authenticate_api_key(api_key: str, requester_id: str) -> bool:
    """Convenience function to authenticate API key"""
    try:
        # Find key by value (in production, this would be indexed)
        for key_id, key in distributed_key_management_security.keys.items():
            if key.key_value == api_key and key.key_type == KeyType.API_KEY:
                success, _ = distributed_key_management_security.use_key(
                    key_id, "authentication", requester_id
                )
                return success
        
        return False
        
    except Exception as e:
        logging.getLogger(__name__).error(f"API key authentication failed: {e}")
        return False