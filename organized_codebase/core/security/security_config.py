"""
Security Configuration Module
=============================

Security-related configuration settings including authentication, encryption, and access control.
Modularized from testmaster_config.py and unified_config.py.

Author: Agent E - Infrastructure Consolidation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set
from pathlib import Path
from enum import Enum

from .C:.Users.kbass.OneDrive.Documents.testmaster.organized_codebase.testing.data_models import ConfigBase


class AuthenticationMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    BASIC = "basic"
    CERTIFICATE = "certificate"
    SAML = "saml"
    NONE = "none"


class EncryptionAlgorithm(Enum):
    """Encryption algorithms."""
    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"
    RSA_4096 = "rsa-4096"
    CHACHA20_POLY1305 = "chacha20-poly1305"
    NONE = "none"


class AccessLevel(Enum):
    """Access control levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"


@dataclass
class AuthenticationConfig(ConfigBase):
    """Authentication configuration."""
    
    # Authentication Settings
    enabled: bool = True
    primary_method: AuthenticationMethod = AuthenticationMethod.API_KEY
    fallback_methods: List[AuthenticationMethod] = field(default_factory=lambda: [
        AuthenticationMethod.JWT,
        AuthenticationMethod.BASIC
    ])
    
    # Token Configuration
    token_expiry_seconds: int = 3600
    refresh_token_expiry_seconds: int = 86400
    max_token_refresh_count: int = 3
    
    # Session Management
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 5
    enforce_single_session: bool = False
    
    # API Key Settings
    api_key_rotation_days: int = 90
    api_key_min_length: int = 32
    require_api_key_headers: bool = True
    
    # OAuth2 Settings
    oauth2_provider_url: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_scopes: List[str] = field(default_factory=lambda: ["read", "write"])
    
    def validate(self) -> List[str]:
        """Validate authentication configuration."""
        errors = []
        
        if self.token_expiry_seconds <= 0:
            errors.append("Token expiry must be positive")
        
        if self.session_timeout_minutes <= 0:
            errors.append("Session timeout must be positive")
        
        if self.api_key_min_length < 16:
            errors.append("API key minimum length should be at least 16")
        
        if self.primary_method == AuthenticationMethod.OAUTH2:
            if not self.oauth2_provider_url or not self.oauth2_client_id:
                errors.append("OAuth2 provider URL and client ID required")
        
        return errors


@dataclass
class EncryptionConfig(ConfigBase):
    """Encryption configuration."""
    
    # Encryption Settings
    enabled: bool = True
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    
    # Key Management
    key_rotation_days: int = 30
    key_derivation_iterations: int = 100000
    use_hardware_security_module: bool = False
    
    # Data Encryption
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    encrypt_sensitive_logs: bool = True
    
    # Certificate Management
    certificate_path: Optional[Path] = None
    private_key_path: Optional[Path] = None
    ca_bundle_path: Optional[Path] = None
    verify_certificates: bool = True
    
    # Encryption Scope
    encrypted_fields: Set[str] = field(default_factory=lambda: {
        "api_key", "password", "token", "secret", "credential"
    })
    
    def validate(self) -> List[str]:
        """Validate encryption configuration."""
        errors = []
        
        if self.key_rotation_days <= 0:
            errors.append("Key rotation days must be positive")
        
        if self.key_derivation_iterations < 10000:
            errors.append("Key derivation iterations should be at least 10000")
        
        if self.algorithm != EncryptionAlgorithm.NONE:
            if self.certificate_path and not self.certificate_path.exists():
                errors.append(f"Certificate path does not exist: {self.certificate_path}")
            
            if self.private_key_path and not self.private_key_path.exists():
                errors.append(f"Private key path does not exist: {self.private_key_path}")
        
        return errors


@dataclass
class AccessControlConfig(ConfigBase):
    """Access control configuration."""
    
    # RBAC Settings
    rbac_enabled: bool = True
    default_role: str = "viewer"
    
    # Role Definitions
    roles: Dict[str, List[str]] = field(default_factory=lambda: {
        "admin": ["*"],
        "developer": ["read", "write", "execute", "test"],
        "tester": ["read", "test", "report"],
        "viewer": ["read"]
    })
    
    # Permission Sets
    permissions: Dict[str, AccessLevel] = field(default_factory=lambda: {
        "config.read": AccessLevel.INTERNAL,
        "config.write": AccessLevel.RESTRICTED,
        "test.execute": AccessLevel.INTERNAL,
        "report.generate": AccessLevel.INTERNAL,
        "api.admin": AccessLevel.CONFIDENTIAL
    })
    
    # IP Restrictions
    ip_whitelist_enabled: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    blocked_ip_ranges: List[str] = field(default_factory=list)
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    def validate(self) -> List[str]:
        """Validate access control configuration."""
        errors = []
        
        if self.default_role not in self.roles:
            errors.append(f"Default role '{self.default_role}' not defined in roles")
        
        if self.rate_limit_per_minute <= 0:
            errors.append("Rate limit per minute must be positive")
        
        if self.rate_limit_per_hour <= 0:
            errors.append("Rate limit per hour must be positive")
        
        return errors


@dataclass
class AuditConfig(ConfigBase):
    """Audit and compliance configuration."""
    
    # Audit Settings
    audit_enabled: bool = True
    audit_level: str = "INFO"
    
    # Event Tracking
    track_authentication: bool = True
    track_authorization: bool = True
    track_data_access: bool = True
    track_configuration_changes: bool = True
    track_api_calls: bool = True
    
    # Audit Storage
    audit_log_path: Path = Path("audit.log")
    audit_retention_days: int = 365
    compress_old_audits: bool = True
    
    # Compliance
    compliance_standards: List[str] = field(default_factory=lambda: [
        "SOC2", "ISO27001"
    ])
    generate_compliance_reports: bool = True
    compliance_check_interval_days: int = 30
    
    # Sensitive Data
    mask_sensitive_data: bool = True
    sensitive_data_patterns: List[str] = field(default_factory=lambda: [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",  # Credit card
        r"\b\d{3}-\d{2}-\d{4}\b"  # SSN
    ])
    
    def validate(self) -> List[str]:
        """Validate audit configuration."""
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.audit_level not in valid_levels:
            errors.append(f"Invalid audit level. Must be one of {valid_levels}")
        
        if self.audit_retention_days <= 0:
            errors.append("Audit retention days must be positive")
        
        if self.compliance_check_interval_days <= 0:
            errors.append("Compliance check interval must be positive")
        
        return errors


@dataclass
class SecurityConfig(ConfigBase):
    """Combined security configuration."""
    
    authentication: AuthenticationConfig = field(default_factory=AuthenticationConfig)
    encryption: EncryptionConfig = field(default_factory=EncryptionConfig)
    access_control: AccessControlConfig = field(default_factory=AccessControlConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    
    # Security Headers
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
    })
    
    # Vulnerability Scanning
    vulnerability_scanning_enabled: bool = True
    scan_interval_days: int = 7
    auto_patch_critical: bool = False
    
    def validate(self) -> List[str]:
        """Validate all security configurations."""
        errors = []
        errors.extend(self.authentication.validate())
        errors.extend(self.encryption.validate())
        errors.extend(self.access_control.validate())
        errors.extend(self.audit.validate())
        
        if self.scan_interval_days <= 0:
            errors.append("Scan interval must be positive")
        
        return errors


__all__ = [
    'AuthenticationMethod',
    'EncryptionAlgorithm',
    'AccessLevel',
    'AuthenticationConfig',
    'EncryptionConfig',
    'AccessControlConfig',
    'AuditConfig',
    'SecurityConfig'
]