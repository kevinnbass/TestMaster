"""
Llama-Agents Derived Service Mesh Security
Extracted from Llama-Agents service mesh patterns and microservice protection
Enhanced for comprehensive service-to-service security and mesh protection
"""

import logging
import ssl
import time
import hashlib
from typing import Dict, Any, Optional, List, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager
from .error_handler import SecurityError, ValidationError, security_error_handler


class ServiceSecurityLevel(Enum):
    """Service security levels based on Llama-Agents patterns"""
    PUBLIC = "public"
    INTERNAL = "internal" 
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class MeshSecurityPolicy(Enum):
    """Service mesh security policies"""
    ALLOW_ALL = "allow_all"
    DENY_ALL = "deny_all"
    WHITELIST = "whitelist"
    AUTHENTICATED_ONLY = "authenticated_only"
    MUTUAL_TLS = "mutual_tls"


@dataclass
class ServiceIdentity:
    """Service identity representation based on Llama-Agents patterns"""
    service_id: str
    service_name: str
    namespace: str = "default"
    security_level: ServiceSecurityLevel = ServiceSecurityLevel.INTERNAL
    certificates: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def identity_key(self) -> str:
        """Generate unique identity key"""
        return f"{self.namespace}/{self.service_name}:{self.service_id}"
    
    @property
    def is_valid(self) -> bool:
        """Check if service identity is valid"""
        required_fields = [self.service_id, self.service_name, self.namespace]
        return all(field and field.strip() for field in required_fields)


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration with security"""
    host: str
    port: int
    protocol: str = "https"
    use_tls: bool = True
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    auth_required: bool = True
    rate_limit: Optional[int] = None
    
    @property
    def base_url(self) -> str:
        """Generate base URL for endpoint"""
        return f"{self.protocol}://{self.host}:{self.port}"
    
    @property
    def is_secure(self) -> bool:
        """Check if endpoint is properly secured"""
        return self.use_tls and self.protocol == "https" and self.auth_required


@dataclass
class CORSPolicy:
    """CORS policy configuration based on Llama-Agents patterns"""
    allowed_origins: Set[str] = field(default_factory=lambda: {"*"})
    allowed_methods: Set[str] = field(default_factory=lambda: {"GET", "POST", "PUT", "DELETE"})
    allowed_headers: Set[str] = field(default_factory=lambda: {"*"})
    allow_credentials: bool = True
    max_age: int = 3600
    expose_headers: Set[str] = field(default_factory=set)
    
    def is_origin_allowed(self, origin: str) -> bool:
        """Check if origin is allowed"""
        if "*" in self.allowed_origins:
            return True
        return origin in self.allowed_origins
    
    def validate_request(self, origin: str, method: str, headers: List[str] = None) -> bool:
        """Validate CORS request"""
        if not self.is_origin_allowed(origin):
            return False
        
        if method not in self.allowed_methods:
            return False
        
        if headers and "*" not in self.allowed_headers:
            for header in headers:
                if header not in self.allowed_headers:
                    return False
        
        return True


class TLSConfiguration:
    """TLS configuration for service mesh security"""
    
    def __init__(self, cert_file: str = None, key_file: str = None,
                 ca_file: str = None, verify_mode: str = "required"):
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.verify_mode = verify_mode
        self.logger = logging.getLogger(__name__)
        
        # SSL context configuration
        self.ssl_context = self._create_ssl_context()
    
    def _create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context based on Llama-Agents patterns"""
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Configure verification mode
            if self.verify_mode == "required":
                context.check_hostname = True
                context.verify_mode = ssl.CERT_REQUIRED
            elif self.verify_mode == "optional":
                context.check_hostname = False
                context.verify_mode = ssl.CERT_OPTIONAL
            else:  # none
                context.check_hostname = False
                context.verify_mode = ssl.CERT_NONE
            
            # Load certificates if provided
            if self.cert_file and self.key_file:
                context.load_cert_chain(self.cert_file, self.key_file)
            
            if self.ca_file:
                context.load_verify_locations(self.ca_file)
            
            # Security settings
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to create SSL context: {e}")
            raise SecurityError(f"TLS configuration error: {str(e)}", "TLS_CONFIG_001")
    
    def validate_certificate(self, cert_data: bytes) -> bool:
        """Validate certificate against CA"""
        try:
            # Basic certificate validation
            import cryptography.x509 as x509
            
            cert = x509.load_pem_x509_certificate(cert_data)
            
            # Check expiration
            if cert.not_valid_after < datetime.utcnow():
                return False
            
            if cert.not_valid_before > datetime.utcnow():
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Certificate validation failed: {e}")
            return False


class ServiceMeshSecurityManager:
    """Comprehensive service mesh security management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.services: Dict[str, ServiceIdentity] = {}
        self.endpoints: Dict[str, ServiceEndpoint] = {}
        self.mesh_policies: Dict[str, MeshSecurityPolicy] = {}
        self.cors_policies: Dict[str, CORSPolicy] = {}
        self.tls_configs: Dict[str, TLSConfiguration] = {}
        
        # Security monitoring
        self.connection_attempts: List[Dict[str, Any]] = []
        self.security_violations: List[Dict[str, Any]] = []
        self.max_history = 10000
        
        # Default configurations
        self._initialize_default_policies()
    
    def register_service(self, service_identity: ServiceIdentity,
                        endpoint: ServiceEndpoint = None) -> bool:
        """Register service with mesh security"""
        try:
            if not service_identity.is_valid:
                raise ValidationError("Invalid service identity")
            
            identity_key = service_identity.identity_key
            
            # Check for duplicate registration
            if identity_key in self.services:
                self.logger.warning(f"Service already registered: {identity_key}")
                return False
            
            # Register service
            self.services[identity_key] = service_identity
            
            if endpoint:
                self.endpoints[identity_key] = endpoint
                
                # Create default CORS policy if needed
                if endpoint.cors_enabled and identity_key not in self.cors_policies:
                    self.cors_policies[identity_key] = CORSPolicy(
                        allowed_origins=set(endpoint.cors_origins)
                    )
            
            # Set default mesh policy
            if identity_key not in self.mesh_policies:
                if service_identity.security_level == ServiceSecurityLevel.PUBLIC:
                    self.mesh_policies[identity_key] = MeshSecurityPolicy.AUTHENTICATED_ONLY
                else:
                    self.mesh_policies[identity_key] = MeshSecurityPolicy.MUTUAL_TLS
            
            self.logger.info(f"Service registered: {identity_key}")
            return True
            
        except Exception as e:
            error = SecurityError(f"Failed to register service: {str(e)}", "MESH_REG_001")
            security_error_handler.handle_error(error)
            return False
    
    def validate_service_communication(self, source_service: str, 
                                     target_service: str,
                                     request_context: Dict[str, Any] = None) -> bool:
        """Validate service-to-service communication"""
        try:
            request_context = request_context or {}
            
            # Check if services are registered
            if source_service not in self.services:
                self._record_security_violation(
                    "unregistered_source", source_service, target_service, request_context
                )
                return False
            
            if target_service not in self.services:
                self._record_security_violation(
                    "unregistered_target", source_service, target_service, request_context
                )
                return False
            
            # Get mesh policy for target service
            policy = self.mesh_policies.get(target_service, MeshSecurityPolicy.DENY_ALL)
            
            if policy == MeshSecurityPolicy.DENY_ALL:
                self._record_security_violation(
                    "policy_deny_all", source_service, target_service, request_context
                )
                return False
            
            if policy == MeshSecurityPolicy.ALLOW_ALL:
                return True
            
            # Check security levels
            source_identity = self.services[source_service]
            target_identity = self.services[target_service]
            
            if policy == MeshSecurityPolicy.AUTHENTICATED_ONLY:
                # Basic authentication check
                if not request_context.get('authenticated', False):
                    self._record_security_violation(
                        "authentication_required", source_service, target_service, request_context
                    )
                    return False
            
            elif policy == MeshSecurityPolicy.MUTUAL_TLS:
                # Check for mutual TLS
                if not request_context.get('mutual_tls', False):
                    self._record_security_violation(
                        "mutual_tls_required", source_service, target_service, request_context
                    )
                    return False
                
                # Validate certificates
                if not self._validate_mtls_certificates(source_service, target_service, request_context):
                    return False
            
            elif policy == MeshSecurityPolicy.WHITELIST:
                # Check whitelist
                whitelist = request_context.get('whitelist', [])
                if source_service not in whitelist:
                    self._record_security_violation(
                        "not_whitelisted", source_service, target_service, request_context
                    )
                    return False
            
            # Record successful communication attempt
            self._record_connection_attempt(source_service, target_service, True, request_context)
            return True
            
        except Exception as e:
            self.logger.error(f"Communication validation error: {e}")
            self._record_security_violation(
                "validation_error", source_service, target_service, 
                {**request_context, 'error': str(e)}
            )
            return False
    
    def validate_cors_request(self, service_identity: str, origin: str, 
                             method: str, headers: List[str] = None) -> bool:
        """Validate CORS request based on Llama-Agents patterns"""
        try:
            if service_identity not in self.cors_policies:
                return False
            
            cors_policy = self.cors_policies[service_identity]
            
            if cors_policy.validate_request(origin, method, headers):
                self.logger.debug(f"CORS request allowed: {origin} -> {service_identity}")
                return True
            else:
                self._record_security_violation(
                    "cors_violation", origin, service_identity,
                    {'method': method, 'headers': headers}
                )
                return False
                
        except Exception as e:
            self.logger.error(f"CORS validation error: {e}")
            return False
    
    def configure_tls(self, service_identity: str, tls_config: TLSConfiguration):
        """Configure TLS for service"""
        try:
            self.tls_configs[service_identity] = tls_config
            self.logger.info(f"TLS configured for service: {service_identity}")
            
        except Exception as e:
            error = SecurityError(f"Failed to configure TLS: {str(e)}", "TLS_CONFIG_002")
            security_error_handler.handle_error(error)
    
    def get_service_security_status(self, service_identity: str) -> Dict[str, Any]:
        """Get security status for specific service"""
        try:
            if service_identity not in self.services:
                return {'error': 'Service not found'}
            
            service = self.services[service_identity]
            endpoint = self.endpoints.get(service_identity)
            policy = self.mesh_policies.get(service_identity)
            cors_policy = self.cors_policies.get(service_identity)
            tls_config = self.tls_configs.get(service_identity)
            
            # Count security events for this service
            recent_attempts = [
                a for a in self.connection_attempts[-100:]
                if a.get('target_service') == service_identity
            ]
            
            recent_violations = [
                v for v in self.security_violations[-100:]
                if v.get('target_service') == service_identity
            ]
            
            return {
                'service_identity': {
                    'service_id': service.service_id,
                    'service_name': service.service_name,
                    'namespace': service.namespace,
                    'security_level': service.security_level.value,
                    'created_at': service.created_at.isoformat()
                },
                'endpoint': {
                    'configured': endpoint is not None,
                    'secure': endpoint.is_secure if endpoint else False,
                    'protocol': endpoint.protocol if endpoint else None,
                    'use_tls': endpoint.use_tls if endpoint else False
                } if endpoint else None,
                'security_policy': policy.value if policy else None,
                'cors_enabled': cors_policy is not None,
                'tls_configured': tls_config is not None,
                'recent_connections': len(recent_attempts),
                'recent_violations': len(recent_violations),
                'security_score': self._calculate_security_score(service_identity)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting service security status: {e}")
            return {'error': str(e)}
    
    def get_mesh_security_overview(self) -> Dict[str, Any]:
        """Get comprehensive mesh security overview"""
        try:
            # Calculate statistics
            total_services = len(self.services)
            secure_endpoints = sum(1 for ep in self.endpoints.values() if ep.is_secure)
            tls_configured = len(self.tls_configs)
            
            # Security level distribution
            security_levels = {}
            for service in self.services.values():
                level = service.security_level.value
                security_levels[level] = security_levels.get(level, 0) + 1
            
            # Policy distribution
            policy_distribution = {}
            for policy in self.mesh_policies.values():
                policy_name = policy.value
                policy_distribution[policy_name] = policy_distribution.get(policy_name, 0) + 1
            
            # Recent activity
            recent_attempts = len([a for a in self.connection_attempts if 
                                 (datetime.utcnow() - datetime.fromisoformat(a['timestamp'])).total_seconds() < 3600])
            recent_violations = len([v for v in self.security_violations if 
                                   (datetime.utcnow() - datetime.fromisoformat(v['timestamp'])).total_seconds() < 3600])
            
            return {
                'service_mesh_stats': {
                    'total_services': total_services,
                    'secure_endpoints': secure_endpoints,
                    'tls_configured_services': tls_configured,
                    'cors_policies': len(self.cors_policies)
                },
                'security_distribution': {
                    'security_levels': security_levels,
                    'mesh_policies': policy_distribution
                },
                'security_activity_1h': {
                    'connection_attempts': recent_attempts,
                    'security_violations': recent_violations,
                    'violation_rate_pct': (recent_violations / max(recent_attempts, 1)) * 100
                },
                'overall_security_score': self._calculate_overall_security_score()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating mesh security overview: {e}")
            return {'error': str(e)}
    
    def _validate_mtls_certificates(self, source_service: str, target_service: str,
                                   request_context: Dict[str, Any]) -> bool:
        """Validate mutual TLS certificates"""
        try:
            # Check if both services have TLS configuration
            source_tls = self.tls_configs.get(source_service)
            target_tls = self.tls_configs.get(target_service)
            
            if not source_tls or not target_tls:
                return False
            
            # Validate certificates from request context
            client_cert = request_context.get('client_certificate')
            if not client_cert:
                return False
            
            # Validate certificate using target's TLS config
            return target_tls.validate_certificate(client_cert.encode())
            
        except Exception as e:
            self.logger.error(f"mTLS certificate validation failed: {e}")
            return False
    
    def _calculate_security_score(self, service_identity: str) -> float:
        """Calculate security score for service (0-100)"""
        score = 0.0
        
        # Base score for registration
        score += 20
        
        # Endpoint security
        endpoint = self.endpoints.get(service_identity)
        if endpoint and endpoint.is_secure:
            score += 25
        
        # TLS configuration
        if service_identity in self.tls_configs:
            score += 20
        
        # Security policy
        policy = self.mesh_policies.get(service_identity)
        if policy == MeshSecurityPolicy.MUTUAL_TLS:
            score += 20
        elif policy == MeshSecurityPolicy.AUTHENTICATED_ONLY:
            score += 10
        
        # CORS configuration
        if service_identity in self.cors_policies:
            score += 10
        
        # Recent violations penalty
        recent_violations = [
            v for v in self.security_violations[-50:]
            if v.get('target_service') == service_identity
        ]
        
        if recent_violations:
            penalty = min(len(recent_violations) * 2, 15)
            score -= penalty
        
        return max(0.0, min(100.0, score))
    
    def _calculate_overall_security_score(self) -> float:
        """Calculate overall mesh security score"""
        if not self.services:
            return 0.0
        
        total_score = sum(self._calculate_security_score(service) 
                         for service in self.services.keys())
        
        return total_score / len(self.services)
    
    def _record_connection_attempt(self, source: str, target: str, 
                                  success: bool, context: Dict[str, Any]):
        """Record connection attempt"""
        attempt = {
            'timestamp': datetime.utcnow().isoformat(),
            'source_service': source,
            'target_service': target,
            'success': success,
            'context': context
        }
        
        self.connection_attempts.append(attempt)
        
        # Limit history
        if len(self.connection_attempts) > self.max_history:
            self.connection_attempts = self.connection_attempts[-self.max_history // 2:]
    
    def _record_security_violation(self, violation_type: str, source: str, 
                                  target: str, context: Dict[str, Any]):
        """Record security violation"""
        violation = {
            'timestamp': datetime.utcnow().isoformat(),
            'violation_type': violation_type,
            'source_service': source,
            'target_service': target,
            'context': context
        }
        
        self.security_violations.append(violation)
        
        # Limit history
        if len(self.security_violations) > self.max_history:
            self.security_violations = self.security_violations[-self.max_history // 2:]
        
        self.logger.warning(f"Security violation: {violation_type} {source} -> {target}")
    
    def _initialize_default_policies(self):
        """Initialize default security policies"""
        # Default CORS policy
        self.default_cors_policy = CORSPolicy(
            allowed_origins={"https://localhost:3000", "https://127.0.0.1:3000"},
            allow_credentials=True
        )
        
        self.logger.info("Default mesh security policies initialized")


# Global service mesh security manager
service_mesh_security = ServiceMeshSecurityManager()


# Convenience functions
def register_secure_service(service_id: str, service_name: str, 
                           namespace: str = "default",
                           security_level: ServiceSecurityLevel = ServiceSecurityLevel.INTERNAL) -> bool:
    """Convenience function to register secure service"""
    identity = ServiceIdentity(
        service_id=service_id,
        service_name=service_name,
        namespace=namespace,
        security_level=security_level
    )
    return service_mesh_security.register_service(identity)


def validate_service_request(source: str, target: str, 
                           authenticated: bool = False,
                           mutual_tls: bool = False) -> bool:
    """Convenience function to validate service request"""
    context = {
        'authenticated': authenticated,
        'mutual_tls': mutual_tls
    }
    return service_mesh_security.validate_service_communication(source, target, context)