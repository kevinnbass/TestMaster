"""
Llama-Agents Derived Network Security Controls
Extracted from Llama-Agents network security patterns and traffic encryption
Enhanced for comprehensive network protection and secure communications
"""

import logging
import ssl
import socket
import hashlib
import hmac
import time
from typing import Dict, Any, Optional, List, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv6Address, AddressValueError
import secrets
from .error_handler import SecurityError, ValidationError, security_error_handler


class NetworkSecurityLevel(Enum):
    """Network security levels based on Llama-Agents patterns"""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"


class ProtocolType(Enum):
    """Network protocol types"""
    HTTP = "http"
    HTTPS = "https"
    TCP = "tcp"
    UDP = "udp"
    WEBSOCKET = "websocket"
    GRPC = "grpc"


class EncryptionLevel(Enum):
    """Network encryption levels"""
    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    STRONG = "strong"
    QUANTUM_RESISTANT = "quantum_resistant"


@dataclass
class NetworkEndpoint:
    """Network endpoint configuration based on Llama-Agents patterns"""
    host: str
    port: int
    protocol: ProtocolType
    security_level: NetworkSecurityLevel = NetworkSecurityLevel.INTERNAL
    encryption_level: EncryptionLevel = EncryptionLevel.STANDARD
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    rate_limit_per_minute: int = 100
    timeout_seconds: int = 30
    
    @property
    def is_secure(self) -> bool:
        """Check if endpoint uses secure protocol"""
        return self.protocol in [ProtocolType.HTTPS, ProtocolType.GRPC] and self.encryption_level != EncryptionLevel.NONE
    
    @property
    def endpoint_url(self) -> str:
        """Generate endpoint URL"""
        return f"{self.protocol.value}://{self.host}:{self.port}"


@dataclass
class TLSConfiguration:
    """TLS configuration for secure communications"""
    version: str = "TLSv1.3"
    cipher_suites: List[str] = field(default_factory=lambda: [
        "TLS_AES_256_GCM_SHA384",
        "TLS_CHACHA20_POLY1305_SHA256",
        "TLS_AES_128_GCM_SHA256"
    ])
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None
    ca_certificate_path: Optional[str] = None
    verify_peer: bool = True
    verify_hostname: bool = True
    
    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context with secure configuration"""
        try:
            context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
            
            # Set minimum TLS version
            if self.version == "TLSv1.3":
                context.minimum_version = ssl.TLSVersion.TLSv1_3
            elif self.version == "TLSv1.2":
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            else:
                context.minimum_version = ssl.TLSVersion.TLSv1_2
            
            # Configure verification
            if self.verify_peer:
                context.verify_mode = ssl.CERT_REQUIRED
            else:
                context.verify_mode = ssl.CERT_NONE
            
            context.check_hostname = self.verify_hostname
            
            # Load certificates
            if self.certificate_path and self.private_key_path:
                context.load_cert_chain(self.certificate_path, self.private_key_path)
            
            if self.ca_certificate_path:
                context.load_verify_locations(self.ca_certificate_path)
            
            # Set cipher suites
            if self.cipher_suites:
                cipher_string = ":".join(self.cipher_suites)
                context.set_ciphers(cipher_string)
            
            return context
            
        except Exception as e:
            raise SecurityError(f"Failed to create SSL context: {str(e)}", "SSL_CONTEXT_001")


@dataclass
class NetworkTrafficRule:
    """Network traffic filtering rule"""
    rule_id: str
    source_ips: List[str] = field(default_factory=list)
    destination_ips: List[str] = field(default_factory=list)
    source_ports: List[int] = field(default_factory=list)
    destination_ports: List[int] = field(default_factory=list)
    protocols: List[ProtocolType] = field(default_factory=list)
    action: str = "allow"  # allow, deny, log
    priority: int = 100
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def matches_traffic(self, source_ip: str, dest_ip: str, 
                       source_port: int, dest_port: int, 
                       protocol: ProtocolType) -> bool:
        """Check if rule matches network traffic"""
        try:
            # Check source IP
            if self.source_ips and source_ip not in self.source_ips:
                if not any(self._ip_in_range(source_ip, ip_range) for ip_range in self.source_ips):
                    return False
            
            # Check destination IP
            if self.destination_ips and dest_ip not in self.destination_ips:
                if not any(self._ip_in_range(dest_ip, ip_range) for ip_range in self.destination_ips):
                    return False
            
            # Check ports
            if self.source_ports and source_port not in self.source_ports:
                return False
            
            if self.destination_ports and dest_port not in self.destination_ports:
                return False
            
            # Check protocol
            if self.protocols and protocol not in self.protocols:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _ip_in_range(self, ip: str, ip_range: str) -> bool:
        """Check if IP is in range (supports CIDR notation)"""
        try:
            if '/' not in ip_range:
                return ip == ip_range
            
            from ipaddress import ip_network, ip_address
            network = ip_network(ip_range, strict=False)
            return ip_address(ip) in network
            
        except (AddressValueError, ValueError):
            return False


class IPWhitelistManager:
    """IP whitelist management based on Llama-Agents patterns"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.whitelisted_ips: Set[str] = set()
        self.blacklisted_ips: Set[str] = set()
        self.ip_access_history: Dict[str, List[datetime]] = {}
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_duration = timedelta(minutes=15)
        self.max_failed_attempts = 5
    
    def add_to_whitelist(self, ip_address: str) -> bool:
        """Add IP to whitelist"""
        try:
            self._validate_ip_address(ip_address)
            self.whitelisted_ips.add(ip_address)
            # Remove from blacklist if present
            self.blacklisted_ips.discard(ip_address)
            self.logger.info(f"Added IP to whitelist: {ip_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add IP to whitelist: {e}")
            return False
    
    def add_to_blacklist(self, ip_address: str, reason: str = "security_violation") -> bool:
        """Add IP to blacklist"""
        try:
            self._validate_ip_address(ip_address)
            self.blacklisted_ips.add(ip_address)
            # Remove from whitelist if present
            self.whitelisted_ips.discard(ip_address)
            self.logger.warning(f"Added IP to blacklist: {ip_address} (reason: {reason})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add IP to blacklist: {e}")
            return False
    
    def is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP is allowed access"""
        try:
            # Check blacklist first
            if ip_address in self.blacklisted_ips:
                return False
            
            # Check if IP is locked out due to failed attempts
            if self._is_ip_locked_out(ip_address):
                return False
            
            # Check whitelist (if empty, allow all except blacklisted)
            if not self.whitelisted_ips:
                return True
            
            return ip_address in self.whitelisted_ips
            
        except Exception as e:
            self.logger.error(f"Error checking IP access: {e}")
            return False
    
    def record_access_attempt(self, ip_address: str, success: bool):
        """Record IP access attempt"""
        try:
            current_time = datetime.utcnow()
            
            if success:
                # Reset failed attempts on successful access
                self.failed_attempts.pop(ip_address, None)
                
                # Record access history
                if ip_address not in self.ip_access_history:
                    self.ip_access_history[ip_address] = []
                
                self.ip_access_history[ip_address].append(current_time)
                
                # Keep only recent history (last 24 hours)
                cutoff_time = current_time - timedelta(hours=24)
                self.ip_access_history[ip_address] = [
                    t for t in self.ip_access_history[ip_address] if t > cutoff_time
                ]
                
            else:
                # Increment failed attempts
                self.failed_attempts[ip_address] = self.failed_attempts.get(ip_address, 0) + 1
                
                # Auto-blacklist if too many failed attempts
                if self.failed_attempts[ip_address] >= self.max_failed_attempts:
                    self.add_to_blacklist(ip_address, "too_many_failed_attempts")
            
        except Exception as e:
            self.logger.error(f"Error recording access attempt: {e}")
    
    def _validate_ip_address(self, ip_address: str):
        """Validate IP address format"""
        try:
            # Try IPv4
            IPv4Address(ip_address)
        except AddressValueError:
            try:
                # Try IPv6
                IPv6Address(ip_address)
            except AddressValueError:
                raise ValidationError(f"Invalid IP address format: {ip_address}")
    
    def _is_ip_locked_out(self, ip_address: str) -> bool:
        """Check if IP is temporarily locked out"""
        if ip_address not in self.failed_attempts:
            return False
        
        failed_count = self.failed_attempts[ip_address]
        if failed_count < self.max_failed_attempts:
            return False
        
        # IP is locked out - check if lockout period has expired
        # For simplicity, we'll use a fixed lockout duration
        # In a real implementation, you'd track the time of last failed attempt
        return True


class TrafficEncryptionManager:
    """Traffic encryption management system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.encryption_keys: Dict[str, bytes] = {}
        self.session_keys: Dict[str, Dict[str, Any]] = {}
        self.key_rotation_interval = timedelta(hours=24)
    
    def generate_session_key(self, session_id: str) -> bytes:
        """Generate secure session key"""
        try:
            # Generate 256-bit key
            session_key = secrets.token_bytes(32)
            
            # Store session key with metadata
            self.session_keys[session_id] = {
                'key': session_key,
                'created_at': datetime.utcnow(),
                'last_used': datetime.utcnow(),
                'usage_count': 0
            }
            
            self.logger.debug(f"Generated session key for session: {session_id}")
            return session_key
            
        except Exception as e:
            raise SecurityError(f"Failed to generate session key: {str(e)}", "KEY_GEN_001")
    
    def encrypt_message(self, message: bytes, session_id: str) -> bytes:
        """Encrypt message using session key"""
        try:
            if session_id not in self.session_keys:
                raise ValidationError(f"Session key not found: {session_id}")
            
            session_info = self.session_keys[session_id]
            session_key = session_info['key']
            
            # Update usage statistics
            session_info['last_used'] = datetime.utcnow()
            session_info['usage_count'] += 1
            
            # Simple XOR encryption for demonstration
            # In production, use proper encryption like AES-GCM
            encrypted = bytearray()
            for i, byte in enumerate(message):
                key_byte = session_key[i % len(session_key)]
                encrypted.append(byte ^ key_byte)
            
            return bytes(encrypted)
            
        except Exception as e:
            raise SecurityError(f"Encryption failed: {str(e)}", "ENCRYPT_001")
    
    def decrypt_message(self, encrypted_message: bytes, session_id: str) -> bytes:
        """Decrypt message using session key"""
        try:
            if session_id not in self.session_keys:
                raise ValidationError(f"Session key not found: {session_id}")
            
            session_info = self.session_keys[session_id]
            session_key = session_info['key']
            
            # Simple XOR decryption (same as encryption for XOR)
            decrypted = bytearray()
            for i, byte in enumerate(encrypted_message):
                key_byte = session_key[i % len(session_key)]
                decrypted.append(byte ^ key_byte)
            
            return bytes(decrypted)
            
        except Exception as e:
            raise SecurityError(f"Decryption failed: {str(e)}", "DECRYPT_001")
    
    def rotate_session_keys(self):
        """Rotate old session keys"""
        try:
            current_time = datetime.utcnow()
            keys_to_rotate = []
            
            for session_id, session_info in self.session_keys.items():
                if current_time - session_info['created_at'] > self.key_rotation_interval:
                    keys_to_rotate.append(session_id)
            
            for session_id in keys_to_rotate:
                old_key_info = self.session_keys.pop(session_id)
                self.logger.info(f"Rotated session key: {session_id} (used {old_key_info['usage_count']} times)")
            
            return len(keys_to_rotate)
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            return 0


class NetworkSecurityControlManager:
    """Comprehensive network security control system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.endpoints: Dict[str, NetworkEndpoint] = {}
        self.traffic_rules: List[NetworkTrafficRule] = []
        self.whitelist_manager = IPWhitelistManager()
        self.encryption_manager = TrafficEncryptionManager()
        self.connection_attempts: List[Dict[str, Any]] = []
        self.blocked_connections: List[Dict[str, Any]] = []
        self.max_history = 10000
        
        # Initialize default security rules
        self._initialize_default_rules()
    
    def register_endpoint(self, endpoint_id: str, endpoint: NetworkEndpoint) -> bool:
        """Register network endpoint with security controls"""
        try:
            # Validate endpoint configuration
            if not self._validate_endpoint_config(endpoint):
                return False
            
            # Register endpoint
            self.endpoints[endpoint_id] = endpoint
            
            # Add default whitelist entries if specified
            for ip in endpoint.allowed_ips:
                self.whitelist_manager.add_to_whitelist(ip)
            
            # Add blacklist entries if specified
            for ip in endpoint.blocked_ips:
                self.whitelist_manager.add_to_blacklist(ip, "endpoint_config")
            
            self.logger.info(f"Registered network endpoint: {endpoint_id} ({endpoint.endpoint_url})")
            return True
            
        except Exception as e:
            error = SecurityError(f"Failed to register endpoint: {str(e)}", "ENDPOINT_REG_001")
            security_error_handler.handle_error(error)
            return False
    
    def validate_connection(self, endpoint_id: str, source_ip: str, 
                           source_port: int = 0, 
                           protocol: ProtocolType = ProtocolType.TCP) -> bool:
        """Validate network connection attempt"""
        try:
            connection_info = {
                'timestamp': datetime.utcnow().isoformat(),
                'endpoint_id': endpoint_id,
                'source_ip': source_ip,
                'source_port': source_port,
                'protocol': protocol.value,
                'allowed': False,
                'reason': 'unknown'
            }
            
            # Check if endpoint exists
            if endpoint_id not in self.endpoints:
                connection_info['reason'] = 'endpoint_not_found'
                self._record_connection_attempt(connection_info)
                return False
            
            endpoint = self.endpoints[endpoint_id]
            
            # Check IP whitelist/blacklist
            if not self.whitelist_manager.is_ip_allowed(source_ip):
                connection_info['reason'] = 'ip_not_allowed'
                self._record_blocked_connection(connection_info)
                return False
            
            # Check traffic rules
            if not self._check_traffic_rules(source_ip, endpoint.host, 
                                           source_port, endpoint.port, protocol):
                connection_info['reason'] = 'traffic_rule_violation'
                self._record_blocked_connection(connection_info)
                return False
            
            # Check rate limiting
            if not self._check_rate_limit(source_ip, endpoint.rate_limit_per_minute):
                connection_info['reason'] = 'rate_limit_exceeded'
                self._record_blocked_connection(connection_info)
                return False
            
            # Connection allowed
            connection_info['allowed'] = True
            connection_info['reason'] = 'allowed'
            self._record_connection_attempt(connection_info)
            
            # Record successful access
            self.whitelist_manager.record_access_attempt(source_ip, True)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Connection validation error: {e}")
            return False
    
    def create_secure_connection(self, endpoint_id: str, 
                               tls_config: TLSConfiguration = None) -> Optional[ssl.SSLContext]:
        """Create secure connection with TLS"""
        try:
            if endpoint_id not in self.endpoints:
                raise ValidationError(f"Endpoint not found: {endpoint_id}")
            
            endpoint = self.endpoints[endpoint_id]
            
            if not endpoint.is_secure:
                self.logger.warning(f"Endpoint {endpoint_id} is not configured for secure connections")
                return None
            
            # Use provided TLS config or create default
            if not tls_config:
                tls_config = TLSConfiguration()
            
            ssl_context = tls_config.create_ssl_context()
            
            self.logger.info(f"Created secure connection context for endpoint: {endpoint_id}")
            return ssl_context
            
        except Exception as e:
            error = SecurityError(f"Failed to create secure connection: {str(e)}", "SECURE_CONN_001")
            security_error_handler.handle_error(error)
            return None
    
    def add_traffic_rule(self, rule: NetworkTrafficRule) -> bool:
        """Add network traffic rule"""
        try:
            # Validate rule
            if not rule.rule_id or not rule.action:
                raise ValidationError("Traffic rule must have ID and action")
            
            # Check for duplicate rule IDs
            if any(r.rule_id == rule.rule_id for r in self.traffic_rules):
                raise ValidationError(f"Traffic rule already exists: {rule.rule_id}")
            
            # Add rule
            self.traffic_rules.append(rule)
            
            # Sort rules by priority (higher priority first)
            self.traffic_rules.sort(key=lambda r: r.priority, reverse=True)
            
            self.logger.info(f"Added traffic rule: {rule.rule_id} (priority: {rule.priority})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add traffic rule: {e}")
            return False
    
    def get_network_security_status(self) -> Dict[str, Any]:
        """Get comprehensive network security status"""
        try:
            # Calculate statistics
            total_endpoints = len(self.endpoints)
            secure_endpoints = sum(1 for ep in self.endpoints.values() if ep.is_secure)
            
            # Connection statistics
            recent_connections = len([c for c in self.connection_attempts 
                                    if (datetime.utcnow() - datetime.fromisoformat(c['timestamp'])).total_seconds() < 3600])
            recent_blocked = len([c for c in self.blocked_connections 
                                if (datetime.utcnow() - datetime.fromisoformat(c['timestamp'])).total_seconds() < 3600])
            
            # IP whitelist statistics
            whitelisted_count = len(self.whitelist_manager.whitelisted_ips)
            blacklisted_count = len(self.whitelist_manager.blacklisted_ips)
            
            # Encryption statistics
            active_sessions = len(self.encryption_manager.session_keys)
            
            return {
                'endpoint_stats': {
                    'total_endpoints': total_endpoints,
                    'secure_endpoints': secure_endpoints,
                    'security_coverage_pct': (secure_endpoints / max(total_endpoints, 1)) * 100
                },
                'connection_stats_1h': {
                    'total_attempts': recent_connections,
                    'blocked_attempts': recent_blocked,
                    'success_rate_pct': ((recent_connections - recent_blocked) / max(recent_connections, 1)) * 100
                },
                'access_control': {
                    'whitelisted_ips': whitelisted_count,
                    'blacklisted_ips': blacklisted_count,
                    'traffic_rules': len(self.traffic_rules)
                },
                'encryption': {
                    'active_sessions': active_sessions,
                    'encryption_enabled': active_sessions > 0
                },
                'overall_security_score': self._calculate_security_score()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating network security status: {e}")
            return {'error': str(e)}
    
    def _validate_endpoint_config(self, endpoint: NetworkEndpoint) -> bool:
        """Validate endpoint configuration"""
        try:
            # Validate host
            if not endpoint.host:
                raise ValidationError("Endpoint host cannot be empty")
            
            # Validate port
            if not (1 <= endpoint.port <= 65535):
                raise ValidationError(f"Invalid port number: {endpoint.port}")
            
            # Validate IP addresses in allowed/blocked lists
            for ip in endpoint.allowed_ips + endpoint.blocked_ips:
                try:
                    IPv4Address(ip)
                except AddressValueError:
                    try:
                        IPv6Address(ip)
                    except AddressValueError:
                        raise ValidationError(f"Invalid IP address: {ip}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Endpoint validation failed: {e}")
            return False
    
    def _check_traffic_rules(self, source_ip: str, dest_ip: str,
                           source_port: int, dest_port: int,
                           protocol: ProtocolType) -> bool:
        """Check if traffic is allowed by rules"""
        for rule in self.traffic_rules:
            if rule.matches_traffic(source_ip, dest_ip, source_port, dest_port, protocol):
                if rule.action == "deny":
                    self.logger.info(f"Traffic denied by rule: {rule.rule_id}")
                    return False
                elif rule.action == "allow":
                    return True
                # Log action doesn't affect the result
        
        # Default allow if no matching deny rules
        return True
    
    def _check_rate_limit(self, source_ip: str, limit_per_minute: int) -> bool:
        """Check rate limiting for source IP"""
        if source_ip not in self.whitelist_manager.ip_access_history:
            return True
        
        current_time = datetime.utcnow()
        minute_ago = current_time - timedelta(minutes=1)
        
        recent_accesses = [
            t for t in self.whitelist_manager.ip_access_history[source_ip]
            if t > minute_ago
        ]
        
        return len(recent_accesses) < limit_per_minute
    
    def _calculate_security_score(self) -> float:
        """Calculate overall network security score"""
        score = 0.0
        
        if not self.endpoints:
            return 0.0
        
        # Endpoint security
        secure_endpoints = sum(1 for ep in self.endpoints.values() if ep.is_secure)
        score += (secure_endpoints / len(self.endpoints)) * 40
        
        # Access control
        if self.whitelist_manager.whitelisted_ips or self.whitelist_manager.blacklisted_ips:
            score += 20
        
        # Traffic rules
        if self.traffic_rules:
            score += 15
        
        # Encryption
        if self.encryption_manager.session_keys:
            score += 15
        
        # Recent security violations penalty
        recent_blocked = len([c for c in self.blocked_connections[-100:]])
        recent_total = len([c for c in self.connection_attempts[-100:]])
        
        if recent_total > 0:
            violation_rate = recent_blocked / recent_total
            penalty = violation_rate * 10
            score = max(0, score - penalty)
        else:
            score += 10  # Bonus for no recent violations
        
        return min(100.0, score)
    
    def _record_connection_attempt(self, connection_info: Dict[str, Any]):
        """Record connection attempt"""
        self.connection_attempts.append(connection_info)
        
        # Limit history
        if len(self.connection_attempts) > self.max_history:
            self.connection_attempts = self.connection_attempts[-self.max_history // 2:]
    
    def _record_blocked_connection(self, connection_info: Dict[str, Any]):
        """Record blocked connection"""
        self.blocked_connections.append(connection_info)
        self._record_connection_attempt(connection_info)
        
        # Limit history
        if len(self.blocked_connections) > self.max_history:
            self.blocked_connections = self.blocked_connections[-self.max_history // 2:]
        
        # Record failed access
        source_ip = connection_info.get('source_ip')
        if source_ip:
            self.whitelist_manager.record_access_attempt(source_ip, False)
    
    def _initialize_default_rules(self):
        """Initialize default security rules"""
        # Default deny rule for suspicious traffic
        default_deny_rule = NetworkTrafficRule(
            rule_id="default_deny_suspicious",
            action="deny",
            priority=1,
            description="Deny suspicious traffic patterns"
        )
        
        self.traffic_rules.append(default_deny_rule)
        self.logger.info("Initialized default network security rules")


# Global network security control manager
network_security_controls = NetworkSecurityControlManager()


# Convenience functions
def register_secure_endpoint(endpoint_id: str, host: str, port: int, 
                           protocol: ProtocolType = ProtocolType.HTTPS) -> bool:
    """Convenience function to register secure endpoint"""
    endpoint = NetworkEndpoint(
        host=host,
        port=port,
        protocol=protocol,
        security_level=NetworkSecurityLevel.INTERNAL,
        encryption_level=EncryptionLevel.STANDARD
    )
    return network_security_controls.register_endpoint(endpoint_id, endpoint)


def validate_network_connection(endpoint_id: str, source_ip: str) -> bool:
    """Convenience function to validate network connection"""
    return network_security_controls.validate_connection(endpoint_id, source_ip)