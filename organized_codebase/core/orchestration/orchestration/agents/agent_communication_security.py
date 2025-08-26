"""
AutoGen Derived Agent Communication Security Module
Extracted from AutoGen agent worker protocols and RPC patterns
Enhanced for secure multi-agent communication
"""

import uuid
import time
import json
import logging
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import SecurityError, security_error_handler


class MessageType(Enum):
    """Message types based on AutoGen patterns"""
    RPC_REQUEST = "rpc_request"
    RPC_RESPONSE = "rpc_response"
    CLOUD_EVENT = "cloud_event"
    CONTROL_MESSAGE = "control_message"
    BROADCAST = "broadcast"


@dataclass
class AgentIdentity:
    """Agent identity based on AutoGen AgentId patterns"""
    agent_type: str
    agent_key: str
    runtime_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        # Validate agent_type and agent_key based on AutoGen patterns
        import re
        type_pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*$'
        key_pattern = r'^[\x20-\x7E]+$'  # ASCII 32-126
        
        if not re.match(type_pattern, self.agent_type):
            raise SecurityError(
                f"Invalid agent type: '{self.agent_type}'. Must be alphanumeric (a-z, 0-9, _) and cannot start with a number.",
                "AGENT_ID_001"
            )
        
        if not re.match(key_pattern, self.agent_key):
            raise SecurityError(
                f"Invalid agent key: '{self.agent_key}'. Must only contain ASCII characters 32-126.",
                "AGENT_ID_002"
            )
    
    @property
    def full_id(self) -> str:
        """Get full agent identifier in type/key format"""
        return f"{self.agent_type}/{self.agent_key}"
    
    def __str__(self) -> str:
        return self.full_id


@dataclass
class SecureMessage:
    """Secure message structure based on AutoGen Message patterns"""
    message_id: str
    message_type: MessageType
    sender: AgentIdentity
    target: Optional[AgentIdentity]
    payload: Dict[str, Any]
    metadata: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    signature: Optional[str] = None
    ttl_seconds: int = 300  # 5 minutes default TTL
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL"""
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for transmission"""
        return {
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender': self.sender.full_id,
            'target': self.target.full_id if self.target else None,
            'payload': self.payload,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat(),
            'signature': self.signature,
            'ttl_seconds': self.ttl_seconds
        }


class AgentCommunicationSecurityManager:
    """Secure communication manager for agent-to-agent interactions"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or "default_secret"  # Should be configured properly
        self.registered_agents: Dict[str, AgentIdentity] = {}
        self.message_log: List[SecureMessage] = []
        self.blocked_agents: Dict[str, datetime] = {}
        self.rate_limits: Dict[str, List[datetime]] = {}
        self.max_messages_per_minute = 100
        self.logger = logging.getLogger(__name__)
        
        # Message validation patterns
        self.suspicious_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(',
        ]
    
    def register_agent(self, agent: AgentIdentity) -> bool:
        """Register an agent for secure communication"""
        try:
            if agent.full_id in self.registered_agents:
                existing = self.registered_agents[agent.full_id]
                if existing.created_at < agent.created_at:
                    # Allow re-registration with newer timestamp
                    self.registered_agents[agent.full_id] = agent
                    self.logger.info(f"Re-registered agent: {agent.full_id}")
                else:
                    self.logger.warning(f"Agent re-registration rejected: {agent.full_id}")
                    return False
            else:
                self.registered_agents[agent.full_id] = agent
                self.logger.info(f"Registered new agent: {agent.full_id}")
            
            return True
            
        except Exception as e:
            error = SecurityError(f"Agent registration failed: {str(e)}", "AGENT_REG_001")
            security_error_handler.handle_error(error)
            return False
    
    def validate_message(self, message: SecureMessage) -> Tuple[bool, Optional[str]]:
        """Validate message security and integrity"""
        try:
            # Check if message has expired
            if message.is_expired:
                return False, "Message has expired"
            
            # Verify sender is registered
            if message.sender.full_id not in self.registered_agents:
                return False, f"Unknown sender: {message.sender.full_id}"
            
            # Check if sender is blocked
            if message.sender.full_id in self.blocked_agents:
                block_time = self.blocked_agents[message.sender.full_id]
                if datetime.utcnow() < block_time:
                    return False, f"Sender is blocked until: {block_time}"
                else:
                    # Unblock expired blocks
                    del self.blocked_agents[message.sender.full_id]
            
            # Rate limiting check
            if not self._check_rate_limit(message.sender.full_id):
                self._block_agent(message.sender.full_id, minutes=5)
                return False, "Rate limit exceeded - agent blocked"
            
            # Validate payload content for suspicious patterns
            payload_str = json.dumps(message.payload)
            for pattern in self.suspicious_patterns:
                import re
                if re.search(pattern, payload_str, re.IGNORECASE):
                    self.logger.warning(f"Suspicious pattern detected in message from {message.sender.full_id}: {pattern}")
                    self._block_agent(message.sender.full_id, minutes=15)
                    return False, f"Suspicious content detected: {pattern}"
            
            # Verify message signature if present
            if message.signature:
                expected_signature = self._calculate_signature(message)
                if message.signature != expected_signature:
                    return False, "Invalid message signature"
            
            return True, None
            
        except Exception as e:
            self.logger.error(f"Message validation error: {e}")
            return False, f"Validation error: {str(e)}"
    
    def secure_send_message(self, message: SecureMessage, sign_message: bool = True) -> bool:
        """Securely send a message with validation and signing"""
        try:
            # Sign the message if requested
            if sign_message:
                message.signature = self._calculate_signature(message)
            
            # Validate message
            is_valid, error = self.validate_message(message)
            if not is_valid:
                error_msg = SecurityError(f"Message validation failed: {error}", "MSG_VAL_001")
                security_error_handler.handle_error(error_msg)
                return False
            
            # Log the message
            self.message_log.append(message)
            
            # Keep only recent messages (last 1000)
            if len(self.message_log) > 1000:
                self.message_log = self.message_log[-1000:]
            
            self.logger.info(f"Message sent securely: {message.message_id} from {message.sender.full_id}")
            return True
            
        except Exception as e:
            error = SecurityError(f"Secure message send failed: {str(e)}", "MSG_SEND_001")
            security_error_handler.handle_error(error)
            return False
    
    def create_rpc_request(self, sender: AgentIdentity, target: AgentIdentity, 
                          method: str, params: Dict[str, Any]) -> SecureMessage:
        """Create a secure RPC request message"""
        payload = {
            'method': method,
            'params': params,
            'rpc_id': str(uuid.uuid4())
        }
        
        return SecureMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RPC_REQUEST,
            sender=sender,
            target=target,
            payload=payload,
            metadata={'rpc_method': method}
        )
    
    def create_rpc_response(self, request_message: SecureMessage, 
                           result: Any = None, error: str = None) -> SecureMessage:
        """Create a secure RPC response message"""
        payload = {
            'rpc_id': request_message.payload.get('rpc_id'),
            'result': result,
            'error': error
        }
        
        return SecureMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RPC_RESPONSE,
            sender=request_message.target,  # Response sender is original target
            target=request_message.sender,   # Response target is original sender
            payload=payload,
            metadata={'response_to': request_message.message_id}
        )
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication security statistics"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        recent_messages = [
            msg for msg in self.message_log 
            if msg.timestamp > hour_ago
        ]
        
        # Count messages by type
        message_counts = {}
        for msg in recent_messages:
            msg_type = msg.message_type.value
            message_counts[msg_type] = message_counts.get(msg_type, 0) + 1
        
        # Count blocked agents
        active_blocks = sum(1 for block_time in self.blocked_agents.values() if now < block_time)
        
        return {
            'registered_agents': len(self.registered_agents),
            'total_messages': len(self.message_log),
            'recent_messages_1h': len(recent_messages),
            'message_types_1h': message_counts,
            'active_blocks': active_blocks,
            'total_blocks': len(self.blocked_agents)
        }
    
    def _calculate_signature(self, message: SecureMessage) -> str:
        """Calculate HMAC signature for message integrity"""
        import hmac
        
        # Create message hash without signature field
        message_data = {
            'message_id': message.message_id,
            'message_type': message.message_type.value,
            'sender': message.sender.full_id,
            'target': message.target.full_id if message.target else None,
            'payload': message.payload,
            'timestamp': message.timestamp.isoformat()
        }
        
        message_str = json.dumps(message_data, sort_keys=True)
        signature = hmac.new(
            self.secret_key.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _check_rate_limit(self, agent_id: str) -> bool:
        """Check if agent is within rate limits"""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        if agent_id not in self.rate_limits:
            self.rate_limits[agent_id] = []
        
        # Clean old entries
        self.rate_limits[agent_id] = [
            timestamp for timestamp in self.rate_limits[agent_id]
            if timestamp > minute_ago
        ]
        
        # Check current count
        if len(self.rate_limits[agent_id]) >= self.max_messages_per_minute:
            return False
        
        # Add current timestamp
        self.rate_limits[agent_id].append(now)
        return True
    
    def _block_agent(self, agent_id: str, minutes: int = 5):
        """Block agent for specified duration"""
        block_until = datetime.utcnow() + timedelta(minutes=minutes)
        self.blocked_agents[agent_id] = block_until
        self.logger.warning(f"Blocked agent {agent_id} until {block_until}")


# Global communication security manager
agent_comm_security = AgentCommunicationSecurityManager()


def register_secure_agent(agent_type: str, agent_key: str) -> Optional[AgentIdentity]:
    """Convenience function to register a secure agent"""
    try:
        agent = AgentIdentity(agent_type, agent_key)
        if agent_comm_security.register_agent(agent):
            return agent
        return None
    except Exception as e:
        logging.getLogger(__name__).error(f"Agent registration failed: {e}")
        return None


def send_secure_message(message: SecureMessage) -> bool:
    """Convenience function to send a secure message"""
    return agent_comm_security.secure_send_message(message)