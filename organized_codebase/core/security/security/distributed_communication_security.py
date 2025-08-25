"""
Swarms Derived Distributed Communication Security Module
Extracted from Swarms Redis communication patterns for secure distributed messaging
Enhanced for encryption, authentication, and fault-tolerant communication
"""

import uuid
import time
import json
import hashlib
import hmac
import logging
import threading
import ssl
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from .error_handler import SecurityError, security_error_handler


class MessagePriority(Enum):
    """Message priority levels for distributed communication"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EncryptionLevel(Enum):
    """Encryption levels for message security"""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    MILITARY = "military"


class MessageStatus(Enum):
    """Status of distributed messages"""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class SecureChannel:
    """Secure communication channel configuration"""
    channel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    channel_name: str = ""
    encryption_level: EncryptionLevel = EncryptionLevel.ADVANCED
    max_message_size: int = 1024 * 1024  # 1MB default
    message_ttl: int = 3600  # 1 hour default
    allowed_participants: Set[str] = field(default_factory=set)
    encryption_key: Optional[bytes] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    is_persistent: bool = True
    
    def __post_init__(self):
        if not self.channel_name:
            self.channel_name = f"channel_{self.channel_id[:8]}"
        
        # Generate encryption key if not provided
        if not self.encryption_key and self.encryption_level != EncryptionLevel.NONE:
            self.encryption_key = Fernet.generate_key()
    
    @property
    def fernet_cipher(self) -> Optional[Fernet]:
        """Get Fernet cipher for encryption/decryption"""
        if self.encryption_key and self.encryption_level != EncryptionLevel.NONE:
            return Fernet(self.encryption_key)
        return None


@dataclass
class DistributedMessage:
    """Secure distributed message with metadata"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    channel_id: str = ""
    content: Any = None
    message_type: str = "data"
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    signature: Optional[str] = None
    encrypted_payload: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if not self.expires_at:
            self.expires_at = self.timestamp + timedelta(hours=1)
        
        if not self.sender_id:
            raise SecurityError("Sender ID is required", "DIST_MSG_001")
    
    @property
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return datetime.utcnow() > self.expires_at if self.expires_at else False
    
    @property
    def can_retry(self) -> bool:
        """Check if message can be retried"""
        return self.retry_count < self.max_retries
    
    def calculate_message_hash(self) -> str:
        """Calculate message hash for integrity verification"""
        message_data = {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'content': json.dumps(self.content) if self.content else None,
            'timestamp': self.timestamp.isoformat()
        }
        
        message_str = json.dumps(message_data, sort_keys=True)
        return hashlib.sha256(message_str.encode()).hexdigest()


@dataclass
class CommunicationNode:
    """Distributed communication node with security features"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_name: str = ""
    host_address: str = "localhost"
    port: int = 6379
    ssl_enabled: bool = True
    password: Optional[str] = None
    connection_timeout: int = 5
    max_connections: int = 100
    trust_score: float = 1.0
    last_heartbeat: Optional[datetime] = None
    is_active: bool = False
    security_violations: int = 0
    
    def __post_init__(self):
        if not self.node_name:
            self.node_name = f"node_{self.node_id[:8]}"
    
    @property
    def endpoint(self) -> str:
        """Get node network endpoint"""
        protocol = "rediss" if self.ssl_enabled else "redis"
        return f"{protocol}://{self.host_address}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if node is healthy"""
        if not self.last_heartbeat:
            return False
        
        threshold = datetime.utcnow() - timedelta(minutes=2)
        return self.last_heartbeat > threshold and self.is_active


class MessageEncryption:
    """Message encryption and decryption utilities"""
    
    @staticmethod
    def generate_key_from_password(password: str, salt: Optional[bytes] = None) -> bytes:
        """Generate encryption key from password using PBKDF2"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    @staticmethod
    def encrypt_message(message: str, cipher: Fernet) -> bytes:
        """Encrypt message content"""
        try:
            return cipher.encrypt(message.encode())
        except Exception as e:
            raise SecurityError(f"Message encryption failed: {str(e)}", "MSG_ENC_001")
    
    @staticmethod
    def decrypt_message(encrypted_data: bytes, cipher: Fernet) -> str:
        """Decrypt message content"""
        try:
            return cipher.decrypt(encrypted_data).decode()
        except Exception as e:
            raise SecurityError(f"Message decryption failed: {str(e)}", "MSG_DEC_001")
    
    @staticmethod
    def sign_message(message: str, secret_key: str) -> str:
        """Create HMAC signature for message integrity"""
        return hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
    
    @staticmethod
    def verify_signature(message: str, signature: str, secret_key: str) -> bool:
        """Verify HMAC signature"""
        expected_signature = MessageEncryption.sign_message(message, secret_key)
        return hmac.compare_digest(signature, expected_signature)


class MessageQueue:
    """Thread-safe message queue with priority handling"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.queues: Dict[MessagePriority, List[DistributedMessage]] = {
            priority: [] for priority in MessagePriority
        }
        self.queue_lock = threading.Lock()
        self.total_messages = 0
        self.logger = logging.getLogger(__name__)
    
    def enqueue(self, message: DistributedMessage) -> bool:
        """Add message to priority queue"""
        with self.queue_lock:
            if self.total_messages >= self.max_size:
                self.logger.warning("Message queue full, dropping message")
                return False
            
            # Clean expired messages first
            self._clean_expired_messages()
            
            self.queues[message.priority].append(message)
            self.total_messages += 1
            
            self.logger.debug(f"Message enqueued: {message.message_id}")
            return True
    
    def dequeue(self) -> Optional[DistributedMessage]:
        """Get highest priority message from queue"""
        with self.queue_lock:
            # Check priorities in descending order
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                
                if self.queues[priority]:
                    message = self.queues[priority].pop(0)
                    self.total_messages -= 1
                    return message
            
            return None
    
    def peek(self, priority: Optional[MessagePriority] = None) -> Optional[DistributedMessage]:
        """Peek at next message without removing it"""
        with self.queue_lock:
            if priority:
                return self.queues[priority][0] if self.queues[priority] else None
            
            # Return highest priority message
            for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                           MessagePriority.NORMAL, MessagePriority.LOW]:
                if self.queues[priority]:
                    return self.queues[priority][0]
            
            return None
    
    def size(self) -> int:
        """Get total queue size"""
        return self.total_messages
    
    def _clean_expired_messages(self):
        """Remove expired messages from queues"""
        for priority_queue in self.queues.values():
            # Remove expired messages
            priority_queue[:] = [msg for msg in priority_queue if not msg.is_expired]
        
        # Recalculate total
        self.total_messages = sum(len(queue) for queue in self.queues.values())


class DistributedCommunicationSecurityManager:
    """Secure distributed communication manager"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or "default_secret"
        self.channels: Dict[str, SecureChannel] = {}
        self.nodes: Dict[str, CommunicationNode] = {}
        self.message_queue = MessageQueue()
        self.message_log: Dict[str, DistributedMessage] = {}
        self.delivery_confirmations: Dict[str, datetime] = {}
        
        # Thread safety
        self.comm_lock = threading.RLock()
        self.processing_active = False
        
        # Message processing
        self.max_workers = 8
        self.processing_interval = 0.1  # seconds
        
        # Security settings
        self.max_message_rate = 1000  # messages per minute per node
        self.rate_tracking: Dict[str, List[datetime]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def create_secure_channel(self, channel: SecureChannel) -> bool:
        """Create secure communication channel"""
        try:
            with self.comm_lock:
                if channel.channel_id in self.channels:
                    raise SecurityError("Channel already exists", "COMM_CHAN_001")
                
                # Validate channel configuration
                if channel.max_message_size > 10 * 1024 * 1024:  # 10MB max
                    raise SecurityError("Message size limit too high", "COMM_CHAN_002")
                
                if channel.message_ttl > 86400:  # 24 hours max
                    raise SecurityError("Message TTL too high", "COMM_CHAN_003")
                
                self.channels[channel.channel_id] = channel
                
                self.logger.info(f"Secure channel created: {channel.channel_name}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Channel creation failed: {str(e)}", "COMM_CHAN_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def register_node(self, node: CommunicationNode) -> bool:
        """Register communication node"""
        try:
            with self.comm_lock:
                # Validate node configuration
                if not 1 <= node.port <= 65535:
                    raise SecurityError("Invalid port number", "COMM_NODE_001")
                
                if node.trust_score < 0.0 or node.trust_score > 1.0:
                    raise SecurityError("Invalid trust score", "COMM_NODE_002")
                
                self.nodes[node.node_id] = node
                node.last_heartbeat = datetime.utcnow()
                node.is_active = True
                
                # Initialize rate tracking
                self.rate_tracking[node.node_id] = []
                
                self.logger.info(f"Communication node registered: {node.node_name}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Node registration failed: {str(e)}", "COMM_NODE_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def send_secure_message(self, message: DistributedMessage) -> bool:
        """Send secure distributed message"""
        try:
            with self.comm_lock:
                # Validate sender
                if message.sender_id not in self.nodes:
                    raise SecurityError("Sender not registered", "COMM_SEND_001")
                
                sender_node = self.nodes[message.sender_id]
                if not sender_node.is_healthy:
                    raise SecurityError("Sender node unhealthy", "COMM_SEND_002")
                
                # Rate limiting check
                if not self._check_rate_limit(message.sender_id):
                    raise SecurityError("Rate limit exceeded", "COMM_SEND_003")
                
                # Validate channel
                if message.channel_id not in self.channels:
                    raise SecurityError("Channel not found", "COMM_SEND_004")
                
                channel = self.channels[message.channel_id]
                
                # Check channel permissions
                if channel.allowed_participants and message.sender_id not in channel.allowed_participants:
                    raise SecurityError("Sender not authorized for channel", "COMM_SEND_005")
                
                # Validate message size
                message_size = len(json.dumps(message.content).encode()) if message.content else 0
                if message_size > channel.max_message_size:
                    raise SecurityError("Message too large", "COMM_SEND_006")
                
                # Encrypt message if channel requires it
                if channel.encryption_level != EncryptionLevel.NONE:
                    self._encrypt_message(message, channel)
                
                # Sign message for integrity
                message_hash = message.calculate_message_hash()
                message.signature = MessageEncryption.sign_message(message_hash, self.secret_key)
                
                # Queue message for processing
                if not self.message_queue.enqueue(message):
                    raise SecurityError("Message queue full", "COMM_SEND_007")
                
                # Log message
                self.message_log[message.message_id] = message
                
                # Update rate tracking
                self.rate_tracking[message.sender_id].append(datetime.utcnow())
                
                self.logger.info(f"Secure message queued: {message.message_id}")
                return True
                
        except Exception as e:
            error = SecurityError(f"Secure message send failed: {str(e)}", "COMM_SEND_FAIL_001")
            security_error_handler.handle_error(error)
            return False
    
    def receive_message(self, node_id: str) -> Optional[DistributedMessage]:
        """Receive message for specific node"""
        try:
            with self.comm_lock:
                if node_id not in self.nodes:
                    return None
                
                # Look for messages addressed to this node
                for priority in [MessagePriority.CRITICAL, MessagePriority.HIGH, 
                               MessagePriority.NORMAL, MessagePriority.LOW]:
                    
                    queue = self.message_queue.queues[priority]
                    for i, message in enumerate(queue):
                        if message.recipient_id == node_id or message.recipient_id is None:
                            # Remove from queue
                            message = queue.pop(i)
                            self.message_queue.total_messages -= 1
                            
                            # Decrypt if necessary
                            if message.channel_id in self.channels:
                                channel = self.channels[message.channel_id]
                                if message.encrypted_payload:
                                    self._decrypt_message(message, channel)
                            
                            # Mark as delivered
                            self.delivery_confirmations[message.message_id] = datetime.utcnow()
                            
                            self.logger.debug(f"Message delivered: {message.message_id} to {node_id}")
                            return message
                
                return None
                
        except Exception as e:
            self.logger.error(f"Message receive failed: {e}")
            return None
    
    def start_processing(self):
        """Start background message processing"""
        if not self.processing_active:
            self.processing_active = True
            threading.Thread(target=self._process_messages, daemon=True).start()
            self.logger.info("Message processing started")
    
    def stop_processing(self):
        """Stop background message processing"""
        self.processing_active = False
        self.logger.info("Message processing stopped")
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication system statistics"""
        with self.comm_lock:
            active_nodes = sum(1 for node in self.nodes.values() if node.is_healthy)
            total_messages = len(self.message_log)
            delivered_messages = len(self.delivery_confirmations)
            
            queue_stats = {
                priority.name.lower(): len(queue) 
                for priority, queue in self.message_queue.queues.items()
            }
            
            return {
                'registered_nodes': len(self.nodes),
                'active_nodes': active_nodes,
                'secure_channels': len(self.channels),
                'total_messages': total_messages,
                'delivered_messages': delivered_messages,
                'delivery_rate': delivered_messages / max(1, total_messages),
                'queue_stats': queue_stats,
                'queue_size': self.message_queue.size()
            }
    
    def _encrypt_message(self, message: DistributedMessage, channel: SecureChannel):
        """Encrypt message content"""
        if not channel.fernet_cipher:
            return
        
        try:
            content_str = json.dumps(message.content) if message.content else ""
            message.encrypted_payload = MessageEncryption.encrypt_message(
                content_str, channel.fernet_cipher
            )
            
            # Clear original content for security
            message.content = None
            
        except Exception as e:
            raise SecurityError(f"Message encryption failed: {str(e)}", "COMM_ENC_001")
    
    def _decrypt_message(self, message: DistributedMessage, channel: SecureChannel):
        """Decrypt message content"""
        if not message.encrypted_payload or not channel.fernet_cipher:
            return
        
        try:
            decrypted_content = MessageEncryption.decrypt_message(
                message.encrypted_payload, channel.fernet_cipher
            )
            
            if decrypted_content:
                message.content = json.loads(decrypted_content)
            
            # Clear encrypted payload
            message.encrypted_payload = None
            
        except Exception as e:
            raise SecurityError(f"Message decryption failed: {str(e)}", "COMM_DEC_001")
    
    def _check_rate_limit(self, node_id: str) -> bool:
        """Check if node is within rate limits"""
        if node_id not in self.rate_tracking:
            self.rate_tracking[node_id] = []
        
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old entries
        self.rate_tracking[node_id] = [
            timestamp for timestamp in self.rate_tracking[node_id]
            if timestamp > minute_ago
        ]
        
        # Check rate limit
        return len(self.rate_tracking[node_id]) < self.max_message_rate
    
    def _process_messages(self):
        """Background message processing loop"""
        while self.processing_active:
            try:
                # Process pending messages
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = []
                    
                    # Get batch of messages to process
                    for _ in range(self.max_workers):
                        message = self.message_queue.dequeue()
                        if not message:
                            break
                        
                        future = executor.submit(self._process_single_message, message)
                        futures.append(future)
                    
                    # Wait for completion
                    for future in as_completed(futures, timeout=5):
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"Message processing error: {e}")
                
                time.sleep(self.processing_interval)
                
            except Exception as e:
                self.logger.error(f"Message processing loop error: {e}")
                time.sleep(1)
    
    def _process_single_message(self, message: DistributedMessage):
        """Process individual message"""
        try:
            # Verify message signature
            if message.signature:
                message_hash = message.calculate_message_hash()
                if not MessageEncryption.verify_signature(message_hash, message.signature, self.secret_key):
                    self.logger.warning(f"Invalid signature for message: {message.message_id}")
                    return
            
            # Message processing logic would go here
            # For now, just log successful processing
            self.logger.debug(f"Message processed: {message.message_id}")
            
        except Exception as e:
            self.logger.error(f"Single message processing failed: {e}")


# Global distributed communication manager
distributed_comm_security = DistributedCommunicationSecurityManager()


def create_communication_channel(channel_name: str, encryption_level: EncryptionLevel = EncryptionLevel.ADVANCED) -> Optional[str]:
    """Convenience function to create secure communication channel"""
    try:
        channel = SecureChannel(
            channel_name=channel_name,
            encryption_level=encryption_level
        )
        
        if distributed_comm_security.create_secure_channel(channel):
            return channel.channel_id
        return None
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Channel creation failed: {e}")
        return None


def send_distributed_message(sender_id: str, channel_id: str, content: Any, 
                           recipient_id: Optional[str] = None,
                           priority: MessagePriority = MessagePriority.NORMAL) -> bool:
    """Convenience function to send distributed message"""
    try:
        message = DistributedMessage(
            sender_id=sender_id,
            recipient_id=recipient_id,
            channel_id=channel_id,
            content=content,
            priority=priority
        )
        
        return distributed_comm_security.send_secure_message(message)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Message send failed: {e}")
        return False