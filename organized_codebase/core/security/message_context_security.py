"""
AutoGen Derived Message Context Security Module
Extracted from AutoGen MessageContext and exception handling patterns
Enhanced for secure message handling and context validation
"""

import uuid
import json
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from .error_handler import (
    SecurityError, security_error_handler,
    ValidationError, AuthenticationError
)


class MessageSecurityLevel(Enum):
    """Message security levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MessageStatus(Enum):
    """Message processing status based on AutoGen patterns"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DROPPED = "dropped"
    UNDELIVERABLE = "undeliverable"


@dataclass
class SecureMessageContext:
    """Secure message context based on AutoGen MessageContext patterns"""
    message_id: str
    sender_id: Optional[str] = None
    topic_id: Optional[str] = None
    is_rpc: bool = False
    security_level: MessageSecurityLevel = MessageSecurityLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    status: MessageStatus = MessageStatus.PENDING
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    @property
    def is_expired(self) -> bool:
        """Check if message context has expired"""
        if self.expires_at:
            return datetime.utcnow() > self.expires_at
        return False
    
    @property
    def age_seconds(self) -> float:
        """Get message age in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    def add_processing_step(self, step: str, details: Dict[str, Any] = None):
        """Add a processing step to the message history"""
        self.processing_history.append({
            'step': step,
            'timestamp': datetime.utcnow().isoformat(),
            'details': details or {}
        })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'topic_id': self.topic_id,
            'is_rpc': self.is_rpc,
            'security_level': self.security_level.value,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'metadata': self.metadata,
            'processing_history': self.processing_history,
            'status': self.status.value,
            'age_seconds': self.age_seconds
        }


class MessageSecurityException(SecurityError):
    """Base class for message security exceptions based on AutoGen patterns"""
    def __init__(self, message: str, context: Optional[SecureMessageContext] = None):
        super().__init__(message, "MSG_SEC_001")
        self.context = context


class MessageCannotBeHandledException(MessageSecurityException):
    """Exception when message cannot be handled - based on AutoGen CantHandleException"""
    def __init__(self, message: str = "The handler cannot process the given message.", 
                 context: Optional[SecureMessageContext] = None):
        super().__init__(message, context)
        self.error_code = "MSG_HANDLE_001"


class MessageUndeliverableException(MessageSecurityException):
    """Exception when message cannot be delivered - based on AutoGen UndeliverableException"""
    def __init__(self, message: str = "The message cannot be delivered.", 
                 context: Optional[SecureMessageContext] = None):
        super().__init__(message, context)
        self.error_code = "MSG_DELIV_001"


class MessageDroppedException(MessageSecurityException):
    """Exception when message is dropped - based on AutoGen MessageDroppedException"""
    def __init__(self, message: str = "The message was dropped.", 
                 context: Optional[SecureMessageContext] = None):
        super().__init__(message, context)
        self.error_code = "MSG_DROP_001"


class MessageSecurityValidator:
    """Message security validation engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.max_message_age_seconds = 3600  # 1 hour
        self.max_metadata_size = 10240  # 10KB
        self.max_processing_steps = 100
        
        # Security patterns to detect
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'shell=True',
            r'<script[^>]*>',
            r'javascript:',
            r'data:text/html'
        ]
        
        # Allowed topics pattern
        self.topic_pattern = r'^[a-zA-Z0-9_\-\.]+$'
    
    def validate_message_context(self, context: SecureMessageContext) -> Tuple[bool, List[str]]:
        """Validate message context for security issues"""
        try:
            errors = []
            
            # Check message age
            if context.age_seconds > self.max_message_age_seconds:
                errors.append(f"Message is too old: {context.age_seconds} seconds")
            
            # Check if expired
            if context.is_expired:
                errors.append("Message context has expired")
            
            # Validate topic format if present
            if context.topic_id:
                import re
                if not re.match(self.topic_pattern, context.topic_id):
                    errors.append(f"Invalid topic format: {context.topic_id}")
            
            # Check metadata size
            metadata_str = json.dumps(context.metadata)
            if len(metadata_str.encode()) > self.max_metadata_size:
                errors.append(f"Metadata too large: {len(metadata_str)} bytes")
            
            # Check processing history size
            if len(context.processing_history) > self.max_processing_steps:
                errors.append(f"Too many processing steps: {len(context.processing_history)}")
            
            # Scan for dangerous patterns in metadata
            for pattern in self.dangerous_patterns:
                import re
                if re.search(pattern, metadata_str, re.IGNORECASE):
                    errors.append(f"Dangerous pattern detected in metadata: {pattern}")
            
            # Validate sender ID format if present
            if context.sender_id and not self._is_valid_sender_id(context.sender_id):
                errors.append(f"Invalid sender ID format: {context.sender_id}")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Message context validation error: {e}")
            return False, [f"Validation error: {str(e)}"]
    
    def _is_valid_sender_id(self, sender_id: str) -> bool:
        """Validate sender ID format"""
        import re
        # Should match agent_type/agent_key format
        pattern = r'^[a-zA-Z_][a-zA-Z0-9_]*/[\x20-\x7E]+$'
        return re.match(pattern, sender_id) is not None


class MessageContextManager:
    """Manager for secure message contexts and processing"""
    
    def __init__(self):
        self.validator = MessageSecurityValidator()
        self.active_contexts: Dict[str, SecureMessageContext] = {}
        self.completed_contexts: List[SecureMessageContext] = []
        self.message_handlers: Dict[str, Callable] = {}
        self.security_policies: Dict[MessageSecurityLevel, Dict[str, Any]] = {
            MessageSecurityLevel.PUBLIC: {
                'max_age_seconds': 3600,
                'require_auth': False,
                'audit_required': False
            },
            MessageSecurityLevel.INTERNAL: {
                'max_age_seconds': 1800,
                'require_auth': True,
                'audit_required': True
            },
            MessageSecurityLevel.CONFIDENTIAL: {
                'max_age_seconds': 900,
                'require_auth': True,
                'audit_required': True
            },
            MessageSecurityLevel.RESTRICTED: {
                'max_age_seconds': 300,
                'require_auth': True,
                'audit_required': True
            }
        }
        self.logger = logging.getLogger(__name__)
    
    def create_secure_context(self, sender_id: Optional[str] = None, 
                            topic_id: Optional[str] = None,
                            security_level: MessageSecurityLevel = MessageSecurityLevel.INTERNAL,
                            expires_in_seconds: Optional[int] = None,
                            metadata: Dict[str, Any] = None) -> SecureMessageContext:
        """Create a new secure message context"""
        try:
            context = SecureMessageContext(
                message_id=str(uuid.uuid4()),
                sender_id=sender_id,
                topic_id=topic_id,
                security_level=security_level,
                metadata=metadata or {}
            )
            
            # Set expiration based on policy or parameter
            if expires_in_seconds:
                context.expires_at = datetime.utcnow() + timedelta(seconds=expires_in_seconds)
            else:
                policy = self.security_policies.get(security_level)
                if policy:
                    context.expires_at = datetime.utcnow() + timedelta(seconds=policy['max_age_seconds'])
            
            # Validate the context
            is_valid, errors = self.validator.validate_message_context(context)
            if not is_valid:
                raise MessageSecurityException(f"Context validation failed: {errors}")
            
            # Store active context
            self.active_contexts[context.message_id] = context
            context.add_processing_step('created', {'security_level': security_level.value})
            
            self.logger.info(f"Created secure message context: {context.message_id}")
            return context
            
        except Exception as e:
            error = MessageSecurityException(f"Context creation failed: {str(e)}")
            security_error_handler.handle_error(error)
            raise error
    
    def process_message_securely(self, context: SecureMessageContext, 
                                message_data: Any,
                                handler_name: str = None) -> Any:
        """Process a message within a secure context"""
        try:
            # Update context status
            context.status = MessageStatus.PROCESSING
            context.add_processing_step('processing_started', {'handler': handler_name})
            
            # Validate context before processing
            is_valid, errors = self.validator.validate_message_context(context)
            if not is_valid:
                context.status = MessageStatus.FAILED
                context.add_processing_step('validation_failed', {'errors': errors})
                raise MessageCannotBeHandledException(f"Context validation failed: {errors}", context)
            
            # Check security policy compliance
            policy = self.security_policies.get(context.security_level)
            if policy and policy.get('require_auth') and not context.sender_id:
                context.status = MessageStatus.FAILED  
                context.add_processing_step('auth_failed', {'reason': 'missing_sender'})
                raise AuthenticationError("Authentication required for this security level")
            
            # Process the message (placeholder for actual message processing)
            result = self._execute_message_handler(context, message_data, handler_name)
            
            # Mark as completed
            context.status = MessageStatus.COMPLETED
            context.add_processing_step('processing_completed', {'result_type': type(result).__name__})
            
            self.logger.info(f"Message processed successfully: {context.message_id}")
            return result
            
        except (MessageSecurityException, AuthenticationError):
            raise
        except Exception as e:
            context.status = MessageStatus.FAILED
            context.add_processing_step('processing_error', {'error': str(e)})
            error = MessageCannotBeHandledException(f"Message processing failed: {str(e)}", context)
            security_error_handler.handle_error(error)
            raise error
    
    def complete_context(self, message_id: str) -> bool:
        """Mark a message context as completed and move to history"""
        try:
            if message_id in self.active_contexts:
                context = self.active_contexts[message_id]
                context.add_processing_step('context_completed')
                
                # Move to completed contexts
                self.completed_contexts.append(context)
                del self.active_contexts[message_id]
                
                # Keep history manageable
                if len(self.completed_contexts) > 1000:
                    self.completed_contexts = self.completed_contexts[-500:]
                
                self.logger.info(f"Message context completed: {message_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error completing context {message_id}: {e}")
            return False
    
    def drop_message(self, message_id: str, reason: str) -> bool:
        """Drop a message with specified reason"""
        try:
            if message_id in self.active_contexts:
                context = self.active_contexts[message_id]
                context.status = MessageStatus.DROPPED
                context.add_processing_step('message_dropped', {'reason': reason})
                
                # Move to completed contexts for audit
                self.completed_contexts.append(context)
                del self.active_contexts[message_id]
                
                self.logger.warning(f"Message dropped: {message_id} - Reason: {reason}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error dropping message {message_id}: {e}")
            return False
    
    def cleanup_expired_contexts(self) -> int:
        """Clean up expired message contexts"""
        expired_count = 0
        expired_ids = []
        
        for message_id, context in self.active_contexts.items():
            if context.is_expired:
                expired_ids.append(message_id)
        
        for message_id in expired_ids:
            self.drop_message(message_id, "expired")
            expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired message contexts")
        
        return expired_count
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get message context statistics"""
        active_count = len(self.active_contexts)
        completed_count = len(self.completed_contexts)
        
        # Count by status
        status_counts = {}
        security_level_counts = {}
        
        for context in self.active_contexts.values():
            status = context.status.value
            level = context.security_level.value
            status_counts[status] = status_counts.get(status, 0) + 1
            security_level_counts[level] = security_level_counts.get(level, 0) + 1
        
        # Recent activity
        hour_ago = datetime.utcnow() - timedelta(hours=1)
        recent_completed = [
            ctx for ctx in self.completed_contexts
            if ctx.created_at > hour_ago
        ]
        
        return {
            'active_contexts': active_count,
            'completed_contexts': completed_count,
            'recent_completed_1h': len(recent_completed),
            'status_breakdown': status_counts,
            'security_level_breakdown': security_level_counts,
            'handlers_registered': len(self.message_handlers)
        }
    
    def _execute_message_handler(self, context: SecureMessageContext, 
                                message_data: Any, handler_name: str) -> Any:
        """Execute message handler (placeholder for actual implementation)"""
        # This would integrate with actual message handlers
        return {'processed': True, 'context_id': context.message_id}


# Global message context manager
message_context_manager = MessageContextManager()


def create_message_context(sender_id: str = None, topic_id: str = None,
                          security_level: MessageSecurityLevel = MessageSecurityLevel.INTERNAL) -> SecureMessageContext:
    """Convenience function to create secure message context"""
    return message_context_manager.create_secure_context(sender_id, topic_id, security_level)


def process_secure_message(context: SecureMessageContext, message_data: Any) -> Any:
    """Convenience function to process message securely"""
    return message_context_manager.process_message_securely(context, message_data)