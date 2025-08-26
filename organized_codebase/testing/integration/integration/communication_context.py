"""
Communication Context Management
===============================

Context management for integration communications including request/response
tracking, session management, and distributed tracing support.

Author: Agent E - Infrastructure Consolidation
"""

from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import time
from datetime import datetime
import json
import threading
from contextlib import asynccontextmanager

from .integration_base import IntegrationContext, IntegrationPriority


class ContextScope(Enum):
    """Context scope enumeration."""
    REQUEST = "request"
    SESSION = "session"
    TRANSACTION = "transaction"
    WORKFLOW = "workflow"
    SYSTEM = "system"
    GLOBAL = "global"


class ContextType(Enum):
    """Context type enumeration."""
    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    BATCH = "batch"
    STREAMING = "streaming"
    EVENT_DRIVEN = "event_driven"


class ContextState(Enum):
    """Context state enumeration."""
    CREATED = "created"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


@dataclass
class ContextMetadata:
    """Context metadata information."""
    
    # Basic metadata
    context_id: str
    context_type: ContextType
    scope: ContextScope
    created_at: datetime
    
    # Hierarchy information
    parent_context_id: Optional[str] = None
    root_context_id: Optional[str] = None
    child_context_ids: List[str] = field(default_factory=list)
    
    # Execution information
    execution_start: Optional[datetime] = None
    execution_end: Optional[datetime] = None
    execution_duration: Optional[float] = None
    
    # Tracing information
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)
    
    # Tags and labels
    tags: Dict[str, str] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def get_duration(self) -> Optional[float]:
        """Get execution duration in seconds."""
        if self.execution_start and self.execution_end:
            return (self.execution_end - self.execution_start).total_seconds()
        return None
    
    def add_tag(self, key: str, value: str):
        """Add tag to context."""
        self.tags[key] = value
    
    def add_label(self, key: str, value: str):
        """Add label to context."""
        self.labels[key] = value
    
    def add_baggage(self, key: str, value: str):
        """Add baggage item for distributed tracing."""
        self.baggage[key] = value


@dataclass
class ContextData:
    """Context data container."""
    
    # Request/response data
    request_data: Optional[Dict[str, Any]] = None
    response_data: Optional[Dict[str, Any]] = None
    
    # Headers and parameters
    headers: Dict[str, str] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Authentication and authorization
    user_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    
    # Session data
    session_data: Dict[str, Any] = field(default_factory=dict)
    
    # Custom attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    # Temporary storage
    temporary_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get attribute value."""
        return self.attributes.get(key, default)
    
    def set_attribute(self, key: str, value: Any):
        """Set attribute value."""
        self.attributes[key] = value
    
    def get_temp_data(self, key: str, default: Any = None) -> Any:
        """Get temporary data value."""
        return self.temporary_data.get(key, default)
    
    def set_temp_data(self, key: str, value: Any):
        """Set temporary data value."""
        self.temporary_data[key] = value
    
    def clear_temp_data(self):
        """Clear all temporary data."""
        self.temporary_data.clear()


class CommunicationContext:
    """
    Communication context for managing integration request/response lifecycle.
    
    Provides distributed tracing, session management, and context propagation
    across integration boundaries.
    """
    
    def __init__(self, 
                 context_type: ContextType = ContextType.SYNCHRONOUS,
                 scope: ContextScope = ContextScope.REQUEST,
                 parent_context: Optional['CommunicationContext'] = None):
        
        # Generate unique context ID
        self.context_id = str(uuid.uuid4())
        
        # Initialize metadata
        self.metadata = ContextMetadata(
            context_id=self.context_id,
            context_type=context_type,
            scope=scope,
            created_at=datetime.now()
        )
        
        # Initialize data container
        self.data = ContextData()
        
        # Context state
        self.state = ContextState.CREATED
        
        # Parent/child relationships
        self.parent_context = parent_context
        self.child_contexts: List['CommunicationContext'] = []
        
        if parent_context:
            self.metadata.parent_context_id = parent_context.context_id
            self.metadata.root_context_id = parent_context.get_root_context_id()
            parent_context.add_child_context(self)
        else:
            self.metadata.root_context_id = self.context_id
        
        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self.state == ContextState.ACTIVE
    
    @property
    def is_completed(self) -> bool:
        """Check if context is completed."""
        return self.state in [ContextState.COMPLETED, ContextState.FAILED, ContextState.CANCELLED]
    
    def get_root_context_id(self) -> str:
        """Get root context ID."""
        return self.metadata.root_context_id or self.context_id
    
    def get_context_path(self) -> List[str]:
        """Get full context path from root to current."""
        path = []
        current = self
        while current:
            path.insert(0, current.context_id)
            current = current.parent_context
        return path
    
    # Context lifecycle management
    
    def start(self):
        """Start context execution."""
        with self._lock:
            if self.state != ContextState.CREATED:
                raise ValueError(f"Context already started or completed: {self.state}")
            
            self.state = ContextState.ACTIVE
            self.metadata.execution_start = datetime.now()
            self._emit_event("context_started")
    
    def complete(self, success: bool = True):
        """Complete context execution."""
        with self._lock:
            if self.is_completed:
                return  # Already completed
            
            self.state = ContextState.COMPLETED if success else ContextState.FAILED
            self.metadata.execution_end = datetime.now()
            self.metadata.execution_duration = self.metadata.get_duration()
            
            # Complete all child contexts
            for child_context in self.child_contexts:
                if not child_context.is_completed:
                    child_context.complete(success)
            
            self._emit_event("context_completed" if success else "context_failed")
    
    def cancel(self):
        """Cancel context execution."""
        with self._lock:
            if self.is_completed:
                return  # Already completed
            
            self.state = ContextState.CANCELLED
            self.metadata.execution_end = datetime.now()
            self.metadata.execution_duration = self.metadata.get_duration()
            
            # Cancel all child contexts
            for child_context in self.child_contexts:
                if not child_context.is_completed:
                    child_context.cancel()
            
            self._emit_event("context_cancelled")
    
    def suspend(self):
        """Suspend context execution."""
        with self._lock:
            if self.state == ContextState.ACTIVE:
                self.state = ContextState.SUSPENDED
                self._emit_event("context_suspended")
    
    def resume(self):
        """Resume context execution."""
        with self._lock:
            if self.state == ContextState.SUSPENDED:
                self.state = ContextState.ACTIVE
                self._emit_event("context_resumed")
    
    # Child context management
    
    def add_child_context(self, child_context: 'CommunicationContext'):
        """Add child context."""
        with self._lock:
            self.child_contexts.append(child_context)
            self.metadata.child_context_ids.append(child_context.context_id)
    
    def create_child_context(self, 
                            context_type: Optional[ContextType] = None,
                            scope: Optional[ContextScope] = None) -> 'CommunicationContext':
        """Create child context."""
        child_context = CommunicationContext(
            context_type=context_type or self.metadata.context_type,
            scope=scope or ContextScope.REQUEST,
            parent_context=self
        )
        
        # Inherit trace information
        child_context.metadata.trace_id = self.metadata.trace_id
        child_context.metadata.baggage = self.metadata.baggage.copy()
        
        return child_context
    
    # Data management
    
    def set_request_data(self, data: Dict[str, Any]):
        """Set request data."""
        with self._lock:
            self.data.request_data = data
    
    def set_response_data(self, data: Dict[str, Any]):
        """Set response data."""
        with self._lock:
            self.data.response_data = data
    
    def add_header(self, key: str, value: str):
        """Add header."""
        with self._lock:
            self.data.headers[key] = value
    
    def add_parameter(self, key: str, value: Any):
        """Add parameter."""
        with self._lock:
            self.data.parameters[key] = value
    
    def set_user_context(self, user_id: str, roles: List[str] = None, permissions: List[str] = None):
        """Set user context."""
        with self._lock:
            self.data.user_id = user_id
            self.data.roles = roles or []
            self.data.permissions = permissions or []
    
    # Distributed tracing support
    
    def set_trace_info(self, trace_id: str, span_id: str):
        """Set distributed trace information."""
        with self._lock:
            self.metadata.trace_id = trace_id
            self.metadata.span_id = span_id
    
    def add_baggage(self, key: str, value: str):
        """Add baggage for distributed tracing."""
        with self._lock:
            self.metadata.add_baggage(key, value)
    
    def get_trace_headers(self) -> Dict[str, str]:
        """Get trace headers for propagation."""
        headers = {}
        
        if self.metadata.trace_id:
            headers['X-Trace-Id'] = self.metadata.trace_id
        
        if self.metadata.span_id:
            headers['X-Span-Id'] = self.metadata.span_id
        
        if self.metadata.baggage:
            headers['X-Baggage'] = json.dumps(self.metadata.baggage)
        
        return headers
    
    # Event handling
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler."""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler."""
        if event_type in self._event_handlers:
            self._event_handlers[event_type].remove(handler)
    
    def _emit_event(self, event_type: str):
        """Emit event to registered handlers."""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(self, event_type)
                except Exception as e:
                    # Log error but don't fail context
                    print(f"Context event handler error: {e}")
    
    # Integration context conversion
    
    def to_integration_context(self) -> IntegrationContext:
        """Convert to IntegrationContext for compatibility."""
        return IntegrationContext(
            integration_id=self.context_id,
            session_id=self.get_root_context_id(),
            request_id=self.context_id,
            metadata=self.data.attributes,
            headers=self.data.headers,
            parameters=self.data.parameters,
            trace_id=self.metadata.trace_id,
            span_id=self.metadata.span_id,
            created_at=self.metadata.created_at,
            started_at=self.metadata.execution_start,
            completed_at=self.metadata.execution_end
        )
    
    # Context information
    
    def get_context_info(self) -> Dict[str, Any]:
        """Get comprehensive context information."""
        return {
            "context_id": self.context_id,
            "state": self.state.value,
            "type": self.metadata.context_type.value,
            "scope": self.metadata.scope.value,
            "created_at": self.metadata.created_at.isoformat(),
            "execution_start": self.metadata.execution_start.isoformat() if self.metadata.execution_start else None,
            "execution_end": self.metadata.execution_end.isoformat() if self.metadata.execution_end else None,
            "duration": self.metadata.execution_duration,
            "parent_context_id": self.metadata.parent_context_id,
            "root_context_id": self.metadata.root_context_id,
            "child_contexts": len(self.child_contexts),
            "trace_id": self.metadata.trace_id,
            "span_id": self.metadata.span_id,
            "user_id": self.data.user_id,
            "tags": self.metadata.tags,
            "labels": self.metadata.labels
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is None:
            self.complete(success=True)
        else:
            self.complete(success=False)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if exc_type is None:
            self.complete(success=True)
        else:
            self.complete(success=False)
    
    def __repr__(self) -> str:
        return (f"CommunicationContext("
                f"id={self.context_id[:8]}, "
                f"state={self.state.value}, "
                f"type={self.metadata.context_type.value}, "
                f"children={len(self.child_contexts)})")


# Context manager for managing communication contexts
class ContextManager:
    """Manager for communication contexts."""
    
    def __init__(self):
        self._contexts: Dict[str, CommunicationContext] = {}
        self._current_context: Optional[CommunicationContext] = None
        self._context_stack: List[CommunicationContext] = []
        self._lock = threading.RLock()
    
    def create_context(self, 
                      context_type: ContextType = ContextType.SYNCHRONOUS,
                      scope: ContextScope = ContextScope.REQUEST,
                      parent_context: Optional[CommunicationContext] = None) -> CommunicationContext:
        """Create new communication context."""
        context = CommunicationContext(context_type, scope, parent_context)
        
        with self._lock:
            self._contexts[context.context_id] = context
        
        return context
    
    def get_context(self, context_id: str) -> Optional[CommunicationContext]:
        """Get context by ID."""
        return self._contexts.get(context_id)
    
    def get_current_context(self) -> Optional[CommunicationContext]:
        """Get current active context."""
        return self._current_context
    
    def set_current_context(self, context: CommunicationContext):
        """Set current active context."""
        with self._lock:
            self._current_context = context
    
    def push_context(self, context: CommunicationContext):
        """Push context onto stack and make it current."""
        with self._lock:
            self._context_stack.append(context)
            self._current_context = context
    
    def pop_context(self) -> Optional[CommunicationContext]:
        """Pop context from stack."""
        with self._lock:
            if self._context_stack:
                context = self._context_stack.pop()
                self._current_context = self._context_stack[-1] if self._context_stack else None
                return context
        return None
    
    @asynccontextmanager
    async def context_scope(self, context: CommunicationContext):
        """Async context manager for scoped context execution."""
        self.push_context(context)
        try:
            async with context:
                yield context
        finally:
            self.pop_context()
    
    def cleanup_completed_contexts(self, max_age_seconds: int = 3600):
        """Clean up completed contexts older than specified age."""
        cutoff_time = datetime.now().timestamp() - max_age_seconds
        
        with self._lock:
            contexts_to_remove = []
            for context_id, context in self._contexts.items():
                if (context.is_completed and 
                    context.metadata.created_at.timestamp() < cutoff_time):
                    contexts_to_remove.append(context_id)
            
            for context_id in contexts_to_remove:
                del self._contexts[context_id]
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """Get context manager statistics."""
        with self._lock:
            total_contexts = len(self._contexts)
            active_contexts = sum(1 for ctx in self._contexts.values() if ctx.is_active)
            completed_contexts = sum(1 for ctx in self._contexts.values() if ctx.is_completed)
            
            return {
                "total_contexts": total_contexts,
                "active_contexts": active_contexts,
                "completed_contexts": completed_contexts,
                "current_context_id": self._current_context.context_id if self._current_context else None,
                "context_stack_depth": len(self._context_stack)
            }


# Global context manager instance
context_manager = ContextManager()


__all__ = [
    'ContextScope',
    'ContextType',
    'ContextState',
    'ContextMetadata',
    'ContextData',
    'CommunicationContext',
    'ContextManager',
    'context_manager'
]