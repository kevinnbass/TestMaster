"""
Async State Manager for TestMaster

Advanced state management system for async operations with
context isolation, hierarchical scoping, and telemetry integration.
"""

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager, contextmanager
import weakref

from .feature_flags import FeatureFlags
from .shared_state import get_shared_state
# Telemetry imports commented out for now - will be restored when telemetry is consolidated
# from ..telemetry import get_telemetry_collector, get_performance_monitor

T = TypeVar('T')

class StateScope(Enum):
    """State scope levels."""
    GLOBAL = "global"
    SESSION = "session"
    TASK = "task"
    CONTEXT = "context"

@dataclass
class AsyncContext:
    """Async execution context."""
    context_id: str
    parent_id: Optional[str] = None
    scope: StateScope = StateScope.CONTEXT
    created_at: datetime = field(default_factory=datetime.now)
    accessed_at: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    cleanup_callbacks: List[Callable] = field(default_factory=list)

@dataclass
class StateEntry:
    """State entry with metadata."""
    key: str
    value: Any
    scope: StateScope
    created_at: datetime
    accessed_at: datetime
    context_id: Optional[str] = None
    expiry: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class AsyncStateManager:
    """
    Advanced async state management system.
    
    Features:
    - Hierarchical context scoping (global -> session -> task -> context)
    - Context isolation for async operations
    - Automatic cleanup and garbage collection
    - State persistence and recovery
    - Comprehensive telemetry and monitoring
    """
    
    def __init__(self, cleanup_interval: float = 300.0):
        """
        Initialize async state manager.
        
        Args:
            cleanup_interval: Cleanup cycle interval in seconds
        """
        self.enabled = FeatureFlags.is_layer_enabled('layer2_monitoring', 'async_processing')
        
        if not self.enabled:
            return
        
        self.cleanup_interval = cleanup_interval
        
        # State storage by scope
        self.global_state: Dict[str, StateEntry] = {}
        self.session_states: Dict[str, Dict[str, StateEntry]] = {}
        self.task_states: Dict[str, Dict[str, StateEntry]] = {}
        self.context_states: Dict[str, Dict[str, StateEntry]] = {}
        
        # Context management
        self.active_contexts: Dict[str, AsyncContext] = {}
        self.context_hierarchy: Dict[str, List[str]] = {}  # parent -> children
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Current context tracking (thread-local and async-local)
        self.current_context_id: Optional[str] = None
        self._context_stack: List[str] = []
        
        # Cleanup management
        self.cleanup_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        self.is_cleanup_running = False
        
        # Integrations
        if FeatureFlags.is_layer_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_layer_enabled('layer3_orchestration', 'telemetry_system'):
            # Telemetry temporarily disabled - will be restored when telemetry is consolidated
            # self.telemetry = get_telemetry_collector()
            # self.performance_monitor = get_performance_monitor()
            self.telemetry = None
            self.performance_monitor = None
        else:
            self.telemetry = None
            self.performance_monitor = None
        
        # Statistics
        self.contexts_created = 0
        self.contexts_cleaned = 0
        self.state_operations = 0
        
        # Start cleanup if in async environment
        try:
            asyncio.get_event_loop()
            self._start_cleanup()
        except RuntimeError:
            # No event loop, will start cleanup when first context is created
            pass
        
        print("Async state manager initialized")
        print(f"   Cleanup interval: {self.cleanup_interval}s")
    

    
    def __bool__(self):
        """Return True to indicate manager is active."""
        return True

    def update_state(self, key: str, value: Any) -> None:
        """Update state with key-value pair."""
        async def _update():
            async with self.lock:
                self.state[key] = value
        
        # Run async method in sync context
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        if loop.is_running():
            # Already in async context
            asyncio.create_task(_update())
        else:
            # Sync context
            loop.run_until_complete(_update())

    def _start_cleanup(self):
        """Start background cleanup task."""
        if not self.enabled or self.is_cleanup_running:
            return
        
        async def cleanup_worker():
            self.is_cleanup_running = True
            
            while not self.shutdown_event.is_set():
                try:
                    # Cleanup expired contexts and states
                    self._cleanup_expired_contexts()
                    self._cleanup_expired_states()
                    
                    # Send telemetry
                    self._send_state_telemetry()
                    
                    # Wait for next cycle
                    await asyncio.sleep(self.cleanup_interval)
                    
                except Exception as e:
                    print(f"State cleanup error: {e}")
                    await asyncio.sleep(60)  # Wait longer on error
            
            self.is_cleanup_running = False
        
        try:
            loop = asyncio.get_event_loop()
            self.cleanup_task = loop.create_task(cleanup_worker())
        except RuntimeError:
            # No event loop available
            pass
    
    def create_context(self, parent_id: str = None, scope: StateScope = StateScope.CONTEXT,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Create a new async context.
        
        Args:
            parent_id: Parent context ID
            scope: Context scope level
            metadata: Additional metadata
            
        Returns:
            Context ID
        """
        if not self.enabled:
            return "disabled"
        
        context_id = str(uuid.uuid4())
        
        context = AsyncContext(
            context_id=context_id,
            parent_id=parent_id,
            scope=scope,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.active_contexts[context_id] = context
            self.contexts_created += 1
            
            # Update hierarchy
            if parent_id:
                if parent_id not in self.context_hierarchy:
                    self.context_hierarchy[parent_id] = []
                self.context_hierarchy[parent_id].append(context_id)
            
            # Initialize context state storage
            if scope == StateScope.CONTEXT:
                self.context_states[context_id] = {}
            elif scope == StateScope.TASK:
                self.task_states[context_id] = {}
            elif scope == StateScope.SESSION:
                self.session_states[context_id] = {}
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="async_context_created",
                component="async_state_manager",
                operation="create_context",
                metadata={
                    "context_id": context_id,
                    "parent_id": parent_id,
                    "scope": scope.value
                }
            )
        
        print(f"Created async context: {context_id} (scope: {scope.value})")
        return context_id
    
    @asynccontextmanager
    async def async_context(self, parent_id: str = None, scope: StateScope = StateScope.CONTEXT,
                           metadata: Dict[str, Any] = None):
        """
        Async context manager for automatic context lifecycle.
        
        Args:
            parent_id: Parent context ID
            scope: Context scope level
            metadata: Additional metadata
        """
        if not self.enabled:
            yield "disabled"
            return
        
        context_id = self.create_context(parent_id, scope, metadata)
        old_context = self.current_context_id
        
        try:
            # Enter context
            self.current_context_id = context_id
            self._context_stack.append(context_id)
            
            # Track context activation
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="async_context_entered",
                    component="async_state_manager",
                    operation="context_enter",
                    metadata={"context_id": context_id}
                )
            
            yield context_id
            
        finally:
            # Exit context
            self.current_context_id = old_context
            if self._context_stack and self._context_stack[-1] == context_id:
                self._context_stack.pop()
            
            # Run cleanup callbacks
            self._run_context_cleanup(context_id)
            
            # Deactivate context
            self.deactivate_context(context_id)
            
            # Track context exit
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="async_context_exited",
                    component="async_state_manager",
                    operation="context_exit",
                    metadata={"context_id": context_id}
                )
    
    @contextmanager
    def sync_context(self, parent_id: str = None, scope: StateScope = StateScope.CONTEXT,
                    metadata: Dict[str, Any] = None):
        """
        Synchronous context manager.
        
        Args:
            parent_id: Parent context ID
            scope: Context scope level
            metadata: Additional metadata
        """
        if not self.enabled:
            yield "disabled"
            return
        
        context_id = self.create_context(parent_id, scope, metadata)
        old_context = self.current_context_id
        
        try:
            self.current_context_id = context_id
            self._context_stack.append(context_id)
            yield context_id
        finally:
            self.current_context_id = old_context
            if self._context_stack and self._context_stack[-1] == context_id:
                self._context_stack.pop()
            
            self._run_context_cleanup(context_id)
            self.deactivate_context(context_id)
    
    def set_state(self, key: str, value: Any, scope: StateScope = None,
                 context_id: str = None, expiry: datetime = None,
                 metadata: Dict[str, Any] = None) -> bool:
        """
        Set state value.
        
        Args:
            key: State key
            value: State value
            scope: State scope (defaults to current context scope)
            context_id: Target context ID
            expiry: Expiration time
            metadata: Additional metadata
            
        Returns:
            True if state was set successfully
        """
        if not self.enabled:
            return False
        
        # Determine scope and context
        if scope is None:
            scope = StateScope.CONTEXT
        
        if context_id is None:
            context_id = self.current_context_id
        
        # Create state entry
        entry = StateEntry(
            key=key,
            value=value,
            scope=scope,
            created_at=datetime.now(),
            accessed_at=datetime.now(),
            context_id=context_id,
            expiry=expiry,
            metadata=metadata or {}
        )
        
        with self.lock:
            # Store in appropriate scope
            if scope == StateScope.GLOBAL:
                self.global_state[key] = entry
            elif scope == StateScope.SESSION:
                session_id = context_id or "default"
                if session_id not in self.session_states:
                    self.session_states[session_id] = {}
                self.session_states[session_id][key] = entry
            elif scope == StateScope.TASK:
                task_id = context_id or "default"
                if task_id not in self.task_states:
                    self.task_states[task_id] = {}
                self.task_states[task_id][key] = entry
            elif scope == StateScope.CONTEXT:
                ctx_id = context_id or "default"
                if ctx_id not in self.context_states:
                    self.context_states[ctx_id] = {}
                self.context_states[ctx_id][key] = entry
            
            self.state_operations += 1
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment("async_state_operations")
        
        return True
    
    def get_state(self, key: str, scope: StateScope = None,
                 context_id: str = None, default: Any = None) -> Any:
        """
        Get state value with hierarchical lookup.
        
        Args:
            key: State key
            scope: Preferred scope (searches all if None)
            context_id: Context ID for scoped lookups
            default: Default value if not found
            
        Returns:
            State value or default
        """
        if not self.enabled:
            return default
        
        if context_id is None:
            context_id = self.current_context_id
        
        with self.lock:
            # If specific scope requested, check only that scope
            if scope is not None:
                entry = self._get_state_from_scope(key, scope, context_id)
                if entry and not self._is_expired(entry):
                    entry.accessed_at = datetime.now()
                    self.state_operations += 1
                    return entry.value
                return default
            
            # Hierarchical lookup: context -> task -> session -> global
            for lookup_scope in [StateScope.CONTEXT, StateScope.TASK, 
                               StateScope.SESSION, StateScope.GLOBAL]:
                entry = self._get_state_from_scope(key, lookup_scope, context_id)
                if entry and not self._is_expired(entry):
                    entry.accessed_at = datetime.now()
                    self.state_operations += 1
                    return entry.value
        
        return default
    
    def _get_state_from_scope(self, key: str, scope: StateScope,
                            context_id: str = None) -> Optional[StateEntry]:
        """Get state entry from specific scope."""
        if scope == StateScope.GLOBAL:
            return self.global_state.get(key)
        elif scope == StateScope.SESSION:
            session_id = context_id or "default"
            session_states = self.session_states.get(session_id, {})
            return session_states.get(key)
        elif scope == StateScope.TASK:
            task_id = context_id or "default"
            task_states = self.task_states.get(task_id, {})
            return task_states.get(key)
        elif scope == StateScope.CONTEXT:
            ctx_id = context_id or "default"
            context_states = self.context_states.get(ctx_id, {})
            return context_states.get(key)
        
        return None
    
    def _is_expired(self, entry: StateEntry) -> bool:
        """Check if state entry is expired."""
        return entry.expiry is not None and datetime.now() > entry.expiry
    
    def delete_state(self, key: str, scope: StateScope = None,
                    context_id: str = None) -> bool:
        """Delete state value."""
        if not self.enabled:
            return False
        
        if context_id is None:
            context_id = self.current_context_id
        
        with self.lock:
            if scope is None:
                # Delete from all scopes
                deleted = False
                for lookup_scope in StateScope:
                    if self._delete_from_scope(key, lookup_scope, context_id):
                        deleted = True
                return deleted
            else:
                return self._delete_from_scope(key, scope, context_id)
    
    def _delete_from_scope(self, key: str, scope: StateScope,
                          context_id: str = None) -> bool:
        """Delete state from specific scope."""
        if scope == StateScope.GLOBAL:
            return self.global_state.pop(key, None) is not None
        elif scope == StateScope.SESSION:
            session_id = context_id or "default"
            session_states = self.session_states.get(session_id, {})
            return session_states.pop(key, None) is not None
        elif scope == StateScope.TASK:
            task_id = context_id or "default"
            task_states = self.task_states.get(task_id, {})
            return task_states.pop(key, None) is not None
        elif scope == StateScope.CONTEXT:
            ctx_id = context_id or "default"
            context_states = self.context_states.get(ctx_id, {})
            return context_states.pop(key, None) is not None
        
        return False
    
    def add_cleanup_callback(self, context_id: str, callback: Callable):
        """Add cleanup callback for context."""
        if not self.enabled:
            return
        
        with self.lock:
            if context_id in self.active_contexts:
                self.active_contexts[context_id].cleanup_callbacks.append(callback)
    
    def _run_context_cleanup(self, context_id: str):
        """Run cleanup callbacks for context."""
        with self.lock:
            if context_id in self.active_contexts:
                context = self.active_contexts[context_id]
                for callback in context.cleanup_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        print(f"Cleanup callback error: {e}")
    
    def deactivate_context(self, context_id: str):
        """Deactivate a context."""
        if not self.enabled:
            return
        
        with self.lock:
            if context_id in self.active_contexts:
                self.active_contexts[context_id].is_active = False
    
    def get_context_info(self, context_id: str) -> Optional[AsyncContext]:
        """Get context information."""
        if not self.enabled:
            return None
        
        with self.lock:
            return self.active_contexts.get(context_id)
    
    def get_active_contexts(self) -> List[AsyncContext]:
        """Get list of active contexts."""
        if not self.enabled:
            return []
        
        with self.lock:
            return [ctx for ctx in self.active_contexts.values() if ctx.is_active]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get state management summary."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            total_states = (
                len(self.global_state) +
                sum(len(states) for states in self.session_states.values()) +
                sum(len(states) for states in self.task_states.values()) +
                sum(len(states) for states in self.context_states.values())
            )
            
            active_contexts = sum(1 for ctx in self.active_contexts.values() if ctx.is_active)
            
            return {
                "enabled": True,
                "is_cleanup_running": self.is_cleanup_running,
                "contexts_created": self.contexts_created,
                "contexts_cleaned": self.contexts_cleaned,
                "active_contexts": active_contexts,
                "total_contexts": len(self.active_contexts),
                "total_states": total_states,
                "state_operations": self.state_operations,
                "current_context": self.current_context_id,
                "context_stack_depth": len(self._context_stack),
                "state_breakdown": {
                    "global": len(self.global_state),
                    "sessions": len(self.session_states),
                    "tasks": len(self.task_states),
                    "contexts": len(self.context_states)
                }
            }
    
    def _cleanup_expired_contexts(self):
        """Clean up expired and inactive contexts."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self.lock:
            expired_contexts = []
            
            for context_id, context in self.active_contexts.items():
                if (not context.is_active and 
                    context.accessed_at < cutoff_time):
                    expired_contexts.append(context_id)
            
            for context_id in expired_contexts:
                # Clean up context states
                self.context_states.pop(context_id, None)
                self.task_states.pop(context_id, None)
                self.session_states.pop(context_id, None)
                
                # Remove from hierarchy
                self.context_hierarchy.pop(context_id, None)
                for children in self.context_hierarchy.values():
                    if context_id in children:
                        children.remove(context_id)
                
                # Remove context
                self.active_contexts.pop(context_id)
                self.contexts_cleaned += 1
    
    def _cleanup_expired_states(self):
        """Clean up expired state entries."""
        current_time = datetime.now()
        
        with self.lock:
            # Clean global states
            expired_keys = [
                key for key, entry in self.global_state.items()
                if self._is_expired(entry)
            ]
            for key in expired_keys:
                self.global_state.pop(key)
            
            # Clean session states
            for session_states in self.session_states.values():
                expired_keys = [
                    key for key, entry in session_states.items()
                    if self._is_expired(entry)
                ]
                for key in expired_keys:
                    session_states.pop(key)
            
            # Clean task states
            for task_states in self.task_states.values():
                expired_keys = [
                    key for key, entry in task_states.items()
                    if self._is_expired(entry)
                ]
                for key in expired_keys:
                    task_states.pop(key)
            
            # Clean context states
            for context_states in self.context_states.values():
                expired_keys = [
                    key for key, entry in context_states.items()
                    if self._is_expired(entry)
                ]
                for key in expired_keys:
                    context_states.pop(key)
    
    def _send_state_telemetry(self):
        """Send state management telemetry."""
        if not self.telemetry:
            return
        
        summary = self.get_state_summary()
        
        self.telemetry.record_event(
            event_type="async_state_summary",
            component="async_state_manager",
            operation="monitoring_cycle",
            metadata={
                "active_contexts": summary.get("active_contexts", 0),
                "total_states": summary.get("total_states", 0),
                "state_operations": summary.get("state_operations", 0),
                "contexts_created": summary.get("contexts_created", 0)
            }
        )
    
    def cleanup(self):
        """Clean up state manager."""
        if not self.enabled:
            return
        
        print("Cleaning up async state manager...")
        
        # Stop cleanup task
        if self.cleanup_task and not self.cleanup_task.done():
            try:
                loop = asyncio.get_event_loop()
                loop.create_task(self.shutdown_event.set())
                self.cleanup_task.cancel()
            except RuntimeError:
                pass
        
        # Run final cleanup
        self._cleanup_expired_contexts()
        self._cleanup_expired_states()
        
        with self.lock:
            contexts_cleaned = len(self.active_contexts)
            states_cleaned = (
                len(self.global_state) +
                sum(len(states) for states in self.session_states.values()) +
                sum(len(states) for states in self.task_states.values()) +
                sum(len(states) for states in self.context_states.values())
            )
            
            # Clear all state
            self.active_contexts.clear()
            self.context_hierarchy.clear()
            self.global_state.clear()
            self.session_states.clear()
            self.task_states.clear()
            self.context_states.clear()
            self._context_stack.clear()
            self.current_context_id = None
        
        print(f"State manager cleanup completed - cleaned {contexts_cleaned} contexts, {states_cleaned} states")

# Global instance
_async_state_manager: Optional[AsyncStateManager] = None

def get_async_state_manager() -> AsyncStateManager:
    """Get the global async state manager instance."""
    global _async_state_manager
    if _async_state_manager is None:
        _async_state_manager = AsyncStateManager()
    return _async_state_manager

# Convenience function
def async_context(parent_id: str = None, scope: StateScope = StateScope.CONTEXT,
                 metadata: Dict[str, Any] = None):
    """
    Create an async context.
    
    Args:
        parent_id: Parent context ID
        scope: Context scope level
        metadata: Additional metadata
    
    Returns:
        Async context manager
    """
    manager = get_async_state_manager()
    return manager.async_context(parent_id, scope, metadata)