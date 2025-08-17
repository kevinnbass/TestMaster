"""
Comprehensive Tracking Manager for TestMaster

Inspired by Agency-Swarm's tracking system, this provides comprehensive
execution tracking for all TestMaster operations including test generation,
verification, monitoring, and orchestration activities.

Features:
- Chain-based execution tracking
- Tool and operation monitoring
- Error tracking and analysis
- Performance metrics collection
- Toggleable via feature flags
"""

import json
import sqlite3
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

from .feature_flags import FeatureFlags
from .shared_state import get_shared_state
from .monitoring_decorators import monitor_performance


class EventType(Enum):
    """Types of events that can be tracked."""
    CHAIN_START = "chain_start"
    CHAIN_END = "chain_end"
    CHAIN_ERROR = "chain_error"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    TOOL_ERROR = "tool_error"
    GENERATION_START = "generation_start"
    GENERATION_END = "generation_end"
    VERIFICATION_START = "verification_start"
    VERIFICATION_END = "verification_end"
    MONITORING_EVENT = "monitoring_event"
    HANDOFF_START = "handoff_start"
    HANDOFF_END = "handoff_end"
    CONFIG_CHANGE = "config_change"


@dataclass
class TrackingEvent:
    """Represents a single tracking event."""
    event_id: str
    run_id: str
    parent_run_id: Optional[str]
    event_type: EventType
    timestamp: datetime
    component: str
    operation: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    memory_usage: Optional[float] = None
    success: bool = True


@dataclass
class ExecutionChain:
    """Represents an execution chain with multiple operations."""
    chain_id: str
    parent_chain_id: Optional[str]
    chain_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    events: List[TrackingEvent] = field(default_factory=list)
    child_chains: List[str] = field(default_factory=list)
    status: str = "running"  # running, completed, failed
    total_duration_ms: Optional[float] = None
    error_count: int = 0
    success_count: int = 0


class TrackingManager:
    """
    Comprehensive tracking manager for TestMaster operations.
    
    Provides chain-based execution tracking, performance monitoring,
    and error analysis across all TestMaster components.
    """
    
    def __init__(self, db_path: str = "testmaster_tracking.db"):
        """
        Initialize tracking manager.
        
        Args:
            db_path: Path to SQLite database for storing tracking data
        """
        self.enabled = FeatureFlags.is_enabled('layer2_monitoring', 'tracking_manager')
        
        if not self.enabled:
            return
            
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self._closed = False
        
        # Configuration
        config = FeatureFlags.get_config('layer2_monitoring', 'tracking_manager')
        self.max_chain_depth = config.get('chain_depth', 5)
        self.auto_cleanup_days = config.get('auto_cleanup_days', 30)
        self.track_memory = config.get('track_memory', True)
        self.track_tokens = config.get('track_tokens', True)
        
        # Runtime state
        self.active_chains: Dict[str, ExecutionChain] = {}
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize database
        self._initialize_database()
        
        # Shared state integration
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        print("Comprehensive tracking manager initialized")
        print(f"   Database: {self.db_path}")
        print(f"   Max chain depth: {self.max_chain_depth}")
        print(f"   Memory tracking: {'enabled' if self.track_memory else 'disabled'}")
    
    def _initialize_database(self):
        """Initialize SQLite database for tracking storage."""
        with self.lock:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Create events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS tracking_events (
                        event_id TEXT PRIMARY KEY,
                        run_id TEXT,
                        parent_run_id TEXT,
                        event_type TEXT,
                        timestamp TEXT,
                        component TEXT,
                        operation TEXT,
                        inputs TEXT,
                        outputs TEXT,
                        metadata TEXT,
                        error TEXT,
                        duration_ms REAL,
                        tokens_used INTEGER,
                        memory_usage REAL,
                        success BOOLEAN
                    )
                """)
                
                # Create chains table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS execution_chains (
                        chain_id TEXT PRIMARY KEY,
                        parent_chain_id TEXT,
                        chain_name TEXT,
                        start_time TEXT,
                        end_time TEXT,
                        status TEXT,
                        total_duration_ms REAL,
                        error_count INTEGER,
                        success_count INTEGER,
                        metadata TEXT
                    )
                """)
                
                # Create indexes for better query performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_run_id ON tracking_events(run_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON tracking_events(timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_component ON tracking_events(component)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_chains_start_time ON execution_chains(start_time)")
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                print(f"Warning: Failed to initialize tracking database: {e}")
    
    @monitor_performance(name="start_chain")
    def start_chain(self, chain_name: str, inputs: Dict[str, Any] = None, 
                   parent_chain_id: str = None) -> str:
        """
        Start tracking an execution chain.
        
        Args:
            chain_name: Name of the chain operation
            inputs: Input parameters for the chain
            parent_chain_id: ID of parent chain if this is a sub-chain
            
        Returns:
            Chain ID for tracking
        """
        if not self.enabled:
            return f"disabled_{uuid4()}"
        
        chain_id = f"chain_{uuid4()}"
        start_time = datetime.now()
        
        # Check chain depth
        depth = self._calculate_chain_depth(parent_chain_id)
        if depth >= self.max_chain_depth:
            print(f"Warning: Maximum chain depth ({self.max_chain_depth}) exceeded")
            return chain_id
        
        chain = ExecutionChain(
            chain_id=chain_id,
            parent_chain_id=parent_chain_id,
            chain_name=chain_name,
            start_time=start_time
        )
        
        with self.lock:
            self.active_chains[chain_id] = chain
            
            # Add to parent chain's children
            if parent_chain_id and parent_chain_id in self.active_chains:
                self.active_chains[parent_chain_id].child_chains.append(chain_id)
        
        # Record chain start event
        self._record_event(
            run_id=chain_id,
            parent_run_id=parent_chain_id,
            event_type=EventType.CHAIN_START,
            component="tracking_manager",
            operation=chain_name,
            inputs=inputs or {},
            metadata={
                "chain_depth": depth,
                "chain_name": chain_name
            }
        )
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment('chains_started')
            self.shared_state.set(f'active_chain_{chain_id}', {
                'name': chain_name,
                'start_time': start_time.isoformat(),
                'depth': depth
            }, ttl=3600)
        
        return chain_id
    
    @monitor_performance(name="end_chain")
    def end_chain(self, chain_id: str, outputs: Dict[str, Any] = None, 
                 success: bool = True, error: str = None):
        """
        End tracking for an execution chain.
        
        Args:
            chain_id: Chain ID to end
            outputs: Output results from the chain
            success: Whether the chain completed successfully
            error: Error message if chain failed
        """
        if not self.enabled or chain_id.startswith("disabled_"):
            return
        
        end_time = datetime.now()
        
        with self.lock:
            if chain_id not in self.active_chains:
                print(f"Warning: Chain {chain_id} not found in active chains")
                return
            
            chain = self.active_chains[chain_id]
            chain.end_time = end_time
            chain.status = "completed" if success else "failed"
            chain.total_duration_ms = (end_time - chain.start_time).total_seconds() * 1000
            
            # Record chain end event
            self._record_event(
                run_id=chain_id,
                parent_run_id=chain.parent_chain_id,
                event_type=EventType.CHAIN_END,
                component="tracking_manager",
                operation=chain.chain_name,
                outputs=outputs or {},
                error=error,
                success=success,
                duration_ms=chain.total_duration_ms
            )
            
            # Store chain to database
            self._store_chain_to_db(chain)
            
            # Remove from active chains
            del self.active_chains[chain_id]
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment('chains_completed')
            self.shared_state.increment('chains_failed' if not success else 'chains_succeeded')
            self.shared_state.delete(f'active_chain_{chain_id}')
    
    @monitor_performance(name="track_operation")
    def track_operation(self, run_id: str, component: str, operation: str,
                       inputs: Dict[str, Any] = None, outputs: Dict[str, Any] = None,
                       parent_run_id: str = None, success: bool = True,
                       error: str = None, duration_ms: float = None,
                       tokens_used: int = None, memory_usage: float = None):
        """
        Track a single operation or tool execution.
        
        Args:
            run_id: Unique ID for this operation run
            component: Component performing the operation
            operation: Name of the operation
            inputs: Input parameters
            outputs: Output results
            parent_run_id: Parent chain or run ID
            success: Whether operation succeeded
            error: Error message if failed
            duration_ms: Operation duration in milliseconds
            tokens_used: Number of tokens used (for AI operations)
            memory_usage: Memory usage in MB
        """
        if not self.enabled:
            return
        
        # Determine event type based on operation
        if operation.startswith('generate'):
            event_type = EventType.GENERATION_START if inputs else EventType.GENERATION_END
        elif operation.startswith('verify'):
            event_type = EventType.VERIFICATION_START if inputs else EventType.VERIFICATION_END
        elif operation.startswith('handoff'):
            event_type = EventType.HANDOFF_START if inputs else EventType.HANDOFF_END
        elif operation.startswith('monitor'):
            event_type = EventType.MONITORING_EVENT
        else:
            event_type = EventType.TOOL_START if inputs else EventType.TOOL_END
        
        self._record_event(
            run_id=run_id,
            parent_run_id=parent_run_id,
            event_type=event_type,
            component=component,
            operation=operation,
            inputs=inputs or {},
            outputs=outputs or {},
            success=success,
            error=error,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            memory_usage=memory_usage
        )
        
        # Update chain statistics if this belongs to a chain
        if parent_run_id and parent_run_id in self.active_chains:
            with self.lock:
                chain = self.active_chains[parent_run_id]
                if success:
                    chain.success_count += 1
                else:
                    chain.error_count += 1
    
    def _record_event(self, run_id: str, parent_run_id: Optional[str],
                     event_type: EventType, component: str, operation: str,
                     inputs: Dict[str, Any] = None, outputs: Dict[str, Any] = None,
                     metadata: Dict[str, Any] = None, success: bool = True,
                     error: str = None, duration_ms: float = None,
                     tokens_used: int = None, memory_usage: float = None):
        """Record a tracking event to database and memory."""
        if not self.enabled:
            return
        
        event = TrackingEvent(
            event_id=f"event_{uuid4()}",
            run_id=run_id,
            parent_run_id=parent_run_id,
            event_type=event_type,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            inputs=inputs or {},
            outputs=outputs or {},
            metadata=metadata or {},
            success=success,
            error=error,
            duration_ms=duration_ms,
            tokens_used=tokens_used,
            memory_usage=memory_usage
        )
        
        # Store to database
        self._store_event_to_db(event)
        
        # Add to active chain if applicable
        if parent_run_id and parent_run_id in self.active_chains:
            with self.lock:
                self.active_chains[parent_run_id].events.append(event)
    
    def _store_event_to_db(self, event: TrackingEvent):
        """Store tracking event to SQLite database."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO tracking_events (
                        event_id, run_id, parent_run_id, event_type, timestamp,
                        component, operation, inputs, outputs, metadata,
                        error, duration_ms, tokens_used, memory_usage, success
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.run_id,
                    event.parent_run_id,
                    event.event_type.value,
                    event.timestamp.isoformat(),
                    event.component,
                    event.operation,
                    json.dumps(event.inputs),
                    json.dumps(event.outputs),
                    json.dumps(event.metadata),
                    event.error,
                    event.duration_ms,
                    event.tokens_used,
                    event.memory_usage,
                    event.success
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"Warning: Failed to store tracking event: {e}")
    
    def _store_chain_to_db(self, chain: ExecutionChain):
        """Store execution chain to SQLite database."""
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO execution_chains (
                        chain_id, parent_chain_id, chain_name, start_time,
                        end_time, status, total_duration_ms, error_count,
                        success_count, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    chain.chain_id,
                    chain.parent_chain_id,
                    chain.chain_name,
                    chain.start_time.isoformat(),
                    chain.end_time.isoformat() if chain.end_time else None,
                    chain.status,
                    chain.total_duration_ms,
                    chain.error_count,
                    chain.success_count,
                    json.dumps({
                        'child_chains': chain.child_chains,
                        'event_count': len(chain.events)
                    })
                ))
                
                conn.commit()
                conn.close()
                
        except Exception as e:
            print(f"Warning: Failed to store chain: {e}")
    
    def _calculate_chain_depth(self, parent_chain_id: str) -> int:
        """Calculate the depth of a chain in the execution hierarchy."""
        if not parent_chain_id:
            return 0
        
        depth = 0
        current_id = parent_chain_id
        
        while current_id and current_id in self.active_chains:
            depth += 1
            current_id = self.active_chains[current_id].parent_chain_id
            
            # Prevent infinite loops
            if depth > self.max_chain_depth:
                break
        
        return depth
    
    def get_active_chains(self) -> List[Dict[str, Any]]:
        """Get information about currently active execution chains."""
        if not self.enabled:
            return []
        
        chains = []
        with self.lock:
            for chain in self.active_chains.values():
                chains.append({
                    'chain_id': chain.chain_id,
                    'chain_name': chain.chain_name,
                    'start_time': chain.start_time.isoformat(),
                    'duration_seconds': (datetime.now() - chain.start_time).total_seconds(),
                    'event_count': len(chain.events),
                    'success_count': chain.success_count,
                    'error_count': chain.error_count,
                    'child_chains': len(chain.child_chains),
                    'depth': self._calculate_chain_depth(chain.parent_chain_id)
                })
        
        return chains
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics."""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Get event statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_events,
                        COUNT(CASE WHEN success = 1 THEN 1 END) as successful_events,
                        COUNT(CASE WHEN success = 0 THEN 1 END) as failed_events,
                        AVG(duration_ms) as avg_duration_ms,
                        SUM(tokens_used) as total_tokens,
                        AVG(memory_usage) as avg_memory_mb
                    FROM tracking_events 
                    WHERE datetime(timestamp) >= datetime('now', '-24 hours')
                """)
                
                event_stats = cursor.fetchone()
                
                # Get chain statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chains,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_chains,
                        COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_chains,
                        AVG(total_duration_ms) as avg_chain_duration_ms,
                        AVG(success_count) as avg_success_count,
                        AVG(error_count) as avg_error_count
                    FROM execution_chains 
                    WHERE datetime(start_time) >= datetime('now', '-24 hours')
                """)
                
                chain_stats = cursor.fetchone()
                
                # Get component breakdown
                cursor.execute("""
                    SELECT component, COUNT(*) as event_count
                    FROM tracking_events 
                    WHERE datetime(timestamp) >= datetime('now', '-24 hours')
                    GROUP BY component
                    ORDER BY event_count DESC
                """)
                
                component_stats = dict(cursor.fetchall())
                
                conn.close()
                
                return {
                    'enabled': True,
                    'active_chains': len(self.active_chains),
                    'max_chain_depth': self.max_chain_depth,
                    'db_path': str(self.db_path),
                    'last_24h': {
                        'events': {
                            'total': event_stats[0] or 0,
                            'successful': event_stats[1] or 0,
                            'failed': event_stats[2] or 0,
                            'success_rate': (event_stats[1] or 0) / max(event_stats[0] or 1, 1) * 100,
                            'avg_duration_ms': event_stats[3] or 0,
                            'total_tokens': event_stats[4] or 0,
                            'avg_memory_mb': event_stats[5] or 0
                        },
                        'chains': {
                            'total': chain_stats[0] or 0,
                            'completed': chain_stats[1] or 0,
                            'failed': chain_stats[2] or 0,
                            'success_rate': (chain_stats[1] or 0) / max(chain_stats[0] or 1, 1) * 100,
                            'avg_duration_ms': chain_stats[3] or 0,
                            'avg_success_count': chain_stats[4] or 0,
                            'avg_error_count': chain_stats[5] or 0
                        },
                        'component_breakdown': component_stats
                    }
                }
                
        except Exception as e:
            print(f"Error getting tracking statistics: {e}")
            return {
                'enabled': True,
                'error': str(e),
                'active_chains': len(self.active_chains) if hasattr(self, 'active_chains') else 0
            }
    
    def cleanup_old_data(self, days: int = None):
        """Clean up tracking data older than specified days."""
        if not self.enabled:
            return
        
        days = days or self.auto_cleanup_days
        
        try:
            with self.lock:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Delete old events
                cursor.execute("""
                    DELETE FROM tracking_events 
                    WHERE datetime(timestamp) < datetime('now', '-{} days')
                """.format(days))
                
                events_deleted = cursor.rowcount
                
                # Delete old chains
                cursor.execute("""
                    DELETE FROM execution_chains 
                    WHERE datetime(start_time) < datetime('now', '-{} days')
                """.format(days))
                
                chains_deleted = cursor.rowcount
                
                conn.commit()
                conn.close()
                
                print(f"Cleaned up {events_deleted} old events and {chains_deleted} old chains")
                
        except Exception as e:
            print(f"Error cleaning up tracking data: {e}")


# Global tracking manager instance
_tracking_manager: Optional[TrackingManager] = None


def get_tracking_manager() -> TrackingManager:
    """Get the global tracking manager instance."""
    global _tracking_manager
    if _tracking_manager is None:
        _tracking_manager = TrackingManager()
    return _tracking_manager


def track_operation(component: str, operation: str):
    """
    Decorator for tracking operations with automatic timing and error handling.
    
    Args:
        component: Component name (e.g., 'test_generator', 'verifier')
        operation: Operation name (e.g., 'generate_test', 'verify_quality')
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracking_manager = get_tracking_manager()
            
            if not tracking_manager.enabled:
                return func(*args, **kwargs)
            
            run_id = f"op_{uuid4()}"
            start_time = time.time()
            
            # Extract parent chain ID from kwargs if available
            parent_run_id = kwargs.pop('_tracking_parent_id', None)
            
            # Prepare inputs (avoid serializing complex objects)
            inputs = {
                'args_count': len(args),
                'kwargs_keys': list(kwargs.keys()),
                'function': func.__name__
            }
            
            try:
                # Track operation start
                tracking_manager.track_operation(
                    run_id=run_id,
                    component=component,
                    operation=f"{operation}_start",
                    inputs=inputs,
                    parent_run_id=parent_run_id
                )
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Track operation end
                outputs = {
                    'result_type': type(result).__name__,
                    'success': True
                }
                
                tracking_manager.track_operation(
                    run_id=run_id,
                    component=component,
                    operation=f"{operation}_end",
                    outputs=outputs,
                    parent_run_id=parent_run_id,
                    success=True,
                    duration_ms=duration_ms
                )
                
                return result
                
            except Exception as e:
                # Calculate duration
                duration_ms = (time.time() - start_time) * 1000
                
                # Track operation error
                tracking_manager.track_operation(
                    run_id=run_id,
                    component=component,
                    operation=f"{operation}_error",
                    parent_run_id=parent_run_id,
                    success=False,
                    error=str(e),
                    duration_ms=duration_ms
                )
                
                raise
        
        return wrapper
    return decorator