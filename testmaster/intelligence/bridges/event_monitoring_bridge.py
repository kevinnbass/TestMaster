"""
Event Monitoring Bridge - Agent 12

This bridge creates a unified event bus system that connects all components
in the TestMaster ecosystem, providing real-time event monitoring, persistence,
and intelligent event correlation with consensus-driven event analysis.

Key Features:
- Unified event bus with multiple publishers and subscribers
- Event persistence with SQLite storage and compression
- Real-time event stream processing and filtering
- Event correlation and pattern detection
- Consensus-driven event analysis and alerting
- Cross-system event bridging and translation
- Performance monitoring and event analytics
"""

import asyncio
import json
import sqlite3
import threading
import time
import gzip
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Union, Set
from enum import Enum
from collections import defaultdict, deque
import uuid
from concurrent.futures import ThreadPoolExecutor
import queue
import statistics

from ..consensus import AgentCoordinator, AgentVote
from ..consensus.agent_coordination import AgentRole
from ...core.shared_state import SharedState
from ...core.feature_flags import FeatureFlags


class EventType(Enum):
    """Event type classifications."""
    SYSTEM = "system"                    # System-level events
    WORKFLOW = "workflow"                # Workflow execution events
    AGENT = "agent"                      # Agent activity events
    SECURITY = "security"                # Security-related events
    PERFORMANCE = "performance"          # Performance monitoring events
    ERROR = "error"                      # Error and exception events
    USER = "user"                        # User interaction events
    INTEGRATION = "integration"          # Cross-system integration events
    CONSENSUS = "consensus"              # Consensus mechanism events
    OPTIMIZATION = "optimization"        # Optimization events


class EventSeverity(Enum):
    """Event severity levels."""
    CRITICAL = 0    # System critical events
    HIGH = 1        # High priority events
    MEDIUM = 2      # Medium priority events
    LOW = 3         # Low priority events
    INFO = 4        # Informational events
    DEBUG = 5       # Debug events


class EventStatus(Enum):
    """Event processing status."""
    RECEIVED = "received"
    PROCESSING = "processing"
    PROCESSED = "processed"
    CORRELATED = "correlated"
    ALERTED = "alerted"
    ARCHIVED = "archived"
    FAILED = "failed"


@dataclass
class TestMasterEvent:
    """Universal event structure for TestMaster."""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    source_system: str
    source_component: str
    timestamp: datetime
    title: str
    description: str
    data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None
    status: EventStatus = EventStatus.RECEIVED
    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    expiry_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary."""
        result = asdict(self)
        result['event_type'] = self.event_type.value
        result['severity'] = self.severity.value
        result['status'] = self.status.value
        result['timestamp'] = self.timestamp.isoformat()
        if self.expiry_time:
            result['expiry_time'] = self.expiry_time.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestMasterEvent':
        """Create event from dictionary."""
        data['event_type'] = EventType(data['event_type'])
        data['severity'] = EventSeverity(data['severity'])
        data['status'] = EventStatus(data['status'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('expiry_time'):
            data['expiry_time'] = datetime.fromisoformat(data['expiry_time'])
        return cls(**data)


@dataclass
class EventSubscription:
    """Event subscription configuration."""
    subscription_id: str
    subscriber_id: str
    event_types: Set[EventType]
    severity_filter: Set[EventSeverity]
    source_filter: Optional[Set[str]] = None
    tag_filter: Optional[Set[str]] = None
    callback: Optional[Callable[[TestMasterEvent], None]] = None
    async_callback: Optional[Callable[[TestMasterEvent], Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    active: bool = True


@dataclass
class EventCorrelation:
    """Event correlation result."""
    correlation_id: str
    related_events: List[str]
    pattern_type: str
    confidence: float
    time_window: timedelta
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class EventBus:
    """Unified event bus for TestMaster."""
    
    def __init__(self, max_events_memory: int = 10000):
        self.max_events_memory = max_events_memory
        self.events: deque = deque(maxlen=max_events_memory)
        self.subscriptions: Dict[str, EventSubscription] = {}
        self.publishers: Set[str] = set()
        self.event_queue = queue.Queue()
        self.lock = threading.RLock()
        
        # Performance metrics
        self.events_published = 0
        self.events_processed = 0
        self.subscription_triggers = 0
        self.processing_errors = 0
        
        # Event processing thread
        self.processing_thread = None
        self.running = False
        
        print("Event Bus initialized")
        print(f"   Max events in memory: {max_events_memory}")
    
    def start_processing(self):
        """Start event processing thread."""
        if self.processing_thread and self.processing_thread.is_alive():
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop event processing."""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5)
    
    def _process_events(self):
        """Process events from queue."""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1)
                if event:
                    self._distribute_event(event)
                    with self.lock:
                        self.events_processed += 1
                        event.status = EventStatus.PROCESSED
            except queue.Empty:
                continue
            except Exception as e:
                with self.lock:
                    self.processing_errors += 1
                print(f"Event processing error: {e}")
    
    def publish_event(self, event: TestMasterEvent) -> str:
        """Publish event to the bus."""
        event.status = EventStatus.RECEIVED
        
        with self.lock:
            self.events.append(event)
            self.events_published += 1
        
        # Queue for processing
        self.event_queue.put(event)
        
        return event.event_id
    
    def _distribute_event(self, event: TestMasterEvent):
        """Distribute event to subscribers."""
        event.status = EventStatus.PROCESSING
        
        for subscription in self.subscriptions.values():
            if not subscription.active:
                continue
            
            if self._matches_subscription(event, subscription):
                try:
                    # Update subscription metrics
                    subscription.trigger_count += 1
                    subscription.last_triggered = datetime.now()
                    
                    # Call callback
                    if subscription.callback:
                        subscription.callback(event)
                    
                    # Handle async callback
                    if subscription.async_callback:
                        asyncio.run(subscription.async_callback(event))
                    
                    with self.lock:
                        self.subscription_triggers += 1
                        
                except Exception as e:
                    print(f"Subscription callback error {subscription.subscription_id}: {e}")
    
    def _matches_subscription(self, event: TestMasterEvent, subscription: EventSubscription) -> bool:
        """Check if event matches subscription criteria."""
        # Check event type
        if event.event_type not in subscription.event_types:
            return False
        
        # Check severity
        if event.severity not in subscription.severity_filter:
            return False
        
        # Check source filter
        if subscription.source_filter:
            if event.source_system not in subscription.source_filter:
                return False
        
        # Check tag filter
        if subscription.tag_filter:
            if not subscription.tag_filter.intersection(set(event.tags)):
                return False
        
        return True
    
    def subscribe(
        self,
        subscriber_id: str,
        event_types: List[EventType],
        callback: Optional[Callable[[TestMasterEvent], None]] = None,
        severity_filter: Optional[List[EventSeverity]] = None,
        source_filter: Optional[List[str]] = None,
        tag_filter: Optional[List[str]] = None,
        async_callback: Optional[Callable[[TestMasterEvent], Any]] = None
    ) -> str:
        """Subscribe to events."""
        subscription_id = f"{subscriber_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        subscription = EventSubscription(
            subscription_id=subscription_id,
            subscriber_id=subscriber_id,
            event_types=set(event_types),
            severity_filter=set(severity_filter or [s for s in EventSeverity]),
            source_filter=set(source_filter) if source_filter else None,
            tag_filter=set(tag_filter) if tag_filter else None,
            callback=callback,
            async_callback=async_callback
        )
        
        with self.lock:
            self.subscriptions[subscription_id] = subscription
        
        print(f"Event subscription created: {subscription_id}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events."""
        with self.lock:
            if subscription_id in self.subscriptions:
                del self.subscriptions[subscription_id]
                print(f"Event subscription removed: {subscription_id}")
                return True
        return False
    
    def register_publisher(self, publisher_id: str):
        """Register event publisher."""
        self.publishers.add(publisher_id)
        print(f"Event publisher registered: {publisher_id}")
    
    def get_events(
        self,
        event_type: Optional[EventType] = None,
        severity: Optional[EventSeverity] = None,
        source_system: Optional[str] = None,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[TestMasterEvent]:
        """Get events with filtering."""
        with self.lock:
            events = list(self.events)
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        if source_system:
            events = [e for e in events if e.source_system == source_system]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        # Sort by timestamp (newest first) and limit
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return events[:limit]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        with self.lock:
            return {
                "events_published": self.events_published,
                "events_processed": self.events_processed,
                "events_in_memory": len(self.events),
                "subscription_triggers": self.subscription_triggers,
                "processing_errors": self.processing_errors,
                "active_subscriptions": len([s for s in self.subscriptions.values() if s.active]),
                "total_subscriptions": len(self.subscriptions),
                "registered_publishers": len(self.publishers),
                "processing_rate": self.events_processed / max(self.events_published, 1),
                "queue_size": self.event_queue.qsize()
            }


class EventPersistence:
    """Event persistence with SQLite and compression."""
    
    def __init__(self, db_path: str = "testmaster_events.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._initialize_database()
        
        print(f"Event persistence initialized: {db_path}")
    
    def _initialize_database(self):
        """Initialize SQLite database for event storage."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    source_system TEXT NOT NULL,
                    source_component TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    data_compressed BLOB,
                    tags TEXT,
                    correlation_id TEXT,
                    parent_event_id TEXT,
                    session_id TEXT,
                    user_id TEXT,
                    workflow_id TEXT,
                    status TEXT,
                    expiry_time TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes separately
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON events(source_system)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_correlation ON events(correlation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_workflow ON events(workflow_id)")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS event_correlations (
                    correlation_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    time_window_seconds INTEGER NOT NULL,
                    description TEXT,
                    related_events TEXT,
                    metadata_compressed BLOB,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for correlations table
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_pattern ON event_correlations(pattern_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_confidence ON event_correlations(confidence)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_correlations_created ON event_correlations(created_at)")
    
    def store_event(self, event: TestMasterEvent):
        """Store event in database."""
        with self.lock:
            try:
                # Compress event data
                data_json = json.dumps(event.data)
                data_compressed = gzip.compress(data_json.encode('utf-8'))
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO events (
                            event_id, event_type, severity, source_system, source_component,
                            timestamp, title, description, data_compressed, tags,
                            correlation_id, parent_event_id, session_id, user_id,
                            workflow_id, status, expiry_time
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.event_id,
                        event.event_type.value,
                        event.severity.value,
                        event.source_system,
                        event.source_component,
                        event.timestamp.isoformat(),
                        event.title,
                        event.description,
                        data_compressed,
                        json.dumps(event.tags),
                        event.correlation_id,
                        event.parent_event_id,
                        event.session_id,
                        event.user_id,
                        event.workflow_id,
                        event.status.value,
                        event.expiry_time.isoformat() if event.expiry_time else None
                    ))
                    
            except Exception as e:
                print(f"Event storage error: {e}")
    
    def retrieve_events(
        self,
        limit: int = 100,
        event_type: Optional[str] = None,
        source_system: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None
    ) -> List[TestMasterEvent]:
        """Retrieve events from database."""
        with self.lock:
            try:
                query = "SELECT * FROM events WHERE 1=1"
                params = []
                
                if event_type:
                    query += " AND event_type = ?"
                    params.append(event_type)
                
                if source_system:
                    query += " AND source_system = ?"
                    params.append(source_system)
                
                if since:
                    query += " AND timestamp >= ?"
                    params.append(since.isoformat())
                
                if until:
                    query += " AND timestamp <= ?"
                    params.append(until.isoformat())
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                
                events = []
                for row in rows:
                    # Decompress data
                    data_compressed = row['data_compressed']
                    if data_compressed:
                        data_json = gzip.decompress(data_compressed).decode('utf-8')
                        data = json.loads(data_json)
                    else:
                        data = {}
                    
                    event = TestMasterEvent(
                        event_id=row['event_id'],
                        event_type=EventType(row['event_type']),
                        severity=EventSeverity(row['severity']),
                        source_system=row['source_system'],
                        source_component=row['source_component'],
                        timestamp=datetime.fromisoformat(row['timestamp']),
                        title=row['title'],
                        description=row['description'] or "",
                        data=data,
                        tags=json.loads(row['tags'] or '[]'),
                        correlation_id=row['correlation_id'],
                        parent_event_id=row['parent_event_id'],
                        session_id=row['session_id'],
                        user_id=row['user_id'],
                        workflow_id=row['workflow_id'],
                        status=EventStatus(row['status']),
                        expiry_time=datetime.fromisoformat(row['expiry_time']) if row['expiry_time'] else None
                    )
                    events.append(event)
                
                return events
                
            except Exception as e:
                print(f"Event retrieval error: {e}")
                return []
    
    def store_correlation(self, correlation: EventCorrelation):
        """Store event correlation."""
        with self.lock:
            try:
                # Compress metadata
                metadata_json = json.dumps(correlation.metadata)
                metadata_compressed = gzip.compress(metadata_json.encode('utf-8'))
                
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO event_correlations (
                            correlation_id, pattern_type, confidence, time_window_seconds,
                            description, related_events, metadata_compressed
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        correlation.correlation_id,
                        correlation.pattern_type,
                        correlation.confidence,
                        int(correlation.time_window.total_seconds()),
                        correlation.description,
                        json.dumps(correlation.related_events),
                        metadata_compressed
                    ))
                    
            except Exception as e:
                print(f"Correlation storage error: {e}")
    
    def cleanup_expired_events(self):
        """Clean up expired events from database."""
        with self.lock:
            try:
                current_time = datetime.now().isoformat()
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute(
                        "DELETE FROM events WHERE expiry_time IS NOT NULL AND expiry_time < ?",
                        (current_time,)
                    )
                    deleted_count = cursor.rowcount
                
                if deleted_count > 0:
                    print(f"Cleaned up {deleted_count} expired events")
                    
            except Exception as e:
                print(f"Event cleanup error: {e}")
    
    def get_storage_metrics(self) -> Dict[str, Any]:
        """Get storage metrics."""
        with self.lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT COUNT(*) FROM events")
                    total_events = cursor.fetchone()[0]
                    
                    cursor = conn.execute("SELECT COUNT(*) FROM event_correlations")
                    total_correlations = cursor.fetchone()[0]
                    
                    cursor = conn.execute("""
                        SELECT event_type, COUNT(*) as count 
                        FROM events 
                        GROUP BY event_type
                    """)
                    events_by_type = dict(cursor.fetchall())
                
                return {
                    "total_events": total_events,
                    "total_correlations": total_correlations,
                    "events_by_type": events_by_type,
                    "database_path": self.db_path
                }
                
            except Exception as e:
                print(f"Storage metrics error: {e}")
                return {}


class EventCorrelationEngine:
    """Event correlation and pattern detection."""
    
    def __init__(self, coordinator: AgentCoordinator):
        self.coordinator = coordinator
        self.correlation_rules: List[Dict[str, Any]] = []
        self.active_correlations: Dict[str, EventCorrelation] = {}
        self.pattern_cache: Dict[str, Any] = {}
        
        self._setup_default_correlation_rules()
        
        print("Event Correlation Engine initialized")
    
    def _setup_default_correlation_rules(self):
        """Setup default correlation rules."""
        # Error burst detection
        self.add_correlation_rule(
            "error_burst",
            lambda events: self._detect_error_burst(events),
            "Multiple errors from same component",
            timedelta(minutes=5)
        )
        
        # Performance degradation pattern
        self.add_correlation_rule(
            "performance_degradation",
            lambda events: self._detect_performance_degradation(events),
            "Performance metrics showing degradation",
            timedelta(minutes=10)
        )
        
        # Security event correlation
        self.add_correlation_rule(
            "security_incident",
            lambda events: self._detect_security_incident(events),
            "Multiple security events indicating incident",
            timedelta(minutes=15)
        )
    
    def add_correlation_rule(
        self,
        pattern_type: str,
        detector_func: Callable[[List[TestMasterEvent]], Optional[float]],
        description: str,
        time_window: timedelta
    ):
        """Add correlation rule."""
        rule = {
            "pattern_type": pattern_type,
            "detector": detector_func,
            "description": description,
            "time_window": time_window,
            "matches": 0,
            "last_match": None
        }
        self.correlation_rules.append(rule)
    
    def analyze_events(self, events: List[TestMasterEvent]) -> List[EventCorrelation]:
        """Analyze events for correlations."""
        correlations = []
        
        for rule in self.correlation_rules:
            # Get events within time window
            cutoff_time = datetime.now() - rule["time_window"]
            recent_events = [e for e in events if e.timestamp >= cutoff_time]
            
            if len(recent_events) < 2:
                continue
            
            # Run detector
            confidence = rule["detector"](recent_events)
            if confidence and confidence > 0.7:  # High confidence threshold
                correlation_id = f"{rule['pattern_type']}_{int(time.time())}"
                
                correlation = EventCorrelation(
                    correlation_id=correlation_id,
                    related_events=[e.event_id for e in recent_events],
                    pattern_type=rule["pattern_type"],
                    confidence=confidence,
                    time_window=rule["time_window"],
                    description=rule["description"],
                    metadata={
                        "event_count": len(recent_events),
                        "time_span": (recent_events[-1].timestamp - recent_events[0].timestamp).total_seconds(),
                        "sources": list(set(e.source_system for e in recent_events))
                    }
                )
                
                correlations.append(correlation)
                rule["matches"] += 1
                rule["last_match"] = datetime.now()
        
        return correlations
    
    def _detect_error_burst(self, events: List[TestMasterEvent]) -> Optional[float]:
        """Detect error burst pattern."""
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        
        if len(error_events) < 3:
            return None
        
        # Group by source component
        by_component = defaultdict(list)
        for event in error_events:
            by_component[event.source_component].append(event)
        
        # Check for components with multiple errors
        for component, comp_events in by_component.items():
            if len(comp_events) >= 3:
                # Calculate time density
                time_span = (comp_events[-1].timestamp - comp_events[0].timestamp).total_seconds()
                if time_span < 300:  # Within 5 minutes
                    return min(0.9, 0.5 + (len(comp_events) * 0.1))
        
        return None
    
    def _detect_performance_degradation(self, events: List[TestMasterEvent]) -> Optional[float]:
        """Detect performance degradation pattern."""
        perf_events = [e for e in events if e.event_type == EventType.PERFORMANCE]
        
        if len(perf_events) < 3:
            return None
        
        # Look for degrading metrics
        response_times = []
        cpu_usage = []
        
        for event in perf_events:
            if 'response_time' in event.data:
                response_times.append(event.data['response_time'])
            if 'cpu_usage' in event.data:
                cpu_usage.append(event.data['cpu_usage'])
        
        # Analyze trends
        confidence = 0.0
        
        if len(response_times) >= 3:
            # Check if response times are increasing
            recent_avg = statistics.mean(response_times[-3:])
            earlier_avg = statistics.mean(response_times[:3])
            if recent_avg > earlier_avg * 1.5:
                confidence += 0.4
        
        if len(cpu_usage) >= 3:
            # Check if CPU usage is high and increasing
            recent_avg = statistics.mean(cpu_usage[-3:])
            if recent_avg > 80:
                confidence += 0.4
        
        return confidence if confidence > 0.7 else None
    
    def _detect_security_incident(self, events: List[TestMasterEvent]) -> Optional[float]:
        """Detect security incident pattern."""
        security_events = [e for e in events if e.event_type == EventType.SECURITY]
        
        if len(security_events) < 2:
            return None
        
        # Look for patterns indicating security incident
        failed_auth_count = len([e for e in security_events if 'authentication' in e.title.lower()])
        suspicious_access = len([e for e in security_events if 'suspicious' in e.description.lower()])
        
        confidence = 0.0
        
        if failed_auth_count >= 3:
            confidence += 0.5
        
        if suspicious_access >= 2:
            confidence += 0.4
        
        # Check for events from different sources (coordinated attack)
        unique_sources = len(set(e.source_system for e in security_events))
        if unique_sources >= 2:
            confidence += 0.3
        
        return min(1.0, confidence) if confidence > 0.7 else None
    
    def get_correlation_metrics(self) -> Dict[str, Any]:
        """Get correlation engine metrics."""
        return {
            "total_rules": len(self.correlation_rules),
            "active_correlations": len(self.active_correlations),
            "rule_performance": [
                {
                    "pattern_type": rule["pattern_type"],
                    "matches": rule["matches"],
                    "last_match": rule["last_match"].isoformat() if rule["last_match"] else None
                }
                for rule in self.correlation_rules
            ]
        }


class EventMonitoringBridge:
    """Main event monitoring bridge orchestrator."""
    
    def __init__(self):
        self.enabled = FeatureFlags.is_enabled('layer4_bridges', 'event_monitoring')
        
        # Core components
        self.event_bus = EventBus()
        self.persistence = EventPersistence()
        self.coordinator = AgentCoordinator()
        self.correlation_engine = EventCorrelationEngine(self.coordinator)
        self.shared_state = SharedState()
        
        # Bridge state
        self.publishers: Dict[str, Any] = {}
        self.subscribers: Dict[str, Any] = {}
        self.event_translators: Dict[str, Callable] = {}
        
        # Performance tracking
        self.start_time = datetime.now()
        self.bridge_events_processed = 0
        
        if not self.enabled:
            return
        
        self._setup_system_integrations()
        self._start_correlation_monitoring()
        
        print("Event Monitoring Bridge initialized")
        print(f"   Event persistence: {self.persistence.db_path}")
        print(f"   Correlation rules: {len(self.correlation_engine.correlation_rules)}")
    
    def _setup_system_integrations(self):
        """Setup integrations with existing TestMaster systems."""
        # Subscribe to workflow events
        self.subscribe_to_workflow_events()
        
        # Subscribe to performance events
        self.subscribe_to_performance_events()
        
        # Subscribe to security events
        self.subscribe_to_security_events()
        
        # Start event bus processing
        self.event_bus.start_processing()
    
    def subscribe_to_workflow_events(self):
        """Subscribe to workflow execution events."""
        def workflow_event_handler(event: TestMasterEvent):
            # Store event and update shared state
            self.persistence.store_event(event)
            self.shared_state.set(f"latest_workflow_event_{event.workflow_id}", {
                "event_id": event.event_id,
                "type": event.event_type.value,
                "timestamp": event.timestamp.isoformat(),
                "status": event.data.get("status", "unknown")
            })
        
        subscription_id = self.event_bus.subscribe(
            "workflow_monitor",
            [EventType.WORKFLOW],
            callback=workflow_event_handler
        )
        
        print(f"Workflow events subscription: {subscription_id}")
    
    def subscribe_to_performance_events(self):
        """Subscribe to performance monitoring events."""
        def performance_event_handler(event: TestMasterEvent):
            self.persistence.store_event(event)
            
            # Check for performance alerts
            if event.severity in [EventSeverity.CRITICAL, EventSeverity.HIGH]:
                self.create_alert_event(
                    f"Performance Alert: {event.title}",
                    f"High priority performance event detected: {event.description}",
                    {"source_event": event.event_id, "metric_data": event.data}
                )
        
        subscription_id = self.event_bus.subscribe(
            "performance_monitor",
            [EventType.PERFORMANCE],
            callback=performance_event_handler
        )
        
        print(f"Performance events subscription: {subscription_id}")
    
    def subscribe_to_security_events(self):
        """Subscribe to security events."""
        def security_event_handler(event: TestMasterEvent):
            self.persistence.store_event(event)
            
            # Always alert on security events
            self.create_alert_event(
                f"Security Event: {event.title}",
                f"Security event detected: {event.description}",
                {"source_event": event.event_id, "security_data": event.data}
            )
        
        subscription_id = self.event_bus.subscribe(
            "security_monitor",
            [EventType.SECURITY],
            callback=security_event_handler,
            severity_filter=[EventSeverity.CRITICAL, EventSeverity.HIGH, EventSeverity.MEDIUM]
        )
        
        print(f"Security events subscription: {subscription_id}")
    
    def _start_correlation_monitoring(self):
        """Start background correlation monitoring."""
        def correlation_worker():
            while self.enabled:
                try:
                    # Get recent events for correlation analysis
                    recent_events = self.event_bus.get_events(limit=500, since=datetime.now() - timedelta(hours=1))
                    
                    if len(recent_events) > 5:
                        correlations = self.correlation_engine.analyze_events(recent_events)
                        
                        for correlation in correlations:
                            # Store correlation
                            self.persistence.store_correlation(correlation)
                            
                            # Create correlation event
                            self.publish_event(
                                EventType.SYSTEM,
                                EventSeverity.MEDIUM,
                                "correlation_engine",
                                "Event Correlation Detected",
                                f"Pattern detected: {correlation.description}",
                                {
                                    "correlation_id": correlation.correlation_id,
                                    "pattern_type": correlation.pattern_type,
                                    "confidence": correlation.confidence,
                                    "related_events": correlation.related_events
                                },
                                correlation_id=correlation.correlation_id
                            )
                    
                    time.sleep(60)  # Check every minute
                    
                except Exception as e:
                    print(f"Correlation monitoring error: {e}")
                    time.sleep(30)
        
        correlation_thread = threading.Thread(target=correlation_worker, daemon=True)
        correlation_thread.start()
    
    def publish_event(
        self,
        event_type: EventType,
        severity: EventSeverity,
        source_component: str,
        title: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> str:
        """Publish event to the monitoring bridge."""
        event = TestMasterEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            severity=severity,
            source_system="testmaster_bridge",
            source_component=source_component,
            timestamp=datetime.now(),
            title=title,
            description=description,
            data=data or {},
            tags=tags or [],
            correlation_id=correlation_id,
            workflow_id=workflow_id,
            session_id=session_id
        )
        
        event_id = self.event_bus.publish_event(event)
        self.bridge_events_processed += 1
        
        return event_id
    
    def create_alert_event(self, title: str, description: str, alert_data: Dict[str, Any]) -> str:
        """Create alert event for high priority situations."""
        return self.publish_event(
            EventType.SYSTEM,
            EventSeverity.HIGH,
            "alert_system",
            title,
            description,
            alert_data,
            tags=["alert", "automated"]
        )
    
    def enable_cross_system_events(
        self,
        source_system: str,
        event_translator: Optional[Callable[[Dict[str, Any]], TestMasterEvent]] = None
    ):
        """Enable event bridging from external systems."""
        if event_translator:
            self.event_translators[source_system] = event_translator
        
        print(f"Cross-system events enabled for: {source_system}")
    
    def ingest_external_event(self, source_system: str, external_event: Dict[str, Any]) -> str:
        """Ingest event from external system."""
        if source_system in self.event_translators:
            # Translate external event format
            try:
                translated_event = self.event_translators[source_system](external_event)
                return self.event_bus.publish_event(translated_event)
            except Exception as e:
                print(f"Event translation error for {source_system}: {e}")
                return ""
        else:
            # Default translation
            event = TestMasterEvent(
                event_id=external_event.get("id", str(uuid.uuid4())),
                event_type=EventType.INTEGRATION,
                severity=EventSeverity.INFO,
                source_system=source_system,
                source_component=external_event.get("component", "unknown"),
                timestamp=datetime.now(),
                title=external_event.get("title", "External Event"),
                description=external_event.get("description", ""),
                data=external_event
            )
            
            return self.event_bus.publish_event(event)
    
    def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive bridge metrics."""
        uptime = datetime.now() - self.start_time
        
        return {
            "bridge_status": "active" if self.enabled else "disabled",
            "uptime_seconds": uptime.total_seconds(),
            "bridge_events_processed": self.bridge_events_processed,
            "event_bus_metrics": self.event_bus.get_metrics(),
            "storage_metrics": self.persistence.get_storage_metrics(),
            "correlation_metrics": self.correlation_engine.get_correlation_metrics(),
            "cross_system_integrations": len(self.event_translators),
            "active_subscriptions": len(self.subscribers)
        }
    
    def optimize_event_system(self):
        """Optimize event monitoring system."""
        # Clean up expired events
        self.persistence.cleanup_expired_events()
        
        # Get and analyze metrics
        metrics = self.get_comprehensive_metrics()
        
        # Store optimization results
        self.shared_state.set("event_system_optimization", {
            "optimized_at": datetime.now().isoformat(),
            "metrics": metrics
        })
        
        print("Event monitoring system optimized")
    
    def shutdown(self):
        """Shutdown event monitoring bridge."""
        # Stop event bus processing
        self.event_bus.stop_processing()
        
        # Store final metrics
        final_metrics = self.get_comprehensive_metrics()
        self.shared_state.set("event_bridge_final_metrics", final_metrics)
        
        # Final cleanup
        self.persistence.cleanup_expired_events()
        
        print("Event Monitoring Bridge shutdown complete")


def get_event_monitoring_bridge() -> EventMonitoringBridge:
    """Get event monitoring bridge instance."""
    return EventMonitoringBridge()