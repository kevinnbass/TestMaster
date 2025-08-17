"""
Live Feedback System for TestMaster

Real-time feedback collection and processing for streaming test generation.
Enables interactive improvement and user-guided enhancement.
"""

import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from queue import Queue, Empty
import json

from ..core.feature_flags import FeatureFlags
from ..core.shared_state import get_shared_state
from ..telemetry import get_telemetry_collector

class FeedbackType(Enum):
    """Types of feedback."""
    QUALITY_RATING = "quality_rating"
    SUGGESTION = "suggestion"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    ISSUE_REPORT = "issue_report"
    APPROVAL = "approval"
    REJECTION = "rejection"

@dataclass
class LiveFeedback:
    """Live feedback entry."""
    feedback_id: str
    session_id: str
    feedback_type: FeedbackType
    content: str
    rating: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    processed: bool = False
    response: Optional[str] = None

@dataclass
class FeedbackSession:
    """Feedback collection session."""
    session_id: str
    target_type: str  # "generation", "enhancement", "validation"
    target_id: str
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    feedback_count: int = 0
    average_rating: float = 0.0
    is_active: bool = True

@dataclass
class FeedbackSummary:
    """Summary of feedback for analysis."""
    session_id: str
    total_feedback: int
    feedback_by_type: Dict[FeedbackType, int] = field(default_factory=dict)
    average_rating: float = 0.0
    common_suggestions: List[str] = field(default_factory=list)
    issues_reported: List[str] = field(default_factory=list)
    approval_rate: float = 0.0

class FeedbackCollector:
    """
    Live feedback collector for streaming generation.
    
    Features:
    - Real-time feedback collection and processing
    - Multiple feedback types and ratings
    - Session-based feedback management
    - Automatic feedback analysis and summarization
    - Integration with streaming generation pipeline
    """
    
    def __init__(self, collection_interval: float = 5.0):
        """
        Initialize feedback collector.
        
        Args:
            collection_interval: Interval for processing feedback in seconds
        """
        self.enabled = FeatureFlags.is_enabled('layer1_test_foundation', 'streaming_generation')
        self.collection_interval = collection_interval
        
        # Initialize all attributes regardless of enabled state
        # Feedback state
        self.active_sessions: Dict[str, FeedbackSession] = {}
        self.feedback_queue: Queue = Queue()
        self.feedback_history: Dict[str, List[LiveFeedback]] = {}
        self.feedback_processors: List[Callable[[LiveFeedback], None]] = []
        
        # Threading
        self.lock = threading.RLock()
        self.collector_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.is_collecting = False
        
        # Statistics
        self.total_feedback = 0
        self.sessions_created = 0
        self.processed_feedback = 0
        
        if not self.enabled:
            return
        
        # Integrations
        if FeatureFlags.is_enabled('layer1_test_foundation', 'shared_state'):
            self.shared_state = get_shared_state()
        else:
            self.shared_state = None
        
        if FeatureFlags.is_enabled('layer3_orchestration', 'telemetry_system'):
            self.telemetry = get_telemetry_collector()
        else:
            self.telemetry = None
        
        # Statistics
        self.total_feedback = 0
        self.sessions_created = 0
        self.processed_feedback = 0
        
        print("Live feedback collector initialized")
        print(f"   Collection interval: {self.collection_interval}s")
    
    def start_feedback_session(self, target_type: str, target_id: str) -> str:
        """
        Start a feedback collection session.
        
        Args:
            target_type: Type of target (generation, enhancement, validation)
            target_id: ID of the target
            
        Returns:
            Session ID for feedback collection
        """
        if not self.enabled:
            raise RuntimeError("Feedback collector is disabled")
        
        session_id = str(uuid.uuid4())
        
        session = FeedbackSession(
            session_id=session_id,
            target_type=target_type,
            target_id=target_id
        )
        
        with self.lock:
            self.active_sessions[session_id] = session
            self.feedback_history[session_id] = []
            self.sessions_created += 1
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="feedback_session_started",
                component="feedback_collector",
                operation="start_session",
                metadata={
                    "session_id": session_id,
                    "target_type": target_type,
                    "target_id": target_id
                }
            )
        
        print(f"Started feedback session: {session_id}")
        return session_id
    
    def submit_feedback(self, session_id: str, feedback_type: FeedbackType,
                       content: str, rating: float = None,
                       metadata: Dict[str, Any] = None) -> str:
        """
        Submit feedback for a session.
        
        Args:
            session_id: Feedback session ID
            feedback_type: Type of feedback
            content: Feedback content
            rating: Optional rating (0.0 - 1.0)
            metadata: Additional metadata
            
        Returns:
            Feedback ID
        """
        if not self.enabled:
            raise RuntimeError("Feedback collector is disabled")
        
        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")
        
        feedback_id = str(uuid.uuid4())
        
        feedback = LiveFeedback(
            feedback_id=feedback_id,
            session_id=session_id,
            feedback_type=feedback_type,
            content=content,
            rating=rating,
            metadata=metadata or {}
        )
        
        # Add to queue for processing
        self.feedback_queue.put(feedback)
        
        # Update session
        with self.lock:
            session = self.active_sessions[session_id]
            session.feedback_count += 1
            session.last_activity = datetime.now()
            
            # Update average rating
            if rating is not None:
                if session.average_rating == 0.0:
                    session.average_rating = rating
                else:
                    # Running average
                    session.average_rating = (
                        (session.average_rating * (session.feedback_count - 1) + rating) /
                        session.feedback_count
                    )
            
            self.total_feedback += 1
        
        # Update shared state
        if self.shared_state:
            self.shared_state.increment("live_feedback_submitted")
        
        print(f"Submitted feedback: {feedback_id} (type: {feedback_type.value})")
        return feedback_id
    
    def start_collection(self):
        """Start background feedback collection."""
        if not self.enabled or self.is_collecting:
            return
        
        def collection_worker():
            self.is_collecting = True
            
            while not self.shutdown_event.is_set():
                try:
                    # Process feedback queue
                    self._process_feedback_queue()
                    
                    # Clean up old sessions
                    self._cleanup_old_sessions()
                    
                    # Send telemetry
                    self._send_feedback_telemetry()
                    
                    # Wait for next cycle
                    if self.shutdown_event.wait(timeout=self.collection_interval):
                        break
                        
                except Exception as e:
                    print(f"Feedback collection error: {e}")
            
            self.is_collecting = False
        
        self.collector_thread = threading.Thread(target=collection_worker, daemon=True)
        self.collector_thread.start()
        
        print("Feedback collection started")
    
    def stop_collection(self):
        """Stop background feedback collection."""
        if not self.enabled or not self.is_collecting:
            return
        
        self.shutdown_event.set()
        
        if self.collector_thread and self.collector_thread.is_alive():
            self.collector_thread.join(timeout=5.0)
        
        print("Feedback collection stopped")
    
    def _process_feedback_queue(self):
        """Process pending feedback from the queue."""
        processed_count = 0
        
        while not self.feedback_queue.empty() and processed_count < 50:
            try:
                feedback = self.feedback_queue.get_nowait()
                self._process_single_feedback(feedback)
                processed_count += 1
                
                with self.lock:
                    self.processed_feedback += 1
                    
            except Empty:
                break
            except Exception as e:
                print(f"Error processing feedback: {e}")
    
    def _process_single_feedback(self, feedback: LiveFeedback):
        """Process a single feedback entry."""
        # Add to history
        with self.lock:
            if feedback.session_id in self.feedback_history:
                self.feedback_history[feedback.session_id].append(feedback)
        
        # Apply feedback processors
        for processor in self.feedback_processors:
            try:
                processor(feedback)
            except Exception as e:
                print(f"Feedback processor error: {e}")
        
        # Mark as processed
        feedback.processed = True
        
        # Send telemetry
        if self.telemetry:
            self.telemetry.record_event(
                event_type="feedback_processed",
                component="feedback_collector",
                operation="process_feedback",
                metadata={
                    "feedback_id": feedback.feedback_id,
                    "session_id": feedback.session_id,
                    "feedback_type": feedback.feedback_type.value,
                    "has_rating": feedback.rating is not None
                }
            )
    
    def _cleanup_old_sessions(self):
        """Clean up old inactive sessions."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        with self.lock:
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.last_activity < cutoff_time:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self.active_sessions.pop(session_id)
                session.is_active = False
                print(f"Closed feedback session: {session_id}")
    
    def add_feedback_processor(self, processor: Callable[[LiveFeedback], None]):
        """Add a feedback processor function."""
        if not self.enabled:
            return
        
        self.feedback_processors.append(processor)
        print("Added feedback processor")
    
    def get_session_feedback(self, session_id: str) -> List[LiveFeedback]:
        """Get all feedback for a session."""
        if not self.enabled:
            return []
        
        with self.lock:
            return self.feedback_history.get(session_id, [])
    
    def get_feedback_summary(self, session_id: str) -> Optional[FeedbackSummary]:
        """Get summary of feedback for a session."""
        if not self.enabled or session_id not in self.feedback_history:
            return None
        
        with self.lock:
            feedback_list = self.feedback_history[session_id]
            
            if not feedback_list:
                return FeedbackSummary(session_id=session_id, total_feedback=0)
            
            # Calculate summary statistics
            feedback_by_type = {}
            ratings = []
            suggestions = []
            issues = []
            approvals = 0
            rejections = 0
            
            for feedback in feedback_list:
                # Count by type
                feedback_type = feedback.feedback_type
                feedback_by_type[feedback_type] = feedback_by_type.get(feedback_type, 0) + 1
                
                # Collect ratings
                if feedback.rating is not None:
                    ratings.append(feedback.rating)
                
                # Collect content by type
                if feedback_type == FeedbackType.SUGGESTION:
                    suggestions.append(feedback.content)
                elif feedback_type == FeedbackType.ISSUE_REPORT:
                    issues.append(feedback.content)
                elif feedback_type == FeedbackType.APPROVAL:
                    approvals += 1
                elif feedback_type == FeedbackType.REJECTION:
                    rejections += 1
            
            # Calculate averages
            average_rating = sum(ratings) / len(ratings) if ratings else 0.0
            approval_rate = approvals / (approvals + rejections) if (approvals + rejections) > 0 else 0.0
            
            # Find common suggestions (simplified)
            common_suggestions = suggestions[:5]  # Top 5 most recent
            
            return FeedbackSummary(
                session_id=session_id,
                total_feedback=len(feedback_list),
                feedback_by_type=feedback_by_type,
                average_rating=average_rating,
                common_suggestions=common_suggestions,
                issues_reported=issues,
                approval_rate=approval_rate
            )
    
    def get_active_sessions(self) -> List[FeedbackSession]:
        """Get list of active feedback sessions."""
        if not self.enabled:
            return []
        
        with self.lock:
            return [session for session in self.active_sessions.values() if session.is_active]
    
    def close_session(self, session_id: str):
        """Close a feedback session."""
        if not self.enabled:
            return
        
        with self.lock:
            if session_id in self.active_sessions:
                self.active_sessions[session_id].is_active = False
                print(f"Closed feedback session: {session_id}")
    
    def get_collector_statistics(self) -> Dict[str, Any]:
        """Get feedback collector statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        with self.lock:
            active_sessions_count = len([s for s in self.active_sessions.values() if s.is_active])
            
            # Calculate average rating across all sessions
            all_ratings = []
            for feedback_list in self.feedback_history.values():
                for feedback in feedback_list:
                    if feedback.rating is not None:
                        all_ratings.append(feedback.rating)
            
            overall_rating = sum(all_ratings) / len(all_ratings) if all_ratings else 0.0
            
            return {
                "enabled": True,
                "is_collecting": self.is_collecting,
                "active_sessions": active_sessions_count,
                "total_sessions": self.sessions_created,
                "total_feedback": self.total_feedback,
                "processed_feedback": self.processed_feedback,
                "overall_average_rating": round(overall_rating, 2),
                "collection_interval": self.collection_interval,
                "feedback_processors": len(self.feedback_processors)
            }
    
    def _send_feedback_telemetry(self):
        """Send feedback collection telemetry."""
        if not self.telemetry:
            return
        
        stats = self.get_collector_statistics()
        
        self.telemetry.record_event(
            event_type="feedback_collection_status",
            component="feedback_collector",
            operation="monitoring_cycle",
            metadata={
                "active_sessions": stats.get("active_sessions", 0),
                "total_feedback": stats.get("total_feedback", 0),
                "overall_rating": stats.get("overall_average_rating", 0)
            }
        )

# Global instance
_feedback_collector: Optional[FeedbackCollector] = None

def get_feedback_collector() -> FeedbackCollector:
    """Get the global feedback collector instance."""
    global _feedback_collector
    if _feedback_collector is None:
        _feedback_collector = FeedbackCollector()
    return _feedback_collector

# Convenience function
def collect_live_feedback(target_type: str, target_id: str) -> str:
    """
    Start collecting live feedback.
    
    Args:
        target_type: Type of target (generation, enhancement, validation)
        target_id: ID of the target
        
    Returns:
        Session ID for feedback collection
    """
    collector = get_feedback_collector()
    return collector.start_feedback_session(target_type, target_id)