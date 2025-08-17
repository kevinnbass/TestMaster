"""
Claude Code Messenger System

Inspired by Agency-Swarm's SendMessage validation and SharedState
patterns for reliable file-based communication with Claude Code.

Features:
- Structured YAML/JSON message format
- Message validation and integrity checking
- Priority and urgency indicators
- Acknowledgment and response tracking
"""

import yaml
import json
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
import threading
import time

from ..core.layer_manager import requires_layer


class MessageType(Enum):
    """Types of messages to Claude Code."""
    STATUS_UPDATE = "status_update"
    BREAKING_TESTS = "breaking_tests"
    IDLE_MODULES = "idle_modules"  
    COVERAGE_GAPS = "coverage_gaps"
    TEST_RESULTS = "test_results"
    SYSTEM_ALERT = "system_alert"
    DIRECTIVE_REQUEST = "directive_request"
    ACKNOWLEDGMENT = "acknowledgment"


class MessagePriority(IntEnum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class TestFailureInfo:
    """Information about a test failure."""
    module: str
    test: str
    failure: str
    last_working: Optional[str] = None
    priority: str = "NORMAL"
    suggested_action: Optional[str] = None


@dataclass
class ModuleAttentionInfo:
    """Information about modules needing attention."""
    path: str
    status: str
    coverage: Optional[float] = None
    risks: List[str] = field(default_factory=list)
    recommendation: Optional[str] = None


@dataclass
class CoverageGapInfo:
    """Information about coverage gaps."""
    module: str
    uncovered_lines: List[Union[int, str]] = field(default_factory=list)
    critical_paths: bool = False
    suggested_tests: List[str] = field(default_factory=list)


@dataclass
class ClaudeMessage:
    """Message to Claude Code."""
    message_id: str
    message_type: MessageType
    priority: MessagePriority
    timestamp: datetime
    sender: str = "TestMaster"
    
    # Message content
    breaking_tests: List[TestFailureInfo] = field(default_factory=list)
    modules_need_attention: List[ModuleAttentionInfo] = field(default_factory=list)
    coverage_gaps: List[CoverageGapInfo] = field(default_factory=list)
    system_status: Dict[str, Any] = field(default_factory=dict)
    message_text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    expires_at: Optional[datetime] = None
    requires_acknowledgment: bool = False
    acknowledged_at: Optional[datetime] = None
    response_received: bool = False


@dataclass 
class ClaudeDirective:
    """Directive from Claude Code to TestMaster."""
    directive_id: str
    timestamp: datetime
    sender: str = "Claude Code"
    
    # Directive content
    monitor_priority: List[Dict[str, Any]] = field(default_factory=list)
    temporary_ignore: List[Dict[str, Any]] = field(default_factory=list)
    test_preferences: List[Dict[str, Any]] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ClaudeMessenger:
    """
    File-based messaging system for Claude Code communication.
    
    Uses Agency-Swarm patterns for message validation and
    SharedState management for status synchronization.
    """
    
    @requires_layer("layer2_monitoring", "claude_communication")
    def __init__(self, message_dir: str = ".testmaster_messages",
                 max_message_age_hours: float = 24.0):
        """
        Initialize Claude messenger.
        
        Args:
            message_dir: Directory for message files
            max_message_age_hours: Maximum age before messages expire
        """
        self.message_dir = Path(message_dir)
        self.message_dir.mkdir(exist_ok=True)
        
        self.max_message_age = timedelta(hours=max_message_age_hours)
        
        # Message tracking (SharedState pattern)
        self._sent_messages: Dict[str, ClaudeMessage] = {}
        self._received_directives: Dict[str, ClaudeDirective] = {}
        
        # File monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Callbacks
        self.on_directive_received: Optional[callable] = None
        self.on_acknowledgment_received: Optional[callable] = None
        
        print(f"📡 Claude messenger initialized")
        print(f"   📁 Message directory: {self.message_dir}")
        print(f"   ⏱️ Message expiry: {max_message_age_hours} hours")
    
    def start_monitoring(self):
        """Start monitoring for Claude Code directives."""
        if self._monitoring:
            print("⚠️ Claude messenger already monitoring")
            return
        
        print("🔍 Starting Claude Code directive monitoring...")
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_directives, daemon=True)
        self._monitor_thread.start()
        
        print("✅ Claude directive monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring for directives."""
        if not self._monitoring:
            return
        
        print("🛑 Stopping Claude directive monitoring...")
        
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        print("✅ Claude directive monitoring stopped")
    
    def send_status_update(self, breaking_tests: List[TestFailureInfo] = None,
                          modules_need_attention: List[ModuleAttentionInfo] = None,
                          coverage_gaps: List[CoverageGapInfo] = None,
                          system_status: Dict[str, Any] = None,
                          priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """
        Send status update to Claude Code.
        
        Args:
            breaking_tests: List of failing tests
            modules_need_attention: List of modules needing attention
            coverage_gaps: List of coverage gaps
            system_status: Overall system status
            priority: Message priority
            
        Returns:
            Message ID
        """
        message = ClaudeMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.STATUS_UPDATE,
            priority=priority,
            timestamp=datetime.now(),
            breaking_tests=breaking_tests or [],
            modules_need_attention=modules_need_attention or [],
            coverage_gaps=coverage_gaps or [],
            system_status=system_status or {},
            requires_acknowledgment=priority >= MessagePriority.HIGH
        )
        
        return self._send_message(message)
    
    def send_breaking_test_alert(self, test_failures: List[TestFailureInfo],
                               priority: MessagePriority = MessagePriority.HIGH) -> str:
        """Send breaking test alert to Claude Code."""
        message = ClaudeMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.BREAKING_TESTS,
            priority=priority,
            timestamp=datetime.now(),
            breaking_tests=test_failures,
            message_text=f"{len(test_failures)} test(s) are failing and need immediate attention",
            requires_acknowledgment=True,
            expires_at=datetime.now() + timedelta(hours=2)  # Urgent
        )
        
        return self._send_message(message)
    
    def send_idle_module_alert(self, idle_modules: List[ModuleAttentionInfo],
                             priority: MessagePriority = MessagePriority.NORMAL) -> str:
        """Send idle module alert to Claude Code."""
        message = ClaudeMessage(
            message_id=self._generate_message_id(), 
            message_type=MessageType.IDLE_MODULES,
            priority=priority,
            timestamp=datetime.now(),
            modules_need_attention=idle_modules,
            message_text=f"{len(idle_modules)} module(s) have been idle for 2+ hours",
            requires_acknowledgment=False
        )
        
        return self._send_message(message)
    
    def send_coverage_gap_report(self, coverage_gaps: List[CoverageGapInfo],
                               priority: MessagePriority = MessagePriority.LOW) -> str:
        """Send coverage gap report to Claude Code."""
        message = ClaudeMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.COVERAGE_GAPS,
            priority=priority,
            timestamp=datetime.now(),
            coverage_gaps=coverage_gaps,
            message_text=f"Found {len(coverage_gaps)} coverage gaps requiring attention"
        )
        
        return self._send_message(message)
    
    def send_system_alert(self, alert_text: str, metadata: Dict[str, Any] = None,
                         priority: MessagePriority = MessagePriority.CRITICAL) -> str:
        """Send system alert to Claude Code."""
        message = ClaudeMessage(
            message_id=self._generate_message_id(),
            message_type=MessageType.SYSTEM_ALERT,
            priority=priority,
            timestamp=datetime.now(),
            message_text=alert_text,
            metadata=metadata or {},
            requires_acknowledgment=True,
            expires_at=datetime.now() + timedelta(hours=1)
        )
        
        return self._send_message(message)
    
    def _send_message(self, message: ClaudeMessage) -> str:
        """Send message to Claude Code via file system."""
        try:
            # Convert to dictionary for YAML serialization
            message_dict = self._message_to_dict(message)
            
            # Write to YAML file
            filename = f"TESTMASTER_STATUS_{message.message_id}.yaml"
            file_path = self.message_dir / filename
            
            with open(file_path, 'w') as f:
                yaml.dump(message_dict, f, default_flow_style=False, sort_keys=False)
            
            # Track sent message
            self._sent_messages[message.message_id] = message
            
            print(f"📤 Sent message: {message.message_type.value} (ID: {message.message_id})")
            return message.message_id
            
        except Exception as e:
            print(f"⚠️ Error sending message: {e}")
            raise
    
    def _message_to_dict(self, message: ClaudeMessage) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        # Convert dataclass to dict
        msg_dict = asdict(message)
        
        # Convert enums to strings
        msg_dict['message_type'] = message.message_type.value
        msg_dict['priority'] = message.priority.name
        
        # Convert timestamps to ISO format
        msg_dict['timestamp'] = message.timestamp.isoformat()
        if message.expires_at:
            msg_dict['expires_at'] = message.expires_at.isoformat()
        if message.acknowledged_at:
            msg_dict['acknowledged_at'] = message.acknowledged_at.isoformat()
        
        # Remove None values and empty lists
        return {k: v for k, v in msg_dict.items() if v is not None and v != []}
    
    def _monitor_directives(self):
        """Monitor for Claude Code directives."""
        while self._monitoring:
            try:
                # Check for new directive files
                directive_files = list(self.message_dir.glob("CLAUDE_DIRECTIVES_*.yaml"))
                
                for file_path in directive_files:
                    self._process_directive_file(file_path)
                
                # Check for acknowledgment files
                ack_files = list(self.message_dir.glob("CLAUDE_ACK_*.yaml"))
                
                for file_path in ack_files:
                    self._process_acknowledgment_file(file_path)
                
                # Clean up old messages
                self._cleanup_old_messages()
                
                # Wait before next check
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"⚠️ Error monitoring directives: {e}")
                time.sleep(10)  # Wait longer on error
    
    def _process_directive_file(self, file_path: Path):
        """Process a Claude Code directive file."""
        try:
            with open(file_path, 'r') as f:
                directive_data = yaml.safe_load(f)
            
            # Convert to directive object
            directive = ClaudeDirective(
                directive_id=directive_data.get('directive_id', f"dir_{int(time.time())}"),
                timestamp=datetime.fromisoformat(directive_data.get('timestamp', datetime.now().isoformat())),
                sender=directive_data.get('sender', 'Claude Code'),
                monitor_priority=directive_data.get('monitor_priority', []),
                temporary_ignore=directive_data.get('temporary_ignore', []),
                test_preferences=directive_data.get('test_preferences', []),
                immediate_actions=directive_data.get('immediate_actions', []),
                metadata=directive_data.get('metadata', {})
            )
            
            # Check if already processed
            if directive.directive_id not in self._received_directives:
                self._received_directives[directive.directive_id] = directive
                
                # Call callback
                if self.on_directive_received:
                    try:
                        self.on_directive_received(directive)
                    except Exception as e:
                        print(f"⚠️ Error in directive callback: {e}")
                
                print(f"📥 Received directive: {directive.directive_id}")
            
            # Archive processed file
            archive_path = self.message_dir / "processed" / file_path.name
            archive_path.parent.mkdir(exist_ok=True)
            file_path.rename(archive_path)
            
        except Exception as e:
            print(f"⚠️ Error processing directive {file_path}: {e}")
    
    def _process_acknowledgment_file(self, file_path: Path):
        """Process a Claude Code acknowledgment file."""
        try:
            with open(file_path, 'r') as f:
                ack_data = yaml.safe_load(f)
            
            message_id = ack_data.get('message_id')
            ack_timestamp = datetime.fromisoformat(ack_data.get('timestamp', datetime.now().isoformat()))
            
            # Update sent message
            if message_id in self._sent_messages:
                message = self._sent_messages[message_id]
                message.acknowledged_at = ack_timestamp
                message.response_received = True
                
                print(f"✅ Message acknowledged: {message_id}")
                
                # Call callback
                if self.on_acknowledgment_received:
                    try:
                        self.on_acknowledgment_received(message_id, ack_data)
                    except Exception as e:
                        print(f"⚠️ Error in acknowledgment callback: {e}")
            
            # Archive acknowledgment file
            archive_path = self.message_dir / "processed" / file_path.name
            archive_path.parent.mkdir(exist_ok=True)
            file_path.rename(archive_path)
            
        except Exception as e:
            print(f"⚠️ Error processing acknowledgment {file_path}: {e}")
    
    def _cleanup_old_messages(self):
        """Clean up expired messages."""
        current_time = datetime.now()
        
        # Remove expired sent messages
        expired_ids = []
        for msg_id, message in self._sent_messages.items():
            if message.expires_at and current_time > message.expires_at:
                expired_ids.append(msg_id)
            elif current_time - message.timestamp > self.max_message_age:
                expired_ids.append(msg_id)
        
        for msg_id in expired_ids:
            del self._sent_messages[msg_id]
        
        if expired_ids:
            print(f"🧹 Cleaned up {len(expired_ids)} expired messages")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        return f"msg_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def get_sent_messages(self, since_hours: int = 24) -> List[ClaudeMessage]:
        """Get sent messages from recent time period."""
        cutoff = datetime.now() - timedelta(hours=since_hours)
        return [
            msg for msg in self._sent_messages.values()
            if msg.timestamp > cutoff
        ]
    
    def get_received_directives(self, since_hours: int = 24) -> List[ClaudeDirective]:
        """Get received directives from recent time period."""
        cutoff = datetime.now() - timedelta(hours=since_hours)
        return [
            directive for directive in self._received_directives.values()
            if directive.timestamp > cutoff
        ]
    
    def get_pending_acknowledgments(self) -> List[ClaudeMessage]:
        """Get messages waiting for acknowledgment."""
        return [
            msg for msg in self._sent_messages.values()
            if msg.requires_acknowledgment and not msg.acknowledged_at
        ]
    
    def mark_message_acknowledged(self, message_id: str):
        """Manually mark a message as acknowledged."""
        if message_id in self._sent_messages:
            self._sent_messages[message_id].acknowledged_at = datetime.now()
            self._sent_messages[message_id].response_received = True
            print(f"✅ Manually marked message acknowledged: {message_id}")
    
    def get_communication_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        sent_count = len(self._sent_messages)
        acknowledged_count = sum(1 for msg in self._sent_messages.values() if msg.acknowledged_at)
        directive_count = len(self._received_directives)
        pending_acks = len(self.get_pending_acknowledgments())
        
        return {
            "messages_sent": sent_count,
            "messages_acknowledged": acknowledged_count,
            "acknowledgment_rate": (acknowledged_count / max(sent_count, 1)) * 100,
            "directives_received": directive_count,
            "pending_acknowledgments": pending_acks,
            "message_directory": str(self.message_dir),
            "monitoring_active": self._monitoring
        }


# Convenience functions for common message creation
def create_test_failure_info(module: str, test: str, failure: str,
                           last_working: str = None, 
                           suggested_action: str = None) -> TestFailureInfo:
    """Create test failure information."""
    return TestFailureInfo(
        module=module,
        test=test,
        failure=failure,
        last_working=last_working,
        suggested_action=suggested_action
    )


def create_module_attention_info(path: str, status: str, coverage: float = None,
                               risks: List[str] = None,
                               recommendation: str = None) -> ModuleAttentionInfo:
    """Create module attention information."""
    return ModuleAttentionInfo(
        path=path,
        status=status,
        coverage=coverage,
        risks=risks or [],
        recommendation=recommendation
    )


def create_coverage_gap_info(module: str, uncovered_lines: List[Union[int, str]] = None,
                           critical_paths: bool = False,
                           suggested_tests: List[str] = None) -> CoverageGapInfo:
    """Create coverage gap information."""
    return CoverageGapInfo(
        module=module,
        uncovered_lines=uncovered_lines or [],
        critical_paths=critical_paths,
        suggested_tests=suggested_tests or []
    )