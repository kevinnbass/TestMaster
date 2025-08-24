"""
Bidirectional Message Queue Management

Inspired by Agency-Swarm's thread-based conversation management
for maintaining message queues, acknowledgments, and responses.

Features:
- Message queue via filesystem
- Acknowledgment and response tracking  
- Error handling and retry logic
- Thread-based conversation management
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from collections import deque
import queue

from core.layer_manager import requires_layer


class MessageStatus(Enum):
    """Message status in the queue."""
    QUEUED = "queued"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESPONDED = "responded"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class QueuePriority(IntEnum):
    """Queue priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class QueueMessage:
    """Message in the queue system."""
    message_id: str
    content: Dict[str, Any]
    priority: QueuePriority
    status: MessageStatus
    created_at: datetime
    
    # Routing
    sender: str
    recipient: str
    message_type: str
    
    # Tracking
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Retry logic
    retry_count: int = 0
    max_retries: int = 3
    
    # Response tracking
    requires_response: bool = False
    response_content: Optional[Dict[str, Any]] = None
    
    # Error handling
    last_error: Optional[str] = None
    error_count: int = 0


@dataclass
class QueueStatistics:
    """Queue statistics."""
    total_messages: int
    queued_messages: int
    sent_messages: int
    acknowledged_messages: int
    responded_messages: int
    failed_messages: int
    expired_messages: int
    avg_queue_time_seconds: float
    avg_response_time_seconds: float
    success_rate: float
    last_updated: datetime = field(default_factory=datetime.now)


class MessageQueue:
    """
    Bidirectional message queue with file-based persistence.
    
    Uses Agency-Swarm's thread-based conversation management
    patterns for reliable message delivery and tracking.
    """
    
    @requires_layer("layer2_monitoring", "message_queue")
    def __init__(self, queue_dir: str = ".testmaster_queue",
                 max_queue_size: int = 1000,
                 default_message_ttl_hours: float = 24.0):
        """
        Initialize message queue.
        
        Args:
            queue_dir: Directory for queue persistence
            max_queue_size: Maximum messages in queue
            default_message_ttl_hours: Default message time-to-live
        """
        self.queue_dir = Path(queue_dir)
        self.queue_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.queue_dir / "outbound").mkdir(exist_ok=True)
        (self.queue_dir / "inbound").mkdir(exist_ok=True)
        (self.queue_dir / "processed").mkdir(exist_ok=True)
        (self.queue_dir / "failed").mkdir(exist_ok=True)
        
        self.max_queue_size = max_queue_size
        self.default_ttl = timedelta(hours=default_message_ttl_hours)
        
        # In-memory queues (Thread-based conversation pattern)
        self._outbound_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self._inbound_queue = queue.Queue(maxsize=max_queue_size)
        
        # Message tracking
        self._active_messages: Dict[str, QueueMessage] = {}
        self._conversation_threads: Dict[str, List[str]] = {}  # recipient -> message_ids
        
        # Queue processing
        self._is_running = False
        self._outbound_thread: Optional[threading.Thread] = None
        self._inbound_thread: Optional[threading.Thread] = None
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'total_messages': 0,
            'queue_times': deque(maxlen=1000),
            'response_times': deque(maxlen=1000),
            'message_counts': {status.value: 0 for status in MessageStatus}
        }
        
        # Callbacks
        self.on_message_sent: Optional[Callable[[QueueMessage], None]] = None
        self.on_message_received: Optional[Callable[[QueueMessage], None]] = None
        self.on_acknowledgment: Optional[Callable[[QueueMessage], None]] = None
        self.on_response: Optional[Callable[[QueueMessage], None]] = None
        self.on_error: Optional[Callable[[QueueMessage, str], None]] = None
        
        print(f"📬 Message queue initialized")
        print(f"   📁 Queue directory: {self.queue_dir}")
        print(f"   📦 Max queue size: {max_queue_size}")
        print(f"   ⏱️ Default TTL: {default_message_ttl_hours} hours")
    
    def start(self):
        """Start queue processing."""
        if self._is_running:
            print("⚠️ Message queue is already running")
            return
        
        print("🚀 Starting message queue processing...")
        
        # Load existing messages from disk
        self._load_persisted_messages()
        
        # Start processing threads
        self._is_running = True
        
        self._outbound_thread = threading.Thread(target=self._process_outbound_queue, daemon=True)
        self._inbound_thread = threading.Thread(target=self._process_inbound_queue, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_messages, daemon=True)
        
        self._outbound_thread.start()
        self._inbound_thread.start()
        self._cleanup_thread.start()
        
        print("✅ Message queue started")
    
    def stop(self):
        """Stop queue processing."""
        if not self._is_running:
            return
        
        print("🛑 Stopping message queue...")
        
        self._is_running = False
        
        # Wait for threads to finish
        for thread in [self._outbound_thread, self._inbound_thread, self._cleanup_thread]:
            if thread:
                thread.join(timeout=5)
        
        # Persist remaining messages
        self._persist_active_messages()
        
        print("✅ Message queue stopped")
    
    def send_message(self, recipient: str, message_type: str, content: Dict[str, Any],
                    priority: QueuePriority = QueuePriority.NORMAL,
                    requires_response: bool = False,
                    ttl_hours: Optional[float] = None,
                    sender: str = "TestMaster") -> str:
        """
        Send a message through the queue.
        
        Args:
            recipient: Message recipient
            message_type: Type of message
            content: Message content
            priority: Message priority
            requires_response: Whether response is required
            ttl_hours: Time-to-live in hours
            sender: Message sender
            
        Returns:
            Message ID
        """
        if not self._is_running:
            raise RuntimeError("Message queue is not running")
        
        # Generate message ID
        message_id = self._generate_message_id()
        
        # Calculate expiry
        ttl = timedelta(hours=ttl_hours) if ttl_hours else self.default_ttl
        expires_at = datetime.now() + ttl
        
        # Create queue message
        queue_message = QueueMessage(
            message_id=message_id,
            content=content,
            priority=priority,
            status=MessageStatus.QUEUED,
            created_at=datetime.now(),
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            expires_at=expires_at,
            requires_response=requires_response
        )
        
        # Add to conversation thread
        if recipient not in self._conversation_threads:
            self._conversation_threads[recipient] = []
        self._conversation_threads[recipient].append(message_id)
        
        # Track message
        self._active_messages[message_id] = queue_message
        self._stats['total_messages'] += 1
        self._stats['message_counts'][MessageStatus.QUEUED.value] += 1
        
        try:
            # Add to outbound queue (negative priority for max-heap behavior)
            self._outbound_queue.put((-priority.value, time.time(), queue_message), block=False)
            
            print(f"📤 Queued message: {message_type} to {recipient} (ID: {message_id})")
            return message_id
            
        except queue.Full:
            # Queue is full - mark as failed
            queue_message.status = MessageStatus.FAILED
            queue_message.last_error = "Queue is full"
            
            self._stats['message_counts'][MessageStatus.FAILED.value] += 1
            
            if self.on_error:
                try:
                    self.on_error(queue_message, "Queue is full")
                except Exception as e:
                    print(f"⚠️ Error in error callback: {e}")
            
            raise RuntimeError(f"Message queue is full (max: {self.max_queue_size})")
    
    def acknowledge_message(self, message_id: str, response_content: Dict[str, Any] = None) -> bool:
        """
        Acknowledge a received message.
        
        Args:
            message_id: ID of message to acknowledge
            response_content: Optional response content
            
        Returns:
            True if acknowledgment was successful
        """
        if message_id not in self._active_messages:
            return False
        
        message = self._active_messages[message_id]
        
        # Update message status
        message.acknowledged_at = datetime.now()
        message.status = MessageStatus.ACKNOWLEDGED
        
        if response_content:
            message.response_content = response_content
            message.responded_at = datetime.now()
            message.status = MessageStatus.RESPONDED
            
            # Track response time
            if message.sent_at:
                response_time = (message.responded_at - message.sent_at).total_seconds()
                self._stats['response_times'].append(response_time)
        
        # Update statistics
        self._stats['message_counts'][MessageStatus.ACKNOWLEDGED.value] += 1
        if response_content:
            self._stats['message_counts'][MessageStatus.RESPONDED.value] += 1
        
        # Call callbacks
        if self.on_acknowledgment:
            try:
                self.on_acknowledgment(message)
            except Exception as e:
                print(f"⚠️ Error in acknowledgment callback: {e}")
        
        if response_content and self.on_response:
            try:
                self.on_response(message)
            except Exception as e:
                print(f"⚠️ Error in response callback: {e}")
        
        print(f"✅ Message acknowledged: {message_id}")
        return True
    
    def get_message_status(self, message_id: str) -> Optional[QueueMessage]:
        """Get status of a specific message."""
        return self._active_messages.get(message_id)
    
    def get_conversation_history(self, recipient: str, limit: int = 50) -> List[QueueMessage]:
        """Get conversation history with a recipient."""
        if recipient not in self._conversation_threads:
            return []
        
        message_ids = self._conversation_threads[recipient][-limit:]
        return [
            self._active_messages[msg_id] 
            for msg_id in message_ids 
            if msg_id in self._active_messages
        ]
    
    def _process_outbound_queue(self):
        """Process outbound message queue."""
        while self._is_running:
            try:
                # Get next message
                try:
                    _, _, queue_message = self._outbound_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Check if queue is still running
                if not self._is_running:
                    break
                
                # Process message
                self._send_message_to_recipient(queue_message)
                self._outbound_queue.task_done()
                
            except Exception as e:
                print(f"⚠️ Error processing outbound queue: {e}")
                time.sleep(1)
    
    def _process_inbound_queue(self):
        """Process inbound message queue."""
        while self._is_running:
            try:
                # Check for new inbound messages
                self._scan_for_inbound_messages()
                
                # Process inbound queue
                try:
                    queue_message = self._inbound_queue.get(timeout=1.0)
                    self._handle_inbound_message(queue_message)
                    self._inbound_queue.task_done()
                except queue.Empty:
                    pass
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"⚠️ Error processing inbound queue: {e}")
                time.sleep(5)
    
    def _send_message_to_recipient(self, queue_message: QueueMessage):
        """Send message to recipient via file system."""
        try:
            # Update message status
            queue_message.status = MessageStatus.SENT
            queue_message.sent_at = datetime.now()
            
            # Track queue time
            queue_time = (queue_message.sent_at - queue_message.created_at).total_seconds()
            self._stats['queue_times'].append(queue_time)
            
            # Create outbound message file
            message_dict = self._message_to_dict(queue_message)
            filename = f"{queue_message.recipient}_{queue_message.message_id}.json"
            outbound_path = self.queue_dir / "outbound" / filename
            
            with open(outbound_path, 'w') as f:
                json.dump(message_dict, f, indent=2, default=str)
            
            # Update statistics
            self._stats['message_counts'][MessageStatus.SENT.value] += 1
            
            # Call callback
            if self.on_message_sent:
                try:
                    self.on_message_sent(queue_message)
                except Exception as e:
                    print(f"⚠️ Error in message sent callback: {e}")
            
            print(f"📤 Sent message: {queue_message.message_id} to {queue_message.recipient}")
            
        except Exception as e:
            # Mark message as failed
            queue_message.status = MessageStatus.FAILED
            queue_message.last_error = str(e)
            queue_message.error_count += 1
            
            self._stats['message_counts'][MessageStatus.FAILED.value] += 1
            
            # Retry logic
            if queue_message.retry_count < queue_message.max_retries:
                queue_message.retry_count += 1
                queue_message.status = MessageStatus.QUEUED
                
                # Re-queue for retry
                try:
                    self._outbound_queue.put(
                        (-queue_message.priority.value, time.time(), queue_message),
                        block=False
                    )
                    print(f"🔄 Retrying message: {queue_message.message_id} (attempt {queue_message.retry_count + 1})")
                except queue.Full:
                    print(f"⚠️ Cannot retry message - queue is full")
            else:
                # Move to failed directory
                self._move_to_failed(queue_message)
            
            if self.on_error:
                try:
                    self.on_error(queue_message, str(e))
                except Exception as cb_error:
                    print(f"⚠️ Error in error callback: {cb_error}")
    
    def _scan_for_inbound_messages(self):
        """Scan for new inbound messages."""
        inbound_dir = self.queue_dir / "inbound"
        
        for message_file in inbound_dir.glob("*.json"):
            try:
                with open(message_file, 'r') as f:
                    message_data = json.load(f)
                
                # Convert to QueueMessage
                queue_message = self._dict_to_message(message_data)
                
                # Add to inbound queue
                try:
                    self._inbound_queue.put(queue_message, block=False)
                    
                    # Archive the file
                    processed_path = self.queue_dir / "processed" / message_file.name
                    message_file.rename(processed_path)
                    
                except queue.Full:
                    print(f"⚠️ Inbound queue full - cannot process {message_file.name}")
                    break
                
            except Exception as e:
                print(f"⚠️ Error processing inbound message {message_file}: {e}")
                # Move to failed directory
                failed_path = self.queue_dir / "failed" / message_file.name
                try:
                    message_file.rename(failed_path)
                except:
                    pass
    
    def _handle_inbound_message(self, queue_message: QueueMessage):
        """Handle an inbound message."""
        # Track message
        self._active_messages[queue_message.message_id] = queue_message
        
        # Add to conversation thread
        sender = queue_message.sender
        if sender not in self._conversation_threads:
            self._conversation_threads[sender] = []
        self._conversation_threads[sender].append(queue_message.message_id)
        
        # Call callback
        if self.on_message_received:
            try:
                self.on_message_received(queue_message)
            except Exception as e:
                print(f"⚠️ Error in message received callback: {e}")
        
        print(f"📥 Received message: {queue_message.message_id} from {queue_message.sender}")
    
    def _cleanup_expired_messages(self):
        """Clean up expired messages."""
        while self._is_running:
            try:
                current_time = datetime.now()
                expired_ids = []
                
                for msg_id, message in self._active_messages.items():
                    if message.expires_at and current_time > message.expires_at:
                        expired_ids.append(msg_id)
                
                # Mark expired messages
                for msg_id in expired_ids:
                    message = self._active_messages[msg_id]
                    message.status = MessageStatus.EXPIRED
                    self._stats['message_counts'][MessageStatus.EXPIRED.value] += 1
                
                if expired_ids:
                    print(f"🧹 Marked {len(expired_ids)} messages as expired")
                
                # Sleep for cleanup interval
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                print(f"⚠️ Error in cleanup: {e}")
                time.sleep(60)
    
    def _message_to_dict(self, message: QueueMessage) -> Dict[str, Any]:
        """Convert message to dictionary."""
        msg_dict = asdict(message)
        
        # Convert enums and dates
        msg_dict['priority'] = message.priority.name
        msg_dict['status'] = message.status.value
        msg_dict['created_at'] = message.created_at.isoformat()
        
        if message.sent_at:
            msg_dict['sent_at'] = message.sent_at.isoformat()
        if message.acknowledged_at:
            msg_dict['acknowledged_at'] = message.acknowledged_at.isoformat()
        if message.responded_at:
            msg_dict['responded_at'] = message.responded_at.isoformat()
        if message.expires_at:
            msg_dict['expires_at'] = message.expires_at.isoformat()
        
        return msg_dict
    
    def _dict_to_message(self, msg_dict: Dict[str, Any]) -> QueueMessage:
        """Convert dictionary to message."""
        # Parse dates
        created_at = datetime.fromisoformat(msg_dict['created_at'])
        sent_at = datetime.fromisoformat(msg_dict['sent_at']) if msg_dict.get('sent_at') else None
        acknowledged_at = datetime.fromisoformat(msg_dict['acknowledged_at']) if msg_dict.get('acknowledged_at') else None
        responded_at = datetime.fromisoformat(msg_dict['responded_at']) if msg_dict.get('responded_at') else None
        expires_at = datetime.fromisoformat(msg_dict['expires_at']) if msg_dict.get('expires_at') else None
        
        return QueueMessage(
            message_id=msg_dict['message_id'],
            content=msg_dict['content'],
            priority=QueuePriority[msg_dict['priority']],
            status=MessageStatus(msg_dict['status']),
            created_at=created_at,
            sender=msg_dict['sender'],
            recipient=msg_dict['recipient'],
            message_type=msg_dict['message_type'],
            sent_at=sent_at,
            acknowledged_at=acknowledged_at,
            responded_at=responded_at,
            expires_at=expires_at,
            retry_count=msg_dict.get('retry_count', 0),
            max_retries=msg_dict.get('max_retries', 3),
            requires_response=msg_dict.get('requires_response', False),
            response_content=msg_dict.get('response_content'),
            last_error=msg_dict.get('last_error'),
            error_count=msg_dict.get('error_count', 0)
        )
    
    def _load_persisted_messages(self):
        """Load messages from disk on startup."""
        try:
            # Load from outbound directory
            outbound_dir = self.queue_dir / "outbound"
            for message_file in outbound_dir.glob("*.json"):
                try:
                    with open(message_file, 'r') as f:
                        message_data = json.load(f)
                    
                    queue_message = self._dict_to_message(message_data)
                    self._active_messages[queue_message.message_id] = queue_message
                    
                    # Re-queue if not expired
                    if queue_message.expires_at and datetime.now() < queue_message.expires_at:
                        queue_message.status = MessageStatus.QUEUED
                        self._outbound_queue.put(
                            (-queue_message.priority.value, time.time(), queue_message),
                            block=False
                        )
                except:
                    pass  # Skip corrupted files
            
            print(f"📁 Loaded {len(self._active_messages)} persisted messages")
            
        except Exception as e:
            print(f"⚠️ Error loading persisted messages: {e}")
    
    def _persist_active_messages(self):
        """Persist active messages to disk."""
        try:
            # Save messages that need persistence
            for message in self._active_messages.values():
                if message.status in [MessageStatus.QUEUED, MessageStatus.SENT]:
                    message_dict = self._message_to_dict(message)
                    filename = f"{message.recipient}_{message.message_id}.json"
                    outbound_path = self.queue_dir / "outbound" / filename
                    
                    with open(outbound_path, 'w') as f:
                        json.dump(message_dict, f, indent=2, default=str)
            
            print(f"💾 Persisted active messages to disk")
            
        except Exception as e:
            print(f"⚠️ Error persisting messages: {e}")
    
    def _move_to_failed(self, message: QueueMessage):
        """Move failed message to failed directory."""
        try:
            message_dict = self._message_to_dict(message)
            filename = f"{message.recipient}_{message.message_id}_failed.json"
            failed_path = self.queue_dir / "failed" / filename
            
            with open(failed_path, 'w') as f:
                json.dump(message_dict, f, indent=2, default=str)
                
        except Exception as e:
            print(f"⚠️ Error moving message to failed directory: {e}")
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID."""
        return f"qmsg_{int(time.time() * 1000)}_{hash(datetime.now()) % 10000}"
    
    def get_queue_statistics(self) -> QueueStatistics:
        """Get queue statistics."""
        # Calculate averages
        avg_queue_time = 0.0
        if self._stats['queue_times']:
            avg_queue_time = sum(self._stats['queue_times']) / len(self._stats['queue_times'])
        
        avg_response_time = 0.0
        if self._stats['response_times']:
            avg_response_time = sum(self._stats['response_times']) / len(self._stats['response_times'])
        
        # Calculate success rate
        total_sent = self._stats['message_counts'][MessageStatus.SENT.value]
        total_failed = self._stats['message_counts'][MessageStatus.FAILED.value]
        success_rate = (total_sent / max(total_sent + total_failed, 1)) * 100
        
        return QueueStatistics(
            total_messages=self._stats['total_messages'],
            queued_messages=self._stats['message_counts'][MessageStatus.QUEUED.value],
            sent_messages=self._stats['message_counts'][MessageStatus.SENT.value],
            acknowledged_messages=self._stats['message_counts'][MessageStatus.ACKNOWLEDGED.value],
            responded_messages=self._stats['message_counts'][MessageStatus.RESPONDED.value],
            failed_messages=self._stats['message_counts'][MessageStatus.FAILED.value],
            expired_messages=self._stats['message_counts'][MessageStatus.EXPIRED.value],
            avg_queue_time_seconds=avg_queue_time,
            avg_response_time_seconds=avg_response_time,
            success_rate=success_rate
        )


# Convenience function for quick message sending
def send_quick_message(recipient: str, message_type: str, content: Dict[str, Any],
                      queue_dir: str = ".testmaster_queue") -> str:
    """Send a quick message without persistent queue."""
    queue = MessageQueue(queue_dir)
    queue.start()
    
    try:
        message_id = queue.send_message(recipient, message_type, content)
        time.sleep(1)  # Give time for processing
        return message_id
    finally:
        queue.stop()